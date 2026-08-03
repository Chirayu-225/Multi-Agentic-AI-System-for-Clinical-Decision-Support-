"""
src/agents/orchestrator.py

Orchestrator Agent — the primary agent in MedSight's multi-agent system.

Uses Gemini 2.5 Flash with native tool calling in a ReAct loop:
  - Receives raw report text + patient context
  - Decides which tools to invoke and in what order
  - Conditionally calls PubMed/OpenFDA based on what it finds
  - Streams its reasoning steps via a callback (for WebSocket UI)

Memory: patient history is retrieved via ChromaDB tool call,
giving the agent longitudinal context across visits.
"""

import os
import json
import asyncio
from typing import Callable

import google.generativeai as genai

from src.tools.tool_registry import TOOL_FUNCTIONS, TOOL_DEFINITIONS


def _to_plain(obj):
    """
    Recursively convert Gemini's protobuf wrapper types (MapComposite for
    nested objects, RepeatedComposite for lists) into plain Python
    dict/list/str/float/etc.

    dict(fc.args) only converts the TOP level of a function-call's
    arguments — any nested list or object inside those args stays a
    protobuf composite type, which json.dumps() (and therefore
    SQLAlchemy's JSON column type) cannot serialize.
    """
    if hasattr(obj, "items"):
        return {k: _to_plain(v) for k, v in obj.items()}
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
        return [_to_plain(v) for v in obj]
    return obj

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

ORCHESTRATOR_SYSTEM_PROMPT = """You are a clinical decision support agent for Indian healthcare settings.

You have access to the following tools:
- extract_entities: extract patient info, diagnoses, medications, lab values from report text
- check_icmr_flags: check lab values against ICMR Indian reference ranges
- detect_trends: check for worsening trends across patient visits
- get_patient_history: retrieve patient's prior reports
- search_pubmed: find clinical guidelines for detected conditions (ONLY call if significant findings found)
- check_drug_safety: check medication safety warnings (ONLY call if medications are detected)

Your job:
1. Always start by extracting entities from the report text
2. Always check ICMR flags on extracted lab values
3. Always check trends if the patient has prior history
4. Only call search_pubmed if you find HIGH or CRITICAL flags — search for the most clinically significant finding
5. Only call check_drug_safety if medications were detected in the report
6. After gathering all data, produce a structured clinical analysis

Be precise. Only reference values that actually exist in the extracted data.
Do not invent clinical values, conditions, or medications not present in the report.
"""


def _build_gemini_tools() -> list:
    """Build Gemini tools using the installed SDK's expected format."""
    import google.generativeai as genai

    declarations = []
    for tool_def in TOOL_DEFINITIONS:
        declarations.append(
            genai.protos.FunctionDeclaration(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        k: _schema_to_proto(v)
                        for k, v in tool_def["parameters"].get("properties", {}).items()
                    },
                    required=tool_def["parameters"].get("required", []),
                ),
            )
        )
    return [genai.protos.Tool(function_declarations=declarations)]


def _schema_to_proto(schema: dict):
    """Convert a JSON schema dict to a Gemini protos.Schema object."""
    import google.generativeai as genai

    type_map = {
        "string":  genai.protos.Type.STRING,
        "integer": genai.protos.Type.INTEGER,
        "number":  genai.protos.Type.NUMBER,
        "boolean": genai.protos.Type.BOOLEAN,
        "array":   genai.protos.Type.ARRAY,
        "object":  genai.protos.Type.OBJECT,
    }
    proto_type = type_map.get(schema.get("type", "string"), genai.protos.Type.STRING)

    kwargs = {
        "type": proto_type,
        "description": schema.get("description", ""),
    }

    if proto_type == genai.protos.Type.ARRAY and "items" in schema:
        kwargs["items"] = _schema_to_proto(schema["items"])

    return genai.protos.Schema(**kwargs)


async def _execute_tool(tool_name: str, tool_args: dict) -> dict:
    """Execute a tool by name and return its result."""
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        result = await TOOL_FUNCTIONS[tool_name](**tool_args)
        return result if isinstance(result, dict) else {"result": result}
    except Exception as e:
        return {"error": str(e), "tool": tool_name}


async def run_orchestrator(
    report_text: str,
    patient_id: str,
    report_date: str = "",
    on_step: Callable[[dict], None] = None,
) -> dict:
    """
    Run the orchestrator agent on a medical report.

    Args:
        report_text:  extracted text from the PDF
        patient_id:   patient identifier for history lookup
        report_date:  report date string YYYY-MM-DD
        on_step:      optional callback called on each agent step
                      (used to stream progress to WebSocket clients)

    Returns:
        {
            entities, rule_flags, trend_flags, risk_summary,
            pubmed_references, drug_safety,
            tool_calls_log, token_usage
        }
    """
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=ORCHESTRATOR_SYSTEM_PROMPT,
        tools=_build_gemini_tools(),
    )

    # Initial message to the agent
    initial_message = f"""Analyse this medical report for patient {patient_id}.

Report date: {report_date}

REPORT TEXT:
{report_text[:4000]}

Patient ID for history lookup: {patient_id}
"""

    chat = model.start_chat()
    tool_calls_log = []
    accumulated_data = {
        "entities": {},
        "rule_flags": [],
        "trend_flags": [],
        "risk_summary": {},
        "pubmed_references": [],
        "drug_safety": [],
    }

    # ── ReAct loop ────────────────────────────────────────────────────────────
    response = await asyncio.to_thread(chat.send_message, initial_message)
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Check if there are tool calls in the response
        tool_calls_in_response = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool_calls_in_response.append(part.function_call)

        if not tool_calls_in_response:
            # No more tool calls — agent is done reasoning
            break

        # Execute all tool calls in this response
        tool_results = []
        for fc in tool_calls_in_response:
            tool_name = fc.name
            tool_args = _to_plain(fc.args)

            if on_step:
                on_step({"type": "tool_call", "tool": tool_name, "args": tool_args})

            result = await _execute_tool(tool_name, tool_args)

            # Store results in accumulated data
            if tool_name == "extract_entities":
                accumulated_data["entities"] = result
            elif tool_name == "check_icmr_flags":
                accumulated_data["rule_flags"] = result.get("flags", [])
                accumulated_data["risk_summary"] = result.get("summary", {})
            elif tool_name == "detect_trends":
                accumulated_data["trend_flags"] = result if isinstance(result, list) else []
            elif tool_name == "search_pubmed":
                refs = result if isinstance(result, list) else [result]
                accumulated_data["pubmed_references"].extend(refs)
            elif tool_name == "check_drug_safety":
                accumulated_data["drug_safety"].append(result)

            tool_calls_log.append({
                "tool": tool_name,
                "args": tool_args,
                "result_summary": str(result)[:200],
            })

            if on_step:
                on_step({"type": "tool_result", "tool": tool_name,
                         "result": str(result)[:200]})

            tool_results.append({
                "function_response": {
                    "name": tool_name,
                    "response": result,
                }
            })

        # Send tool results back to the agent
        response = await asyncio.to_thread(
            chat.send_message,
            tool_results,
        )

    accumulated_data["tool_calls_log"] = tool_calls_log
    accumulated_data["token_usage"] = {
        "model": MODEL,
        "iterations": iteration,
    }

    if on_step:
        on_step({"type": "orchestrator_complete"})

    return accumulated_data