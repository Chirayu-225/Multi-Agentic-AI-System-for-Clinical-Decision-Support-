"""The background job that analyses an uploaded report end to end."""
import asyncio
import logging
import secrets
import time
from datetime import datetime

from sqlalchemy.exc import IntegrityError

from src.agents import prompts
from src.agents.orchestrator import run_orchestrator
from src.agents.summarizer import directive_actions, generate_summaries, is_placeholder
from src.agents.translator import LANGUAGES, translate_patient_summary
from src.agents.verifier import verify_and_correct
from src.api.auth import Principal, ROLE_CLINICIAN, fingerprint
from src.core.config import env_num, require_review
from src.core.llm import begin_call_tracking, begin_model_tracking, llm_unavailable_for
from src.db.models import AgentRun, LlmCall, Patient, Report
from src.db.session import SessionLocal
from src.evidence.grounding import build_evidence, collect_allowed_numbers, find_ungrounded_numbers
from src.ingestion.document_loader import PDF_EXTS, load_document
from src.jobs import outbox, queue as jobq
from src.monitoring import observability as obs
from src.pipeline.history_sync import apply_history_event, history_text, outbox_config, outbox_result
from src.pipeline.progress import make_job_callback
from src.security import audit
from src.security.deidentification import redact, rehydrate
from src.security.llm_guardrails import detect_hidden_pdf_text, scan_output_for_leakage

logger = logging.getLogger("medsight")


LOW_OCR_CONFIDENCE = 0.6


def _call_cost_usd(tokens_in: int, tokens_out: int) -> float:
    """NOTIONAL cost: what these tokens would cost at paid-tier prices. On the free tier the real
    spend is zero; the number exists so a prompt change can be judged by what it would cost at
    scale. Prices are per million tokens and configurable (LLM_PRICE_INPUT_PER_M / _OUTPUT_PER_M)."""
    return round((tokens_in * env_num("LLM_PRICE_INPUT_PER_M", 0.30)
                  + tokens_out * env_num("LLM_PRICE_OUTPUT_PER_M", 2.50)) / 1_000_000, 8)


def _persist_llm_calls(report_id: str, calls: list[dict]) -> None:
    """Write the model-call trace. Best effort: tracing must never fail an upload."""
    if not calls:
        return
    db = SessionLocal()
    try:
        db.add_all([LlmCall(
            report_id=report_id, purpose=c["purpose"], model=c["model"],
            prompt_version=prompts.version_of(c["purpose"]), attempt=c["attempt"], status=c["status"],
            error_class=c["error_class"], latency_ms=c["latency_ms"], input_tokens=c["input_tokens"],
            output_tokens=c["output_tokens"], cost_usd=_call_cost_usd(c["input_tokens"], c["output_tokens"]),
        ) for c in list(calls)])
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.exception("could not record the model-call trace for %s", report_id)
    finally:
        db.close()


def _flag_severity(f: dict) -> str:
    return str(f.get("severity", "")).replace("Severity.", "").upper()


def save_preliminary(report_id: str, patient_id: str, filename: str | None, report_date: str,
                      source_type: str, pages: int, uploaded_by: str, data: dict) -> bool:
    """Write the deterministic results as a 'preliminary' report. Runs before any model call, so a
    critical value is on a clinician's screen within seconds of the upload and stays there if the
    model provider is down. Idempotent: returns False if a row already exists (a retry)."""
    db = SessionLocal()
    try:
        if db.query(Report.id).filter(Report.id == report_id).first():
            return False
        if not db.query(Patient.id).filter(Patient.id == patient_id).first():
            db.add(Patient(id=patient_id))
        flags = data.get("rule_flags", [])
        entities = data.get("entities", {})
        db.add(Report(
            id=report_id, patient_id=patient_id, filename=filename, report_date=report_date,
            source_type=source_type, pages=pages, entities=entities, rule_flags=flags,
            trend_flags=data.get("trend_flags", []), baseline_flags=data.get("baseline_flags", []),
            risk_summary=data.get("risk_summary", {}), lab_values=entities.get("lab_values", []),
            ingestion_warnings=list(data.get("ingestion_warnings", [])) + [
                "PRELIMINARY: lab flags and drug interactions only. The summaries have not been "
                "generated yet and this report cannot be approved until they are."],
            security_warnings=[], drug_interactions=data.get("drug_interactions", {}) or {},
            # Never released to a patient from here, whatever REQUIRE_CLINICIAN_REVIEW says.
            review_status="pending_review", analysis_status="preliminary",
            has_critical=any(_flag_severity(f) == "CRITICAL" for f in flags),
            uploaded_by=uploaded_by,
        ))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            return False
        obs.inc("reports_preliminary_total")
        return True
    finally:
        db.close()


def _defer_if_ai_down(job: jobq.JobSnapshot, report_id: str) -> None:
    """If the model provider is considered down, wait for it instead of producing a report with
    no summaries. The preliminary report is already visible meanwhile. After JOB_DEFER_MAX_SECONDS
    we stop waiting and finish with the degraded (clearly labelled) output rather than never."""
    wait = llm_unavailable_for()
    if wait <= 0:
        return
    limit = env_num("JOB_DEFER_MAX_SECONDS", 3600.0)
    age = (jobq.utcnow() - job.created_at).total_seconds() if job.created_at else 0.0
    if age >= limit:
        logger.warning("AI service still unavailable after %.0fs; finishing %s without waiting", age, report_id)
        return
    raise jobq.DeferJob(min(max(wait, 5.0), 300.0) + secrets.randbelow(5), "AI service unavailable")


async def process_upload(job: jobq.JobSnapshot, report_progress) -> None:
    """The analysis pipeline for one queued upload, run by a JobWorker with its own DB session.

    Failure handling lives in the worker: this raises, the worker retries with backoff and finally
    fails the job. Delivery is at-least-once, so this must be safe to run twice for the same
    upload: it returns early if an earlier attempt already saved the report, and the final write
    treats a primary-key clash (a second worker got there first) as success."""
    report_id, patient_id, ext = job.id, job.patient_id, job.ext or ""
    filename, report_date, language = job.filename, job.report_date or "", job.language
    tmp_path, ip = job.spool_path, job.ip or "unknown"
    principal = Principal(subject=job.owner, role=ROLE_CLINICIAN, patient_id=None)
    started = time.monotonic()
    models_used = begin_model_tracking()   # every model that answers during this upload
    llm_trace = begin_call_tracking(max_calls=env_num("LLM_MAX_CALLS_PER_REPORT", 12, int))
    caller_fp = fingerprint(principal.subject)
    ws_callback = make_job_callback(report_id, report_progress)
    db = SessionLocal()
    try:
        existing = db.query(Report.id, Report.analysis_status).filter(Report.id == report_id).first()
        if existing and (existing.analysis_status or "complete") != "preliminary":
            logger.info("report already saved by an earlier attempt; nothing to redo")
            return

        # ── Step 1: Ingest ─────────────────────────────────────────────────
        try:
            doc = await asyncio.to_thread(load_document, tmp_path)
        except Exception as exc:
            # An unreadable file will be just as unreadable on the next attempt: fail now.
            raise jobq.PermanentJobError("document could not be read") from exc
        raw_text = doc["raw_text"]
        raw_text = doc["raw_text"]
        clean_date = str(report_date).split(" ")[0].split("T")[0] if report_date else ""

        ingestion_extra: list[str] = []
        security_extra: list[str] = []
        conf = doc.get("ocr_confidence")
        if conf is not None and conf < LOW_OCR_CONFIDENCE:
            ingestion_extra.append(
                f"Low OCR confidence ({conf:.0%}) - digits may have been misread. "
                f"Verify every value against the original document."
            )
        if ext in PDF_EXTS:
            security_extra += await asyncio.to_thread(detect_hidden_pdf_text, tmp_path)

        # ── Step 2: Orchestrator agent (tool calling + ReAct) ──────────────
        ws_callback({"type": "pipeline_start", "report_id": report_id})

        agent_data = await run_orchestrator(
            report_text=raw_text,
            patient_id=patient_id,
            report_date=clean_date,
            on_step=ws_callback,
            on_preliminary=lambda snap: asyncio.to_thread(
                save_preliminary, report_id, patient_id, filename, clean_date,
                doc["source_type"], doc["pages"], principal.subject, snap),
        )

        # Pop the redaction map out immediately — it exists only to keep
        # patient identifiers out of Gemini calls (see deidentification.py)
        # and must never be logged, persisted, or included in agent_data
        # past this point. It's threaded explicitly into the next agent
        # calls, then consumed once by rehydrate() below.
        redaction_map = agent_data.pop("_redaction_map", {})

        # Log pipeline shape only — never patient content (names, lab
        # values, diagnoses, raw report text).
        logger.info(
            "orchestrator complete report_id=%s entity_keys=%s rule_flags=%d "
            "trend_flags=%d tools=%s",
            report_id,
            list(agent_data.get("entities", {}).keys()),
            len(agent_data.get("rule_flags", [])),
            len(agent_data.get("trend_flags", [])),
            [t["tool"] for t in agent_data.get("tool_calls_log", [])],
        )

        entities    = agent_data.get("entities", {})
        rule_flags  = agent_data.get("rule_flags", [])
        trend_flags = agent_data.get("trend_flags", [])
        baseline_flags = agent_data.get("baseline_flags", [])
        forecast_flags = agent_data.get("forecast_flags", [])
        similar_history = agent_data.get("similar_history", [])
        risk_summary = agent_data.get("risk_summary", {})
        pubmed_refs = agent_data.get("pubmed_references", [])
        drug_safety = agent_data.get("drug_safety", [])
        interactions = agent_data.get("drug_interactions", {}) or {}
        interaction_list = interactions.get("interactions", [])
        ingestion_warnings = list(agent_data.get("ingestion_warnings", [])) + ingestion_extra
        security_warnings = list(agent_data.get("security_warnings", [])) + security_extra
        llm_calls = int(agent_data.get("llm_calls", 0))

        # Provider down: keep the preliminary report, wait, and do the model steps later.
        _defer_if_ai_down(job, report_id)

        # ── Step 3: Generate summaries ─────────────────────────────────────
        # NOTE: summaries returned here still contain redaction placeholders
        # (e.g. "[REDACTED_NAME_1]") in place of the patient's real name —
        # see deidentification.py. Do not use these strings user-facing
        # until rehydrate() has run, below.
        ws_callback({"type": "generating_summaries"})
        summaries = await generate_summaries(
            entities, rule_flags, trend_flags, pubmed_refs, drug_safety, baseline_flags,
            redaction_map=redaction_map, drug_interactions=interaction_list,
        )
        llm_calls += 1

        # ── Step 4: Verification agent ─────────────────────────────────────
        ws_callback({"type": "verifier_start"})
        verified = await verify_and_correct(
            clinical_summary=summaries["clinical_summary"],
            patient_summary=summaries["patient_summary"],
            entities=entities,
            flags=rule_flags,
            trend_flags=trend_flags,
            redaction_map=redaction_map,
            drug_interactions=interaction_list,
        )
        llm_calls += 1
        ws_callback({"type": "verifier_complete",
                     "clinical_corrected": verified["clinical_corrected"],
                     "patient_corrected": verified["patient_corrected"]})

        # The provider may have failed during the steps above (they degrade instead of raising).
        _defer_if_ai_down(job, report_id)

        # ── Step 4b: deterministic numeric-grounding check ─────────────────
        # Independent of the (LLM) Verifier: every number in a summary must exist
        # somewhere in the structured ground truth.
        allowed = collect_allowed_numbers(
            entities, rule_flags, trend_flags, baseline_flags, forecast_flags,
            pubmed_refs, interaction_list, extra_texts=[clean_date],
        )
        grounding_warnings: list[str] = []
        for label, text in (("Clinical summary", verified["clinical_summary"]),
                            ("Patient summary", verified["patient_summary"]),
                            ("Longitudinal narrative", summaries.get("longitudinal_narrative", ""))):
            bad = find_ungrounded_numbers(text, allowed)
            if bad:
                grounding_warnings.append(
                    f"{label} contains number(s) not found in the extracted data: {', '.join(bad)}"
                )

        # A summary that was never produced, leaves out a critical result, or does not say it is urgent
        # is said in plain words at the top of the report, not only as a missing "verified" badge.
        for issue in verified.get("content_issues", []):
            ingestion_warnings.append(f"AI summary: {issue}")
        # Follow-up actions are not audited by the verifier. Ones worded as treatment orders are
        # kept as written (so the clinician sees what the model said) and flagged (SAFETY.md H18).
        orders = directive_actions(summaries.get("followup_actions", []))
        if orders:
            ingestion_warnings.append(
                f"AI follow-up actions: {len(orders)} of them read as treatment orders, not suggestions "
                f"(\"{orders[0][:80]}\"). They are AI suggestions only; treatment is the clinician's decision.")

        # ── Step 4c: optional translation of the (audited) patient summary ─
        translation = None
        if language != "en" and not is_placeholder(verified["patient_summary"]):
            ws_callback({"type": "translating", "language": language})
            translation = await translate_patient_summary(verified["patient_summary"], language)
            llm_calls += 1
            if not translation["ok"]:
                ingestion_warnings.append(
                    f"{LANGUAGES[language]['name']} translation was rejected by automated checks "
                    f"({'; '.join(translation['issues'])}) - the English summary is shown instead."
                )

        # Rehydrate — the ONE point in the whole pipeline where redaction
        # placeholders are converted back to the patient's real identifiers,
        # now that every Gemini call for this report is done. Everything
        # from here on (DB row, API response) uses real values again — Gemini
        # itself just never saw them.
        if redaction_map:
            verified["clinical_summary"] = rehydrate(verified["clinical_summary"], redaction_map)
            verified["patient_summary"] = rehydrate(verified["patient_summary"], redaction_map)
            summaries["longitudinal_narrative"] = rehydrate(
                summaries.get("longitudinal_narrative", ""), redaction_map
            )
            summaries["followup_actions"] = [
                rehydrate(a, redaction_map) for a in summaries.get("followup_actions", [])
            ]
            summaries["explained_flags"] = [
                {**f, "explanation": rehydrate(f.get("explanation", ""), redaction_map)}
                for f in summaries.get("explained_flags", [])
            ]
            if translation and translation.get("text"):
                translation["text"] = rehydrate(translation["text"], redaction_map)

        # Output-side guardrail: scan the (now-verified) summaries for signs
        # the model complied with an injected instruction. Independent of
        # the Verifier agent's hallucination audit above — different threat
        # model (adversarial input vs. ordinary model error).
        security_warnings += [
            f"Clinical summary: {w}" for w in scan_output_for_leakage(verified["clinical_summary"])
        ]
        security_warnings += [
            f"Patient summary: {w}" for w in scan_output_for_leakage(verified["patient_summary"])
        ]
        if security_warnings:
            logger.warning(
                "security_warnings report_id=%s caller=%s count=%d",
                report_id, caller_fp, len(security_warnings),
            )
            audit.record(actor=principal.subject, role=principal.role, action="security.injection_suspected",
                         resource=f"report:{report_id}", ip=ip, outcome="success",
                         detail={"warnings": len(security_warnings)})

        # ── Evidence: link each extracted fact to its source line ──────────
        evidence = build_evidence(
            raw_text, entities, rule_flags,
            summaries={"clinical": verified["clinical_summary"], "patient": verified["patient_summary"]},
        )

        # ── Step 5: Persist ────────────────────────────────────────────────
        ws_callback({"type": "saving"})
        patient =db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            patient = Patient(id=patient_id)
            db.add(patient)

        review_required = require_review()
        lab_values = entities.get("lab_values", [])
        ver_blob = {
            "clinical_corrected": verified["clinical_corrected"],
            "patient_corrected": verified["patient_corrected"],
            "clinical_unverified": verified.get("clinical_unverified", False),
            "patient_unverified": verified.get("patient_unverified", False),
            "clinical_issues": verified.get("clinical_issues", ""),
            "patient_issues": verified.get("patient_issues", ""),
            "clinical_missing_critical": verified.get("clinical_missing_critical", []),
            "content_issues": verified.get("content_issues", []),
            "directive_actions": orders,
        }
        # The preliminary row (saved earlier in this attempt or in a deferred one) is replaced in
        # place: same id, original timestamp.
        prelim = db.query(Report.created_at).filter(
            Report.id == report_id, Report.analysis_status == "preliminary").first()
        prelim_created_at = prelim.created_at if prelim else None
        if prelim:
            db.query(Report).filter(Report.id == report_id, Report.analysis_status == "preliminary").delete(
                synchronize_session=False)
            db.flush()
        report = Report(
            id=report_id, created_at=prelim_created_at or datetime.utcnow(),
            analysis_status="complete",
            has_critical=any(_flag_severity(f) == "CRITICAL" for f in rule_flags),
            patient_id=patient_id, filename=filename, report_date=clean_date,
            source_type=doc["source_type"], has_hindi=str(doc["has_hindi"]), pages=doc["pages"],
            entities=entities, rule_flags=rule_flags, trend_flags=trend_flags,
            baseline_flags=baseline_flags, forecast_flags=forecast_flags,
            similar_history=similar_history, risk_summary=risk_summary, lab_values=lab_values,
            ingestion_warnings=ingestion_warnings, security_warnings=security_warnings,
            clinical_summary=verified["clinical_summary"], patient_summary=verified["patient_summary"],
            explained_flags=summaries["explained_flags"], followup_actions=summaries["followup_actions"],
            longitudinal_narrative=summaries.get("longitudinal_narrative", ""),
            verification=ver_blob,
            review_status="pending_review" if review_required else "approved",
            reviewed_by=None if review_required else "auto (review not required)",
            reviewed_at=None if review_required else datetime.utcnow(),
            uploaded_by=principal.subject,
            evidence=evidence, grounding_warnings=grounding_warnings,
            drug_interactions=interactions, drug_safety=drug_safety, pubmed_references=pubmed_refs,
            patient_summary_translated=translation["text"] if translation and translation.get("text") else None,
            translation_language=language if translation and translation.get("text") else None,
            translation_check=({"ok": translation["ok"], "issues": translation["issues"]}
                               if translation and translation.get("text") else None),
        )
        db.add(report)
        db.flush()  # agent_runs.report_id is a FK; no ORM relationship orders the inserts
        outbox.add(db, kind=outbox.HISTORY_INDEX, report_id=report_id, patient_id=patient_id, payload={
            # Redacted text only: names, phone numbers, MRNs etc. never reach the vector store.
            "text": history_text(redact(raw_text, redaction_map)), "lab_values": lab_values,
            "report_date": clean_date})
        db.add(AgentRun(
            report_id=report_id, agent_name="orchestrator+summarizer+verifier",
            tool_calls=agent_data.get("tool_calls_log", []),
            was_corrected=str(verified["clinical_corrected"] or verified["patient_corrected"]),
            caller_fingerprint=caller_fp,
            duration_ms=int((time.monotonic() - started) * 1000), llm_calls=llm_calls,
            # e.g. "gemini-3.8-flash" or, if the fallback answered any call, both models
            model=",".join(sorted(models_used)) or (agent_data.get("token_usage") or {}).get("model"),
            input_tokens=sum(c["input_tokens"] for c in llm_trace),
            output_tokens=sum(c["output_tokens"] for c in llm_trace),
            prompt_versions={k: f"{k}@{v}" for k, v in prompts.current_versions().items()},
        ))
        try:
            db.commit()
        except IntegrityError:
            # Another worker saved this report first (lease handed over mid-run). That is a success.
            db.rollback()
            if db.query(Report.id).filter(Report.id == report_id).first():
                logger.warning("report saved concurrently by another worker; finishing")
                return
            raise

        # ── Step 6: apply the history-store write staged in the outbox ──────
        # Best effort right now; if it fails the event stays in the outbox and the relay retries it
        # (a failure here must never become a 500: the report is saved and a re-upload would
        # duplicate it). outbox_result records the outcome on the report.
        cfg = outbox_config()
        await asyncio.to_thread(
            outbox.drain, SessionLocal, apply_history_event, lease_seconds=cfg["lease_seconds"],
            max_attempts=cfg["max_attempts"], base_seconds=cfg["base_seconds"], report_id=report_id,
            on_result=outbox_result)

        audit.record(actor=principal.subject, role=principal.role, action="report.upload",
                     resource=f"report:{report_id}", ip=ip,
                     detail={"patient_id": patient_id, "source_type": doc["source_type"],
                             "pages": doc["pages"], "llm_calls": llm_calls,
                             "review_status": report.review_status})
        obs.inc("reports_analysed_total")
    finally:
        _persist_llm_calls(report_id, llm_trace)    # also on failure, deferral and cancellation
        db.close()
