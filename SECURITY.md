# Security

This is a final-year academic/portfolio project, not a certified production system —
nothing here is a claim of HIPAA, DPDP, or GDPR compliance. What follows is an honest map
of which real AI-security risks this codebase mitigates, how, how each claim is tested, and
where it deliberately doesn't, so the security work is legible on its own terms.

## Threat model

The attack surface an ordinary web-app checklist misses: this app's "user input" is
free-text PDF/photo content pasted into an LLM's context, which then drives real tool
calls (external API requests, database writes) and real clinical output shown to a human.
The mitigations target: prompt injection through a tampered document, sensitive data
leaving the system in a third-party model call, an agent doing more than it was asked, the
model being wrong with confidence, and unauthorised or unattributable access to patient
data.

## Mapped to the OWASP Top 10 for LLM Applications (2026 edition)

Uses the 2026 numbering (verify against the current
[OWASP list](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
before citing; it is revised periodically).

| # | Risk | Status | Mitigation and evidence |
|---|------|--------|-------------------------|
| LLM01 | Prompt Injection | **Implemented, measured** | `llm_guardrails.py`: text is NFKC-normalised and zero-width/bidi characters stripped before 20+ pattern checks (override, role markers, exfiltration, non-English); line-wrapped phrases are caught; untrusted text is delimited with an instruction-hierarchy reminder; PyMuPDF detects hidden PDF text (white, sub-2pt, off-page). Detected and surfaced to the clinician, not silently blocked. **Evidence:** 31-payload adversarial corpus 100% detected, 0% false positives on 16 benign clinical strings and on 200 synthetic reports (`tests/adversarial_corpus.py`, `EVAL_RESULTS.md`). **Known misses, pinned as tests:** base64-encoded, leetspeak, paraphrased, and field-split payloads are not detected. A pattern list cannot stop a determined attacker; it raises cost and informs the reviewer. |
| LLM02 | Sensitive Information Disclosure | **Implemented, measured** | `deidentification.py` replaces name, treating doctor, hospital, phone, email, ID numbers with placeholders before any Gemini prompt, and restores them once server-side. Matching is case-insensitive, whitespace-flexible and word-bounded; a label-based fallback and a letterhead heuristic cover NER misses. Translation runs on the redacted text. Audit events never contain PHI. **Evidence:** 0% of identifiers survive when the NER finds them; **14% of reports leak a name when the NER misses it** (regex-only worst case, `EVAL_RESULTS.md`). OCR is local EasyOCR only: a hosted vision model would bypass de-identification, so none is used. FHIR export omits the patient name unless explicitly requested. |
| LLM03 | Excessive Agency | **Implemented** | Extraction, ICMR flagging, trends and the drug-interaction check are a fixed deterministic pipeline. The model only chooses among three read-only tools, with sanitised arguments, deduplicated and capped. **No model output reaches a patient without a named clinician approving it** (see Access control). |
| LLM04 | Supply Chain | **Implemented, with documented exceptions** | CI (`.github/workflows/security.yml`) runs `pip-audit`, `npm audit`, `bandit`, `gitleaks`, plus Dependabot and a pre-commit config. Unused dependencies were removed (aiohttp, pandas) and vulnerable ones bumped. See "Dependency audit" below for what is accepted and why. |
| LLM05 | Data and Model Poisoning | **Not applicable / partial** | Hosted model, no fine-tuning. Each patient's ChromaDB history is queried only by that `patient_id`. |
| LLM06 | Unbounded Consumption | **Partial** | Per-caller upload limit, per-IP and per-username login limits, upload size cap (20 MB), 40M-pixel image cap, text and chunk caps, retry caps, and a tool-call dedup guard. In-process only: a multi-instance deployment needs a shared store. |
| LLM07 | Misinformation | **Implemented, measured** | Layered: (1) the Verifier audits both summaries against ground truth and fails closed; (2) a deterministic **numeric grounding** check flags any number in a summary that is absent from the extracted data; (3) **evidence linking** maps each extracted fact to its source line so a clinician can check it; (4) mandatory human review. **Evidence:** the numeric check catches 100% of novel values and 99.5% of fabricated labs with 0% false alarms on clean summaries, but **0% of value swaps and wrong-direction statements**, because they reuse numbers already present. Only the LLM Verifier addresses those, and its accuracy is not measured offline (`--live` mode exists but needs a key). |
| LLM08 | Hidden Context Exposure | **Partial** | Structural delimiting, and output scanning for prompt-probing shapes. No further defence. |
| LLM09 | Vector and Embedding Weaknesses | **Partial** | Queries scoped by `patient_id` by construction. Not adversarially tested against embedding-space attacks. |
| LLM10 | Improper Output Handling | **Implemented, enforced in CI** | The Angular client renders model and server text only with `{{ }}` interpolation. Evidence highlighting splits the snippet into three text nodes in TypeScript. CI fails the build if `innerHTML`, `bypassSecurityTrust*`, `DomSanitizer`, `eval(` or `document.write` appears under `frontend/src`. Output is also scanned for external URLs (a markdown-image exfiltration channel). Translations are validated before display (see below). |

## Access control

- **Accounts and roles.** JWT bearer login (HS256, algorithm pinned, issuer and expiry
  required, per-token `jti`). Roles: **clinician** (everything, and the only role that can
  approve/reject, read the audit log or metrics, or delete a patient), **patient** (read
  only, bound to one `patient_id`, sees approved reports only), **service** (the legacy
  shared `X-API-Key`: may upload and read, can never approve or read audit data;
  disable with `ALLOW_API_KEY_AUTH=false`).
- **Passwords.** scrypt with per-user salt, constant-time verification with a dummy hash
  for unknown users (no user enumeration by timing), minimum length 10, lockout for 15
  minutes after 5 failures (HTTP 423), and per-IP and per-username login rate limits.
- **Object-level authorisation.** A patient asking for another patient's report, or an
  unapproved one, gets 404, not 403, so existence is not leaked. Tested end to end in
  `tests/test_api_flow.py` (about 45 API tests).
- **Fail closed.** No credentials configured means no access, not open access. A missing
  `JWT_SECRET` produces a random per-process secret with a warning; a secret under 32
  characters is refused.
- **Human-in-the-loop release gate.** New reports are `pending_review`. Legacy rows with
  no status are treated as pending. Approve/reject is written to the audit log **first**
  with `critical=True`: if the audit write fails, the approval does not happen. Rejection
  requires a note.

## Audit log

Table `audit_events` is hash-chained: `hash_n = sha256(hash_{n-1} || canonical_json(event))`.
`GET /admin/audit/verify` recomputes the chain and reports the first bad row. It detects
edited rows and deletions in the middle of the chain.

**Limits, stated plainly:** (1) someone with database write access who *truncates the tail*
or rewrites the whole chain consistently cannot be detected from inside the database. The
verify endpoint returns `head_hash` so you can record it somewhere the DB admin cannot
edit; (2) this is tamper-evident, not tamper-proof; (3) events contain actor, action,
resource id and outcome, never PHI.

## Translation safety

Patient summaries can be produced in Hindi, Kannada, Tamil, Telugu, Marathi and Bengali.
The English summary is audited by the Verifier first; the translation is made from the
redacted text and then validated **deterministically**: the multiset of numbers must equal
the English text's, placeholders must be preserved, the text must be mostly in the target
script, and it must contain no URL or injection-shaped content. A failed check discards
the translation and the English summary is shown. This does **not** verify medical
accuracy of the wording; the UI labels it machine-translated.

## Medication checks

`tools/drug_interactions.py` is a curated, deterministic rule set (16 drug-drug rules,
drug-lab rules, single-drug advisories, brand-to-generic map). It is **illustrative, not a
licensed interaction database**, it cannot see doses, and an empty result is not a
guarantee of safety; the UI says so. Allergy checking is not implemented.

## FHIR export

`GET /report/{id}/fhir` returns a FHIR R4 Bundle (Patient, Observation with LOINC/UCUM,
Condition, MedicationStatement, DiagnosticReport). The report is `preliminary` until a
clinician approves it. Verified structurally in tests only, **not validated with the HL7
validator**.

## De-identification: scope and limits

Detection is regex plus GLiNER entity extraction, not a dedicated PII library (use
[Presidio](https://microsoft.github.io/presidio/) for real PHI). Not HIPAA Safe Harbor:
age and gender are kept as clinical context, ages 90+ are not specially handled, and
free-text dates, addresses and device identifiers beyond the ID-number pattern are not
covered. The measured worst case is a 14% name leak when the NER fails.

## Dependency audit

CI gates: `pip-audit` (fails on anything new), `npm audit --omit=dev --audit-level=critical`,
`bandit -ll`, `gitleaks`.

- **Python:** the ML stack (`torch`, `transformers`, `sentence-transformers`, `chromadb`)
  has open advisories listed in `backend/.pip-audit-ignore`. These versions are pinned
  together because GLiNER and the embedding model were validated against them; upgrading
  needs a re-test of extraction quality on a machine with the models, which CI cannot do.
  They process only text the server itself extracted and are not exposed to network input
  directly. The risk is accepted until that upgrade; the list must shrink, not grow.
- **Frontend:** the production-dependency critical gate passes. The full tree still shows
  77 advisories (4 critical, all in dev tooling that is not shipped) and Angular 18.2.x has
  advisories fixed only in a newer major version. `npm audit fix --force` would cause a
  breaking upgrade, so this is deferred as a planned migration.

## Known gaps

- **JWTs cannot be revoked** before they expire (default 60 minutes). The SPA keeps the
  token in `sessionStorage`, so an XSS bug would expose it; hence the CI sink check.
  A production system would use short-lived tokens with a refresh/revocation list or
  httpOnly cookies.
- **The shared service key** is a bearer secret that, if baked into a frontend bundle, is
  public. Leave it out of the production build and set `ALLOW_API_KEY_AUTH=false`.
- **Evaluation data is synthetic and self-authored.** The numbers show where logic holds
  or breaks on controlled inputs; they are not clinical validation, and not a benchmark
  against third-party data.
- **Value swaps** (a summary that swaps two real values) are invisible to the numeric
  check; only the LLM Verifier addresses them.
- **Rate limiting** and lockout counters are in process memory (single instance only).
- **No TLS in the repo.** Terminate TLS in front of nginx; the WebSocket carries its token
  as a query parameter, which can appear in proxy logs.
- **Audit tail truncation** is undetectable without an external anchor (see above).
- **No multi-tenancy or consent management**, no data-retention policy, no encryption at
  rest beyond what the database or disk provides.
- **Frontend not browser-tested** by an automated suite; it is verified by a production
  build and the CI sink check.
