# Hazard log

What can go wrong, how badly, what stops it, and which automated test proves the control still
works. `backend/tests/test_safety_log.py` fails the build if a hazard has no test, if a referenced
test no longer exists, or if a hazard lacks a stated residual risk. A control without a test is a
hope, not a control.

Scope and honesty: this is a decision-support prototype run on synthetic data. It is not a medical
device and has had no clinical validation. Severity is the worst plausible harm if the control
failed. "Residual risk" is what the control does NOT cover.

## H1 - A critical result is not seen promptly
- Severity: catastrophic
- Cause: model provider slow or down; the report waits behind AI steps.
- Controls: lab flags are computed locally and saved as a preliminary report before any model call;
  critical reports sort first in the review queue; a circuit breaker stops queued reports from each
  waiting out their own retries.
- Tests: `tests/test_fast_path.py::test_preliminary_report_exists_before_the_first_model_step`, `tests/test_fast_path.py::test_provider_outage_defers_the_job_and_keeps_the_critical_result_visible`, `tests/test_fast_path.py::test_critical_reports_are_listed_first_in_the_review_queue`, `tests/test_resilience.py::test_with_retry_trips_the_breaker_and_later_calls_fail_without_calling_the_provider`
- Residual risk: a clinician must actually look at the queue; there is no paging or push alert.
  These controls make the flags visible; H16 and H17 cover the banner and the summary beside them.

## H2 - Unreviewed or incomplete AI output reaches a patient
- Severity: major
- Cause: a report is released before a clinician approves it, or before its summaries exist.
- Controls: every report is pending until a named clinician approves; patients cannot read pending
  reports; rows without a status are treated as pending; a preliminary report cannot be approved;
  warnings must be acknowledged.
- Tests: `tests/test_review_flow.py::test_approval_blocked_until_warnings_acknowledged`, `tests/test_api_flow.py::test_patient_cannot_see_unapproved_report_then_can_after_approval`, `tests/test_api_flow.py::test_legacy_report_without_status_is_treated_as_pending`, `tests/test_fast_path.py::test_provider_outage_defers_the_job_and_keeps_the_critical_result_visible`
- Residual risk: approval is only as careful as the clinician; REQUIRE_CLINICIAN_REVIEW=false
  disables the gate by design for demos.

## H3 - A summary states a number that is not in the report
- Severity: major
- Cause: model hallucination.
- Controls: a deterministic check that every number in a summary exists in the structured data;
  an independent LLM verifier.
- Tests: `tests/test_api_flow.py::test_fabricated_number_is_caught_by_grounding`, `tests/test_clinical_modules.py::test_ungrounded_number_detected_and_grounded_ignored`
- Residual risk: the numeric check cannot see a value swapped between two tests or a wrong
  direction ("high" for "low"); only the verifier can, and it is itself a model.

## H4 - The verifier fails silently and the summary looks audited
- Severity: major
- Cause: malformed or missing verifier output, rate limit, provider error.
- Controls: structured output validated against a schema; one repair attempt; anything still
  invalid, or any provider error, marks the summary unverified and forces acknowledgement.
- Tests: `tests/test_verifier_structured.py::test_still_invalid_after_the_repair_fails_closed_and_never_calls_a_third_time`, `tests/test_verifier_structured.py::test_fail_without_a_correction_is_invalid_and_gets_repaired`, `tests/test_verifier_structured.py::test_provider_error_fails_closed`
- Residual risk: a verifier that answers well-formed but wrong is not detected here (see H13).

## H5 - The wrong reference range is applied
- Severity: major
- Cause: sex or age missing or written unusually; child judged against adult ranges.
- Controls: normalised demographics; unknown sex uses the widest range and says so; paediatric
  ranges; disagreement with the lab's printed range is flagged.
- Tests: `tests/test_range_selection.py::test_unknown_sex_never_silently_uses_male_range`, `tests/test_range_selection.py::test_child_gets_paediatric_range`, `tests/test_range_selection.py::test_printed_range_discordance_is_flagged`
- Residual risk: tests without a paediatric range fall back to adult ranges (labelled as such).

## H6 - "Nothing was read" looks like "everything is normal"
- Severity: major
- Cause: OCR or parsing extracts no values; an empty flag list renders as all clear.
- Controls: an explicit warning when no numeric lab values were extracted.
- Tests: `tests/test_pipeline_integration.py::test_empty_extraction_is_never_reported_as_all_clear`
- Residual risk: a partially extracted report shows only what was read; missing rows are not listed.

## H7 - A dangerous drug interaction is missed
- Severity: major
- Cause: brand names, unrecognised medications.
- Controls: brand-to-generic resolution; unrecognised medications are reported, not ignored.
- Tests: `tests/test_clinical_modules.py::test_major_drug_drug_interactions_detected_via_brand_names`, `tests/test_clinical_modules.py::test_unrecognized_medications_are_reported_not_silently_ignored`
- Residual risk: the rule set is curated and small, not a drug database.

## H8 - Patient identifiers leak to the model, the vector store or the audit log
- Severity: major
- Cause: names, phone numbers and IDs in report text or filenames.
- Controls: redaction before every model call; redacted text only in the vector store and the
  outbox; audit events hold identifiers and counts, never content.
- Tests: `tests/test_deidentification.py::test_no_identifier_survives_redaction`, `tests/test_erasure.py::test_upload_stores_redacted_text_only`, `tests/test_api_flow.py::test_audit_never_contains_phi`, `tests/test_outbox.py::test_payload_holds_redacted_text_only`
- Residual risk: NER can miss a name it does not recognise; regex-only redaction is measured as
  weaker in eval/run_eval.py.

## H9 - Instructions hidden in a report steer the model
- Severity: major
- Cause: prompt injection in uploaded documents, including white or tiny text in PDFs.
- Controls: input scanning, untrusted-data delimiters, output scanning, hidden-text detection.
- Tests: `tests/test_guardrails_adversarial.py::test_attack_is_detected`, `tests/test_guardrails_adversarial.py::test_hidden_white_text_detected`, `tests/test_api_flow.py::test_hidden_white_text_in_pdf_is_flagged`
- Residual risk: known evasions are documented in the adversarial corpus, not eliminated.

## H10 - A report is lost or analysed twice when a process crashes
- Severity: major
- Cause: crash, restart or lease handover mid-analysis.
- Controls: durable database queue with leases and heartbeats; compare-and-swap claims; idempotent
  final write.
- Tests: `tests/test_job_queue.py::test_a_job_left_running_by_a_crashed_process_is_picked_up_after_restart`, `tests/test_job_queue.py::test_each_job_is_claimed_by_exactly_one_of_many_concurrent_workers`, `tests/test_job_pipeline_behaviour.py::test_transient_failure_is_retried_and_the_report_is_saved_exactly_once`
- Residual risk: delivery is at-least-once, so model work can be repeated (costing quota).

## H11 - The audit trail is altered, forked or silently incomplete
- Severity: major
- Cause: tampering, concurrent writers, an unwritable audit store.
- Controls: hash chain with a unique-predecessor constraint; critical actions fail closed when the
  audit record cannot be written.
- Tests: `tests/test_api_flow.py::test_editing_an_audit_row_breaks_the_chain`, `tests/test_audit_chain_concurrency.py::test_concurrent_appends_never_fork_the_chain`, `tests/test_api_flow.py::test_critical_action_fails_closed_when_audit_unwritable`
- Residual risk: truncating the tail of the chain is only caught with an external anchor.

## H12 - A rejected or erased report keeps influencing a patient's history
- Severity: major
- Cause: the history-store write is not in the database transaction and can land late.
- Controls: transactional outbox; rejection cancels pending writes; a write racing a rejection is
  undone; erasure deletes outbox rows and vectors.
- Tests: `tests/test_outbox.py::test_rejection_cancels_a_pending_write_and_nothing_is_stored`, `tests/test_outbox.py::test_write_that_races_with_a_rejection_is_undone`, `tests/test_outbox.py::test_patient_erasure_removes_outbox_events`, `tests/test_erasure.py::test_delete_patient_clears_history_store`
- Residual risk: none known beyond backups, which are outside this system.

## H13 - A prompt or model change makes quality worse and nobody notices
- Severity: major
- Cause: editing a prompt, upgrading or swapping the model.
- Controls: every call records its prompt version and model; a small frozen evaluation set run on
  demand before such a change, with a regression verdict; clinician edit rates tracked per version.
- Tests: `tests/test_frozen_gate.py::test_a_worse_verifier_is_reported_as_a_regression_and_does_not_update_the_baseline`, `tests/test_llm_tracing.py::test_calls_are_persisted_with_versions_tokens_and_notional_cost`, `tests/test_ai_quality.py::test_ai_quality_endpoint_groups_by_the_prompt_version_the_report_was_made_with`
- Residual risk: the gate is on demand, so it only works if it is run; 8 synthetic cases show
  direction, not significance.

## H14 - Runaway model use exhausts the free quota and stops the service
- Severity: moderate
- Cause: a retry or tool-calling loop; many workers calling at once.
- Controls: per-report call budget; shared rate limiter; circuit breaker.
- Tests: `tests/test_llm_tracing.py::test_per_report_budget_stops_a_runaway_loop`, `tests/test_resilience.py::test_rate_limiter_is_shared_by_concurrent_callers`
- Residual risk: the rate limiter is per process; several processes share one key's quota unaware
  of each other.

## H15 - A report is saved but missing from the patient's trend history
- Severity: moderate
- Cause: the vector store is down or errors at write time.
- Controls: the write is retried from the outbox; the report shows a warning until it succeeds, and
  a different one if it is given up on.
- Tests: `tests/test_outbox.py::test_failed_write_stays_queued_with_a_visible_warning_then_succeeds_on_retry`, `tests/test_outbox.py::test_gives_up_after_max_attempts_and_says_so`
- Residual risk: after giving up, trend and baseline checks omit the report until it is re-indexed.

## H16 - The overall risk level understates the lab flags
- Severity: catastrophic
- Cause: a severity reaches the code that totals them in a form it does not recognise. The tool
  layer writes "Severity.CRITICAL"; the total compared against the enum, so no lab flag counted and
  a report with two critical values was shown as "Risk Level NORMAL, 0 Critical".
- Controls: one function reads every form a severity takes; a severity that cannot be read is
  counted and never treated as normal; the risk summary is recomputed from the stored flags every
  time a report is read, so reports saved before the fix are shown correctly too.
- Tests: `tests/test_critical_not_hidden.py::test_critical_lab_flags_from_the_real_tool_path_make_the_report_critical`, `tests/test_critical_not_hidden.py::test_the_orchestrator_reports_critical_for_a_critical_report`, `tests/test_critical_not_hidden.py::test_a_severity_that_cannot_be_read_is_never_treated_as_normal`, `tests/test_api_flow.py::test_a_report_stored_before_the_fix_is_shown_with_the_right_level`
- Residual risk: a test the reference table does not know raises no flag, so it cannot raise the
  level; forecast flags are not part of the level.

## H17 - A failed or incomplete AI summary reads as an all-clear
- Severity: catastrophic
- Cause: the model answers in a format the reader does not expect, or returns nothing, and the gap
  is filled with reassuring text ("No significant findings", "Routine monitoring recommended"); or
  the summary leaves a critical result out. An audit that only looks for false claims passes both.
- Controls: the reader accepts the header styles models use; a section it cannot read becomes
  "Summary unavailable" and no follow-up action is invented; in code, not by the model, a
  placeholder is never verified and a clinical summary that mentions a critical lab flag neither by
  value nor by name is not verified. A summary can also name every value and still say nothing:
  live, Groq wrote "a hemoglobin of 6.2 g/dL and a glucose level of 540 mg/dL" and nothing else. So,
  when a result is CRITICAL, the clinical summary must also say it is critical or urgent, and the
  patient summary must tell the patient to contact their doctor; the prompts ask for both, and the
  verifier's rewrite is told to keep them. The reason is shown on the report and must be acknowledged
  before approval; the evaluation gate counts unusable and silent ("thin") summaries and will not
  save either as a baseline.
- Tests: `tests/test_critical_not_hidden.py::test_the_live_answer_is_no_longer_verified`, `tests/test_critical_not_hidden.py::test_a_correction_that_drops_the_urgency_is_not_verified`, `tests/test_api_flow.py::test_critical_report_whose_summaries_only_list_the_values_must_be_acknowledged`, `tests/test_frozen_gate.py::test_summaries_that_list_critical_values_without_saying_so_fail_the_gate`, `tests/test_critical_not_hidden.py::test_an_unreadable_answer_becomes_summary_unavailable_never_an_all_clear`, `tests/test_critical_not_hidden.py::test_the_original_failure_is_no_longer_verified`, `tests/test_critical_not_hidden.py::test_a_placeholder_is_never_verified_and_costs_no_model_call`, `tests/test_critical_not_hidden.py::test_a_correction_that_drops_a_critical_result_is_not_verified_either`, `tests/test_api_flow.py::test_critical_report_with_an_unreadable_summary_is_shown_as_critical_and_unverified`, `tests/test_frozen_gate.py::test_summaries_that_cannot_be_read_fail_the_gate_and_never_become_the_baseline`
- Residual risk: the completeness check covers CRITICAL lab flags only, not HIGH ones or trends. A
  summary can name a critical result and still describe it wrongly; that is left to the model audit
  (H3, H4). Unusual wording for a test can cause a false "not verified", which costs a clinician an
  acknowledgement and is the safe direction to be wrong in. The urgency and doctor checks look for
  words, not meaning: "critical" in a sentence that plays the result down would still pass.

## H18 - AI follow-up actions read as treatment orders
- Severity: major
- Cause: the summarizer is asked for follow-up actions and, from two lab values alone, writes orders:
  "Admit the patient ... initiate transfusion", "Start diabetic emergency protocol: IV insulin
  infusion" (seen live, 2026-11-17). The verifier audits the summaries, not the actions, and the
  actions are shown to the clinician as a numbered list with no further check.
- Controls: the prompt asks for suggestions that start with "Consider" and forbids orders to admit,
  start, give, prescribe or stop treatment, and drug doses. In code, an action whose first word (after
  "urgently" and similar) is such a verb is found; it is kept as written, so the clinician sees what
  the model said, and the report carries a warning that must be acknowledged before release. The
  evaluation gate counts these actions and will not save a baseline that has them.
- Tests: `tests/test_critical_not_hidden.py::test_treatment_orders_are_found`, `tests/test_critical_not_hidden.py::test_suggestions_and_tests_are_not_orders`, `tests/test_api_flow.py::test_treatment_orders_in_the_actions_are_kept_but_must_be_acknowledged`, `tests/test_frozen_gate.py::test_actions_written_as_treatment_orders_fail_the_gate`
- Residual risk: the check reads the first word only. "Consider admitting and starting insulin" passes,
  and so does an order phrased as a noun ("Transfusion of 2 units"). Whether a suggestion is clinically
  right is not checked at all; that is the clinician's job, which is why every report is reviewed (H2).
