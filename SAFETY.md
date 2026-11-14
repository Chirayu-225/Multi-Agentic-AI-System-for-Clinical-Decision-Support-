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
