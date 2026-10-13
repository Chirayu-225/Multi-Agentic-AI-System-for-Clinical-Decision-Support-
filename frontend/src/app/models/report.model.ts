// src/app/models/report.model.ts

export interface Patient {
  name: string | null;
  age: string | null;
  gender: string | null;
}

export interface LabValue {
  test: string;
  value: number | null;
  unit: string;
}

export interface RiskFlag {
  test: string;
  value: number;
  unit: string;
  severity: string;
  direction: string;
  message: string;
  explanation?: string;
  normal_range?: { min: number; max: number; unit: string };
  source?: string;
}

export interface TrendFlag {
  test: string;
  trend_direction: string;
  values_over_time: number[];
  total_change_pct: number;
  unit: string;
  severity: string;
  message: string;
  explanation?: string;
}

export interface BaselineFlag {
  test: string;
  canonical: string;
  current_value: number;
  patient_mean: number;
  patient_stdev: number;
  z_score: number;
  pct_change_from_mean: number;
  unit: string;
  direction: string;
  severity: string;
  message: string;
  explanation?: string;
}

export interface ForecastFlag {
  test: string;
  canonical: string;
  current_value: number;
  slope_per_visit: number;
  unit: string;
  target_threshold: number;
  target_label: string;
  visits_to_cross: number;
  estimated_days_to_cross: number | null;
  severity: string;
  message: string;
}

export interface SimilarHistoryItem {
  report_id: string;
  report_date: string;
  filename: string;
  similarity_pct: number;
  snippet: string;
}

export interface RiskSummary {
  overall_severity: string;
  color: string;
  total_flags: number;
  critical_count: number;
  high_count: number;
  trend_count: number;
  baseline_count?: number;
  requires_immediate_attention: boolean;
}

export interface Verification {
  clinical_summary_corrected: boolean;
  patient_summary_corrected: boolean;
  clinical_unverified?: boolean;
  patient_unverified?: boolean;
}

export interface Translation {
  language: string;
  text: string;
  check?: unknown;
}

export interface LLMOutput {
  clinical_summary: string;
  patient_summary: string;
  explained_flags: RiskFlag[];
  followup_actions: string[];
  verification: Verification;
  longitudinal_narrative?: string;
  translation?: Translation | null;
}

export interface PubMedReference {
  pmid: string;
  title: string;
  authors: string;
  journal: string;
  pub_date: string;
  url: string;
}

export interface DrugSafety {
  medication: string;
  has_warnings: boolean;
  boxed_warning?: string;
  warnings_snippet?: string;
  source: string;
}

export interface AgentMetadata {
  tool_calls: Array<{ tool: string; args: any; result_summary: string }>;
  pubmed_found: number;
  drug_safety_checked: number;
  duration_ms?: number | null;
  llm_calls?: number | null;
}

// ── Medication interactions ─────────────────────────────────────────────────

export type InteractionType = 'drug-drug' | 'drug-lab' | 'drug-advisory';
export type InteractionSeverity = 'major' | 'moderate' | 'minor';

export interface DrugInteraction {
  type: InteractionType;
  severity: InteractionSeverity;
  drugs: string[];
  effect: string;
  advice: string;
  lab?: string;
  lab_value?: number;
  source: string;
}

export interface DrugInteractions {
  interactions: DrugInteraction[];
  checked_medications: string[];
  unrecognized_medications: string[];
  disclaimer: string;
}

// ── Evidence (source-grounding of summary claims) ───────────────────────────

export type EvidenceKind = 'lab' | 'diagnosis' | 'medication' | 'symptom';

export interface EvidenceItem {
  id: string;
  kind: EvidenceKind;
  label: string;
  value?: number;
  severity?: string;
  found: boolean;
  snippet: string;
  match_start: number;
  match_end: number;
  cited_in: Array<'clinical' | 'patient'>;
}

// ── Human review ────────────────────────────────────────────────────────────

export type ReviewStatus = 'pending_review' | 'approved' | 'rejected';

export interface ReviewInfo {
  status: ReviewStatus;
  reviewed_by: string | null;
  reviewed_at: string | null;
  note?: string | null;
}

export interface ReviewResult {
  report_id: string;
  status: ReviewStatus;
  reviewed_by: string | null;
  reviewed_at: string | null;
}

export interface PendingReviewItem {
  report_id: string;
  patient_id: string;
  filename: string;
  report_date: string;
  created_at: string;
  overall_severity: string;
  security_warnings: number;
  grounding_warnings: number;
  interaction_alerts: number;
}

export interface PendingReviews {
  count: number;
  reports: PendingReviewItem[];
}

export interface TranslateResult {
  report_id: string;
  language: string;
  text: string;
}

export interface ReportAnalysis {
  report_id: string;
  patient_id: string;
  filename: string;
  report_date?: string;
  source_type: string;
  has_hindi: boolean;
  pages: number;
  entities: {
    patient: Patient;
    clinical: { diagnoses: string[]; symptoms: string[]; procedures: string[] };
    medications: Array<{ name: string; dosage: string | null }>;
    lab_values: LabValue[];
  };
  rule_flags: RiskFlag[];
  trend_flags: TrendFlag[];
  baseline_flags?: BaselineFlag[];
  forecast_flags?: ForecastFlag[];
  similar_history?: SimilarHistoryItem[];
  risk_summary: RiskSummary;
  pubmed_references?: PubMedReference[];
  drug_safety?: DrugSafety[];
  llm_output: LLMOutput;
  agent_metadata: AgentMetadata;
  drug_interactions?: DrugInteractions;
  ingestion_warnings?: string[];
  security_warnings?: string[];
  grounding_warnings?: string[];
  evidence?: EvidenceItem[];
  review?: ReviewInfo;
  disclaimer?: string;
}

// ── Patient-facing report (role = patient) ──────────────────────────────────

export interface PatientLabValue {
  test: string;
  value: number | null;
  unit: string;
  severity: string;
  normal_range: { min: number; max: number; unit: string } | null;
  direction: 'high' | 'low' | 'normal' | 'unknown';
}

export interface PatientReportView {
  report_id: string;
  patient_id: string;
  report_date: string;
  filename: string;
  review: ReviewInfo;
  risk_summary: { overall_severity: string; color: string };
  patient_summary: string;
  patient_summary_translated: { language: string; text: string } | null;
  lab_values: PatientLabValue[];
  disclaimer: string;
}

export interface MyReportItem {
  report_id: string;
  report_date: string;
  filename: string;
  overall_severity: string;
}

export interface MyReports {
  patient_id: string;
  reports: MyReportItem[];
}

// ── Trends ──────────────────────────────────────────────────────────────────

export interface TrendPoint {
  date: string;
  value: number;
  report_id: string;
}

export interface TrendSeries {
  test: string;
  canonical?: string;
  unit: string;
  ref_low: number | null;
  ref_high: number | null;
  critical_low: number | null;
  critical_high: number | null;
  points: TrendPoint[];
}

export interface PatientTrends {
  patient_id: string;
  report_count: number;
  series: TrendSeries[];
}

// ── Admin: metrics + audit ──────────────────────────────────────────────────

export interface AdminMetrics {
  generated_at: string;
  reports: {
    total: number;
    last_24h: number;
    by_review_status: { pending_review: number; approved: number; rejected: number };
    uploads_7d: Array<{ date: string; count: number }>;
  };
  pipeline: {
    avg_duration_ms: number | null;
    p95_duration_ms: number | null;
    avg_llm_calls: number | null;
    avg_tool_calls: number | null;
  };
  safety: {
    verifier_correction_rate: number | null;
    verifier_unverified_rate: number | null;
    reports_with_security_warnings: number;
    reports_with_grounding_warnings: number;
    reports_with_interaction_alerts: number;
  };
  security_24h: {
    events: number;
    denied: number;
    login_failures: number;
    authz_denied: number;
    rate_limited: number;
    injection_suspected: number;
    by_action: Record<string, number>;
  };
}

export interface AuditEvent {
  id: number;
  ts: string;
  actor: string;
  role: string;
  action: string;
  resource: string;
  outcome: 'success' | 'denied' | 'failure';
  ip: string;
  detail: string;
  hash: string;
}

export interface AuditVerification {
  ok: boolean;
  checked: number;
  first_bad_id: number | null;
  reason: string | null;
  head_hash: string | null;
}

// ── Auth ────────────────────────────────────────────────────────────────────

export type Role = 'clinician' | 'patient' | 'service';

export interface AuthUser {
  username: string;
  role: Role;
  displayName: string;
  patientId: string | null;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  role: 'clinician' | 'patient';
  username: string;
  display_name: string;
  patient_id: string | null;
}

export interface AgentStep {
  type: string;
  tool?: string;
  args?: any;
  result?: string;
  clinical_corrected?: boolean;
  patient_corrected?: boolean;
  report_id?: string;
}

export interface PatientHistoryReport {
  report_id: string;
  filename: string;
  report_date: string;
  risk_summary: RiskSummary;
  lab_values: LabValue[];
  diagnoses: string[];
  review_status?: ReviewStatus;
}

export interface PatientHistory {
  patient_id: string;
  report_count: number;
  reports: PatientHistoryReport[];
}

export type SeverityLevel = 'NORMAL' | 'MODERATE' | 'HIGH' | 'CRITICAL';
