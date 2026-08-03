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

export interface RiskSummary {
  overall_severity: string;
  color: string;
  total_flags: number;
  critical_count: number;
  high_count: number;
  trend_count: number;
  requires_immediate_attention: boolean;
}

export interface Verification {
  clinical_summary_corrected: boolean;
  patient_summary_corrected: boolean;
}

export interface LLMOutput {
  clinical_summary: string;
  patient_summary: string;
  explained_flags: RiskFlag[];
  followup_actions: string[];
  verification: Verification;
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
}

export interface ReportAnalysis {
  report_id: string;
  patient_id: string;
  filename: string;
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
  risk_summary: RiskSummary;
  pubmed_references: PubMedReference[];
  drug_safety: DrugSafety[];
  llm_output: LLMOutput;
  agent_metadata: AgentMetadata;
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
}

export interface PatientHistory {
  patient_id: string;
  report_count: number;
  reports: PatientHistoryReport[];
}

export type SeverityLevel = 'NORMAL' | 'MODERATE' | 'HIGH' | 'CRITICAL';
