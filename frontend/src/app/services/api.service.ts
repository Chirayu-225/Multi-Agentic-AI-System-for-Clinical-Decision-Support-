// src/app/services/api.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, Subject } from 'rxjs';
import { environment } from '../../environments/environment';
import { AuthService } from './auth.service';
import {
  ReportAnalysis,
  PatientHistory,
  PatientTrends,
  PatientReportView,
  MyReports,
  PendingReviews,
  ReviewResult,
  TranslateResult,
  AdminMetrics,
  AuditEvent,
  AuditVerification,
  AgentStep,
} from '../models/report.model';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly base = environment.apiUrl;
  private wsSubjects = new Map<string, Subject<AgentStep>>();

  constructor(private http: HttpClient, private auth: AuthService) {}

  // ── Reports ───────────────────────────────────────────────────────────────

  uploadReport(
    file: File,
    patientId: string,
    reportDate: string,
    reportId: string,
    language?: string
  ): Observable<ReportAnalysis> {
    const form = new FormData();
    form.append('file', file);
    form.append('patient_id', patientId);
    form.append('report_date', reportDate);
    form.append('report_id', reportId);
    if (language) form.append('language', language);
    return this.http.post<ReportAnalysis>(`${this.base}/report/upload`, form);
  }

  getReport(reportId: string): Observable<ReportAnalysis> {
    return this.http.get<ReportAnalysis>(`${this.base}/report/${reportId}`);
  }

  /** Patient-role view of an approved report (no clinician fields). */
  getPatientReport(reportId: string): Observable<PatientReportView> {
    return this.http.get<PatientReportView>(`${this.base}/report/${reportId}`);
  }

  reviewReport(
    reportId: string,
    decision: 'approve' | 'reject',
    note: string
  ): Observable<ReviewResult> {
    return this.http.post<ReviewResult>(`${this.base}/report/${reportId}/review`, {
      decision,
      note,
    });
  }

  translateReport(reportId: string, language: string): Observable<TranslateResult> {
    return this.http.post<TranslateResult>(
      `${this.base}/report/${reportId}/translate`,
      null,
      { params: { language } }
    );
  }

  getFhir(reportId: string): Observable<unknown> {
    return this.http.get<unknown>(`${this.base}/report/${reportId}/fhir`);
  }

  getPendingReviews(): Observable<PendingReviews> {
    return this.http.get<PendingReviews>(`${this.base}/reviews/pending`);
  }

  getMyReports(): Observable<MyReports> {
    return this.http.get<MyReports>(`${this.base}/me/reports`);
  }

  // ── Admin ─────────────────────────────────────────────────────────────────

  getMetrics(): Observable<AdminMetrics> {
    return this.http.get<AdminMetrics>(`${this.base}/admin/metrics`);
  }

  getAudit(limit = 50): Observable<{ events: AuditEvent[] }> {
    return this.http.get<{ events: AuditEvent[] }>(`${this.base}/admin/audit`, {
      params: { limit },
    });
  }

  verifyAudit(): Observable<AuditVerification> {
    return this.http.get<AuditVerification>(`${this.base}/admin/audit/verify`);
  }

  // ── Patients ──────────────────────────────────────────────────────────────

  getPatientHistory(patientId: string): Observable<PatientHistory> {
    return this.http.get<PatientHistory>(
      `${this.base}/patient/${encodeURIComponent(patientId)}/history`
    );
  }

  getPatientTrends(patientId: string): Observable<PatientTrends> {
    return this.http.get<PatientTrends>(
      `${this.base}/patient/${encodeURIComponent(patientId)}/trends`
    );
  }

  listPatients(): Observable<{ patients: string[] }> {
    return this.http.get<{ patients: string[] }>(`${this.base}/patients`);
  }

  deletePatient(patientId: string): Observable<{ message: string }> {
    return this.http.delete<{ message: string }>(
      `${this.base}/patient/${patientId}`
    );
  }

  // ── WebSocket — real-time agent streaming ─────────────────────────────────

  connectAgentStream(reportId: string): Subject<AgentStep> {
    const subject = new Subject<AgentStep>();
    this.wsSubjects.set(reportId, subject);

    // Native WebSocket can't set custom headers, so the credential goes in
    // the query string — the one place a token is allowed in a URL.
    const token = this.auth.token();
    const wsUrl =
      environment.wsUrl +
      `/ws/${reportId}` +
      (token
        ? `?token=${encodeURIComponent(token)}`
        : environment.apiKey
          ? `?api_key=${encodeURIComponent(environment.apiKey)}`
          : '');
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const step: AgentStep = JSON.parse(event.data);
        subject.next(step);
      } catch {}
    };

    ws.onerror = () => subject.error('WebSocket error');
    ws.onclose = () => subject.complete();

    return subject;
  }

  disconnectAgentStream(reportId: string): void {
    this.wsSubjects.get(reportId)?.complete();
    this.wsSubjects.delete(reportId);
  }

  // ── Health ────────────────────────────────────────────────────────────────

  health(): Observable<{ status: string }> {
    return this.http.get<{ status: string }>(`${this.base}/health`);
  }
}
