// src/app/services/api.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, Subject } from 'rxjs';
import { environment } from '../../environments/environment';
import {
  ReportAnalysis,
  PatientHistory,
  AgentStep,
} from '../models/report.model';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly base = environment.apiUrl;
  private wsSubjects = new Map<string, Subject<AgentStep>>();

  constructor(private http: HttpClient) {}

  // ── Reports ───────────────────────────────────────────────────────────────

  uploadReport(
    file: File,
    patientId: string,
    reportDate: string,
    reportId: string
  ): Observable<ReportAnalysis> {
    const form = new FormData();
    form.append('file', file);
    form.append('patient_id', patientId);
    form.append('report_date', reportDate);
    form.append('report_id', reportId);
    return this.http.post<ReportAnalysis>(`${this.base}/report/upload`, form);
  }

  getReport(reportId: string): Observable<ReportAnalysis> {
    return this.http.get<ReportAnalysis>(`${this.base}/report/${reportId}`);
  }

  // ── Patients ──────────────────────────────────────────────────────────────

  getPatientHistory(patientId: string): Observable<PatientHistory> {
    return this.http.get<PatientHistory>(
      `${this.base}/patient/${patientId}/history`
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

    const wsUrl = environment.wsUrl + `/ws/${reportId}`;
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
