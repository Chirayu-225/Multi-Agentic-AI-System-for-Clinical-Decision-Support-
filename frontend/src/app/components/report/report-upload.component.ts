// src/app/components/report/report-upload.component.ts
import { Component, OnDestroy, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';

import { ApiService } from '../../services/api.service';
import { AuthService } from '../../services/auth.service';
import { ReviewStateService } from '../../services/review-state.service';
import { errorMessage } from '../../services/error-message';
import { LANGUAGES } from '../../services/languages';
import { ReportAnalysis, AgentStep } from '../../models/report.model';
import { AgentActivityComponent } from '../shared/agent-activity.component';
import { ReportResultComponent } from './report-result.component';

const ALLOWED_EXT = ['.pdf', '.jpg', '.jpeg', '.png', '.webp'];
const MAX_BYTES = 20 * 1024 * 1024;
const PATIENT_ID_RE = /^[A-Za-z0-9_.\- ]{1,64}$/;

function newReportId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    // crypto.randomUUID needs a secure context; fall back on plain http.
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = Math.floor(Math.random() * 16);
      return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
    });
  }
}

@Component({
  selector: 'app-report-upload',
  standalone: true,
  imports: [FormsModule, AgentActivityComponent, ReportResultComponent],
  template: `
    <div class="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6">

      <!-- Header -->
      <div class="mb-8">
        <p class="text-xs font-semibold text-blue-400 uppercase tracking-widest mb-1">Clinical Analysis</p>
        <h1 class="text-2xl sm:text-3xl font-bold text-white tracking-tight">Analyse Medical Report</h1>
        <p class="text-gray-500 mt-1 text-sm">
          Upload a patient report (PDF or photo) to extract findings, flag risks against ICMR
          reference ranges, and generate an AI-verified clinical summary. Results are held for
          clinician review before the patient can see them.
        </p>
      </div>

      <!-- Upload form -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-4 mb-6">
        <div>
          <span class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Report file
          </span>
          <label
            (dragover)="onDragOver($event)"
            (dragleave)="onDragLeave($event)"
            (drop)="onDrop($event)"
            [class]="dropzoneClasses()">
            @if (selectedFile()) {
              <span class="text-emerald-400 text-sm">✓</span>
            }
            <span class="text-gray-400 text-sm px-3 text-center truncate max-w-full">
              {{ selectedFile() ? selectedFile()!.name : (isDragging() ? 'Drop file to attach' : 'Drag & drop or click to upload') }}
            </span>
            <span class="text-xs text-gray-600">PDF or photo (JPG/PNG/WebP) · max 20 MB</span>
            <input type="file" accept=".pdf,.jpg,.jpeg,.png,.webp" class="sr-only"
                   aria-label="Report file (PDF or photo)" (change)="onFileSelect($event)" />
          </label>
        </div>

        <div>
          <label for="patient-id"
                 class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Patient ID
          </label>
          <input
            id="patient-id"
            [(ngModel)]="patientId"
            maxlength="64"
            placeholder="e.g. PT-2026-001"
            [attr.aria-invalid]="!!patientIdError()"
            aria-describedby="patient-id-help"
            class="w-full bg-gray-900 border rounded-lg px-4 py-3
                   text-gray-200 font-mono text-sm focus:outline-none focus:border-blue-500
                   focus:ring-1 focus:ring-blue-500 transition-colors"
            [class]="patientIdError() ? 'border-red-500/60' : 'border-gray-700'"
          />
          <p id="patient-id-help" class="mt-1.5 text-xs"
             [class]="patientIdError() ? 'text-red-300' : 'text-gray-600'">
            {{ patientIdError() || "Use a pseudonymous ID (e.g. MRN), not the patient's name." }}
          </p>
        </div>

        <div>
          <label for="report-date"
                 class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Report Date
          </label>
          <input
            id="report-date"
            type="date"
            [(ngModel)]="reportDate"
            class="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3
                   text-gray-200 font-mono text-sm focus:outline-none focus:border-blue-500
                   focus:ring-1 focus:ring-blue-500 transition-colors"
          />
        </div>

        <div>
          <label for="summary-language"
                 class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Patient summary language
          </label>
          <select
            id="summary-language"
            [(ngModel)]="language"
            class="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3
                   text-gray-200 text-sm focus:outline-none focus:border-blue-500
                   focus:ring-1 focus:ring-blue-500 transition-colors">
            <option value="">English</option>
            @for (l of languages; track l.code) {
              <option [value]="l.code">{{ l.name }}</option>
            }
          </select>
        </div>
      </div>

      <button
        [disabled]="!selectedFile() || !patientId.trim() || !!patientIdError() || loading()"
        (click)="runAnalysis()"
        class="w-full py-3 rounded-lg font-semibold text-sm tracking-wide transition-all
               bg-gradient-to-r from-blue-600 to-sky-500 text-white
               hover:shadow-lg hover:shadow-blue-500/25 hover:-translate-y-0.5
               disabled:opacity-40 disabled:cursor-not-allowed disabled:transform-none">
        {{ loading() ? 'Analysing...' : 'Run Analysis' }}
      </button>

      <!-- Upload / file error -->
      @if (uploadError()) {
        <div role="alert" class="mt-4 rounded-xl p-3 border border-red-500/40 bg-red-500/10 text-sm text-red-300">
          {{ uploadError() }}
        </div>
      }

      <!-- Agent activity stream -->
      @if (loading() || agentSteps().length) {
        <app-agent-activity [steps]="agentSteps()" [loading]="loading()" class="mt-6 block" />
      }

      <!-- Idle empty state -->
      @if (!loading() && !result() && !agentSteps().length) {
        <div class="mt-8 border border-dashed border-gray-800 rounded-xl px-6 py-12
                    flex flex-col items-center text-center bg-gray-900/30">
          <div class="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-600/20 to-sky-500/20
                      border border-blue-500/20 flex items-center justify-center text-2xl mb-4">
            📋
          </div>
          <p class="text-sm font-medium text-gray-400 mb-1.5">No analysis yet</p>
          <p class="text-xs text-gray-600 max-w-sm leading-relaxed">
            Upload a report (PDF or photo) and enter a patient ID above, then run the analysis to see
            extracted findings, ICMR-grounded risk flags and an AI-verified clinical summary here.
          </p>
        </div>
      }

      <!-- Results -->
      @if (result(); as res) {
        <div class="mt-8">
          <app-report-result [report]="res" [canReview]="canReview()" />
        </div>
      }
    </div>
  `,
})
export class ReportUploadComponent implements OnDestroy {
  selectedFile   = signal<File | null>(null);
  patientId      = '';
  reportDate     = new Date().toISOString().split('T')[0];
  language       = '';
  loading        = signal(false);
  result         = signal<ReportAnalysis | null>(null);
  agentSteps     = signal<AgentStep[]>([]);
  uploadError    = signal<string | null>(null);
  isDragging     = signal(false);

  readonly languages = LANGUAGES;

  private sub?: Subscription;
  private uploadSub?: Subscription;

  constructor(
    private api: ApiService,
    private auth: AuthService,
    private reviewState: ReviewStateService,
  ) {}

  canReview(): boolean {
    return this.auth.hasRole('clinician');
  }

  patientIdError(): string | null {
    const v = this.patientId;
    if (!v) return null;
    return PATIENT_ID_RE.test(v)
      ? null
      : 'Use 1–64 letters, digits, spaces, "-", "_" or "." only.';
  }

  private validateFile(file: File): string | null {
    const name = file.name.toLowerCase();
    if (!ALLOWED_EXT.some((ext) => name.endsWith(ext))) {
      return 'Unsupported file type. Please choose a PDF or a photo (JPG, PNG or WebP).';
    }
    if (file.size > MAX_BYTES) {
      return 'File is too large (maximum 20 MB).';
    }
    if (file.size === 0) {
      return 'That file is empty.';
    }
    return null;
  }

  private acceptFile(file: File): void {
    const problem = this.validateFile(file);
    if (problem) {
      this.selectedFile.set(null);
      this.uploadError.set(problem);
      return;
    }
    this.uploadError.set(null);
    this.selectedFile.set(file);
  }

  onFileSelect(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files?.length) {
      this.acceptFile(input.files[0]);
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragging.set(true);
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.isDragging.set(false);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragging.set(false);
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      this.acceptFile(file);
    }
  }

  dropzoneClasses(): string {
    const base = 'flex flex-col items-center justify-center w-full h-28 gap-1 ' +
                 'border-2 border-dashed rounded-xl cursor-pointer transition-colors ' +
                 'focus-within:ring-2 focus-within:ring-blue-500';
    if (this.isDragging()) {
      return `${base} border-blue-500 bg-blue-500/5`;
    }
    if (this.selectedFile()) {
      return `${base} border-emerald-600 bg-emerald-500/5`;
    }
    return `${base} border-gray-700 bg-gray-900 hover:border-blue-500`;
  }

  runAnalysis(): void {
    const file = this.selectedFile();
    const pid = this.patientId.trim();
    if (!file || !pid || this.patientIdError() || this.loading()) return;

    const reportId = newReportId();
    this.loading.set(true);
    this.result.set(null);
    this.agentSteps.set([]);
    this.uploadError.set(null);

    // Live agent streaming (staff only; a failure here must not block the upload)
    this.sub?.unsubscribe();
    try {
      const stream = this.api.connectAgentStream(reportId);
      this.sub = stream.subscribe({
        next: (step) => this.agentSteps.update(steps => [...steps, step]),
        error: () => {},
      });
    } catch {
      /* streaming is optional */
    }

    this.uploadSub?.unsubscribe();
    this.uploadSub = this.api
      .uploadReport(file, pid, this.reportDate, reportId, this.language || undefined)
      .subscribe({
        next: (res) => {
          this.result.set(res);
          this.loading.set(false);
          this.api.disconnectAgentStream(reportId);
          this.reviewState.refresh();
        },
        error: (err) => {
          this.loading.set(false);
          this.api.disconnectAgentStream(reportId);
          const status = err?.status;
          if (status === 401) {
            this.uploadError.set('Your session has expired — please sign in again.');
          } else if (status === 413) {
            this.uploadError.set(errorMessage(err, 'File is too large (maximum 20 MB).'));
          } else if (status === 429) {
            this.uploadError.set(errorMessage(err, 'Too many uploads — please wait a bit and try again.'));
          } else {
            this.uploadError.set(
              errorMessage(err, 'Something went wrong processing this report. Please try again.')
            );
          }
        },
      });
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
    this.uploadSub?.unsubscribe();
  }
}
