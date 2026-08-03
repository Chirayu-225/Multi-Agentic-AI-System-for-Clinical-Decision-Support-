// src/app/components/report/report-upload.component.ts
import {
  Component, OnDestroy, signal, computed
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';

import { ApiService } from '../../services/api.service';
import { SeverityService } from '../../services/severity.service';
import { ReportAnalysis, AgentStep, RiskFlag, TrendFlag } from '../../models/report.model';
import { RiskCardComponent } from '../shared/risk-card.component';
import { AgentActivityComponent } from '../shared/agent-activity.component';
import { PubmedCardComponent } from '../shared/pubmed-card.component';

@Component({
  selector: 'app-report-upload',
  standalone: true,
  imports: [CommonModule, FormsModule, RiskCardComponent,
            AgentActivityComponent, PubmedCardComponent],
  template: `
    <div class="min-h-screen bg-gray-950 text-gray-100 p-6">

      <!-- Header -->
      <div class="mb-8">
        <p class="text-xs font-semibold text-blue-400 uppercase tracking-widest mb-1">Clinical Analysis</p>
        <h1 class="text-3xl font-bold text-white tracking-tight">Analyse Medical Report</h1>
        <p class="text-gray-500 mt-1 text-sm">
          Upload a patient report to extract findings, flag risks against ICMR reference ranges,
          and generate an AI-verified clinical summary.
        </p>
      </div>

      <!-- Upload form -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        <div class="lg:col-span-1">
          <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Report PDF
          </label>
          <label class="flex flex-col items-center justify-center w-full h-28
                        border-2 border-dashed border-gray-700 rounded-xl
                        bg-gray-900 cursor-pointer hover:border-blue-500 transition-colors">
            <span class="text-gray-500 text-sm">
              {{ selectedFile() ? selectedFile()!.name : 'Drag & drop or click to upload' }}
            </span>
            <span class="text-xs text-gray-600 mt-1">PDF only · max 200MB</span>
            <input type="file" accept=".pdf" class="hidden" (change)="onFileSelect($event)" />
          </label>
        </div>

        <div>
          <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Patient ID
          </label>
          <input
            [(ngModel)]="patientId"
            placeholder="e.g. PT-2026-001"
            class="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3
                   text-gray-200 font-mono text-sm focus:outline-none focus:border-blue-500
                   focus:ring-1 focus:ring-blue-500 transition-colors"
          />
        </div>

        <div>
          <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Report Date
          </label>
          <input
            type="date"
            [(ngModel)]="reportDate"
            class="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3
                   text-gray-200 font-mono text-sm focus:outline-none focus:border-blue-500
                   focus:ring-1 focus:ring-blue-500 transition-colors"
          />
        </div>
      </div>

      <button
        [disabled]="!selectedFile() || !patientId || loading()"
        (click)="runAnalysis()"
        class="w-full py-3 rounded-lg font-semibold text-sm tracking-wide transition-all
               bg-gradient-to-r from-blue-600 to-sky-500 text-white
               hover:shadow-lg hover:shadow-blue-500/25 hover:-translate-y-0.5
               disabled:opacity-40 disabled:cursor-not-allowed disabled:transform-none">
        {{ loading() ? 'Analysing...' : 'Run Analysis' }}
      </button>

      <!-- Agent activity stream -->
      @if (loading() || agentSteps().length) {
        <app-agent-activity [steps]="agentSteps()" [loading]="loading()" class="mt-6 block" />
      }

      <!-- Results -->
      @if (result()) {
        <div class="mt-8">
          <!-- Risk banner -->
          <div class="rounded-xl p-4 mb-6 border flex items-center justify-between"
               [style.background]="sev.getColor(result()!.risk_summary.overall_severity) + '0D'"
               [style.border-color]="sev.getColor(result()!.risk_summary.overall_severity) + '40'">
            <div class="flex items-center gap-3">
              <div class="w-3 h-3 rounded-full animate-pulse"
                   [style.background]="sev.getColor(result()!.risk_summary.overall_severity)"></div>
              <div>
                <div class="font-bold text-base"
                     [style.color]="sev.getColor(result()!.risk_summary.overall_severity)">
                  {{ sev.getBannerTitle(result()!.risk_summary.overall_severity) }}
                </div>
                <div class="text-xs text-gray-500 mt-0.5">
                  {{ result()!.filename }} · {{ result()!.pages }} page(s) · {{ result()!.source_type | titlecase }}
                </div>
              </div>
            </div>
            <div class="flex gap-2 flex-wrap">
              <span class="px-2 py-1 rounded text-xs font-semibold bg-red-500/20 text-red-400">
                {{ result()!.risk_summary.critical_count }} Critical
              </span>
              <span class="px-2 py-1 rounded text-xs font-semibold bg-orange-500/20 text-orange-400">
                {{ result()!.risk_summary.high_count }} High
              </span>
              <span class="px-2 py-1 rounded text-xs font-semibold bg-yellow-500/20 text-yellow-400">
                {{ result()!.risk_summary.trend_count }} Trends
              </span>
              <span class="px-2 py-1 rounded text-xs font-semibold bg-gray-700 text-gray-400">
                {{ result()!.risk_summary.total_flags }} Total
              </span>
            </div>
          </div>

          <!-- Two-column layout -->
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

            <!-- Left: Risk flags + actions -->
            <div>
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3">Risk Flags</p>

              @if (!abnormalFlags().length && !result()!.trend_flags.length) {
                <app-risk-card [flag]="null" [isAllClear]="true" />
              }
              @for (flag of abnormalFlags(); track flag.test) {
                <app-risk-card [flag]="flag" />
              }
              @for (trend of result()!.trend_flags; track trend.test) {
                <app-risk-card [flag]="null" [trend]="trend" />
              }

              <!-- Follow-up actions -->
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3 mt-6">Recommended Actions</p>
              @for (action of result()!.llm_output.followup_actions; track $index) {
                <div class="flex gap-3 bg-gray-900 border border-gray-800 rounded-lg
                            px-4 py-3 mb-2 text-sm text-gray-300">
                  <span class="font-mono text-xs text-blue-400 font-semibold shrink-0 pt-0.5">
                    {{ ($index + 1).toString().padStart(2, '0') }}
                  </span>
                  {{ action }}
                </div>
              }

              <!-- PubMed references -->
              @if (result()!.pubmed_references?.length) {
                <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                           pb-2 border-b border-gray-800 mb-3 mt-6">Clinical Evidence</p>
                @for (ref of result()!.pubmed_references; track ref.pmid) {
                  <app-pubmed-card [reference]="ref" />
                }
              }
            </div>

            <!-- Right: Summaries + entities -->
            <div>
              <!-- Clinical summary -->
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3">Clinical Summary</p>
              <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-4">
                <div class="flex items-center gap-2 mb-3">
                  <div class="w-0.5 h-3 bg-blue-500 rounded"></div>
                  <span class="text-xs font-semibold text-blue-400 uppercase tracking-wider">
                    For Healthcare Provider
                  </span>
                  @if (result()!.llm_output.verification.clinical_summary_corrected) {
                    <span class="ml-auto text-xs font-semibold bg-yellow-500/20 text-yellow-400
                                  px-2 py-0.5 rounded">✓ verified-agent corrected</span>
                  } @else {
                    <span class="ml-auto text-xs font-semibold bg-green-500/20 text-green-400
                                  px-2 py-0.5 rounded">✓ verified</span>
                  }
                </div>
                <p class="text-gray-300 text-sm leading-relaxed">
                  {{ result()!.llm_output.clinical_summary }}
                </p>
              </div>

              <!-- Patient summary -->
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3">Patient Summary</p>
              <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-4">
                <div class="flex items-center gap-2 mb-3">
                  <div class="w-0.5 h-3 bg-blue-500 rounded"></div>
                  <span class="text-xs font-semibold text-blue-400 uppercase tracking-wider">
                    In Plain Language
                  </span>
                  @if (result()!.llm_output.verification.patient_summary_corrected) {
                    <span class="ml-auto text-xs font-semibold bg-yellow-500/20 text-yellow-400
                                  px-2 py-0.5 rounded">✓ verified-agent corrected</span>
                  } @else {
                    <span class="ml-auto text-xs font-semibold bg-green-500/20 text-green-400
                                  px-2 py-0.5 rounded">✓ verified</span>
                  }
                </div>
                <p class="text-gray-300 text-sm leading-relaxed">
                  {{ result()!.llm_output.patient_summary }}
                </p>
              </div>

              <!-- Patient info -->
              @if (result()!.entities.patient.name) {
                <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                           pb-2 border-b border-gray-800 mb-3">Patient Info</p>
                <div class="grid grid-cols-3 gap-3 mb-4">
                  @for (field of patientFields(); track field.label) {
                    @if (field.value) {
                      <div class="bg-gray-900 border border-gray-800 rounded-lg p-3 text-center">
                        <p class="text-xs text-gray-500 uppercase tracking-wider mb-1">{{ field.label }}</p>
                        <p class="font-semibold font-mono text-gray-200">{{ field.value }}</p>
                      </div>
                    }
                  }
                </div>
              }

              <!-- Diagnoses -->
              @if (result()!.entities.clinical.diagnoses.length) {
                <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                           pb-2 border-b border-gray-800 mb-3">Diagnoses</p>
                <div class="flex flex-wrap gap-2 mb-4">
                  @for (dx of result()!.entities.clinical.diagnoses; track dx) {
                    <span class="bg-gray-800 border border-blue-900 text-blue-300
                                  font-mono text-xs px-3 py-1.5 rounded-md">{{ dx }}</span>
                  }
                </div>
              }

              <!-- Agent metadata -->
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3">Agent Activity</p>
              <div class="flex flex-wrap gap-2">
                <span class="bg-gray-900 border border-gray-800 font-mono text-xs
                              px-3 py-1.5 rounded-md text-gray-400">
                  Source <span class="text-blue-400">{{ result()!.source_type | titlecase }}</span>
                </span>
                <span class="bg-gray-900 border border-gray-800 font-mono text-xs
                              px-3 py-1.5 rounded-md text-gray-400">
                  Tools called <span class="text-blue-400">{{ result()!.agent_metadata.tool_calls.length }}</span>
                </span>
                <span class="bg-gray-900 border border-gray-800 font-mono text-xs
                              px-3 py-1.5 rounded-md text-gray-400">
                  PubMed refs <span class="text-blue-400">{{ result()!.agent_metadata.pubmed_found }}</span>
                </span>
                <span class="bg-gray-900 border border-gray-800 font-mono text-xs
                              px-3 py-1.5 rounded-md text-gray-400">
                  Hindi <span class="text-blue-400">{{ result()!.has_hindi ? 'Yes' : 'No' }}</span>
                </span>
              </div>
            </div>
          </div>
        </div>
      }
    </div>
  `,
})
export class ReportUploadComponent implements OnDestroy {
  selectedFile   = signal<File | null>(null);
  patientId      = '';
  reportDate     = new Date().toISOString().split('T')[0];
  loading        = signal(false);
  result         = signal<ReportAnalysis | null>(null);
  agentSteps     = signal<AgentStep[]>([]);

  private sub?: Subscription;

  constructor(
    private api: ApiService,
    public sev: SeverityService,
  ) {}

  onFileSelect(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files?.length) {
      this.selectedFile.set(input.files[0]);
    }
  }

  abnormalFlags() {
    return (this.result()?.rule_flags ?? []).filter(
      f => !['NORMAL', 'Severity.NORMAL'].includes(f.severity)
    );
  }

  patientFields() {
    const p = this.result()?.entities.patient;
    return [
      { label: 'Name',   value: p?.name },
      { label: 'Age',    value: p?.age },
      { label: 'Gender', value: p?.gender },
    ];
  }

  runAnalysis(): void {
    if (!this.selectedFile() || !this.patientId) return;

    const reportId = crypto.randomUUID();
    this.loading.set(true);
    this.result.set(null);
    this.agentSteps.set([]);

    // Connect WebSocket for live agent streaming
    const stream = this.api.connectAgentStream(reportId);
    this.sub = stream.subscribe({
      next: (step) => this.agentSteps.update(steps => [...steps, step]),
      error: () => {},
    });

    this.api
      .uploadReport(this.selectedFile()!, this.patientId, this.reportDate, reportId)
      .subscribe({
        next: (res) => {
          this.result.set(res);
          this.loading.set(false);
          this.api.disconnectAgentStream(reportId);
        },
        error: (err) => {
          console.error(err);
          this.loading.set(false);
        },
      });
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }
}
