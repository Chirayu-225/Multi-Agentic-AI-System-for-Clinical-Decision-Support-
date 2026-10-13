// src/app/components/report/report-result.component.ts
// Reusable analysis result view (upload page + review queue). Every piece of
// model/server text is rendered with {{ }} interpolation only.
import { Component, EventEmitter, Input, Output, computed, signal } from '@angular/core';
import { DatePipe, TitleCasePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { ApiService } from '../../services/api.service';
import { SeverityService } from '../../services/severity.service';
import { ReviewStateService } from '../../services/review-state.service';
import { errorMessage } from '../../services/error-message';
import { downloadJson } from '../../services/download';
import { ReportAnalysis, ReviewInfo, ReviewStatus } from '../../models/report.model';
import { RiskCardComponent } from '../shared/risk-card.component';
import { PubmedCardComponent } from '../shared/pubmed-card.component';
import { TranslatableSummaryComponent } from '../shared/translatable-summary.component';
import { InteractionsCardComponent } from './interactions-card.component';
import { EvidencePanelComponent } from './evidence-panel.component';

@Component({
  selector: 'app-report-result',
  standalone: true,
  imports: [
    DatePipe, TitleCasePipe, FormsModule, RiskCardComponent, PubmedCardComponent,
    TranslatableSummaryComponent, InteractionsCardComponent, EvidencePanelComponent,
  ],
  template: `
    @if (data(); as r) {
      <div>

        <!-- Review status banner -->
        @if (review(); as rv) {
          @switch (rv.status) {
            @case ('pending_review') {
              <div class="rounded-xl p-4 mb-4 border border-amber-500/40 bg-amber-500/10">
                <p class="text-sm font-semibold text-amber-300">
                  Awaiting clinician approval — hidden from the patient
                </p>
                @if (canReview) {
                  <label for="review-note-{{ r.report_id }}"
                         class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mt-3 mb-1.5">
                    Note <span class="normal-case font-normal text-gray-500">(required to reject, at least 3 characters)</span>
                  </label>
                  <textarea id="review-note-{{ r.report_id }}" rows="2" maxlength="1000"
                            [(ngModel)]="note"
                            class="w-full bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm
                                   text-gray-200 focus:outline-none focus:border-blue-500 focus:ring-1
                                   focus:ring-blue-500"></textarea>
                  <div class="flex flex-wrap gap-2 mt-3">
                    <button type="button" (click)="submitReview('approve')" [disabled]="busy()"
                            class="px-4 py-2 rounded-lg text-sm font-semibold bg-emerald-600
                                   hover:bg-emerald-500 text-white transition-colors disabled:opacity-40
                                   disabled:cursor-not-allowed">
                      {{ busy() === 'approve' ? 'Approving...' : 'Approve' }}
                    </button>
                    <button type="button" (click)="submitReview('reject')"
                            [disabled]="busy() !== null || note.trim().length < 3"
                            class="px-4 py-2 rounded-lg text-sm font-semibold bg-red-600/80
                                   hover:bg-red-500 text-white transition-colors disabled:opacity-40
                                   disabled:cursor-not-allowed">
                      {{ busy() === 'reject' ? 'Rejecting...' : 'Reject' }}
                    </button>
                  </div>
                  @if (reviewError()) {
                    <p role="alert" class="mt-3 text-sm text-red-300">{{ reviewError() }}</p>
                  }
                }
              </div>
            }
            @case ('approved') {
              <div class="rounded-xl p-4 mb-4 border border-emerald-500/40 bg-emerald-500/10">
                <p class="text-sm font-semibold text-emerald-300">
                  Approved by {{ rv.reviewed_by || 'a clinician' }}
                  @if (rv.reviewed_at) { on {{ rv.reviewed_at | date:'medium' }} }
                </p>
                @if (rv.note) { <p class="text-xs text-emerald-200/80 mt-1">{{ rv.note }}</p> }
              </div>
            }
            @case ('rejected') {
              <div class="rounded-xl p-4 mb-4 border border-red-500/40 bg-red-500/10">
                <p class="text-sm font-semibold text-red-300">
                  Rejected by {{ rv.reviewed_by || 'a clinician' }}
                  @if (rv.reviewed_at) { on {{ rv.reviewed_at | date:'medium' }} }
                </p>
                @if (rv.note) { <p class="text-sm text-red-200/90 mt-1">Note: {{ rv.note }}</p> }
              </div>
            }
          }
        }

        <!-- Ingestion warnings (e.g. report text/entities truncated) -->
        @if (r.ingestion_warnings?.length) {
          <div class="rounded-xl p-3 mb-4 border border-amber-500/30 bg-amber-500/10
                      text-xs text-amber-300 space-y-1">
            @for (w of r.ingestion_warnings; track $index) {
              <div class="flex gap-2"><span aria-hidden="true">⚠️</span><span>{{ w }}</span></div>
            }
          </div>
        }

        <!-- Security warnings (suspected prompt injection / output leakage) -->
        @if (r.security_warnings?.length) {
          <div class="rounded-xl p-3 mb-4 border border-red-500/40 bg-red-500/10
                      text-xs text-red-300 space-y-1">
            @for (w of r.security_warnings; track $index) {
              <div class="flex gap-2"><span aria-hidden="true">🛡️</span><span>{{ w }} — review this report's source document before trusting its output.</span></div>
            }
          </div>
        }

        <!-- Risk banner -->
        <div class="rounded-xl p-4 mb-3 border flex flex-wrap items-center justify-between gap-3 shadow-lg shadow-black/20"
             [style.background]="sev.getColor(r.risk_summary.overall_severity) + '0D'"
             [style.border-color]="sev.getColor(r.risk_summary.overall_severity) + '40'">
          <div class="flex items-center gap-3 min-w-0">
            <div class="w-3 h-3 rounded-full animate-pulse shrink-0"
                 [style.background]="sev.getColor(r.risk_summary.overall_severity)"></div>
            <div class="min-w-0">
              <div class="font-bold text-base"
                   [style.color]="sev.getColor(r.risk_summary.overall_severity)">
                {{ sev.getBannerTitle(r.risk_summary.overall_severity) }}
              </div>
              <div class="text-xs text-gray-500 mt-0.5 break-words">
                {{ r.filename }} · {{ r.pages }} page(s) · {{ r.source_type | titlecase }}
              </div>
            </div>
          </div>
          <div class="flex gap-2 flex-wrap">
            <span class="px-2 py-1 rounded text-xs font-semibold bg-red-500/20 text-red-400">
              {{ r.risk_summary.critical_count }} Critical
            </span>
            <span class="px-2 py-1 rounded text-xs font-semibold bg-orange-500/20 text-orange-400">
              {{ r.risk_summary.high_count }} High
            </span>
            <span class="px-2 py-1 rounded text-xs font-semibold bg-yellow-500/20 text-yellow-400">
              {{ r.risk_summary.trend_count }} Trends
            </span>
            @if (r.risk_summary.baseline_count) {
              <span class="px-2 py-1 rounded text-xs font-semibold bg-indigo-500/20 text-indigo-400">
                {{ r.risk_summary.baseline_count }} Baseline
              </span>
            }
            <span class="px-2 py-1 rounded text-xs font-semibold bg-gray-700 text-gray-400">
              {{ r.risk_summary.total_flags }} Total
            </span>
          </div>
        </div>

        <!-- Toolbar: timing + FHIR export -->
        <div class="flex flex-wrap items-center justify-between gap-2 mb-5">
          <p class="text-[11px] text-gray-600">{{ perfLine(r) }}</p>
          <div class="flex items-center gap-3">
            @if (fhirError()) {
              <span role="alert" class="text-xs text-red-300">{{ fhirError() }}</span>
            }
            <button type="button" (click)="exportFhir(r)" [disabled]="fhirBusy()"
                    class="px-3 py-1.5 rounded-lg text-xs font-semibold border border-gray-700
                           text-gray-300 hover:border-blue-600 hover:text-white transition-colors
                           disabled:opacity-40 disabled:cursor-not-allowed">
              {{ fhirBusy() ? 'Preparing...' : 'Export FHIR (JSON)' }}
            </button>
          </div>
        </div>

        <!-- Grounding warnings -->
        @if (r.grounding_warnings?.length) {
          <div class="rounded-xl p-4 mb-5 border border-amber-500/40 bg-amber-500/10">
            <p class="text-sm font-semibold text-amber-300">Numbers not found in the extracted data</p>
            <p class="text-xs text-amber-200/80 mt-0.5">Check these against the source report.</p>
            <ul class="mt-2 space-y-1 text-xs text-amber-200 list-disc pl-5">
              @for (w of r.grounding_warnings; track $index) {
                <li>{{ w }}</li>
              }
            </ul>
          </div>
        }

        <!-- Two-column layout -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

          <!-- Left: Risk flags + actions -->
          <div class="min-w-0">
            <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                       pb-2 border-b border-gray-800 mb-3">Risk Flags</p>

            @if (!abnormalFlags().length && !r.trend_flags.length && !(r.baseline_flags?.length)) {
              <app-risk-card [flag]="null" [isAllClear]="true" />
            }
            @for (flag of abnormalFlags(); track flag.test) {
              <app-risk-card [flag]="flag" />
            }
            @for (trend of r.trend_flags; track trend.test) {
              <app-risk-card [flag]="null" [trend]="trend" />
            }
            @for (b of r.baseline_flags; track b.test) {
              <app-risk-card [flag]="null" [baseline]="b" />
            }

            <!-- Longitudinal trajectory narrative -->
            @if (r.llm_output.longitudinal_narrative) {
              <div class="bg-indigo-500/10 border border-indigo-500/30 rounded-lg
                          px-4 py-3 mt-3 text-sm text-indigo-200 leading-relaxed">
                <div class="flex items-center gap-2 mb-1.5">
                  <span class="text-xs font-semibold text-indigo-400 uppercase tracking-wider">
                    📈 Trajectory Across Visits
                  </span>
                </div>
                {{ r.llm_output.longitudinal_narrative }}
              </div>
            }

            <!-- Slope-based forecast — deliberately lower-confidence styling -->
            @for (f of r.forecast_flags; track f.test) {
              <div class="rounded-xl p-4 mb-3 border border-dashed border-gray-700 bg-gray-900/50">
                <div class="flex items-center justify-between mb-1.5">
                  <span class="text-xs font-bold uppercase tracking-widest text-gray-400">
                    🔮 Illustrative Projection · {{ f.test | titlecase }}
                  </span>
                </div>
                <p class="text-sm text-gray-400 leading-relaxed">{{ f.message }}</p>
              </div>
            }

            <!-- Medication interactions -->
            @if (r.drug_interactions; as di) {
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3 mt-6">Medication Interactions</p>
              <app-interactions-card [data]="di" />
            }

            <!-- Drug safety (OpenFDA) -->
            @if (r.drug_safety?.length) {
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3 mt-6">Drug Safety (OpenFDA)</p>
              @for (d of r.drug_safety; track d.medication) {
                <div class="bg-gray-900 border border-gray-800 rounded-lg px-4 py-3 mb-2">
                  <div class="flex items-center justify-between gap-2">
                    <span class="text-sm font-medium text-gray-200">{{ d.medication }}</span>
                    @if (d.has_warnings) {
                      <span class="text-[11px] font-semibold bg-amber-500/20 text-amber-300 px-2 py-0.5 rounded">Warnings on label</span>
                    } @else {
                      <span class="text-[11px] font-semibold bg-gray-700/60 text-gray-400 px-2 py-0.5 rounded">No alerts found</span>
                    }
                  </div>
                  @if (d.boxed_warning) {
                    <p class="text-xs text-red-300 mt-2 leading-relaxed">Boxed warning: {{ d.boxed_warning }}</p>
                  }
                  @if (d.warnings_snippet) {
                    <p class="text-xs text-gray-500 mt-2 leading-relaxed">{{ d.warnings_snippet }}</p>
                  }
                  <p class="text-[11px] text-gray-600 mt-1.5">Source: {{ d.source }}</p>
                </div>
              }
            }

            <!-- Follow-up actions -->
            <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                       pb-2 border-b border-gray-800 mb-3 mt-6">Recommended Actions</p>
            @for (action of r.llm_output.followup_actions; track $index) {
              <div class="flex gap-3 bg-gray-900 border border-gray-800 rounded-lg
                          px-4 py-3 mb-2 text-sm text-gray-300">
                <span class="font-mono text-xs text-blue-400 font-semibold shrink-0 pt-0.5">
                  {{ ($index + 1).toString().padStart(2, '0') }}
                </span>
                {{ action }}
              </div>
            }

            <!-- PubMed references -->
            @if (r.pubmed_references?.length) {
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3 mt-6">Clinical Evidence</p>
              @for (ref of r.pubmed_references; track ref.pmid) {
                <app-pubmed-card [reference]="ref" />
              }
            }

            <!-- Similar past episodes -->
            @if (r.similar_history?.length) {
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3 mt-6">Similar Past Episodes</p>
              @for (item of r.similar_history; track item.report_id) {
                <div class="bg-gray-900 border border-gray-800 rounded-lg px-4 py-3 mb-2">
                  <div class="flex items-center justify-between gap-2 mb-1">
                    <span class="text-xs font-mono text-gray-500 break-all">
                      {{ item.filename }} · {{ item.report_date || 'undated' }}
                    </span>
                    <span class="text-xs font-semibold bg-purple-500/20 text-purple-400
                                  px-2 py-0.5 rounded shrink-0">
                      {{ item.similarity_pct }}% similar
                    </span>
                  </div>
                  <p class="text-sm text-gray-400 leading-relaxed italic">"{{ item.snippet }}"</p>
                </div>
              }
            }
          </div>

          <!-- Right: Summaries + entities -->
          <div class="min-w-0">
            <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                       pb-2 border-b border-gray-800 mb-3">Clinical Summary</p>
            <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-4">
              <div class="flex flex-wrap items-center gap-2 mb-3">
                <div class="w-0.5 h-3 bg-blue-500 rounded"></div>
                <span class="text-xs font-semibold text-blue-400 uppercase tracking-wider">
                  For Healthcare Provider
                </span>
                @if (r.llm_output.verification.clinical_unverified) {
                  <span class="ml-auto text-xs font-semibold bg-red-500/20 text-red-400
                                px-2 py-0.5 rounded">⚠ not verified — review manually</span>
                } @else if (r.llm_output.verification.clinical_summary_corrected) {
                  <span class="ml-auto text-xs font-semibold bg-yellow-500/20 text-yellow-400
                                px-2 py-0.5 rounded">✓ verified-agent corrected</span>
                } @else {
                  <span class="ml-auto text-xs font-semibold bg-green-500/20 text-green-400
                                px-2 py-0.5 rounded">✓ verified</span>
                }
              </div>
              <p class="text-gray-300 text-sm leading-relaxed">{{ r.llm_output.clinical_summary }}</p>
            </div>

            <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                       pb-2 border-b border-gray-800 mb-3">Patient Summary</p>
            <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-4">
              <div class="flex flex-wrap items-center gap-2 mb-3">
                <div class="w-0.5 h-3 bg-blue-500 rounded"></div>
                <span class="text-xs font-semibold text-blue-400 uppercase tracking-wider">
                  In Plain Language
                </span>
                @if (r.llm_output.verification.patient_unverified) {
                  <span class="ml-auto text-xs font-semibold bg-red-500/20 text-red-400
                                px-2 py-0.5 rounded">⚠ not verified — review manually</span>
                } @else if (r.llm_output.verification.patient_summary_corrected) {
                  <span class="ml-auto text-xs font-semibold bg-yellow-500/20 text-yellow-400
                                px-2 py-0.5 rounded">✓ verified-agent corrected</span>
                } @else {
                  <span class="ml-auto text-xs font-semibold bg-green-500/20 text-green-400
                                px-2 py-0.5 rounded">✓ verified</span>
                }
              </div>
              <app-translatable-summary
                [reportId]="r.report_id"
                [text]="r.llm_output.patient_summary"
                [translation]="r.llm_output.translation" />
            </div>

            <!-- Patient info -->
            @if (r.entities.patient.name) {
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3">Patient Info</p>
              <div class="grid grid-cols-3 gap-3 mb-4">
                @for (field of patientFields(); track field.label) {
                  @if (field.value) {
                    <div class="bg-gray-900 border border-gray-800 rounded-lg p-3 text-center min-w-0">
                      <p class="text-xs text-gray-500 uppercase tracking-wider mb-1">{{ field.label }}</p>
                      <p class="font-semibold font-mono text-gray-200 break-words">{{ field.value }}</p>
                    </div>
                  }
                }
              </div>
            }

            <!-- Diagnoses -->
            @if (r.entities.clinical.diagnoses.length) {
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3">Diagnoses</p>
              <div class="flex flex-wrap gap-2 mb-4">
                @for (dx of r.entities.clinical.diagnoses; track dx) {
                  <span class="bg-gray-800 border border-blue-900 text-blue-300
                                font-mono text-xs px-3 py-1.5 rounded-md">{{ dx }}</span>
                }
              </div>
            }

            <!-- Agent metadata -->
            <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                       pb-2 border-b border-gray-800 mb-3">Agent Activity</p>
            <div class="flex flex-wrap gap-2">
              <span class="bg-gray-900 border border-gray-800 font-mono text-xs px-3 py-1.5 rounded-md text-gray-400">
                Source <span class="text-blue-400">{{ r.source_type | titlecase }}</span>
              </span>
              <span class="bg-gray-900 border border-gray-800 font-mono text-xs px-3 py-1.5 rounded-md text-gray-400">
                Tools called <span class="text-blue-400">{{ r.agent_metadata.tool_calls.length }}</span>
              </span>
              <span class="bg-gray-900 border border-gray-800 font-mono text-xs px-3 py-1.5 rounded-md text-gray-400">
                PubMed refs <span class="text-blue-400">{{ r.agent_metadata.pubmed_found }}</span>
              </span>
              <span class="bg-gray-900 border border-gray-800 font-mono text-xs px-3 py-1.5 rounded-md text-gray-400">
                Hindi <span class="text-blue-400">{{ r.has_hindi ? 'Yes' : 'No' }}</span>
              </span>
            </div>
          </div>
        </div>

        <!-- Evidence panel (collapsed by default) -->
        @if (r.evidence?.length) {
          <div class="mt-6">
            <app-evidence-panel [evidence]="r.evidence!" />
          </div>
        }

        @if (r.disclaimer) {
          <p class="mt-6 text-[11px] text-gray-600 leading-relaxed">{{ r.disclaimer }}</p>
        }
      </div>
    }
  `,
})
export class ReportResultComponent {
  @Input() canReview = false;
  @Output() reviewed = new EventEmitter<ReviewStatus>();

  data = signal<ReportAnalysis | null>(null);
  review = signal<ReviewInfo | null>(null);
  busy = signal<'approve' | 'reject' | null>(null);
  reviewError = signal<string | null>(null);
  fhirBusy = signal(false);
  fhirError = signal<string | null>(null);
  note = '';

  @Input({ required: true }) set report(v: ReportAnalysis) {
    this.data.set(v);
    this.review.set(v.review ?? null);
    this.busy.set(null);
    this.reviewError.set(null);
    this.fhirError.set(null);
    this.note = '';
  }

  abnormalFlags = computed(() =>
    (this.data()?.rule_flags ?? []).filter(
      (f) => !['NORMAL', 'Severity.NORMAL'].includes(f.severity)
    )
  );

  patientFields = computed(() => {
    const p = this.data()?.entities.patient;
    return [
      { label: 'Name', value: p?.name },
      { label: 'Age', value: p?.age },
      { label: 'Gender', value: p?.gender },
    ];
  });

  constructor(
    private api: ApiService,
    public sev: SeverityService,
    private reviewState: ReviewStateService,
  ) {}

  perfLine(r: ReportAnalysis): string {
    const ms = r.agent_metadata?.duration_ms;
    const calls = r.agent_metadata?.llm_calls;
    const parts: string[] = [];
    if (typeof ms === 'number') parts.push(`Analysed in ${(ms / 1000).toFixed(1)} s`);
    if (typeof calls === 'number') parts.push(`${calls} model call${calls === 1 ? '' : 's'}`);
    return parts.join(' · ');
  }

  submitReview(decision: 'approve' | 'reject'): void {
    const r = this.data();
    if (!r || this.busy()) return;
    const note = this.note.trim();
    if (decision === 'reject' && note.length < 3) return;

    this.busy.set(decision);
    this.reviewError.set(null);
    this.api.reviewReport(r.report_id, decision, note).subscribe({
      next: (res) => {
        this.review.set({
          status: res.status,
          reviewed_by: res.reviewed_by,
          reviewed_at: res.reviewed_at,
          note: note || null,
        });
        this.busy.set(null);
        this.note = '';
        this.reviewState.refresh();
        this.reviewed.emit(res.status);
      },
      error: (err) => {
        this.busy.set(null);
        this.reviewError.set(errorMessage(err, 'Could not save the review. Please try again.'));
      },
    });
  }

  exportFhir(r: ReportAnalysis): void {
    if (this.fhirBusy()) return;
    this.fhirBusy.set(true);
    this.fhirError.set(null);
    this.api.getFhir(r.report_id).subscribe({
      next: (bundle) => {
        this.fhirBusy.set(false);
        try {
          downloadJson(`report-${r.report_id}-fhir.json`, bundle);
        } catch {
          this.fhirError.set('Could not save the file.');
        }
      },
      error: (err) => {
        this.fhirBusy.set(false);
        this.fhirError.set(errorMessage(err, 'FHIR export failed.'));
      },
    });
  }
}
