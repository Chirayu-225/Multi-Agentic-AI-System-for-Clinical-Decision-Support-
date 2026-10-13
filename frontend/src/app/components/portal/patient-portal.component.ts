// src/app/components/portal/patient-portal.component.ts  (patient role only)
// Shows only what the API returns for a patient: released (approved) reports in
// plain language. No clinician-facing fields exist in these responses.
import { Component, OnDestroy, OnInit, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { Subscription } from 'rxjs';

import { ApiService } from '../../services/api.service';
import { SeverityService } from '../../services/severity.service';
import { errorMessage } from '../../services/error-message';
import {
  MyReportItem, PatientLabValue, PatientReportView, TrendSeries,
} from '../../models/report.model';
import { TrendChartComponent } from '../shared/trend-chart.component';
import { TranslatableSummaryComponent } from '../shared/translatable-summary.component';

interface Chip { text: string; cls: string }

const CHIP_CLASSES: Record<string, string> = {
  NORMAL: 'bg-green-500/15 text-green-300 border-green-500/30',
  MODERATE: 'bg-yellow-500/15 text-yellow-300 border-yellow-500/30',
  HIGH: 'bg-orange-500/15 text-orange-300 border-orange-500/30',
  CRITICAL: 'bg-red-500/15 text-red-300 border-red-500/30',
};

@Component({
  selector: 'app-patient-portal',
  standalone: true,
  imports: [DatePipe, TrendChartComponent, TranslatableSummaryComponent],
  template: `
    <div class="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6">

      <div class="mb-6">
        <p class="text-xs font-semibold text-blue-400 uppercase tracking-widest mb-1">Your health records</p>
        <h1 class="text-2xl sm:text-3xl font-bold text-white tracking-tight">My reports</h1>
        <p class="text-gray-500 mt-1 text-sm">
          Reports appear here after your care team has reviewed them.
        </p>
      </div>

      @if (listLoading()) {
        <div class="border border-gray-800 rounded-xl px-6 py-12 flex flex-col items-center bg-gray-900/30" role="status">
          <div class="flex gap-1.5 mb-4">
            <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></span>
            <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.15s]"></span>
            <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.3s]"></span>
          </div>
          <p class="text-sm text-gray-500">Loading your reports...</p>
        </div>
      } @else if (listError()) {
        <div role="alert" class="rounded-xl p-3 border border-red-500/40 bg-red-500/10 text-sm text-red-300">
          {{ listError() }}
        </div>
      } @else if (!reports().length) {
        <div class="border border-dashed border-gray-800 rounded-xl px-6 py-12
                    flex flex-col items-center text-center bg-gray-900/30">
          <div class="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-600/20 to-sky-500/20
                      border border-blue-500/20 flex items-center justify-center text-2xl mb-4" aria-hidden="true">📄</div>
          <p class="text-sm font-medium text-gray-400 mb-1.5">No reports have been released to you yet</p>
          <p class="text-xs text-gray-600 max-w-sm leading-relaxed">
            Your care team reviews every report before it appears here.
          </p>
        </div>
      } @else {
        <div class="grid grid-cols-1 lg:grid-cols-[17rem_minmax(0,1fr)] gap-6">

          <!-- Report list -->
          <nav aria-label="Your reports">
            <ul class="space-y-2">
              @for (it of reports(); track it.report_id) {
                <li>
                  <button type="button" (click)="select(it.report_id)"
                          [attr.aria-current]="selectedId() === it.report_id ? 'true' : null"
                          class="w-full text-left rounded-xl px-4 py-3 border transition-colors
                                 focus:outline-none focus:ring-1 focus:ring-blue-500"
                          [class]="selectedId() === it.report_id
                            ? 'bg-blue-600/10 border-blue-600/50'
                            : 'bg-gray-900 border-gray-800 hover:border-gray-700'">
                    <p class="text-sm font-medium text-gray-200">{{ it.report_date || 'Undated report' }}</p>
                    <p class="text-xs text-gray-500 break-all">{{ it.filename }}</p>
                    <span class="inline-block mt-2 text-[11px] font-semibold px-2 py-0.5 rounded border"
                          [class]="overall(it.overall_severity).cls">
                      {{ overall(it.overall_severity).text }}
                    </span>
                  </button>
                </li>
              }
            </ul>
          </nav>

          <!-- Detail -->
          <div class="min-w-0">
            @if (detailLoading()) {
              <p class="text-sm text-gray-500 py-6" role="status">Loading report...</p>
            }
            @if (detailError()) {
              <div role="alert" class="rounded-xl p-3 border border-red-500/40 bg-red-500/10 text-sm text-red-300">
                {{ detailError() }}
              </div>
            }
            @if (view(); as v) {
              <div class="bg-gray-900 border border-gray-800 rounded-xl p-5 mb-5">
                <div class="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h2 class="text-lg font-semibold text-white">Your report from {{ v.report_date || 'an unknown date' }}</h2>
                    <p class="text-xs text-gray-500 mt-0.5 break-all">{{ v.filename }}</p>
                    @if (v.review.reviewed_by) {
                      <p class="text-xs text-gray-500 mt-1">
                        Reviewed by {{ v.review.reviewed_by }}
                        @if (v.review.reviewed_at) { on {{ v.review.reviewed_at | date:'mediumDate' }} }
                      </p>
                    }
                  </div>
                  <span class="text-xs font-semibold px-3 py-1 rounded-full border"
                        [class]="overall(v.risk_summary.overall_severity).cls">
                    {{ overall(v.risk_summary.overall_severity).text }}
                  </span>
                </div>
              </div>

              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3">In plain language</p>
              <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-5">
                <app-translatable-summary
                  [reportId]="v.report_id"
                  [text]="v.patient_summary"
                  [translation]="v.patient_summary_translated"
                  [preferTranslated]="true" />
              </div>

              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3">Your results</p>
              @if (v.lab_values.length) {
                <div class="bg-gray-900 border border-gray-800 rounded-xl overflow-x-auto mb-5">
                  <table class="min-w-full text-sm">
                    <caption class="sr-only">Your lab results</caption>
                    <thead>
                      <tr class="text-left text-xs text-gray-500 border-b border-gray-800">
                        <th scope="col" class="px-4 py-2.5 font-semibold">Test</th>
                        <th scope="col" class="px-4 py-2.5 font-semibold">Your result</th>
                        <th scope="col" class="px-4 py-2.5 font-semibold">Normal range</th>
                        <th scope="col" class="px-4 py-2.5 font-semibold">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      @for (l of v.lab_values; track $index) {
                        <tr class="border-b border-gray-800/60 last:border-b-0">
                          <td class="px-4 py-2.5 text-gray-200">{{ l.test }}</td>
                          <td class="px-4 py-2.5 font-mono text-gray-100 whitespace-nowrap">
                            {{ l.value === null ? '—' : l.value }} {{ l.unit }}
                          </td>
                          <td class="px-4 py-2.5 font-mono text-gray-400 whitespace-nowrap">
                            @if (l.normal_range; as nr) { {{ nr.min }}–{{ nr.max }} {{ nr.unit || l.unit }} } @else { — }
                          </td>
                          <td class="px-4 py-2.5">
                            <span class="text-[11px] font-semibold px-2 py-0.5 rounded border whitespace-nowrap"
                                  [class]="labChip(l).cls">{{ labChip(l).text }}</span>
                          </td>
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
              } @else {
                <p class="text-sm text-gray-600 mb-5">No individual lab values were found in this report.</p>
              }

              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                         pb-2 border-b border-gray-800 mb-3">How your results change over time</p>
              @if (trendsLoading()) {
                <p class="text-sm text-gray-600" role="status">Loading charts...</p>
              } @else if (trendsError()) {
                <p role="alert" class="text-sm text-red-300">{{ trendsError() }}</p>
              } @else if (!trends().length) {
                <p class="text-sm text-gray-600">Charts will appear once there are results to compare.</p>
              } @else {
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  @for (s of trends(); track s.test + '|' + s.unit) {
                    <app-trend-chart [series]="s" />
                  }
                </div>
              }

              @if (v.disclaimer) {
                <p class="mt-6 text-[11px] text-gray-600 leading-relaxed">{{ v.disclaimer }}</p>
              }
            }
          </div>
        </div>
      }
    </div>
  `,
})
export class PatientPortalComponent implements OnInit, OnDestroy {
  reports = signal<MyReportItem[]>([]);
  listLoading = signal(true);
  listError = signal<string | null>(null);

  selectedId = signal<string | null>(null);
  view = signal<PatientReportView | null>(null);
  detailLoading = signal(false);
  detailError = signal<string | null>(null);

  trends = signal<TrendSeries[]>([]);
  trendsLoading = signal(false);
  trendsError = signal<string | null>(null);
  private trendsFor: string | null = null;

  private subs: Subscription[] = [];
  private detailSub?: Subscription;
  private trendSub?: Subscription;

  constructor(private api: ApiService, private sev: SeverityService) {}

  ngOnInit(): void {
    this.subs.push(
      this.api.getMyReports().subscribe({
        next: (r) => {
          const list = r.reports ?? [];
          this.reports.set(list);
          this.listLoading.set(false);
          if (list.length) this.select(list[0].report_id);
        },
        error: (err) => {
          this.listLoading.set(false);
          this.listError.set(errorMessage(err, 'Could not load your reports.'));
        },
      })
    );
  }

  select(id: string): void {
    this.detailSub?.unsubscribe();
    this.selectedId.set(id);
    this.view.set(null);
    this.detailError.set(null);
    this.detailLoading.set(true);
    this.detailSub = this.api.getPatientReport(id).subscribe({
      next: (v) => {
        this.view.set(v);
        this.detailLoading.set(false);
        this.loadTrends(v.patient_id);
      },
      error: (err) => {
        this.detailLoading.set(false);
        this.detailError.set(errorMessage(err, 'Could not load this report.'));
      },
    });
  }

  private loadTrends(patientId: string): void {
    if (this.trendsFor === patientId) return;
    this.trendsFor = patientId;
    this.trendSub?.unsubscribe();
    this.trendsLoading.set(true);
    this.trendsError.set(null);
    this.trendSub = this.api.getPatientTrends(patientId).subscribe({
      next: (t) => {
        this.trends.set(
          [...(t.series ?? [])]
            .filter((s) => s.points?.length)
            .sort((a, b) => b.points.length - a.points.length)
        );
        this.trendsLoading.set(false);
      },
      error: (err) => {
        this.trendsFor = null;
        this.trendsLoading.set(false);
        this.trendsError.set(errorMessage(err, 'Could not load your charts.'));
      },
    });
  }

  overall(raw: string): Chip {
    const level = this.sev.normalize(raw);
    const text =
      level === 'NORMAL' ? 'Everything looks in range'
      : level === 'MODERATE' ? 'A few things to keep an eye on'
      : level === 'HIGH' ? 'Some results need follow-up'
      : 'Needs urgent attention';
    return { text, cls: CHIP_CLASSES[level] };
  }

  labChip(l: PatientLabValue): Chip {
    const level = this.sev.normalize(l.severity);
    const dir = l.direction;
    let text: string;
    switch (level) {
      case 'NORMAL':
        text = 'Normal';
        break;
      case 'MODERATE':
        text = dir === 'high' ? 'Slightly high' : dir === 'low' ? 'Slightly low' : 'Slightly outside normal';
        break;
      case 'HIGH':
        text = dir === 'high' ? 'High' : dir === 'low' ? 'Low' : 'Outside normal range';
        break;
      default:
        text = 'Needs urgent attention';
    }
    return { text, cls: CHIP_CLASSES[level] };
  }

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
    this.detailSub?.unsubscribe();
    this.trendSub?.unsubscribe();
  }
}
