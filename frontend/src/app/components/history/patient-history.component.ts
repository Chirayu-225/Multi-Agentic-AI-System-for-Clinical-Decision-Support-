// src/app/components/history/patient-history.component.ts
import { Component, OnDestroy, computed, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { ApiService } from '../../services/api.service';
import { SeverityService } from '../../services/severity.service';
import { errorMessage } from '../../services/error-message';
import {
  PatientHistory, PatientHistoryReport, PatientTrends, ReviewStatus, TrendSeries,
} from '../../models/report.model';
import { TrendChartComponent } from '../shared/trend-chart.component';

const TRENDS_SHOWN = 12;

@Component({
  selector: 'app-patient-history',
  standalone: true,
  imports: [FormsModule, TrendChartComponent],
  template: `
    <div class="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6">

      <!-- Header -->
      <div class="mb-8">
        <p class="text-xs font-semibold text-blue-400 uppercase tracking-widest mb-1">
          Longitudinal Tracking
        </p>
        <h1 class="text-2xl sm:text-3xl font-bold text-white tracking-tight">Patient History</h1>
        <p class="text-gray-500 mt-1 text-sm">
          View lab value trends and risk patterns across multiple visits.
        </p>
      </div>

      <!-- Search -->
      <div class="max-w-md mb-6">
        <label for="history-patient-id"
               class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Patient ID
        </label>
        <div class="flex gap-3">
          <input
            id="history-patient-id"
            [(ngModel)]="patientId"
            placeholder="e.g. PT-2026-001"
            (keydown.enter)="loadHistory()"
            class="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3
                   text-gray-200 font-mono text-sm focus:outline-none focus:border-blue-500
                   focus:ring-1 focus:ring-blue-500 transition-colors"
          />
          <button
            (click)="loadHistory()"
            [disabled]="!patientId || loading()"
            class="px-5 py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-40
                   rounded-lg text-sm font-semibold transition-colors">
            Search
          </button>
        </div>
      </div>

      @if (error()) {
        <div role="alert" class="mb-4 rounded-xl p-3 border border-red-500/40 bg-red-500/10 text-sm text-red-300">
          {{ error() }}
        </div>
      }

      <!-- Idle empty state — before any search has been made -->
      @if (!history() && !loading() && !error()) {
        <div class="border border-dashed border-gray-800 rounded-xl px-6 py-12
                    flex flex-col items-center text-center bg-gray-900/30">
          <div class="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-600/20 to-sky-500/20
                      border border-blue-500/20 flex items-center justify-center text-2xl mb-4">
            📈
          </div>
          <p class="text-sm font-medium text-gray-400 mb-1.5">No patient loaded</p>
          <p class="text-xs text-gray-600 max-w-sm leading-relaxed">
            Enter a patient ID above to view their lab value trends, personal-baseline
            deviations and visit-by-visit report history.
          </p>
        </div>
      }

      <!-- Loading state (first search only — a re-search keeps the existing results visible) -->
      @if (loading() && !history()) {
        <div class="border border-gray-800 rounded-xl px-6 py-12
                    flex flex-col items-center text-center bg-gray-900/30">
          <div class="flex gap-1.5 mb-4">
            <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></span>
            <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.15s]"></span>
            <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.3s]"></span>
          </div>
          <p class="text-sm text-gray-500">Loading patient history...</p>
        </div>
      }

      <!-- Results -->
      @if (history()) {
        <div class="mb-4 text-sm text-gray-500">
          <span class="font-mono text-blue-400">{{ history()!.patient_id }}</span>
          · {{ history()!.report_count }} report(s) on record
        </div>

        @if (history()!.reports.length === 0) {
          <div class="bg-gray-900 border border-dashed border-gray-800 rounded-xl p-12
                      flex flex-col items-center text-center">
            <div class="w-12 h-12 rounded-xl bg-gray-800 flex items-center justify-center text-xl mb-3">
              📄
            </div>
            <p class="text-gray-400 text-sm font-medium mb-1">No reports found</p>
            <p class="text-gray-600 text-xs max-w-xs leading-relaxed">
              This patient ID has no analysed reports yet. Run an analysis from the
              Analyse Report page to start building their history.
            </p>
          </div>
        } @else {

          <!-- Lab trends -->
          <div class="mb-6">
            <div class="flex items-center justify-between mb-3 pb-2 border-b border-gray-800">
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest">Lab trends</p>
              @if (series().length > shownLimit) {
                <button type="button" (click)="showAll.set(!showAll())"
                        class="text-xs text-blue-400 hover:text-blue-300 transition-colors">
                  {{ showAll() ? 'Show fewer' : 'Show all (' + series().length + ')' }}
                </button>
              }
            </div>
            @if (trendsLoading()) {
              <p class="text-sm text-gray-600 py-4" role="status">Loading lab trends...</p>
            } @else if (trendsError()) {
              <p role="alert" class="text-sm text-red-300">{{ trendsError() }}</p>
            } @else if (!series().length) {
              <p class="text-sm text-gray-600 py-4">
                No numeric lab results with a recorded unit were found for this patient yet.
              </p>
            } @else {
              <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                @for (s of visibleSeries(); track s.test + '|' + s.unit) {
                  <app-trend-chart [series]="s" />
                }
              </div>
              @if (maxPoints() < 2) {
                <p class="text-xs text-gray-600 mt-3">
                  Upload more reports for this patient to see how values change over time.
                </p>
              }
            }
          </div>

          <!-- Visit history cards -->
          <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                     pb-2 border-b border-gray-800 mb-4">Visit History</p>

          @for (report of sortedReports().slice().reverse(); track report.report_id) {
            <div class="bg-gray-900 border border-gray-800 rounded-xl mb-3 overflow-hidden
                        hover:border-gray-700 transition-colors">
              <button
                class="w-full flex items-center justify-between px-5 py-4 text-left
                       hover:bg-gray-800/50 transition-colors"
                (click)="toggleReport(report.report_id)">
                <div class="flex items-center gap-3">
                  <span class="text-gray-400 text-sm">📄</span>
                  <div>
                    <p class="text-sm font-medium text-gray-200 break-all">{{ report.filename }}</p>
                    <p class="text-xs text-gray-500 font-mono">{{ report.report_date }}</p>
                  </div>
                </div>
                <div class="flex flex-wrap items-center justify-end gap-2 sm:gap-3">
                  @if (report.review_status) {
                    <span class="text-[11px] font-semibold px-2 py-0.5 rounded border"
                          [class]="reviewChipClass(report.review_status)">
                      {{ reviewLabel(report.review_status) }}
                    </span>
                  }
                  @if (report.risk_summary?.overall_severity) {
                    <span class="text-xs font-semibold px-2 py-0.5 rounded font-mono"
                          [style.color]="sev.getColor(report.risk_summary.overall_severity)"
                          [style.background]="sev.getColor(report.risk_summary.overall_severity) + '20'">
                      {{ sev.normalize(report.risk_summary.overall_severity) }}
                    </span>
                  }
                  <span class="text-gray-600 text-sm">
                    {{ expandedReports().has(report.report_id) ? '▲' : '▼' }}
                  </span>
                </div>
              </button>

              @if (expandedReports().has(report.report_id)) {
                <div class="px-5 pb-4 border-t border-gray-800">
                  @if (report.diagnoses?.length) {
                    <div class="mt-3 mb-2">
                      <span class="text-xs text-gray-500">Diagnoses: </span>
                      @for (dx of report.diagnoses; track dx) {
                        <span class="text-xs bg-gray-800 text-blue-300 px-2 py-0.5
                                      rounded font-mono mr-1">{{ dx }}</span>
                      }
                    </div>
                  }
                  @if (report.lab_values?.length) {
                    <div class="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-3">
                      @for (lab of report.lab_values; track lab.test) {
                        @if (lab.value !== null) {
                          <div class="bg-gray-800 rounded-lg px-3 py-2">
                            <p class="text-xs text-gray-500">{{ lab.test }}</p>
                            <p class="text-sm font-mono font-semibold text-gray-200">
                              {{ lab.value }} {{ lab.unit }}
                            </p>
                          </div>
                        }
                      }
                    </div>
                  }
                </div>
              }
            </div>
          }
        }
      }
    </div>
  `,
})
export class PatientHistoryComponent implements OnDestroy {
  patientId  = '';
  loading    = signal(false);
  error      = signal<string | null>(null);
  history    = signal<PatientHistory | null>(null);
  expandedReports = signal(new Set<string>());

  trends        = signal<PatientTrends | null>(null);
  trendsLoading = signal(false);
  trendsError   = signal<string | null>(null);
  showAll       = signal(false);
  readonly shownLimit = TRENDS_SHOWN;

  // Series with the most readings first (stable for ties).
  series = computed<TrendSeries[]>(() =>
    [...(this.trends()?.series ?? [])]
      .filter((s) => s.points?.length)
      .sort((a, b) => b.points.length - a.points.length)
  );
  visibleSeries = computed(() =>
    this.showAll() ? this.series() : this.series().slice(0, TRENDS_SHOWN)
  );
  maxPoints = computed(() => Math.max(0, ...this.series().map((s) => s.points.length)));

  private histSub?: Subscription;
  private trendSub?: Subscription;

  constructor(private api: ApiService, public sev: SeverityService) {}

  loadHistory(): void {
    const pid = this.patientId.trim();
    if (!pid) return;
    this.histSub?.unsubscribe();
    this.trendSub?.unsubscribe();
    this.loading.set(true);
    this.error.set(null);
    this.trendsError.set(null);
    this.trendsLoading.set(true);
    this.showAll.set(false);

    this.histSub = this.api.getPatientHistory(pid).subscribe({
      next: (h) => {
        this.history.set(h);
        this.loading.set(false);
      },
      error: (err) => {
        this.history.set(null);
        this.trends.set(null);
        this.loading.set(false);
        this.trendsLoading.set(false);
        this.trendSub?.unsubscribe();
        this.error.set(errorMessage(err, 'Could not load this patient\'s history.'));
      },
    });

    this.trendSub = this.api.getPatientTrends(pid).subscribe({
      next: (t) => {
        this.trends.set(t);
        this.trendsLoading.set(false);
      },
      error: (err) => {
        this.trends.set(null);
        this.trendsLoading.set(false);
        this.trendsError.set(errorMessage(err, 'Could not load lab trends.'));
      },
    });
  }

  sortedReports(): PatientHistoryReport[] {
    return [...(this.history()?.reports ?? [])].sort(
      (a, b) => a.report_date.localeCompare(b.report_date)
    );
  }

  reviewLabel(s: ReviewStatus): string {
    return s === 'approved' ? 'Approved' : s === 'rejected' ? 'Rejected' : 'Pending review';
  }

  reviewChipClass(s: ReviewStatus): string {
    return s === 'approved'
      ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
      : s === 'rejected'
        ? 'bg-red-500/15 text-red-300 border-red-500/30'
        : 'bg-amber-500/15 text-amber-300 border-amber-500/30';
  }

  toggleReport(id: string): void {
    this.expandedReports.update(set => {
      const next = new Set(set);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  ngOnDestroy(): void {
    this.histSub?.unsubscribe();
    this.trendSub?.unsubscribe();
  }
}
