// src/app/components/review/review-queue.component.ts  (clinician only)
import { Component, OnDestroy, OnInit, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { Subscription } from 'rxjs';

import { ApiService } from '../../services/api.service';
import { SeverityService } from '../../services/severity.service';
import { errorMessage } from '../../services/error-message';
import { PendingReviewItem, ReportAnalysis, ReviewStatus } from '../../models/report.model';
import { ReportResultComponent } from '../report/report-result.component';

@Component({
  selector: 'app-review-queue',
  standalone: true,
  imports: [DatePipe, ReportResultComponent],
  template: `
    <div class="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6">

      <div class="mb-6 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p class="text-xs font-semibold text-blue-400 uppercase tracking-widest mb-1">Human in the loop</p>
          <h1 class="text-2xl sm:text-3xl font-bold text-white tracking-tight">Review queue</h1>
          <p class="text-gray-500 mt-1 text-sm">
            Reports stay hidden from patients until a clinician approves them.
          </p>
        </div>
        @if (!selected()) {
          <button type="button" (click)="load()" [disabled]="loading()"
                  class="px-4 py-2 rounded-lg text-sm font-semibold border border-gray-700 text-gray-300
                         hover:border-blue-600 hover:text-white transition-colors disabled:opacity-40">
            {{ loading() ? 'Refreshing...' : 'Refresh' }}
          </button>
        }
      </div>

      @if (notice()) {
        <div role="status" class="mb-4 rounded-xl p-3 border border-emerald-500/30 bg-emerald-500/10 text-sm text-emerald-300">
          {{ notice() }}
        </div>
      }

      @if (selectedId()) {
        <!-- Detail view -->
        <button type="button" (click)="closeDetail()"
                class="mb-4 text-sm text-gray-400 hover:text-white transition-colors">
          ← Back to queue
        </button>

        @if (detailLoading()) {
          <div class="border border-gray-800 rounded-xl px-6 py-12 flex flex-col items-center bg-gray-900/30" role="status">
            <div class="flex gap-1.5 mb-4">
              <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></span>
              <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.15s]"></span>
              <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.3s]"></span>
            </div>
            <p class="text-sm text-gray-500">Loading report...</p>
          </div>
        }
        @if (detailError()) {
          <div role="alert" class="rounded-xl p-3 border border-red-500/40 bg-red-500/10 text-sm text-red-300">
            {{ detailError() }}
          </div>
        }
        @if (selected(); as rep) {
          <app-report-result [report]="rep" [canReview]="true" (reviewed)="onReviewed($event)" />
        }
      } @else {
        <!-- List view -->
        @if (loading() && !items().length) {
          <div class="border border-gray-800 rounded-xl px-6 py-12 flex flex-col items-center bg-gray-900/30" role="status">
            <div class="flex gap-1.5 mb-4">
              <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></span>
              <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.15s]"></span>
              <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.3s]"></span>
            </div>
            <p class="text-sm text-gray-500">Loading queue...</p>
          </div>
        } @else if (error()) {
          <div role="alert" class="rounded-xl p-3 border border-red-500/40 bg-red-500/10 text-sm text-red-300">
            {{ error() }}
          </div>
        } @else if (!items().length) {
          <div class="border border-dashed border-gray-800 rounded-xl px-6 py-12
                      flex flex-col items-center text-center bg-gray-900/30">
            <div class="w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/20
                        flex items-center justify-center text-2xl mb-4" aria-hidden="true">✅</div>
            <p class="text-sm font-medium text-gray-400 mb-1.5">Nothing waiting for review</p>
            <p class="text-xs text-gray-600 max-w-sm leading-relaxed">
              New uploads will appear here until they are approved or rejected.
            </p>
          </div>
        } @else {
          <p class="text-xs text-gray-500 mb-3">{{ items().length }} report(s) waiting</p>
          <ul class="space-y-2">
            @for (it of items(); track it.report_id) {
              <li>
                <button type="button" (click)="open(it)"
                        class="w-full text-left bg-gray-900 border border-gray-800 rounded-xl px-4 py-3
                               hover:border-blue-700 focus:outline-none focus:border-blue-500
                               focus:ring-1 focus:ring-blue-500 transition-colors">
                  <div class="flex flex-wrap items-center justify-between gap-2">
                    <div class="min-w-0">
                      <p class="text-sm font-medium text-gray-200">
                        <span class="font-mono text-blue-400">{{ it.patient_id }}</span>
                        <span class="text-gray-600"> · </span>
                        <span class="break-all">{{ it.filename }}</span>
                      </p>
                      <p class="text-xs text-gray-500 font-mono mt-0.5">
                        report {{ it.report_date || 'undated' }} · uploaded {{ it.created_at | date:'medium' }}
                      </p>
                    </div>
                    <div class="flex flex-wrap items-center gap-1.5">
                      <span class="text-xs font-semibold px-2 py-0.5 rounded font-mono"
                            [style.color]="sev.getColor(it.overall_severity)"
                            [style.background]="sev.getColor(it.overall_severity) + '20'">
                        {{ sev.normalize(it.overall_severity) }}
                      </span>
                      @if (it.security_warnings > 0) {
                        <span class="text-[11px] font-semibold px-2 py-0.5 rounded bg-red-500/20 text-red-300">
                          {{ it.security_warnings }} security
                        </span>
                      }
                      @if (it.grounding_warnings > 0) {
                        <span class="text-[11px] font-semibold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300">
                          {{ it.grounding_warnings }} grounding
                        </span>
                      }
                      @if (it.interaction_alerts > 0) {
                        <span class="text-[11px] font-semibold px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">
                          {{ it.interaction_alerts }} interaction{{ it.interaction_alerts === 1 ? '' : 's' }}
                        </span>
                      }
                    </div>
                  </div>
                </button>
              </li>
            }
          </ul>
        }
      }
    </div>
  `,
})
export class ReviewQueueComponent implements OnInit, OnDestroy {
  items = signal<PendingReviewItem[]>([]);
  loading = signal(false);
  error = signal<string | null>(null);
  notice = signal<string | null>(null);

  selectedId = signal<string | null>(null);
  selected = signal<ReportAnalysis | null>(null);
  detailLoading = signal(false);
  detailError = signal<string | null>(null);

  private listSub?: Subscription;
  private detailSub?: Subscription;

  constructor(private api: ApiService, public sev: SeverityService) {}

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.listSub?.unsubscribe();
    this.loading.set(true);
    this.error.set(null);
    this.listSub = this.api.getPendingReviews().subscribe({
      next: (r) => {
        this.items.set(r.reports ?? []);
        this.loading.set(false);
      },
      error: (err) => {
        this.loading.set(false);
        this.error.set(errorMessage(err, 'Could not load the review queue.'));
      },
    });
  }

  open(it: PendingReviewItem): void {
    this.detailSub?.unsubscribe();
    this.notice.set(null);
    this.selectedId.set(it.report_id);
    this.selected.set(null);
    this.detailError.set(null);
    this.detailLoading.set(true);
    this.detailSub = this.api.getReport(it.report_id).subscribe({
      next: (rep) => {
        this.selected.set(rep);
        this.detailLoading.set(false);
      },
      error: (err) => {
        this.detailLoading.set(false);
        this.detailError.set(errorMessage(err, 'Could not load this report.'));
      },
    });
  }

  closeDetail(): void {
    this.detailSub?.unsubscribe();
    this.selectedId.set(null);
    this.selected.set(null);
    this.detailError.set(null);
    this.detailLoading.set(false);
  }

  onReviewed(status: ReviewStatus): void {
    const id = this.selectedId();
    const item = this.items().find((i) => i.report_id === id);
    this.items.update((list) => list.filter((i) => i.report_id !== id));
    this.closeDetail();
    this.notice.set(
      `Report${item ? ' for ' + item.patient_id : ''} ${status === 'approved' ? 'approved and released to the patient' : 'rejected'}.`
    );
  }

  ngOnDestroy(): void {
    this.listSub?.unsubscribe();
    this.detailSub?.unsubscribe();
  }
}
