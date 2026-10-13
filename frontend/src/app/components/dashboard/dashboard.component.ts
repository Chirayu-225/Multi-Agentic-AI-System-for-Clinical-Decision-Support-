// src/app/components/dashboard/dashboard.component.ts  (clinician only)
// Operations dashboard: pipeline health, safety signals, security events and
// the tamper-evident audit log. Charts are hand-written inline SVG with one
// restrained colour; all text is interpolated, never rendered as markup.
import { Component, OnDestroy, OnInit, computed, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { Subscription } from 'rxjs';

import { ApiService } from '../../services/api.service';
import { errorMessage } from '../../services/error-message';
import { AdminMetrics, AuditEvent, AuditVerification } from '../../models/report.model';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// Bar chart geometry (viewBox units)
const CW = 420, CH = 190;
const PAD = { l: 28, r: 8, t: 20, b: 28 };

interface Bar { x: number; y: number; w: number; h: number; count: number; label: string; tip: string }

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [DatePipe],
  template: `
    <div class="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6">

      <div class="mb-6 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p class="text-xs font-semibold text-blue-400 uppercase tracking-widest mb-1">Operations</p>
          <h1 class="text-2xl sm:text-3xl font-bold text-white tracking-tight">Dashboard</h1>
          <p class="text-gray-500 mt-1 text-sm">Pipeline health, safety signals and security events.</p>
        </div>
        <div class="flex items-center gap-3">
          @if (updatedAt()) {
            <span class="text-xs text-gray-600 font-mono">Updated {{ updatedAt() | date:'HH:mm:ss' }}</span>
          }
          <button type="button" (click)="refresh()" [disabled]="loading()"
                  class="px-4 py-2 rounded-lg text-sm font-semibold border border-gray-700 text-gray-300
                         hover:border-blue-600 hover:text-white transition-colors disabled:opacity-40">
            {{ loading() ? 'Refreshing...' : 'Refresh' }}
          </button>
        </div>
      </div>

      @if (error()) {
        <div role="alert" class="mb-4 rounded-xl p-3 border border-red-500/40 bg-red-500/10 text-sm text-red-300">
          {{ error() }}
        </div>
      }

      @if (!metrics() && loading()) {
        <div class="border border-gray-800 rounded-xl px-6 py-12 flex flex-col items-center bg-gray-900/30" role="status">
          <div class="flex gap-1.5 mb-4">
            <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></span>
            <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.15s]"></span>
            <span class="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.3s]"></span>
          </div>
          <p class="text-sm text-gray-500">Loading metrics...</p>
        </div>
      }

      @if (metrics(); as m) {
        <!-- Stat tiles -->
        <div class="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3 mb-6">
          <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p class="text-[11px] text-gray-500 uppercase tracking-wider">Reports</p>
            <p class="text-2xl font-bold font-mono text-gray-100 mt-1">{{ m.reports.total }}</p>
            <p class="text-[11px] text-gray-600 mt-1">
              {{ m.reports.by_review_status.approved }} approved · {{ m.reports.by_review_status.rejected }} rejected
            </p>
          </div>
          <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p class="text-[11px] text-gray-500 uppercase tracking-wider">Last 24 h</p>
            <p class="text-2xl font-bold font-mono text-gray-100 mt-1">{{ m.reports.last_24h }}</p>
            <p class="text-[11px] text-gray-600 mt-1">uploads</p>
          </div>
          <div class="bg-gray-900 border rounded-xl p-4"
               [class]="m.reports.by_review_status.pending_review > 0 ? 'border-amber-500/40' : 'border-gray-800'">
            <p class="text-[11px] text-gray-500 uppercase tracking-wider">Pending review</p>
            <p class="text-2xl font-bold font-mono mt-1"
               [class]="m.reports.by_review_status.pending_review > 0 ? 'text-amber-300' : 'text-gray-100'">
              {{ m.reports.by_review_status.pending_review }}
            </p>
            <p class="text-[11px] text-gray-600 mt-1">awaiting a clinician</p>
          </div>
          <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p class="text-[11px] text-gray-500 uppercase tracking-wider">Avg pipeline time</p>
            <p class="text-2xl font-bold font-mono text-gray-100 mt-1">{{ secs(m.pipeline.avg_duration_ms) }}</p>
            <p class="text-[11px] text-gray-600 mt-1">p95 {{ secs(m.pipeline.p95_duration_ms) }}</p>
          </div>
          <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p class="text-[11px] text-gray-500 uppercase tracking-wider">Model calls / report</p>
            <p class="text-2xl font-bold font-mono text-gray-100 mt-1">{{ num(m.pipeline.avg_llm_calls) }}</p>
            <p class="text-[11px] text-gray-600 mt-1">average</p>
          </div>
          <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p class="text-[11px] text-gray-500 uppercase tracking-wider">Tool calls / report</p>
            <p class="text-2xl font-bold font-mono text-gray-100 mt-1">{{ num(m.pipeline.avg_tool_calls) }}</p>
            <p class="text-[11px] text-gray-600 mt-1">average</p>
          </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
          <!-- Uploads, last 7 days -->
          <div class="lg:col-span-2 bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3">Uploads, last 7 days</p>
            <svg class="w-full h-auto block" [attr.viewBox]="'0 0 ' + CW + ' ' + CH" role="img"
                 [attr.aria-label]="barAria()">
              <line [attr.x1]="PAD.l" [attr.x2]="CW - PAD.r" [attr.y1]="baseY" [attr.y2]="baseY"
                    stroke="#374151" stroke-width="1"/>
              @for (b of bars(); track b.label) {
                <rect [attr.x]="b.x" [attr.y]="b.y" [attr.width]="b.w" [attr.height]="b.h"
                      rx="4" fill="#38bdf8" fill-opacity="0.85">
                  <title>{{ b.tip }}</title>
                </rect>
                <text [attr.x]="b.x + b.w / 2" [attr.y]="b.y - 5" text-anchor="middle"
                      font-size="10" fill="#d1d5db" font-family="JetBrains Mono, monospace">{{ b.count }}</text>
                <text [attr.x]="b.x + b.w / 2" [attr.y]="baseY + 16" text-anchor="middle"
                      font-size="9" fill="#6b7280">{{ b.label }}</text>
              }
            </svg>
          </div>

          <!-- Safety -->
          <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3">Safety</p>
            <div class="space-y-4">
              <div>
                <div class="flex justify-between text-sm">
                  <span class="text-gray-400">Verifier correction rate</span>
                  <span class="font-mono text-gray-100">{{ pct(m.safety.verifier_correction_rate) }}</span>
                </div>
                <div class="h-1.5 bg-gray-800 rounded mt-1.5" aria-hidden="true">
                  <div class="h-1.5 bg-sky-400 rounded" [style.width.%]="pctNum(m.safety.verifier_correction_rate)"></div>
                </div>
              </div>
              <div>
                <div class="flex justify-between text-sm">
                  <span class="text-gray-400">Unverified rate</span>
                  <span class="font-mono text-gray-100">{{ pct(m.safety.verifier_unverified_rate) }}</span>
                </div>
                <div class="h-1.5 bg-gray-800 rounded mt-1.5" aria-hidden="true">
                  <div class="h-1.5 bg-sky-400 rounded" [style.width.%]="pctNum(m.safety.verifier_unverified_rate)"></div>
                </div>
              </div>
              <dl class="text-sm space-y-1.5 pt-1">
                <div class="flex justify-between"><dt class="text-gray-400">Reports with security warnings</dt>
                  <dd class="font-mono text-gray-100">{{ m.safety.reports_with_security_warnings }}</dd></div>
                <div class="flex justify-between"><dt class="text-gray-400">Reports with grounding warnings</dt>
                  <dd class="font-mono text-gray-100">{{ m.safety.reports_with_grounding_warnings }}</dd></div>
                <div class="flex justify-between"><dt class="text-gray-400">Reports with interaction alerts</dt>
                  <dd class="font-mono text-gray-100">{{ m.safety.reports_with_interaction_alerts }}</dd></div>
              </dl>
            </div>
          </div>
        </div>

        <!-- Security, last 24 h -->
        <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-6">
          <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3">Security (last 24h)</p>
          <div class="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-3 mb-4">
            @for (t of securityTiles(); track t.label) {
              <div class="bg-gray-950/60 border border-gray-800 rounded-lg p-3">
                <p class="text-[11px] text-gray-500">{{ t.label }}</p>
                <p class="text-xl font-bold font-mono mt-0.5" [class]="t.value > 0 && t.warn ? 'text-amber-300' : 'text-gray-100'">{{ t.value }}</p>
              </div>
            }
          </div>
          @if (actions().length) {
            <p class="text-[11px] text-gray-500 uppercase tracking-wider mb-2">Events by action</p>
            <ul class="space-y-1.5">
              @for (a of actions(); track a.name) {
                <li class="flex items-center gap-3 text-xs">
                  <span class="w-40 sm:w-56 shrink-0 truncate font-mono text-gray-400" [attr.title]="a.name">{{ a.name }}</span>
                  <span class="flex-1 h-2 bg-gray-800 rounded" aria-hidden="true">
                    <span class="block h-2 bg-sky-400/80 rounded" [style.width.%]="a.pct"></span>
                  </span>
                  <span class="w-10 text-right font-mono text-gray-300">{{ a.count }}</span>
                </li>
              }
            </ul>
          } @else {
            <p class="text-sm text-gray-600">No security events recorded in the last 24 hours.</p>
          }
        </div>
      }

      <!-- Audit log -->
      <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <div class="flex flex-wrap items-center justify-between gap-3 mb-3">
          <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest">Audit log</p>
          <button type="button" (click)="verify()" [disabled]="verifying()"
                  class="px-3 py-1.5 rounded-lg text-xs font-semibold border border-gray-700 text-gray-300
                         hover:border-blue-600 hover:text-white transition-colors disabled:opacity-40">
            {{ verifying() ? 'Verifying...' : 'Verify audit chain' }}
          </button>
        </div>

        @if (verifyError()) {
          <p role="alert" class="mb-3 text-sm text-red-300">{{ verifyError() }}</p>
        }
        @if (verification(); as v) {
          @if (v.ok) {
            <div class="mb-3 rounded-lg p-3 border border-emerald-500/40 bg-emerald-500/10" role="status">
              <p class="text-sm font-semibold text-emerald-300">Chain intact — {{ v.checked }} events verified</p>
              @if (v.head_hash) {
                <p class="text-xs text-emerald-200/80 mt-1">
                  Head hash <code class="font-mono">{{ v.head_hash.slice(0, 16) }}</code>
                </p>
                <p class="text-[11px] text-gray-500 mt-1">
                  Record this value somewhere the database admin cannot edit to detect tail truncation
                </p>
              }
            </div>
          } @else {
            <div class="mb-3 rounded-lg p-3 border border-red-500/50 bg-red-500/10" role="alert">
              <p class="text-sm font-semibold text-red-300">
                Chain BROKEN at event #{{ v.first_bad_id }}: {{ v.reason || 'hash mismatch' }}
              </p>
            </div>
          }
        }

        @if (auditError()) {
          <p role="alert" class="text-sm text-red-300">{{ auditError() }}</p>
        } @else if (auditLoading() && !events().length) {
          <p class="text-sm text-gray-600 py-4" role="status">Loading audit log...</p>
        } @else if (!events().length) {
          <p class="text-sm text-gray-600 py-4">No audit events yet.</p>
        } @else {
          <div class="overflow-x-auto -mx-4 px-4">
            <table class="min-w-full text-xs">
              <caption class="sr-only">Most recent audit events</caption>
              <thead>
                <tr class="text-left text-gray-500 border-b border-gray-800">
                  <th scope="col" class="py-2 pr-4 font-semibold">Time</th>
                  <th scope="col" class="py-2 pr-4 font-semibold">Actor</th>
                  <th scope="col" class="py-2 pr-4 font-semibold">Role</th>
                  <th scope="col" class="py-2 pr-4 font-semibold">Action</th>
                  <th scope="col" class="py-2 pr-4 font-semibold">Resource</th>
                  <th scope="col" class="py-2 pr-4 font-semibold">Outcome</th>
                  <th scope="col" class="py-2 font-semibold">IP</th>
                </tr>
              </thead>
              <tbody>
                @for (e of events(); track e.id) {
                  <tr class="border-b border-gray-800/60 text-gray-300">
                    <td class="py-1.5 pr-4 font-mono whitespace-nowrap text-gray-400">{{ e.ts | date:'MMM d, HH:mm:ss' }}</td>
                    <td class="py-1.5 pr-4 whitespace-nowrap">{{ e.actor }}</td>
                    <td class="py-1.5 pr-4 whitespace-nowrap text-gray-400">{{ e.role }}</td>
                    <td class="py-1.5 pr-4 font-mono whitespace-nowrap" [attr.title]="e.detail">{{ e.action }}</td>
                    <td class="py-1.5 pr-4 font-mono text-gray-400 max-w-[14rem] truncate" [attr.title]="e.resource">{{ e.resource }}</td>
                    <td class="py-1.5 pr-4">
                      <span class="px-2 py-0.5 rounded font-semibold" [class]="outcomeClass(e)">{{ e.outcome }}</span>
                    </td>
                    <td class="py-1.5 font-mono text-gray-500 whitespace-nowrap">{{ e.ip }}</td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        }
      </div>
    </div>
  `,
})
export class DashboardComponent implements OnInit, OnDestroy {
  readonly CW = CW; readonly CH = CH; readonly PAD = PAD;
  readonly baseY = CH - PAD.b;

  metrics = signal<AdminMetrics | null>(null);
  loading = signal(false);
  error = signal<string | null>(null);
  updatedAt = signal<Date | null>(null);

  events = signal<AuditEvent[]>([]);
  auditLoading = signal(false);
  auditError = signal<string | null>(null);

  verification = signal<AuditVerification | null>(null);
  verifying = signal(false);
  verifyError = signal<string | null>(null);

  private subs: Subscription[] = [];
  private verifySub?: Subscription;

  bars = computed<Bar[]>(() => {
    const days = this.metrics()?.reports.uploads_7d ?? [];
    if (!days.length) return [];
    const max = Math.max(1, ...days.map((d) => d.count));
    const plotH = CH - PAD.t - PAD.b;
    const slot = (CW - PAD.l - PAD.r) / days.length;
    const w = Math.min(36, slot * 0.6);
    return days.map((d, i) => {
      const h = d.count > 0 ? Math.max(3, (d.count / max) * plotH) : 0;
      return {
        x: PAD.l + i * slot + (slot - w) / 2,
        y: this.baseY - h,
        w, h,
        count: d.count,
        label: this.shortDay(d.date),
        tip: `${d.date}: ${d.count} upload${d.count === 1 ? '' : 's'}`,
      };
    });
  });

  barAria = computed(() => {
    const days = this.metrics()?.reports.uploads_7d ?? [];
    return 'Uploads per day, last 7 days: ' + days.map((d) => `${d.date} ${d.count}`).join(', ');
  });

  securityTiles = computed(() => {
    const s = this.metrics()?.security_24h;
    if (!s) return [];
    return [
      { label: 'Events', value: s.events, warn: false },
      { label: 'Denied', value: s.denied, warn: true },
      { label: 'Login failures', value: s.login_failures, warn: true },
      { label: 'Authorization denials', value: s.authz_denied, warn: true },
      { label: 'Rate-limited', value: s.rate_limited, warn: true },
      { label: 'Injection suspected', value: s.injection_suspected, warn: true },
    ];
  });

  actions = computed(() => {
    const by = this.metrics()?.security_24h.by_action ?? {};
    const rows = Object.entries(by).sort((a, b) => b[1] - a[1]);
    const max = Math.max(1, ...rows.map((r) => r[1]));
    return rows.map(([name, count]) => ({ name, count, pct: (count / max) * 100 }));
  });

  ngOnInit(): void {
    this.refresh();
  }

  refresh(): void {
    this.subs.forEach((s) => s.unsubscribe());
    this.subs = [];
    this.loading.set(true);
    this.error.set(null);
    this.auditLoading.set(true);
    this.auditError.set(null);

    this.subs.push(
      this.api.getMetrics().subscribe({
        next: (m) => {
          this.metrics.set(m);
          this.loading.set(false);
          this.updatedAt.set(new Date());
        },
        error: (err) => {
          this.loading.set(false);
          this.error.set(errorMessage(err, 'Could not load metrics.'));
        },
      }),
      this.api.getAudit(50).subscribe({
        next: (r) => {
          this.events.set(r.events ?? []);
          this.auditLoading.set(false);
        },
        error: (err) => {
          this.auditLoading.set(false);
          this.auditError.set(errorMessage(err, 'Could not load the audit log.'));
        },
      })
    );
  }

  verify(): void {
    this.verifying.set(true);
    this.verifyError.set(null);
    this.verification.set(null);
    this.verifySub?.unsubscribe();
    this.verifySub = this.api.verifyAudit().subscribe({
      next: (v) => {
        this.verification.set(v);
        this.verifying.set(false);
      },
      error: (err) => {
        this.verifying.set(false);
        this.verifyError.set(errorMessage(err, 'Could not verify the audit chain.'));
      },
    });
  }

  secs(ms: number | null): string {
    return ms === null || ms === undefined ? '—' : `${(ms / 1000).toFixed(1)} s`;
  }

  num(v: number | null): string {
    return v === null || v === undefined ? '—' : String(Math.round(v * 10) / 10);
  }

  pct(v: number | null): string {
    return v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`;
  }

  pctNum(v: number | null): number {
    return v === null || v === undefined ? 0 : Math.min(100, Math.max(0, v * 100));
  }

  outcomeClass(e: AuditEvent): string {
    return e.outcome === 'success'
      ? 'bg-emerald-500/15 text-emerald-300'
      : e.outcome === 'denied'
        ? 'bg-amber-500/15 text-amber-300'
        : 'bg-red-500/15 text-red-300';
  }

  private shortDay(iso: string): string {
    const m = /^\d{4}-(\d{2})-(\d{2})/.exec(iso);
    if (!m) return iso;
    return `${Number(m[2])} ${MONTHS[Number(m[1]) - 1] ?? ''}`.trim();
  }

  constructor(private api: ApiService) {}

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
    this.verifySub?.unsubscribe();
  }
}
