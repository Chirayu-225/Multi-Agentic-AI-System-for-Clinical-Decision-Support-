// src/app/components/shared/trend-chart.component.ts
// Hand-written inline-SVG trend card for one lab series: reference band,
// optional critical lines, polyline + markers, axis labels and tooltips.
import { Component, Input, computed, signal } from '@angular/core';
import { TrendSeries } from '../../models/report.model';

const W = 320;
const H = 176;
const M = { left: 42, right: 18, top: 16, bottom: 28 };
const X_PAD = 12;
const PLOT_W = W - M.left - M.right;
const PLOT_H = H - M.top - M.bottom;

const COL_IN = '#2dd4bf';   // inside reference band
const COL_OUT = '#f59e0b';  // outside reference band
const COL_CRIT = '#ef4444'; // at/beyond a critical limit

type Zone = 'in' | 'high' | 'low' | 'unknown';

interface PlotPoint {
  x: number;
  y: number;
  value: number;
  color: string;
  zone: Zone;
  critical: boolean;
  tip: string;
  last: boolean;
}

interface Tick { y: number; label: string }
interface XLabel { x: number; text: string }
interface Band { y: number; h: number }
interface CritLine { y: number; label: string }

interface Model {
  points: PlotPoint[];
  line: string;
  ticks: Tick[];
  xLabels: XLabel[];
  band: Band | null;
  critLines: CritLine[];
  lastLabel: { x: number; y: number; text: string } | null;
  latestZone: Zone;
  latestText: string;
  aria: string;
  refText: string;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function shortDate(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (!m) return iso;
  const mon = MONTHS[Number(m[2]) - 1];
  return mon ? `${Number(m[3])} ${mon} ${m[1].slice(2)}` : iso;
}

function fmt(v: number, decimals = 2): string {
  const r = Number(v.toFixed(decimals));
  return String(r);
}

function niceStep(raw: number): number {
  const exp = Math.floor(Math.log10(raw));
  const f = raw / Math.pow(10, exp);
  const nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return nf * Math.pow(10, exp);
}

function zoneOf(v: number, lo: number | null, hi: number | null): Zone {
  if (lo === null && hi === null) return 'unknown';
  if (hi !== null && v > hi) return 'high';
  if (lo !== null && v < lo) return 'low';
  return 'in';
}

@Component({
  selector: 'app-trend-chart',
  standalone: true,
  template: `
    @if (m(); as c) {
      <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <div class="flex items-start justify-between gap-2 mb-1">
          <div class="min-w-0">
            <p class="text-sm font-semibold text-gray-200 truncate">{{ s().test }}</p>
            <p class="text-[11px] text-gray-600 font-mono">
              {{ s().unit }}@if (c.refText) { · ref {{ c.refText }} }
            </p>
          </div>
          <div class="text-right shrink-0">
            <p class="text-sm font-mono font-semibold text-gray-100">{{ c.latestText }}</p>
            @switch (c.latestZone) {
              @case ('in')   { <span class="text-[10px] font-semibold text-teal-300">In range</span> }
              @case ('high') { <span class="text-[10px] font-semibold text-amber-300">Above range</span> }
              @case ('low')  { <span class="text-[10px] font-semibold text-amber-300">Below range</span> }
            }
          </div>
        </div>

        <svg class="w-full h-auto block" [attr.viewBox]="'0 0 ' + W + ' ' + H"
             role="img" [attr.aria-label]="c.aria">
          <!-- reference band -->
          @if (c.band) {
            <rect [attr.x]="ML" [attr.y]="c.band.y" [attr.width]="PW" [attr.height]="c.band.h"
                  fill="#34d399" fill-opacity="0.12"/>
          }
          <!-- y ticks + grid -->
          @for (t of c.ticks; track t.label) {
            <line [attr.x1]="ML" [attr.x2]="ML + PW" [attr.y1]="t.y" [attr.y2]="t.y"
                  stroke="#1f2937" stroke-width="1"/>
            <text [attr.x]="ML - 6" [attr.y]="t.y + 3" text-anchor="end"
                  font-size="9" fill="#6b7280" font-family="JetBrains Mono, monospace">{{ t.label }}</text>
          }
          <!-- critical limits -->
          @for (cl of c.critLines; track cl.label) {
            <line [attr.x1]="ML" [attr.x2]="ML + PW" [attr.y1]="cl.y" [attr.y2]="cl.y"
                  stroke="#ef4444" stroke-width="1" stroke-dasharray="4 3"/>
            <text [attr.x]="ML + PW" [attr.y]="cl.y - 3" text-anchor="end"
                  font-size="8" fill="#f87171">{{ cl.label }}</text>
          }
          <!-- axis baseline -->
          <line [attr.x1]="ML" [attr.x2]="ML + PW" [attr.y1]="MT + PH" [attr.y2]="MT + PH"
                stroke="#374151" stroke-width="1"/>
          <!-- x labels -->
          @for (xl of c.xLabels; track $index) {
            <text [attr.x]="xl.x" [attr.y]="MT + PH + 16" text-anchor="middle"
                  font-size="9" fill="#6b7280">{{ xl.text }}</text>
          }
          <!-- line -->
          @if (c.points.length > 1) {
            <polyline [attr.points]="c.line" fill="none" stroke="#7dd3fc" stroke-opacity="0.8"
                      stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          }
          <!-- markers -->
          @for (p of c.points; track $index) {
            <circle [attr.cx]="p.x" [attr.cy]="p.y" [attr.r]="p.last ? 5.5 : 4"
                    [attr.fill]="p.color" stroke="#111827" stroke-width="2">
              <title>{{ p.tip }}</title>
            </circle>
          }
          <!-- latest value label -->
          @if (c.lastLabel) {
            <text [attr.x]="c.lastLabel.x" [attr.y]="c.lastLabel.y" text-anchor="middle"
                  font-size="11" font-weight="600" fill="#f3f4f6"
                  font-family="JetBrains Mono, monospace">{{ c.lastLabel.text }}</text>
          }
        </svg>
      </div>
    }
  `,
})
export class TrendChartComponent {
  readonly W = W; readonly H = H;
  readonly ML = M.left; readonly MT = M.top; readonly PW = PLOT_W; readonly PH = PLOT_H;

  s = signal<TrendSeries>({
    test: '', unit: '', ref_low: null, ref_high: null,
    critical_low: null, critical_high: null, points: [],
  });

  @Input({ required: true }) set series(v: TrendSeries) {
    this.s.set(v);
  }

  m = computed<Model | null>(() => this.build(this.s()));

  private build(s: TrendSeries): Model | null {
    const pts = (s.points ?? [])
      .filter((p) => Number.isFinite(p.value))
      .slice()
      .sort((a, b) => a.date.localeCompare(b.date));
    if (!pts.length) return null;

    const lo = s.ref_low, hi = s.ref_high;
    const vals = pts.map((p) => p.value);

    // y-range: the data plus the reference band; critical limits only drawn if inside.
    const cand = [...vals];
    if (lo !== null) cand.push(lo);
    if (hi !== null) cand.push(hi);
    let min = Math.min(...cand);
    let max = Math.max(...cand);
    if (max - min < 1e-9) {
      const d = Math.abs(max) * 0.1 || 1;
      min -= d; max += d;
    }
    const pad = (max - min) * 0.1;
    min -= pad; max += pad;
    const yOf = (v: number) => M.top + ((max - v) / (max - min)) * PLOT_H;
    const clampY = (y: number) => Math.min(M.top + PLOT_H, Math.max(M.top, y));

    // y ticks
    const step = niceStep((max - min) / 3);
    const decimals = Math.min(3, Math.max(0, -Math.floor(Math.log10(step))));
    const ticks: Tick[] = [];
    for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-6; t += step) {
      ticks.push({ y: yOf(t), label: fmt(t, decimals) });
    }

    // x positions: proportional to time when every date parses, else evenly spaced
    const times = pts.map((p) => Date.parse(p.date));
    const useTime = times.every((t) => Number.isFinite(t));
    const tMin = Math.min(...times), tMax = Math.max(...times);
    const innerW = PLOT_W - X_PAD * 2;
    const xOf = (i: number): number => {
      if (pts.length === 1) return M.left + PLOT_W / 2;
      if (useTime && tMax > tMin) return M.left + X_PAD + ((times[i] - tMin) / (tMax - tMin)) * innerW;
      return M.left + X_PAD + (i / (pts.length - 1)) * innerW;
    };

    const unit = s.unit ? ` ${s.unit}` : '';
    const plot: PlotPoint[] = pts.map((p, i) => {
      const zone = zoneOf(p.value, lo, hi);
      const critical =
        (s.critical_low !== null && p.value <= s.critical_low) ||
        (s.critical_high !== null && p.value >= s.critical_high);
      const color = critical ? COL_CRIT : zone === 'high' || zone === 'low' ? COL_OUT : COL_IN;
      const flag = critical ? ' (critical)' : zone === 'high' ? ' (above range)' : zone === 'low' ? ' (below range)' : '';
      return {
        x: xOf(i), y: yOf(p.value), value: p.value, color, zone, critical,
        tip: `${p.date}: ${fmt(p.value)}${unit}${flag}`,
        last: i === pts.length - 1,
      };
    });

    // reference band, clipped to the plot area
    let band: Band | null = null;
    if (lo !== null || hi !== null) {
      const top = hi !== null ? clampY(yOf(hi)) : M.top;
      const bottom = lo !== null ? clampY(yOf(lo)) : M.top + PLOT_H;
      if (bottom - top > 0.5) band = { y: top, h: bottom - top };
    }

    // critical lines only when inside the plotted range
    const critLines: CritLine[] = [];
    if (s.critical_high !== null && s.critical_high >= min && s.critical_high <= max) {
      critLines.push({ y: yOf(s.critical_high), label: 'critical high' });
    }
    if (s.critical_low !== null && s.critical_low >= min && s.critical_low <= max) {
      critLines.push({ y: yOf(s.critical_low), label: 'critical low' });
    }

    // x labels: all when <= 4 points, else first/last; skip any that would collide
    const idxs = pts.length <= 4 ? pts.map((_, i) => i) : [0, pts.length - 1];
    const xLabels: XLabel[] = [];
    for (const i of idxs) {
      const x = plot[i].x;
      const prev = xLabels[xLabels.length - 1];
      if (prev && x - prev.x < 50) {
        if (i !== pts.length - 1) continue;
        xLabels.pop(); // the latest date always wins over a crowded neighbour
      }
      xLabels.push({ x, text: shortDate(pts[i].date) });
    }

    const last = plot[plot.length - 1];
    const lastLabel = {
      x: last.x,
      y: last.y - 10 < M.top - 2 ? last.y + 18 : last.y - 10,
      text: fmt(last.value),
    };

    const refText =
      lo !== null && hi !== null ? `${fmt(lo)}–${fmt(hi)}`
      : hi !== null ? `≤ ${fmt(hi)}`
      : lo !== null ? `≥ ${fmt(lo)}` : '';

    const first = pts[0], latest = pts[pts.length - 1];
    const zoneWords: Record<Zone, string> = {
      in: 'within the reference range', high: 'above the reference range',
      low: 'below the reference range', unknown: 'with no reference range available',
    };
    let movement = '';
    if (pts.length > 1) {
      const dir = latest.value > first.value ? 'rising' : latest.value < first.value ? 'falling' : 'unchanged';
      movement = `, ${dir} from ${fmt(first.value)} to ${fmt(latest.value)}`;
    }
    const aria =
      `${s.test} trend${s.unit ? ' in ' + s.unit : ''}: ${pts.length} reading${pts.length === 1 ? '' : 's'} ` +
      `from ${shortDate(first.date)} to ${shortDate(latest.date)}${movement}. ` +
      `Latest value ${fmt(latest.value)} is ${zoneWords[last.zone]}.`;

    return {
      points: plot,
      line: plot.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '),
      ticks, xLabels, band, critLines, lastLabel,
      latestZone: last.zone,
      latestText: fmt(latest.value) + unit,
      aria, refText,
    };
  }
}
