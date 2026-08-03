// src/app/components/shared/risk-card.component.ts
import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RiskFlag, TrendFlag } from '../../models/report.model';
import { SeverityService } from '../../services/severity.service';

@Component({
  selector: 'app-risk-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <!-- All clear -->
    @if (isAllClear) {
      <div class="rounded-xl p-4 mb-3 border border-l-4 border-green-500 bg-green-500/5">
        <div class="flex items-center justify-between mb-1">
          <span class="text-xs font-bold uppercase tracking-widest text-green-400">All Clear</span>
          <span class="text-xs font-semibold font-mono bg-green-500/20 text-green-400 px-2 py-0.5 rounded">NORMAL</span>
        </div>
        <p class="text-sm text-gray-400">All extracted lab values are within ICMR reference ranges.</p>
      </div>
    }

    <!-- Rule-based flag -->
    @if (flag && !trend) {
      <div class="rounded-xl p-4 mb-3 border border-l-4 transition-transform hover:translate-x-0.5"
           [style.border-color]="color"
           [style.background]="color + '0D'">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-bold uppercase tracking-widest" [style.color]="color">
            {{ flag!.test }}
          </span>
          <span class="text-xs font-semibold font-mono px-2 py-0.5 rounded"
                [style.background]="color + '20'"
                [style.color]="color">
            {{ sev.normalize(flag!.severity) }} · {{ flag!.value }} {{ flag!.unit }}
          </span>
        </div>
        <p class="text-sm text-gray-400 leading-relaxed">{{ flag!.message }}</p>
        @if (flag!.explanation) {
          <div class="mt-3 pt-3 border-t border-gray-800 text-xs text-gray-500 italic leading-relaxed">
            ↳ {{ flag!.explanation }}
          </div>
        }
      </div>
    }

    <!-- Trend flag -->
    @if (trend) {
      <div class="rounded-xl p-4 mb-3 border border-l-4 transition-transform hover:translate-x-0.5"
           [style.border-color]="trendColor"
           [style.background]="trendColor + '0D'">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-bold uppercase tracking-widest" [style.color]="trendColor">
            📈 Trend · {{ trend!.test | titlecase }}
          </span>
          <span class="text-xs font-semibold font-mono px-2 py-0.5 rounded"
                [style.background]="trendColor + '20'"
                [style.color]="trendColor">
            {{ trend!.total_change_pct }}% change
          </span>
        </div>
        <p class="text-xs font-mono text-gray-500 mb-2">
          {{ trend!.values_over_time.join(' → ') }} {{ trend!.unit }}
        </p>
        <p class="text-sm text-gray-400 leading-relaxed">{{ trend!.message }}</p>
        @if (trend!.explanation) {
          <div class="mt-3 pt-3 border-t border-gray-800 text-xs text-gray-500 italic leading-relaxed">
            ↳ {{ trend!.explanation }}
          </div>
        }
      </div>
    }
  `,
})
export class RiskCardComponent {
  @Input() flag: RiskFlag | null = null;
  @Input() trend: TrendFlag | null = null;
  @Input() isAllClear = false;

  constructor(public sev: SeverityService) {}

  get color(): string {
    return this.flag ? this.sev.getColor(this.flag.severity) : '#22C55E';
  }

  get trendColor(): string {
    return this.trend ? this.sev.getColor(this.trend.severity) : '#EAB308';
  }
}
