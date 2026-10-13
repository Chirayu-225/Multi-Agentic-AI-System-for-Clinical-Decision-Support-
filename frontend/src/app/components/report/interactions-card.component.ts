// src/app/components/report/interactions-card.component.ts
import { Component, Input } from '@angular/core';
import { DrugInteraction, DrugInteractions } from '../../models/report.model';

const TYPE_LABELS: Record<string, string> = {
  'drug-drug': 'Drug–drug',
  'drug-lab': 'Drug–lab',
  'drug-advisory': 'Advisory',
};

const SEV: Record<string, { label: string; cls: string }> = {
  major: { label: 'Major', cls: 'bg-red-500/20 text-red-300 border-red-500/40' },
  moderate: { label: 'Moderate', cls: 'bg-amber-500/20 text-amber-300 border-amber-500/40' },
  minor: { label: 'Minor', cls: 'bg-gray-700/60 text-gray-300 border-gray-600' },
};

@Component({
  selector: 'app-interactions-card',
  standalone: true,
  template: `
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
      @if (!data.interactions.length) {
        <p class="text-sm text-gray-400">
          No interactions found by the curated rule set (not a guarantee of safety).
        </p>
      }
      @for (i of data.interactions; track $index) {
        <div class="py-3 border-b border-gray-800 last:border-b-0 first:pt-0">
          <div class="flex flex-wrap items-center gap-2 mb-1.5">
            <span class="text-[11px] font-semibold px-2 py-0.5 rounded border" [class]="sevClass(i)">
              {{ sevLabel(i) }}
            </span>
            <span class="text-[11px] text-gray-400 bg-gray-800 px-2 py-0.5 rounded">{{ typeLabel(i) }}</span>
            <span class="text-sm font-medium text-gray-200">{{ i.drugs.join(' + ') }}</span>
            @if (i.lab) {
              <span class="text-xs font-mono text-gray-500">
                {{ i.lab }}@if (i.lab_value !== undefined && i.lab_value !== null) { = {{ i.lab_value }} }
              </span>
            }
          </div>
          <p class="text-sm text-gray-400 leading-relaxed">{{ i.effect }}</p>
          <p class="text-sm text-gray-300 leading-relaxed mt-1">
            <span class="text-xs font-semibold text-blue-400 uppercase tracking-wider">Advice </span>{{ i.advice }}
          </p>
          @if (i.source) {
            <p class="text-[11px] text-gray-600 mt-1">Source: {{ i.source }}</p>
          }
        </div>
      }

      @if (data.checked_medications.length) {
        <p class="text-[11px] text-gray-600 mt-3">
          Checked: {{ data.checked_medications.join(', ') }}
        </p>
      }
      @if (data.unrecognized_medications.length) {
        <p class="text-[11px] text-gray-500 mt-1">
          Not recognised by the rule set (not checked): {{ data.unrecognized_medications.join(', ') }}
        </p>
      }
      <p class="text-[11px] text-gray-600 mt-3 leading-relaxed">{{ data.disclaimer }}</p>
    </div>
  `,
})
export class InteractionsCardComponent {
  @Input({ required: true }) data!: DrugInteractions;

  typeLabel(i: DrugInteraction): string {
    return TYPE_LABELS[i.type] ?? i.type;
  }
  sevLabel(i: DrugInteraction): string {
    return SEV[i.severity]?.label ?? i.severity;
  }
  sevClass(i: DrugInteraction): string {
    return SEV[i.severity]?.cls ?? SEV['minor'].cls;
  }
}
