// src/app/components/report/evidence-panel.component.ts
// Collapsible list of the source evidence behind summary claims. The matched
// span is highlighted by splitting the snippet into three strings in TypeScript
// and rendering each in its own <span> — no markup is ever built from text.
import { Component, Input, computed, signal } from '@angular/core';
import { EvidenceItem } from '../../models/report.model';
import { SeverityService } from '../../services/severity.service';

interface Row {
  item: EvidenceItem;
  before: string;
  match: string;
  after: string;
  citedIn: string[];
  color: string | null;
  sevText: string | null;
}

@Component({
  selector: 'app-evidence-panel',
  standalone: true,
  template: `
    <div class="bg-gray-900 border border-gray-800 rounded-xl">
      <button type="button" (click)="open.set(!open())" [attr.aria-expanded]="open()"
              [attr.aria-controls]="panelId"
              class="w-full flex items-center justify-between px-4 py-3 text-left
                     hover:bg-gray-800/50 transition-colors rounded-xl">
        <span class="text-xs font-semibold text-gray-400 uppercase tracking-widest">
          Evidence from the source report
          <span class="ml-2 normal-case tracking-normal font-mono text-gray-600">{{ rows().length }}</span>
        </span>
        <span class="text-gray-600 text-sm" aria-hidden="true">{{ open() ? '▲' : '▼' }}</span>
      </button>

      @if (open()) {
        <div [id]="panelId" class="px-4 pb-4 border-t border-gray-800">
          @for (r of rows(); track r.item.id) {
            <div class="py-3 border-b border-gray-800/70 last:border-b-0">
              <div class="flex flex-wrap items-center gap-2 mb-1.5">
                <span class="text-[10px] font-semibold uppercase tracking-wider bg-gray-800
                             text-blue-300 px-2 py-0.5 rounded">{{ r.item.kind }}</span>
                <span class="text-sm text-gray-200">
                  {{ r.item.label }}@if (r.item.value !== undefined && r.item.value !== null) {
                    <span class="font-mono text-gray-400"> · {{ r.item.value }}</span>
                  }
                </span>
                @if (r.sevText) {
                  <span class="text-[11px] font-semibold font-mono px-2 py-0.5 rounded"
                        [style.color]="r.color" [style.background]="r.color + '20'">{{ r.sevText }}</span>
                }
                @for (c of r.citedIn; track c) {
                  <span class="text-[10px] text-gray-400 border border-gray-700 px-1.5 py-0.5 rounded">
                    cited in {{ c }}
                  </span>
                }
              </div>

              @if (!r.item.found) {
                <p class="text-xs font-semibold text-amber-300">Not found in source text</p>
              }
              @if (r.before || r.match || r.after) {
                <p class="text-xs text-gray-500 font-mono leading-relaxed break-words
                          bg-gray-950/60 border border-gray-800 rounded-md px-3 py-2">
                  <span>{{ r.before }}</span><span class="bg-yellow-400/25 text-yellow-100 rounded-sm px-0.5">{{ r.match }}</span><span>{{ r.after }}</span>
                </p>
              }
            </div>
          } @empty {
            <p class="text-sm text-gray-600 pt-3">No evidence items were recorded.</p>
          }
        </div>
      }
    </div>
  `,
})
export class EvidencePanelComponent {
  private static seq = 0;
  readonly panelId = `evidence-panel-${++EvidencePanelComponent.seq}`;

  open = signal(false);
  private items = signal<EvidenceItem[]>([]);

  @Input({ required: true }) set evidence(v: EvidenceItem[]) {
    this.items.set(v ?? []);
  }

  rows = computed<Row[]>(() => this.items().map((item) => this.toRow(item)));

  constructor(private sev: SeverityService) {}

  private toRow(item: EvidenceItem): Row {
    const snip = item.snippet ?? '';
    const s = item.match_start, e = item.match_end;
    let before = snip, match = '', after = '';
    if (item.found && Number.isInteger(s) && Number.isInteger(e) && s >= 0 && e > s && e <= snip.length) {
      before = snip.slice(0, s);
      match = snip.slice(s, e);
      after = snip.slice(e);
    }
    const citedIn = (item.cited_in ?? []).map((c) => (c === 'clinical' ? 'Clinical' : c === 'patient' ? 'Patient' : String(c)));
    return {
      item, before, match, after, citedIn,
      color: item.severity ? this.sev.getColor(item.severity) : null,
      sevText: item.severity ? this.sev.normalize(item.severity) : null,
    };
  }
}
