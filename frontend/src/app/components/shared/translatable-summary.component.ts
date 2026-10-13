// src/app/components/shared/translatable-summary.component.ts
// Patient-summary text with an English/<language> toggle and an on-demand
// "Translate" dropdown. All text is rendered via interpolation.
import { Component, Input, OnChanges, OnDestroy, SimpleChanges, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { ApiService } from '../../services/api.service';
import { errorMessage } from '../../services/error-message';
import { LANGUAGES, languageName } from '../../services/languages';

@Component({
  selector: 'app-translatable-summary',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="flex flex-wrap items-center gap-2 mb-3">
      @if (current(); as t) {
        <div class="inline-flex rounded-lg border border-gray-700 overflow-hidden text-xs"
             role="group" aria-label="Summary language">
          <button type="button" (click)="showTranslated.set(false)"
                  [attr.aria-pressed]="!showTranslated()"
                  class="px-3 py-1.5 transition-colors"
                  [class]="!showTranslated() ? 'bg-blue-600 text-white' : 'bg-gray-900 text-gray-400 hover:bg-gray-800'">
            English
          </button>
          <button type="button" (click)="showTranslated.set(true)"
                  [attr.aria-pressed]="showTranslated()"
                  class="px-3 py-1.5 border-l border-gray-700 transition-colors"
                  [class]="showTranslated() ? 'bg-blue-600 text-white' : 'bg-gray-900 text-gray-400 hover:bg-gray-800'">
            {{ nameOf(t.language) }}
          </button>
        </div>
      }

      <label class="sr-only" [attr.for]="selectId">Translate summary</label>
      <select [id]="selectId" [(ngModel)]="pick" (ngModelChange)="translate($event)"
              [disabled]="loading()"
              class="bg-gray-900 border border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-300
                     focus:outline-none focus:border-blue-500 disabled:opacity-50">
        <option value="">Translate…</option>
        @for (l of languages; track l.code) {
          <option [value]="l.code">{{ l.name }}</option>
        }
      </select>

      @if (loading()) {
        <span class="inline-flex items-center gap-1.5 text-xs text-gray-500" role="status">
          <span class="w-3 h-3 border-2 border-gray-600 border-t-blue-400 rounded-full animate-spin"></span>
          Translating…
        </span>
      }
    </div>

    @if (error()) {
      <div role="alert" class="mb-3 rounded-lg p-2.5 border border-red-500/40 bg-red-500/10 text-xs text-red-300">
        {{ error() }}
      </div>
    }

    @if (showTranslated() && current(); as t) {
      <p class="text-gray-300 text-sm leading-relaxed whitespace-pre-line">{{ t.text }}</p>
      <p class="mt-2 text-[11px] text-gray-600">
        Machine-translated; numbers were automatically checked against the English summary.
      </p>
    } @else {
      <p class="text-gray-300 text-sm leading-relaxed whitespace-pre-line">{{ text }}</p>
    }
  `,
})
export class TranslatableSummaryComponent implements OnChanges, OnDestroy {
  private static seq = 0;

  @Input({ required: true }) reportId!: string;
  @Input({ required: true }) text = '';
  @Input() translation: { language: string; text: string } | null | undefined = null;
  /** Open on the translated text when one already exists (patient portal). */
  @Input() preferTranslated = false;

  readonly languages = LANGUAGES;
  readonly selectId = `translate-${++TranslatableSummaryComponent.seq}`;

  current = signal<{ language: string; text: string } | null>(null);
  showTranslated = signal(false);
  loading = signal(false);
  error = signal<string | null>(null);
  pick = '';

  private sub?: Subscription;

  nameOf = languageName;

  ngOnChanges(ch: SimpleChanges): void {
    if (ch['reportId'] || ch['translation']) {
      this.sub?.unsubscribe();
      this.loading.set(false);
      this.error.set(null);
      this.pick = '';
      const t = this.translation ?? null;
      this.current.set(t);
      this.showTranslated.set(!!t && this.preferTranslated);
    }
  }

  translate(code: string): void {
    if (!code) return;
    this.sub?.unsubscribe();
    this.loading.set(true);
    this.error.set(null);
    this.sub = this.api.translateReport(this.reportId, code).subscribe({
      next: (res) => {
        this.current.set({ language: res.language || code, text: res.text });
        this.showTranslated.set(true);
        this.loading.set(false);
        this.pick = '';
      },
      error: (err) => {
        this.loading.set(false);
        this.pick = '';
        this.error.set(errorMessage(err, 'Translation failed. Please try again.'));
      },
    });
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  constructor(private api: ApiService) {}
}
