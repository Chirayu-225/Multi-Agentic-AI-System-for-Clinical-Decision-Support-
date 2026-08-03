// src/app/components/shared/pubmed-card.component.ts
import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PubMedReference } from '../../models/report.model';

@Component({
  selector: 'app-pubmed-card',
  standalone: true,
  imports: [CommonModule],
  template: `
    <a [href]="reference.url" target="_blank" rel="noopener noreferrer"
       class="block bg-gray-900 border border-gray-800 rounded-lg p-3 mb-2
              hover:border-blue-700 transition-colors group">
      <div class="flex items-start gap-2">
        <span class="text-blue-400 text-xs font-mono shrink-0 mt-0.5">PubMed</span>
        <div>
          <p class="text-sm text-gray-300 group-hover:text-white transition-colors leading-snug">
            {{ reference.title }}
          </p>
          <p class="text-xs text-gray-600 mt-1">
            {{ reference.authors }} · {{ reference.journal }} · {{ reference.pub_date }}
          </p>
        </div>
        <span class="ml-auto text-gray-600 group-hover:text-blue-400 transition-colors shrink-0">↗</span>
      </div>
    </a>
  `,
})
export class PubmedCardComponent {
  @Input() reference!: PubMedReference;
}
