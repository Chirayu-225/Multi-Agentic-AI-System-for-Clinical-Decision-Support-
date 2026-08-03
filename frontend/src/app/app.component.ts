// src/app/app.component.ts
import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReportUploadComponent } from './components/report/report-upload.component';
import { PatientHistoryComponent } from './components/history/patient-history.component';

type Page = 'analyse' | 'history' | 'about';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, ReportUploadComponent, PatientHistoryComponent],
  template: `
    <div class="flex h-screen bg-gray-950 overflow-hidden">

      <!-- Sidebar -->
      <aside class="w-56 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">

        <!-- Logo -->
        <div class="p-5 border-b border-gray-800">
          <div class="flex items-center gap-3 mb-1">
            <div class="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-600 to-sky-500
                        flex items-center justify-center text-lg shadow-lg shadow-blue-500/20">
              🩺
            </div>
            <span class="text-lg font-bold text-white tracking-tight">MedSight</span>
          </div>
          <p class="text-xs text-gray-600 pl-12 leading-tight">
            CLINICAL INTELLIGENCE
          </p>
        </div>

        <!-- Nav -->
        <nav class="flex-1 p-3 space-y-1">
          @for (item of navItems; track item.id) {
            <button
              (click)="activePage.set(item.id)"
              class="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm
                     transition-all text-left"
              [class.bg-blue-600]="activePage() === item.id"
              [class.text-white]="activePage() === item.id"
              [class.text-gray-400]="activePage() !== item.id"
              [class.hover:bg-gray-800]="activePage() !== item.id">
              <span>{{ item.icon }}</span>
              <span class="font-medium">{{ item.label }}</span>
              @if (activePage() === item.id) {
                <span class="ml-auto w-1.5 h-1.5 rounded-full bg-white/70"></span>
              }
            </button>
          }
        </nav>

        <!-- Multi-agent badge -->
        <div class="p-4 border-t border-gray-800">
          <div class="bg-gray-800 rounded-lg p-3 text-xs text-gray-500 leading-relaxed">
            <div class="flex items-center gap-2 mb-2">
              <span class="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"></span>
              <span class="font-semibold text-gray-400">Multi-Agent Active</span>
            </div>
            <p>Orchestrator + Verification agents running on Gemini 2.5 Flash</p>
            <div class="mt-2 flex gap-1">
              <span class="bg-blue-500/20 text-blue-400 px-1.5 py-0.5 rounded text-xs">PubMed</span>
              <span class="bg-purple-500/20 text-purple-400 px-1.5 py-0.5 rounded text-xs">OpenFDA</span>
            </div>
          </div>
        </div>
      </aside>

      <!-- Main content -->
      <main class="flex-1 overflow-y-auto">
        @switch (activePage()) {
          @case ('analyse') { <app-report-upload /> }
          @case ('history') { <app-patient-history /> }
          @case ('about')   { <ng-container *ngTemplateOutlet="aboutTpl" /> }
        }
      </main>
    </div>

    <!-- About page template -->
    <ng-template #aboutTpl>
      <div class="p-6 max-w-2xl">
        <p class="text-xs font-semibold text-blue-400 uppercase tracking-widest mb-1">About</p>
        <h1 class="text-3xl font-bold text-white tracking-tight mb-2">MedSight</h1>
        <p class="text-gray-500 text-sm mb-8">
          Multi-Agent Clinical Intelligence System for Indian Healthcare
        </p>

        <div class="space-y-4">
          @for (feature of features; track feature.title) {
            <div class="bg-gray-900 border border-gray-800 rounded-xl p-5">
              <div class="text-xl mb-2">{{ feature.icon }}</div>
              <h3 class="font-semibold text-gray-200 mb-1">{{ feature.title }}</h3>
              <p class="text-sm text-gray-500 leading-relaxed">{{ feature.desc }}</p>
            </div>
          }
        </div>

        <div class="mt-6 bg-gray-900 border border-gray-800 rounded-xl p-5">
          <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3">Tech Stack</p>
          <div class="flex flex-wrap gap-2">
            @for (tag of stack; track tag) {
              <span class="bg-gray-800 border border-gray-700 text-gray-300
                            font-mono text-xs px-3 py-1.5 rounded-md">{{ tag }}</span>
            }
          </div>
        </div>
      </div>
    </ng-template>
  `,
})
export class AppComponent {
  activePage = signal<Page>('analyse');

  navItems = [
    { id: 'analyse' as Page, label: 'Analyse Report', icon: '🩺' },
    { id: 'history' as Page, label: 'Patient History', icon: '📈' },
    { id: 'about'   as Page, label: 'About',           icon: 'ℹ️' },
  ];

  features = [
    {
      icon: '🤖',
      title: 'Multi-Agent Architecture',
      desc: 'An Orchestrator agent with tool-calling handles pipeline decisions. A Verification agent independently audits outputs for hallucinated clinical claims before delivery.',
    },
    {
      icon: '🇮🇳',
      title: 'ICMR-Grounded Risk Flagging',
      desc: 'Lab values checked against Indian Council of Medical Research reference ranges, not Western defaults. Gender-specific thresholds included.',
    },
    {
      icon: '📈',
      title: 'Longitudinal Trend Detection',
      desc: 'Detects consistently worsening values across multiple visits via ChromaDB, even when individual readings appear borderline.',
    },
    {
      icon: '📚',
      title: 'Real Clinical Evidence',
      desc: 'Agent automatically searches PubMed for relevant clinical guidelines when significant findings are detected. OpenFDA checked for medication safety alerts.',
    },
  ];

  stack = [
    'Angular', 'TypeScript', 'TailwindCSS',
    'FastAPI', 'SQLAlchemy', 'PostgreSQL',
    'ChromaDB', 'Gemini 2.5 Flash',
    'GLiNER', 'EasyOCR', 'PyMuPDF',
    'PubMed API', 'OpenFDA API', 'Docker',
  ];
}
