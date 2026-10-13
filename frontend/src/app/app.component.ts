// src/app/app.component.ts
import { Component, computed, effect, signal, untracked } from '@angular/core';
import { NgTemplateOutlet } from '@angular/common';
import { ReportUploadComponent } from './components/report/report-upload.component';
import { PatientHistoryComponent } from './components/history/patient-history.component';
import { ReviewQueueComponent } from './components/review/review-queue.component';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { PatientPortalComponent } from './components/portal/patient-portal.component';
import { LoginComponent } from './components/auth/login.component';
import { AuthService } from './services/auth.service';
import { ReviewStateService } from './services/review-state.service';
import { Role } from './models/report.model';

type Page = 'analyse' | 'review' | 'history' | 'dashboard' | 'portal' | 'about';

interface NavItem { id: Page; label: string; icon: string; roles: Role[] }

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    NgTemplateOutlet, ReportUploadComponent, PatientHistoryComponent, ReviewQueueComponent,
    DashboardComponent, PatientPortalComponent, LoginComponent,
  ],
  template: `
    @if (!auth.isAuthenticated()) {
      <app-login />
    } @else {
    <div class="flex flex-col md:flex-row h-dvh bg-gray-950 overflow-hidden">

      <!-- Mobile top bar -->
      <header class="md:hidden shrink-0 flex items-center gap-3 px-4 py-3 bg-gray-900 border-b border-gray-800">
        <button type="button" (click)="menuOpen.set(true)" aria-label="Open menu"
                [attr.aria-expanded]="menuOpen()"
                class="w-9 h-9 rounded-lg border border-gray-700 text-gray-300 hover:bg-gray-800">☰</button>
        <p class="text-sm font-bold text-white tracking-tight">Agentic Medical Assistant</p>
      </header>

      @if (menuOpen()) {
        <button type="button" class="md:hidden fixed inset-0 z-30 bg-black/60" aria-label="Close menu"
                (click)="menuOpen.set(false)"></button>
      }

      <!-- Sidebar -->
      <aside class="fixed inset-y-0 left-0 z-40 w-64 md:static md:w-56 md:translate-x-0 transition-transform
                    bg-gray-900 border-r border-gray-800 flex flex-col shrink-0 overflow-y-auto"
             [class.-translate-x-full]="!menuOpen()">

        <!-- Logo -->
        <div class="p-5 border-b border-gray-800">
          <div class="flex items-center gap-3">
            <div class="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-600 to-sky-500
                        flex items-center justify-center text-lg shadow-lg shadow-blue-500/20 shrink-0">
              🩺
            </div>
            <div class="leading-tight">
              <p class="text-sm font-bold text-white tracking-tight">Agentic Medical</p>
              <p class="text-sm font-bold text-white tracking-tight">Assistant</p>
            </div>
          </div>
          <p class="text-xs text-gray-600 mt-2 leading-tight">
            Report analysis &amp; risk flagging
          </p>
        </div>

        <!-- Nav -->
        <nav class="flex-1 p-3 space-y-1">
          @for (item of navItems(); track item.id) {
            <button
              type="button"
              (click)="go(item.id)"
              [attr.aria-current]="currentPage() === item.id ? 'page' : null"
              class="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm
                     transition-all text-left"
              [class.bg-blue-600]="currentPage() === item.id"
              [class.text-white]="currentPage() === item.id"
              [class.text-gray-400]="currentPage() !== item.id"
              [class.hover:bg-gray-800]="currentPage() !== item.id">
              <span aria-hidden="true">{{ item.icon }}</span>
              <span class="font-medium">{{ item.label }}</span>
              @if (item.id === 'review' && (reviewState.pendingCount() ?? 0) > 0) {
                <span class="ml-auto min-w-5 text-center text-[11px] font-bold px-1.5 py-0.5 rounded-full
                             bg-amber-500 text-gray-950"
                      [attr.aria-label]="reviewState.pendingCount() + ' reports pending review'">
                  {{ reviewState.pendingCount() }}
                </span>
              } @else if (currentPage() === item.id) {
                <span class="ml-auto w-1.5 h-1.5 rounded-full bg-white/70"></span>
              }
            </button>
          }
        </nav>

        <!-- Signed-in user -->
        <div class="px-4 pb-3">
          <div class="bg-gray-800/60 border border-gray-800/80 rounded-lg p-3">
            <p class="text-sm font-medium text-gray-200 truncate">{{ auth.user()?.displayName }}</p>
            <div class="flex items-center justify-between gap-2 mt-1.5">
              <span class="text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded
                           bg-blue-500/15 text-blue-300">{{ auth.role() }}</span>
              <button type="button" (click)="signOut()"
                      class="text-xs text-gray-400 hover:text-white underline underline-offset-2 transition-colors">
                Sign out
              </button>
            </div>
          </div>
        </div>

        <!-- Architecture summary -->
        <div class="p-4 border-t border-gray-800 hidden md:block">
          <div class="bg-gray-800/60 border border-gray-800/80 rounded-lg p-3 text-xs text-gray-500 leading-relaxed">
            <p class="font-semibold text-gray-400 mb-1.5 tracking-wide">Multi-Agent Pipeline</p>
            <p>Orchestrator + Verification agents running on Gemini 2.5 Flash</p>
            <div class="mt-2 flex gap-1">
              <span class="bg-blue-500/20 text-blue-400 px-1.5 py-0.5 rounded text-xs">PubMed</span>
              <span class="bg-purple-500/20 text-purple-400 px-1.5 py-0.5 rounded text-xs">OpenFDA</span>
            </div>
          </div>
        </div>
      </aside>

      <!-- Main content -->
      <main class="flex-1 min-w-0 flex flex-col overflow-hidden">
        <div class="flex-1 overflow-y-auto overflow-x-hidden">
          @switch (currentPage()) {
            @case ('analyse')   { <app-report-upload /> }
            @case ('review')    { <app-review-queue /> }
            @case ('history')   { <app-patient-history /> }
            @case ('dashboard') { <app-dashboard /> }
            @case ('portal')    { <app-patient-portal /> }
            @case ('about')     { <ng-container *ngTemplateOutlet="aboutTpl" /> }
          }
        </div>

        <!-- AI disclaimer — shown on every page, kept low-key but always visible -->
        <footer class="shrink-0 px-4 sm:px-6 py-2 border-t border-gray-800/60 text-[11px] text-gray-600
                       flex items-start gap-1.5">
          <span class="text-gray-700">⚠</span>
          @if (auth.role() === 'patient') {
            <span>
              AI-generated explanation, reviewed by your care team — not a diagnosis and not a
              substitute for professional medical advice. Talk to your doctor about any questions.
            </span>
          } @else {
            <span>
              AI-generated analysis for clinician review only — not a diagnosis, not a
              substitute for professional medical judgment. Verify every finding against
              the source report before acting on it.
            </span>
          }
        </footer>
      </main>
    </div>
    }

    <!-- About page template -->
    <ng-template #aboutTpl>
      <div class="p-4 sm:p-6 max-w-2xl">
        <p class="text-xs font-semibold text-blue-400 uppercase tracking-widest mb-1">About</p>
        <h1 class="text-2xl sm:text-3xl font-bold text-white tracking-tight mb-2">Agentic Medical Assistant</h1>
        <p class="text-gray-500 text-sm mb-8">
          A multi-agent pipeline for analysing medical reports, with ICMR-grounded risk
          flagging for Indian healthcare contexts.
        </p>

        <div class="space-y-4">
          @for (feature of features; track feature.title) {
            <div class="bg-gray-900 border border-gray-800 rounded-xl p-5
                        hover:border-gray-700 transition-colors">
              <div class="w-9 h-9 rounded-lg bg-gray-800 flex items-center justify-center text-lg mb-3">
                {{ feature.icon }}
              </div>
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
                            font-mono text-xs px-3 py-1.5 rounded-md
                            hover:border-blue-700 hover:text-blue-300 transition-colors">{{ tag }}</span>
            }
          </div>
        </div>
      </div>
    </ng-template>
  `,
})
export class AppComponent {
  activePage = signal<Page>('analyse');
  menuOpen = signal(false);

  private allNav: NavItem[] = [
    { id: 'analyse',   label: 'Analyse',      icon: '🩺', roles: ['clinician', 'service'] },
    { id: 'review',    label: 'Review queue', icon: '✅', roles: ['clinician'] },
    { id: 'history',   label: 'Patients',     icon: '📈', roles: ['clinician', 'service'] },
    { id: 'dashboard', label: 'Dashboard',    icon: '📊', roles: ['clinician'] },
    { id: 'portal',    label: 'My reports',   icon: '📄', roles: ['patient'] },
    { id: 'about',     label: 'About',        icon: 'ℹ️', roles: ['clinician', 'service', 'patient'] },
  ];

  navItems = computed(() => {
    const role = this.auth.role();
    return role ? this.allNav.filter((n) => n.roles.includes(role)) : [];
  });

  /** The page actually shown — never one the current role may not open. */
  currentPage = computed<Page>(() => {
    const items = this.navItems();
    const wanted = this.activePage();
    return items.some((n) => n.id === wanted) ? wanted : (items[0]?.id ?? 'about');
  });

  constructor(public auth: AuthService, public reviewState: ReviewStateService) {
    // New sign-in (or sign-out) -> land on the role's first page; poll the
    // pending-review count for clinicians only.
    effect(() => {
      const user = this.auth.user();
      untracked(() => {
        this.menuOpen.set(false);
        this.activePage.set(this.navItems()[0]?.id ?? 'about');
        if (user?.role === 'clinician') {
          this.reviewState.start();
        } else {
          this.reviewState.stop();
        }
      });
    }, { allowSignalWrites: true });
  }

  go(page: Page): void {
    this.activePage.set(page);
    this.menuOpen.set(false);
  }

  signOut(): void {
    this.auth.logout();
  }

  features = [
    {
      icon: '🤖',
      title: 'Multi-Agent Architecture',
      desc: 'An Orchestrator agent with tool-calling handles pipeline decisions. A Verification agent independently audits outputs for hallucinated clinical claims before delivery — and fails CLOSED (flags "not verified") rather than silently passing if the audit itself cannot run.',
    },
    {
      icon: '🇮🇳',
      title: 'ICMR-Grounded Risk Flagging',
      desc: 'Lab values checked against Indian Council of Medical Research reference ranges, not Western defaults. Gender-specific thresholds included.',
    },
    {
      icon: '📈',
      title: 'Four-Tier Longitudinal Intelligence',
      desc: '1) Trend detection across visits, 2) deviation from this patient\'s own historical baseline (not just the population range), 3) illustrative slope-based threshold projections, 4) semantic recall of similar past episodes from report text — via ChromaDB.',
    },
    {
      icon: '📚',
      title: 'Real Clinical Evidence',
      desc: 'Agent automatically searches PubMed for relevant clinical guidelines when significant findings are detected. OpenFDA checked for medication safety alerts.',
    },
    {
      icon: '👩‍⚕️',
      title: 'Clinician Approval Before Release',
      desc: 'Every analysis starts as "pending review" and stays hidden from the patient until a clinician approves it. Source evidence for each claim, numeric grounding checks, a medication-interaction rule set and FHIR export support that review.',
    },
    {
      icon: '🛡️',
      title: 'LLM-Specific Security Hardening',
      desc: 'Untrusted document text is delimited and scanned for prompt-injection patterns before reaching the model; agent-chosen tool arguments are sanitized before hitting external APIs; model output is scanned for injection compliance; uploads are rate-limited per caller. Role-based access and a hash-chained audit log record who did what.',
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
