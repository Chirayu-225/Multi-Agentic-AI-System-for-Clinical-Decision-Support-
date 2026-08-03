// src/app/components/history/patient-history.component.ts
import { Component, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { SeverityService } from '../../services/severity.service';
import { PatientHistory, PatientHistoryReport, LabValue } from '../../models/report.model';

@Component({
  selector: 'app-patient-history',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="min-h-screen bg-gray-950 text-gray-100 p-6">

      <!-- Header -->
      <div class="mb-8">
        <p class="text-xs font-semibold text-blue-400 uppercase tracking-widest mb-1">
          Longitudinal Tracking
        </p>
        <h1 class="text-3xl font-bold text-white tracking-tight">Patient History</h1>
        <p class="text-gray-500 mt-1 text-sm">
          View lab value trends and risk patterns across multiple visits.
        </p>
      </div>

      <!-- Search -->
      <div class="max-w-md mb-6">
        <label class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Patient ID
        </label>
        <div class="flex gap-3">
          <input
            [(ngModel)]="patientId"
            placeholder="e.g. PT-2026-001"
            (keydown.enter)="loadHistory()"
            class="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3
                   text-gray-200 font-mono text-sm focus:outline-none focus:border-blue-500
                   focus:ring-1 focus:ring-blue-500 transition-colors"
          />
          <button
            (click)="loadHistory()"
            [disabled]="!patientId || loading()"
            class="px-5 py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-40
                   rounded-lg text-sm font-semibold transition-colors">
            Search
          </button>
        </div>
      </div>

      <!-- Results -->
      @if (history()) {
        <div class="mb-4 text-sm text-gray-500">
          <span class="font-mono text-blue-400">{{ history()!.patient_id }}</span>
          · {{ history()!.report_count }} report(s) on record
        </div>

        @if (history()!.reports.length === 0) {
          <div class="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
            <p class="text-gray-600 text-sm">No reports found for this patient ID.</p>
          </div>
        } @else {

          <!-- Lab trend chart -->
          @if (availableTests().length) {
            <div class="bg-gray-900 border border-gray-800 rounded-xl p-5 mb-6">
              <div class="flex items-center justify-between mb-4">
                <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest">
                  Lab Value Trend
                </p>
                <select
                  [(ngModel)]="selectedTest"
                  (ngModelChange)="onTestChange()"
                  class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5
                         text-gray-300 text-sm font-mono focus:outline-none focus:border-blue-500">
                  @for (test of availableTests(); track test) {
                    <option [value]="test">{{ test }}</option>
                  }
                </select>
              </div>

              <!-- SVG sparkline chart -->
              @if (chartPoints().length >= 2) {
                <div class="relative h-40">
                  <svg class="w-full h-full" viewBox="0 0 600 160" preserveAspectRatio="none">
                    <!-- Grid lines -->
                    @for (line of [0, 1, 2, 3]; track line) {
                      <line [attr.x1]="0" [attr.y1]="line * 40"
                            [attr.x2]="600" [attr.y2]="line * 40"
                            stroke="#1F2937" stroke-width="1"/>
                    }
                    <!-- Area fill -->
                    <path [attr.d]="areaPath()" [attr.fill]="trendColor() + '15'"/>
                    <!-- Line -->
                    <polyline
                      [attr.points]="linePoints()"
                      fill="none"
                      [attr.stroke]="trendColor()"
                      stroke-width="2.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"/>
                    <!-- Dots -->
                    @for (pt of chartPoints(); track $index) {
                      <circle
                        [attr.cx]="pt.x"
                        [attr.cy]="pt.y"
                        r="5"
                        [attr.fill]="trendColor()"
                        stroke="#030712"
                        stroke-width="2"/>
                    }
                  </svg>
                  <!-- X-axis labels -->
                  <div class="flex justify-between mt-1 px-1">
                    @for (report of sortedReports(); track report.report_id) {
                      <span class="text-xs text-gray-600 font-mono">
                        {{ report.report_date | slice:0:7 }}
                      </span>
                    }
                  </div>
                </div>
              } @else {
                <p class="text-sm text-gray-600 text-center py-8">
                  Upload more reports for this patient to see trends.
                </p>
              }
            </div>
          }

          <!-- Visit history cards -->
          <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest
                     pb-2 border-b border-gray-800 mb-4">Visit History</p>

          @for (report of sortedReports().slice().reverse(); track report.report_id) {
            <div class="bg-gray-900 border border-gray-800 rounded-xl mb-3 overflow-hidden">
              <button
                class="w-full flex items-center justify-between px-5 py-4 text-left
                       hover:bg-gray-800/50 transition-colors"
                (click)="toggleReport(report.report_id)">
                <div class="flex items-center gap-3">
                  <span class="text-gray-400 text-sm">📄</span>
                  <div>
                    <p class="text-sm font-medium text-gray-200">{{ report.filename }}</p>
                    <p class="text-xs text-gray-500 font-mono">{{ report.report_date }}</p>
                  </div>
                </div>
                <div class="flex items-center gap-3">
                  @if (report.risk_summary?.overall_severity) {
                    <span class="text-xs font-semibold px-2 py-0.5 rounded font-mono"
                          [style.color]="sev.getColor(report.risk_summary.overall_severity)"
                          [style.background]="sev.getColor(report.risk_summary.overall_severity) + '20'">
                      {{ sev.normalize(report.risk_summary.overall_severity) }}
                    </span>
                  }
                  <span class="text-gray-600 text-sm">
                    {{ expandedReports().has(report.report_id) ? '▲' : '▼' }}
                  </span>
                </div>
              </button>

              @if (expandedReports().has(report.report_id)) {
                <div class="px-5 pb-4 border-t border-gray-800">
                  @if (report.diagnoses?.length) {
                    <div class="mt-3 mb-2">
                      <span class="text-xs text-gray-500">Diagnoses: </span>
                      @for (dx of report.diagnoses; track dx) {
                        <span class="text-xs bg-gray-800 text-blue-300 px-2 py-0.5
                                      rounded font-mono mr-1">{{ dx }}</span>
                      }
                    </div>
                  }
                  @if (report.lab_values?.length) {
                    <div class="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-3">
                      @for (lab of report.lab_values; track lab.test) {
                        @if (lab.value !== null) {
                          <div class="bg-gray-800 rounded-lg px-3 py-2">
                            <p class="text-xs text-gray-500">{{ lab.test }}</p>
                            <p class="text-sm font-mono font-semibold text-gray-200">
                              {{ lab.value }} {{ lab.unit }}
                            </p>
                          </div>
                        }
                      }
                    </div>
                  }
                </div>
              }
            </div>
          }
        }
      }
    </div>
  `,
})
export class PatientHistoryComponent {
  patientId  = '';
  loading    = signal(false);
  history    = signal<PatientHistory | null>(null);
  selectedTest = '';
  expandedReports = signal(new Set<string>());

  constructor(private api: ApiService, public sev: SeverityService) {}

  loadHistory(): void {
    if (!this.patientId) return;
    this.loading.set(true);
    this.api.getPatientHistory(this.patientId).subscribe({
      next: (h) => {
        this.history.set(h);
        this.loading.set(false);
        const tests = this.availableTests();
        if (tests.length) this.selectedTest = tests[0];
      },
      error: () => this.loading.set(false),
    });
  }

  sortedReports(): PatientHistoryReport[] {
    return [...(this.history()?.reports ?? [])].sort(
      (a, b) => a.report_date.localeCompare(b.report_date)
    );
  }

  availableTests(): string[] {
    const tests = new Set<string>();
    this.sortedReports().forEach(r =>
      r.lab_values?.forEach(l => { if (l.value !== null) tests.add(l.test); })
    );
    return [...tests];
  }

  onTestChange(): void {}

  chartPoints(): Array<{ x: number; y: number; value: number }> {
    const reports = this.sortedReports();
    const values: number[] = [];
    reports.forEach(r => {
      const lab = r.lab_values?.find(l => l.test === this.selectedTest && l.value !== null);
      if (lab) values.push(Number(lab.value));
    });
    if (values.length < 2) return [];

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const W = 600, H = 140, PAD = 20;

    return values.map((v, i) => ({
      x: PAD + (i / (values.length - 1)) * (W - PAD * 2),
      y: H - PAD - ((v - min) / range) * (H - PAD * 2),
      value: v,
    }));
  }

  linePoints(): string {
    return this.chartPoints().map(p => `${p.x},${p.y}`).join(' ');
  }

  areaPath(): string {
    const pts = this.chartPoints();
    if (!pts.length) return '';
    const H = 140;
    return `M${pts[0].x},${H} ` +
      pts.map(p => `L${p.x},${p.y}`).join(' ') +
      ` L${pts[pts.length - 1].x},${H} Z`;
  }

  trendColor(): string {
    const pts = this.chartPoints();
    if (pts.length < 2) return '#60A5FA';
    return pts[pts.length - 1].value < pts[0].value ? '#EF4444' :
           pts[pts.length - 1].value > pts[0].value ? '#22C55E' : '#60A5FA';
  }

  toggleReport(id: string): void {
    this.expandedReports.update(set => {
      const next = new Set(set);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }
}
