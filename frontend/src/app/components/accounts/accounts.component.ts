// src/app/components/accounts/accounts.component.ts  (clinician only)
// Create a login for a patient and reset a forgotten password. The temporary password is
// shown once, right after it is issued; it is never stored or retrievable again.
// All text is interpolated, never rendered as markup.
import { Component, OnInit, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { ApiService } from '../../services/api.service';
import { errorMessage } from '../../services/error-message';
import { PatientAccount, PatientAccountWithSecret } from '../../models/report.model';

@Component({
  selector: 'app-accounts',
  standalone: true,
  imports: [FormsModule, DatePipe],
  template: `
    <div class="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6 max-w-3xl">
      <p class="text-xs font-semibold text-blue-400 uppercase tracking-widest mb-1">Administration</p>
      <h1 class="text-2xl sm:text-3xl font-bold text-white tracking-tight">Patient accounts</h1>
      <p class="text-gray-500 mt-1 mb-6 text-sm">
        Give a patient a login to read their approved reports. You choose the username; the system
        issues a temporary password that the patient must replace at first sign-in.
      </p>

      @if (secret()) {
        <div role="status" class="mb-6 rounded-xl p-4 border border-amber-500/40 bg-amber-500/10">
          <p class="text-sm font-semibold text-amber-300 mb-1">
            Temporary password for {{ secret()!.username }}
          </p>
          <p class="font-mono text-lg text-white select-all break-all">{{ secret()!.temporary_password }}</p>
          <p class="text-xs text-amber-200/80 mt-2">
            Shown once. Give it to the patient securely (in person or by phone, not in a shared chat).
            They will be asked to choose their own password when they sign in.
          </p>
          <button type="button" (click)="secret.set(null)"
                  class="mt-3 text-xs underline underline-offset-2 text-amber-200 hover:text-white">
            I have noted it, hide
          </button>
        </div>
      }

      @if (error()) {
        <div role="alert" class="mb-4 rounded-xl p-3 border border-red-500/40 bg-red-500/10 text-sm text-red-300">
          {{ error() }}
        </div>
      }

      <form (ngSubmit)="create()" class="bg-gray-900 border border-gray-800 rounded-xl p-5 mb-8">
        <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3">New patient account</p>
        <div class="grid sm:grid-cols-3 gap-3">
          <div>
            <label for="acc-patient" class="block text-xs text-gray-400 mb-1">Patient ID</label>
            <select id="acc-patient" name="patient" [(ngModel)]="patientId" required
                    class="w-full bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200">
              <option value="" disabled>Choose...</option>
              @for (p of patients(); track p) { <option [value]="p">{{ p }}</option> }
            </select>
          </div>
          <div>
            <label for="acc-user" class="block text-xs text-gray-400 mb-1">Username</label>
            <input id="acc-user" name="username" [(ngModel)]="username" required minlength="3"
                   autocapitalize="none" spellcheck="false" autocomplete="off"
                   class="w-full bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200" />
          </div>
          <div>
            <label for="acc-name" class="block text-xs text-gray-400 mb-1">Display name (optional)</label>
            <input id="acc-name" name="displayName" [(ngModel)]="displayName" maxlength="100" autocomplete="off"
                   class="w-full bg-gray-950 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200" />
          </div>
        </div>
        <button type="submit" [disabled]="!patientId || username.trim().length < 3 || busy()"
                class="mt-4 px-4 py-2 rounded-lg text-sm font-semibold bg-gradient-to-r from-blue-600 to-sky-500
                       text-white disabled:opacity-40 disabled:cursor-not-allowed">
          {{ busy() ? 'Working...' : 'Create account' }}
        </button>
        @if (!patients().length && !loading()) {
          <p class="text-xs text-gray-500 mt-3">No patients yet. Upload a report first; accounts attach to existing patients.</p>
        }
      </form>

      <p class="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3">Existing accounts</p>
      @if (loading()) {
        <p class="text-sm text-gray-500">Loading...</p>
      } @else if (!accounts().length) {
        <p class="text-sm text-gray-500">No patient accounts yet.</p>
      } @else {
        <div class="space-y-2">
          @for (a of accounts(); track a.username) {
            <div class="bg-gray-900 border border-gray-800 rounded-xl px-4 py-3 flex flex-wrap items-center gap-3">
              <div class="min-w-0 flex-1">
                <p class="text-sm font-medium text-gray-200 truncate">{{ a.display_name }}
                  <span class="text-gray-600 font-mono text-xs">({{ a.username }})</span></p>
                <p class="text-xs text-gray-500">
                  Patient {{ a.patient_id }} ·
                  @if (a.must_change_password) { waiting for first sign-in }
                  @else if (a.last_login_at) { last sign-in {{ a.last_login_at | date:'d MMM y, HH:mm' }} }
                  @else { never signed in }
                </p>
              </div>
              <button type="button" (click)="reset(a)" [disabled]="busy()"
                      class="px-3 py-1.5 rounded-lg text-xs font-semibold border border-gray-700 text-gray-300
                             hover:border-blue-600 hover:text-white transition-colors disabled:opacity-40">
                Reset password
              </button>
            </div>
          }
        </div>
      }
    </div>
  `,
})
export class AccountsComponent implements OnInit {
  patients = signal<string[]>([]);
  accounts = signal<PatientAccount[]>([]);
  secret = signal<PatientAccountWithSecret | null>(null);
  loading = signal(true);
  busy = signal(false);
  error = signal<string | null>(null);

  patientId = '';
  username = '';
  displayName = '';

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.listPatients().subscribe({
      next: (r) => this.patients.set(r.patients),
      error: (e) => this.error.set(errorMessage(e, 'Could not load patients.')),
    });
    this.loadAccounts();
  }

  private loadAccounts(): void {
    this.api.listAccounts().subscribe({
      next: (r) => { this.accounts.set(r.users); this.loading.set(false); },
      error: (e) => { this.loading.set(false); this.error.set(errorMessage(e, 'Could not load accounts.')); },
    });
  }

  create(): void {
    if (!this.patientId || this.username.trim().length < 3 || this.busy()) return;
    this.busy.set(true);
    this.error.set(null);
    this.api.createAccount(this.username.trim().toLowerCase(), this.patientId, this.displayName.trim()).subscribe({
      next: (r) => {
        this.busy.set(false);
        this.secret.set(r);
        this.username = this.displayName = '';
        this.loadAccounts();
      },
      error: (e) => { this.busy.set(false); this.error.set(errorMessage(e, 'Could not create the account.')); },
    });
  }

  reset(a: PatientAccount): void {
    if (this.busy()) return;
    this.busy.set(true);
    this.error.set(null);
    this.api.resetAccountPassword(a.username).subscribe({
      next: (r) => { this.busy.set(false); this.secret.set(r); this.loadAccounts(); },
      error: (e) => { this.busy.set(false); this.error.set(errorMessage(e, 'Could not reset the password.')); },
    });
  }
}
