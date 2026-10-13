// src/app/services/auth.service.ts
// Holds the signed-in user + JWT. The token lives in memory (signals) and in
// sessionStorage only — never localStorage, never a cookie, never logged.
// Every storage access is wrapped in try/catch so the app keeps working
// (for the current page load) when storage is blocked or throws.
import { Injectable, computed, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';
import { environment } from '../../environments/environment';
import { AuthUser, LoginResponse, Role } from '../models/report.model';

const STORAGE_KEY = 'ama.session.v1';
const MAX_TIMEOUT_MS = 2_147_000_000; // setTimeout upper bound (~24.8 days)

interface StoredSession {
  token: string | null;
  user: AuthUser;
  expiresAt: number | null; // epoch ms; null for the service-key session
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  private http = inject(HttpClient);

  readonly user = signal<AuthUser | null>(null);
  readonly token = signal<string | null>(null);

  readonly isAuthenticated = computed(() => this.user() !== null);
  readonly role = computed<Role | null>(() => this.user()?.role ?? null);
  /** True when the build ships a service API key (limited-access fallback). */
  readonly serviceKeyAvailable = !!environment.apiKey;

  private expiryTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    this.restore();
  }

  hasRole(...roles: Role[]): boolean {
    const r = this.role();
    return r !== null && roles.includes(r);
  }

  login(username: string, password: string): Observable<LoginResponse> {
    return this.http
      .post<LoginResponse>(`${environment.apiUrl}/auth/login`, { username, password })
      .pipe(
        tap((res) => {
          const user: AuthUser = {
            username: res.username,
            role: res.role,
            displayName: res.display_name || res.username,
            patientId: res.patient_id ?? null,
          };
          const expiresAt = Date.now() + Math.max(0, res.expires_in) * 1000;
          this.setSession(res.access_token, user, expiresAt);
        })
      );
  }

  /** Limited fallback: act with the shared service key (no JWT, no review rights). */
  useServiceKey(): void {
    if (!environment.apiKey) return;
    this.setSession(
      null,
      { username: 'service', role: 'service', displayName: 'Service key', patientId: null },
      null
    );
  }

  logout(): void {
    this.clearTimer();
    this.token.set(null);
    this.user.set(null);
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      /* storage unavailable — in-memory state is already cleared */
    }
  }

  // ── internals ─────────────────────────────────────────────────────────────

  private setSession(token: string | null, user: AuthUser, expiresAt: number | null): void {
    this.token.set(token);
    this.user.set(user);
    this.persist({ token, user, expiresAt });
    this.scheduleExpiry(expiresAt);
  }

  private persist(session: StoredSession): void {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    } catch {
      /* storage blocked/full — session simply won't survive a reload */
    }
  }

  private restore(): void {
    let raw: string | null = null;
    try {
      raw = sessionStorage.getItem(STORAGE_KEY);
    } catch {
      return;
    }
    if (!raw) return;
    try {
      const s = JSON.parse(raw) as Partial<StoredSession>;
      const u = s.user;
      const validUser =
        !!u && typeof u.username === 'string' &&
        (u.role === 'clinician' || u.role === 'patient' || u.role === 'service');
      if (!validUser) throw new Error('malformed');

      if (u!.role === 'service') {
        if (!environment.apiKey) throw new Error('service key not configured');
        this.token.set(null);
        this.user.set(u!);
        return;
      }
      if (typeof s.token !== 'string' || typeof s.expiresAt !== 'number' || s.expiresAt <= Date.now()) {
        throw new Error('expired');
      }
      this.token.set(s.token);
      this.user.set(u!);
      this.scheduleExpiry(s.expiresAt);
    } catch {
      this.logout();
    }
  }

  private scheduleExpiry(expiresAt: number | null): void {
    this.clearTimer();
    if (expiresAt === null) return;
    const ms = Math.min(Math.max(expiresAt - Date.now(), 0), MAX_TIMEOUT_MS);
    this.expiryTimer = setTimeout(() => this.logout(), ms);
  }

  private clearTimer(): void {
    if (this.expiryTimer !== null) {
      clearTimeout(this.expiryTimer);
      this.expiryTimer = null;
    }
  }
}
