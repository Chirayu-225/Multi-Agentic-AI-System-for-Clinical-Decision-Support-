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
  token: string;
  user: AuthUser;
  expiresAt: number; // epoch ms
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  private http = inject(HttpClient);

  readonly user = signal<AuthUser | null>(null);
  readonly token = signal<string | null>(null);

  readonly isAuthenticated = computed(() => this.user() !== null);
  readonly role = computed<Role | null>(() => this.user()?.role ?? null);
  /** Signed in with a temporary password: only changing it is allowed until it is done. */
  readonly mustChangePassword = computed(() => this.user()?.mustChangePassword === true);

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
      .pipe(tap((res) => this.applySession(res)));
  }

  /** Change your own password. On success the server returns a fresh, unrestricted token. */
  changePassword(currentPassword: string, newPassword: string): Observable<LoginResponse> {
    return this.http
      .post<LoginResponse>(`${environment.apiUrl}/auth/change-password`, {
        current_password: currentPassword,
        new_password: newPassword,
      })
      .pipe(tap((res) => this.applySession(res)));
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

  private applySession(res: LoginResponse): void {
    const user: AuthUser = {
      username: res.username,
      role: res.role,
      displayName: res.display_name || res.username,
      patientId: res.patient_id ?? null,
      mustChangePassword: res.must_change_password === true,
    };
    const expiresAt = Date.now() + Math.max(0, res.expires_in) * 1000;
    this.token.set(res.access_token);
    this.user.set(user);
    this.persist({ token: res.access_token, user, expiresAt });
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
        !!u && typeof u.username === 'string' && (u.role === 'clinician' || u.role === 'patient');
      if (!validUser) throw new Error('malformed');
      if (typeof s.token !== 'string' || typeof s.expiresAt !== 'number' || s.expiresAt <= Date.now()) {
        throw new Error('expired');
      }
      this.token.set(s.token);
      this.user.set({ ...u!, mustChangePassword: u!.mustChangePassword === true });
      this.scheduleExpiry(s.expiresAt);
    } catch {
      this.logout();
    }
  }

  private scheduleExpiry(expiresAt: number): void {
    this.clearTimer();
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
