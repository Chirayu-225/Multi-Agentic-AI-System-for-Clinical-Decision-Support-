// src/app/services/review-state.service.ts
// Shared count of reports awaiting clinician review (sidebar badge).
import { Injectable, inject, signal } from '@angular/core';
import { ApiService } from './api.service';
import { AuthService } from './auth.service';

const POLL_MS = 60_000;

@Injectable({ providedIn: 'root' })
export class ReviewStateService {
  private api = inject(ApiService);
  private auth = inject(AuthService);

  readonly pendingCount = signal<number | null>(null);
  private timer: ReturnType<typeof setInterval> | null = null;

  refresh(): void {
    if (!this.auth.hasRole('clinician')) return;
    this.api.getPendingReviews().subscribe({
      next: (r) => this.pendingCount.set(r.count),
      error: () => {},
    });
  }

  /** Begin polling (clinician only). Safe to call repeatedly. */
  start(): void {
    if (this.timer !== null) return;
    this.refresh();
    this.timer = setInterval(() => this.refresh(), POLL_MS);
  }

  stop(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.pendingCount.set(null);
  }
}
