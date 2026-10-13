// src/app/components/auth/login.component.ts
import { Component, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../../services/auth.service';
import { errorMessage } from '../../services/error-message';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="min-h-dvh bg-gray-950 flex items-center justify-center p-4">
      <div class="w-full max-w-sm">

        <div class="flex items-center justify-center gap-3 mb-6">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-sky-500
                      flex items-center justify-center text-xl shadow-lg shadow-blue-500/20 shrink-0"
               aria-hidden="true">🩺</div>
          <div class="leading-tight">
            <p class="text-base font-bold text-white tracking-tight">Agentic Medical Assistant</p>
            <p class="text-xs text-gray-600">Report analysis &amp; risk flagging</p>
          </div>
        </div>

        <form (ngSubmit)="submit()" class="bg-gray-900 border border-gray-800 rounded-2xl p-6
                                           shadow-lg shadow-black/30" autocomplete="on">
          <h1 class="text-lg font-semibold text-white mb-1">Sign in</h1>
          <p class="text-xs text-gray-500 mb-5">Use the account issued by your care team.</p>

          <label for="login-username"
                 class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Username
          </label>
          <input id="login-username" name="username" type="text"
                 [(ngModel)]="username" autocomplete="username" required
                 autocapitalize="none" spellcheck="false"
                 class="w-full bg-gray-950 border border-gray-700 rounded-lg px-4 py-2.5 mb-4
                        text-gray-200 text-sm focus:outline-none focus:border-blue-500
                        focus:ring-1 focus:ring-blue-500 transition-colors" />

          <label for="login-password"
                 class="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Password
          </label>
          <input id="login-password" name="password" type="password"
                 [(ngModel)]="password" autocomplete="current-password" required
                 class="w-full bg-gray-950 border border-gray-700 rounded-lg px-4 py-2.5 mb-4
                        text-gray-200 text-sm focus:outline-none focus:border-blue-500
                        focus:ring-1 focus:ring-blue-500 transition-colors" />

          @if (error()) {
            <div role="alert"
                 class="mb-4 rounded-lg p-3 border border-red-500/40 bg-red-500/10 text-sm text-red-300">
              {{ error() }}
            </div>
          }

          <button type="submit" [disabled]="!username || !password || loading()"
                  class="w-full py-2.5 rounded-lg font-semibold text-sm tracking-wide transition-all
                         bg-gradient-to-r from-blue-600 to-sky-500 text-white
                         hover:shadow-lg hover:shadow-blue-500/25
                         disabled:opacity-40 disabled:cursor-not-allowed">
            {{ loading() ? 'Signing in...' : 'Sign in' }}
          </button>

          @if (auth.serviceKeyAvailable) {
            <button type="button" (click)="auth.useServiceKey()"
                    class="w-full mt-4 text-xs text-gray-500 hover:text-gray-300 underline
                           underline-offset-2 transition-colors">
              Continue with service key (limited access: cannot approve reports)
            </button>
          }
        </form>
      </div>
    </div>
  `,
})
export class LoginComponent {
  username = '';
  password = '';
  loading = signal(false);
  error = signal<string | null>(null);

  constructor(public auth: AuthService) {}

  submit(): void {
    if (!this.username || !this.password || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    this.auth.login(this.username.trim(), this.password).subscribe({
      next: () => {
        this.loading.set(false);
        this.password = '';
      },
      error: (err) => {
        this.loading.set(false);
        this.password = '';
        this.error.set(
          errorMessage(err, 'Sign-in failed. Check your username and password and try again.')
        );
      },
    });
  }
}
