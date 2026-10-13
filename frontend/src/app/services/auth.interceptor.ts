// src/app/services/auth.interceptor.ts
// Attaches credentials to API requests only:
//   * Authorization: Bearer <jwt>  when a session token exists
//   * X-API-Key                    otherwise, when the build has a service key
// Credentials are never sent to a URL outside the API origin. A 401 from any
// call other than /auth/login ends the session (expired / revoked token); the
// error is re-thrown so callers still see it.
import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, throwError } from 'rxjs';
import { environment } from '../../environments/environment';
import { AuthService } from './auth.service';

function isApiRequest(url: string): boolean {
  return environment.apiUrl ? url.startsWith(environment.apiUrl) : url.startsWith('/');
}

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(AuthService);

  if (!isApiRequest(req.url)) {
    return next(req);
  }

  const token = auth.token();
  let outgoing = req;
  if (token) {
    outgoing = req.clone({ setHeaders: { Authorization: `Bearer ${token}` } });
  } else if (environment.apiKey) {
    outgoing = req.clone({ setHeaders: { 'X-API-Key': environment.apiKey } });
  }

  const isLogin = /\/auth\/login(\?|$)/.test(req.url);
  return next(outgoing).pipe(
    catchError((err: unknown) => {
      if (err instanceof HttpErrorResponse && err.status === 401 && !isLogin) {
        auth.logout();
      }
      return throwError(() => err);
    })
  );
};
