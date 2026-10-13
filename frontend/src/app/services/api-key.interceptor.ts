// src/app/services/api-key.interceptor.ts
// Attaches the shared-secret X-API-Key header to every outgoing HTTP
// request. See backend/src/api/auth.py for what this does and doesn't
// protect against — it's a demo-grade gate, not per-user authentication.
import { HttpInterceptorFn } from '@angular/common/http';
import { environment } from '../../environments/environment';

export const apiKeyInterceptor: HttpInterceptorFn = (req, next) => {
  if (!environment.apiKey) {
    return next(req);
  }
  return next(
    req.clone({ setHeaders: { 'X-API-Key': environment.apiKey } })
  );
};
