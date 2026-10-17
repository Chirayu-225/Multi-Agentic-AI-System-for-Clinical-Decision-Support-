// src/environments/environment.ts
export const environment = {
  production: false,
  apiUrl: 'http://localhost:8001',
  wsUrl:  'ws://localhost:8001',
  // Must match backend/.env's API_KEY for local dev. This is a shared
  // secret baked into the built JS bundle, NOT real per-user auth — see
  // backend/src/api/auth.py for what it does and doesn't protect against.
  apiKey: '',
};
