// src/environments/environment.prod.ts
export const environment = {
  production: true,
  // Same-origin — nginx proxies /report, /patient(s), /health, /ws to the backend container
  apiUrl: '',
  wsUrl:  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`,
  // Set this to the same value as API_KEY in backend/.env BEFORE running
  // `npm run build:prod` / `docker-compose up --build` — Angular has no
  // runtime env-var injection, so this is compiled into the JS bundle.
  // This is a shared secret gating the API, not per-user authentication.
  apiKey: '',
};
