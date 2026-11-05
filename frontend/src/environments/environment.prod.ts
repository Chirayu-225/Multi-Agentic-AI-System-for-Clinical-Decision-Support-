// src/environments/environment.prod.ts
export const environment = {
  production: true,
  // Same-origin — nginx proxies /report, /patient(s), /health, /ws to the backend container
  apiUrl: '',
  wsUrl:  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`,
};
