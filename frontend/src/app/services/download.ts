// src/app/services/download.ts
// Saves JSON to disk via a Blob + temporary <a download>; the object URL is
// revoked straight away.
export function downloadJson(filename: string, data: unknown): void {
  const safeName = filename.replace(/[^A-Za-z0-9._-]/g, '_');
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = safeName;
  a.rel = 'noopener';
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
