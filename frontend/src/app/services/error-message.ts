// src/app/services/error-message.ts
// Turns an HttpErrorResponse (or anything thrown) into display text.
// Backend errors are {detail: string}; 422s may carry detail as [{msg}].
// The result is always rendered with {{ }} interpolation, never as markup.
export function errorMessage(err: unknown, fallback: string): string {
  const e = err as { status?: number; error?: { detail?: unknown } } | null;
  if (e && e.status === 0) {
    return 'Cannot reach the server. Check your connection and try again.';
  }
  const detail = e?.error?.detail;
  if (typeof detail === 'string' && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((d) => (d && typeof d === 'object' && typeof (d as { msg?: unknown }).msg === 'string'
        ? (d as { msg: string }).msg : null))
      .filter((m): m is string => !!m);
    if (msgs.length) return msgs.join('; ');
  }
  return fallback;
}
