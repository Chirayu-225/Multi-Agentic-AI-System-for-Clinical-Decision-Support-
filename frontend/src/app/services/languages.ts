// src/app/services/languages.ts
export interface LanguageOption {
  code: string;
  name: string;
}

export const LANGUAGES: LanguageOption[] = [
  { code: 'hi', name: 'Hindi' },
  { code: 'kn', name: 'Kannada' },
  { code: 'ta', name: 'Tamil' },
  { code: 'te', name: 'Telugu' },
  { code: 'mr', name: 'Marathi' },
  { code: 'bn', name: 'Bengali' },
];

/** Accepts either a code ("hi") or a name ("Hindi") and returns a display name. */
export function languageName(codeOrName: string): string {
  const v = (codeOrName || '').trim();
  const hit = LANGUAGES.find(
    (l) => l.code === v.toLowerCase() || l.name.toLowerCase() === v.toLowerCase()
  );
  return hit ? hit.name : v;
}
