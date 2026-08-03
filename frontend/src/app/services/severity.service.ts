// src/app/services/severity.service.ts
import { Injectable } from '@angular/core';
import { SeverityLevel } from '../models/report.model';

export interface SeverityConfig {
  level: SeverityLevel;
  color: string;       // hex
  bgClass: string;     // tailwind bg
  borderClass: string; // tailwind border
  textClass: string;   // tailwind text
  label: string;
  icon: string;
}

const SEVERITY_MAP: Record<SeverityLevel, SeverityConfig> = {
  NORMAL: {
    level: 'NORMAL',
    color: '#22C55E',
    bgClass: 'bg-green-500/10',
    borderClass: 'border-green-500',
    textClass: 'text-green-400',
    label: 'Normal',
    icon: '✓',
  },
  MODERATE: {
    level: 'MODERATE',
    color: '#EAB308',
    bgClass: 'bg-yellow-500/10',
    borderClass: 'border-yellow-500',
    textClass: 'text-yellow-400',
    label: 'Moderate',
    icon: '⚠',
  },
  HIGH: {
    level: 'HIGH',
    color: '#F97316',
    bgClass: 'bg-orange-500/10',
    borderClass: 'border-orange-500',
    textClass: 'text-orange-400',
    label: 'High',
    icon: '⚠',
  },
  CRITICAL: {
    level: 'CRITICAL',
    color: '#EF4444',
    bgClass: 'bg-red-500/10',
    borderClass: 'border-red-500',
    textClass: 'text-red-400',
    label: 'Critical',
    icon: '🔴',
  },
};

@Injectable({ providedIn: 'root' })
export class SeverityService {
  normalize(raw: string): SeverityLevel {
    const s = raw.toUpperCase().split('.').pop() || 'NORMAL';
    return (SEVERITY_MAP[s as SeverityLevel] ? s : 'NORMAL') as SeverityLevel;
  }

  getConfig(raw: string): SeverityConfig {
    return SEVERITY_MAP[this.normalize(raw)];
  }

  getColor(raw: string): string {
    return this.getConfig(raw).color;
  }

  getBannerTitle(raw: string): string {
    const level = this.normalize(raw);
    return level === 'CRITICAL'
      ? '⚠ CRITICAL — Immediate Attention Required'
      : `Risk Level · ${level}`;
  }
}
