// src/app/components/shared/agent-activity.component.ts
import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AgentStep } from '../../models/report.model';

const STEP_LABELS: Record<string, { label: string; icon: string }> = {
  pipeline_start:      { label: 'Pipeline started',           icon: '🚀' },
  tool_call:           { label: 'Agent calling tool',          icon: '🔧' },
  tool_result:         { label: 'Tool returned result',        icon: '✓'  },
  generating_summaries:{ label: 'Generating summaries',        icon: '📝' },
  verifier_start:      { label: 'Verification agent started',  icon: '🔍' },
  verifier_complete:   { label: 'Verification complete',       icon: '✅' },
  orchestrator_complete:{ label: 'Orchestrator complete',      icon: '🤖' },
  pipeline_complete:   { label: 'Analysis complete',           icon: '🏁' },
};

@Component({
  selector: 'app-agent-activity',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div class="flex items-center gap-2 mb-3">
        <div class="w-0.5 h-4 bg-blue-500 rounded"></div>
        <span class="text-xs font-semibold text-blue-400 uppercase tracking-widest">
          Agent Activity
        </span>
        @if (loading) {
          <span class="ml-auto flex items-center gap-1 text-xs text-gray-500">
            <span class="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce"></span>
            <span class="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:0.15s]"></span>
            <span class="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:0.3s]"></span>
            Live
          </span>
        }
      </div>

      <div class="space-y-1.5 max-h-48 overflow-y-auto">
        @for (step of steps; track $index) {
          <div class="flex items-start gap-2 text-xs">
            <span class="shrink-0 mt-0.5">{{ getIcon(step) }}</span>
            <span class="text-gray-400">{{ getLabel(step) }}</span>
            @if (step.tool) {
              <code class="ml-1 bg-gray-800 text-blue-300 px-1.5 py-0.5 rounded font-mono text-xs">
                {{ step.tool }}()
              </code>
            }
          </div>
        }

        @if (!steps.length && loading) {
          <p class="text-xs text-gray-600 italic">Waiting for agent to start...</p>
        }
      </div>
    </div>
  `,
})
export class AgentActivityComponent {
  @Input() steps: AgentStep[] = [];
  @Input() loading = false;

  getIcon(step: AgentStep): string {
    return STEP_LABELS[step.type]?.icon ?? '·';
  }

  getLabel(step: AgentStep): string {
    return STEP_LABELS[step.type]?.label ?? step.type;
  }
}
