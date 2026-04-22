"use client";

import clsx from "clsx";
import {
  AlignCenter,
  Check,
  CircleDashed,
  Cog,
  Crosshair,
  FileSearch,
  GitBranch,
  Loader2,
  ScanLine,
  ShieldCheck,
  Sparkles,
  Target,
  X,
} from "lucide-react";
import { NodeEvent } from "@/lib/types";

const NODE_LABELS: Record<string, { label: string; Icon: any }> = {
  load: { label: "Load & render", Icon: FileSearch },
  identify: { label: "Identify form", Icon: Target },
  plan: { label: "Plan lane", Icon: GitBranch },
  extract_widgets: { label: "Lane A · widgets", Icon: Cog },
  extract_digital: { label: "Lane B · digital text", Icon: ScanLine },
  digital_downgrade: { label: "Lane B → C", Icon: GitBranch },
  align: { label: "Align template", Icon: AlignCenter },
  extract_scan: { label: "Lane C · batched OCR", Icon: ScanLine },
  validate: { label: "Validate", Icon: ShieldCheck },
  reflect: { label: "Reflect", Icon: Sparkles },
  rescue: { label: "Rescue (VLM)", Icon: Crosshair },
  finalize: { label: "Finalize", Icon: Check },
};

const ORDER = [
  "load",
  "identify",
  "plan",
  "extract_widgets",
  "extract_digital",
  "digital_downgrade",
  "align",
  "extract_scan",
  "validate",
  "reflect",
  "rescue",
  "finalize",
];

export interface ProgressStreamProps {
  active: boolean;
  events: NodeEvent[];
  lane?: string;
  planReason?: string;
  error?: string;
  onCancel?: () => void;
}

export function ProgressStream({
  active,
  events,
  lane,
  planReason,
  error,
  onCancel,
}: ProgressStreamProps) {
  const ran = new Set(events.map((e) => e.node));
  const timings = events.at(-1)?.timings || {};
  const total = Object.values(timings).reduce((a, b) => a + (b || 0), 0);

  return (
    <div className="panel p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {active ? (
            <Loader2 className="text-accent-500 animate-spin" size={18} />
          ) : (
            <Check className="text-ok-500" size={18} />
          )}
          <div className="font-semibold">
            {active ? "Running agentic pipeline" : "Pipeline complete"}
          </div>
          {lane && (
            <span className="chip-accent ml-2" title={planReason}>
              Lane {lane}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-ink-500">
          {total > 0 && <span>{total.toFixed(2)}s</span>}
          {active && onCancel && (
            <button className="btn-ghost" onClick={onCancel}>
              <X size={14} /> cancel
            </button>
          )}
        </div>
      </div>

      {planReason && (
        <div className="mt-3 text-xs text-ink-500">
          <span className="field-label mr-2">plan</span>
          {planReason}
        </div>
      )}

      <ol className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-3">
        {ORDER.map((node) => {
          const done = ran.has(node);
          const skipped =
            !done && !active && !shouldShow(node, ran, lane);
          if (skipped) return null;
          const { label, Icon } =
            NODE_LABELS[node] || { label: node, Icon: CircleDashed };
          return (
            <li
              key={node}
              className={clsx(
                "flex items-center gap-2 rounded-xl px-3 py-2 text-sm transition",
                done
                  ? "bg-ok-100 text-ok-500"
                  : active
                  ? "bg-ink-50 text-ink-500"
                  : "bg-ink-50 text-ink-300",
              )}
            >
              {done ? (
                <Check size={14} />
              ) : active ? (
                <CircleDashed className="animate-pulseDot" size={14} />
              ) : (
                <CircleDashed size={14} />
              )}
              <span className="flex-1 truncate">{label}</span>
              {timings[node] !== undefined && (
                <span className="font-mono text-[11px] text-ink-500">
                  {timings[node].toFixed(2)}s
                </span>
              )}
            </li>
          );
        })}
      </ol>

      {error && (
        <div className="mt-4 rounded-xl border border-bad-500/20 bg-bad-100 px-3 py-2 text-sm text-bad-500">
          {error}
        </div>
      )}
    </div>
  );
}

function shouldShow(node: string, ran: Set<string>, lane?: string) {
  if (ran.has(node)) return true;
  // Hide irrelevant lane nodes after completion
  if (lane === "A" && (node === "align" || node === "extract_scan" ||
      node === "extract_digital" || node === "digital_downgrade"))
    return false;
  if (lane === "B" && (node === "extract_widgets" || node === "extract_scan"))
    return false;
  if (lane === "C" && (node === "extract_widgets" || node === "extract_digital" ||
      node === "digital_downgrade"))
    return false;
  return true;
}
