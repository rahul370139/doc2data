"use client";

import clsx from "clsx";
import { CheckCircle2, ClipboardCopy, Download, XCircle } from "lucide-react";
import { ExtractionResult, RescueAttempt } from "@/lib/types";

export function DebugPanel({ result }: { result: ExtractionResult }) {
  const debug = result.debug || {};
  const timings = debug.timings || {};
  const trace = debug.trace || [];
  const rescueLog = (debug.rescue_log || []) as RescueAttempt[];
  const total = Object.values(timings).reduce((a, b) => a + (b || 0), 0);

  const copy = () =>
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));

  const download = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `doc2data-${result.form_type || "result"}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="panel p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="font-semibold text-ink-800">Debug trace</div>
        <div className="flex items-center gap-2">
          <button className="btn-ghost" onClick={copy}>
            <ClipboardCopy size={14} /> copy JSON
          </button>
          <button className="btn-secondary" onClick={download}>
            <Download size={14} /> download
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Form" value={result.form_type || "—"} />
        <Stat label="Lane" value={result.lane || "—"} />
        <Stat label="Method" value={shorten(result.extraction_method || "")} />
        <Stat
          label="Total"
          value={
            debug.total_latency_sec
              ? `${debug.total_latency_sec.toFixed(2)}s`
              : total
              ? `${total.toFixed(2)}s`
              : "—"
          }
        />
        <Stat
          label="Alignment"
          value={
            debug.alignment_used
              ? `${(debug.alignment_quality || 0).toFixed(2)}`
              : "—"
          }
        />
        <Stat
          label="Rescue iters"
          value={String(debug.rescue_iterations ?? 0)}
        />
        <Stat
          label="VLM rescues"
          value={String(debug.vlm_rescue_count ?? 0)}
        />
        <Stat
          label="Fields"
          value={String((result.field_details || []).length)}
        />
      </div>

      <div>
        <div className="field-label mb-2">Node timings</div>
        <div className="space-y-1.5">
          {Object.entries(timings).map(([node, t]) => (
            <div
              key={node}
              className="flex items-center gap-3 rounded-lg px-2 py-1 text-sm hover:bg-ink-50"
            >
              <span className="font-mono text-[11px] w-24 text-ink-500">
                {node}
              </span>
              <div className="flex-1 h-1.5 rounded-full bg-ink-100 overflow-hidden">
                <div
                  className="h-full bg-accent-500"
                  style={{
                    width: `${(((t as number) / (total || 1)) * 100).toFixed(1)}%`,
                  }}
                />
              </div>
              <span className="w-16 text-right font-mono text-[11px] text-ink-600">
                {(t as number).toFixed(2)}s
              </span>
            </div>
          ))}
        </div>
      </div>

      {trace.length > 0 && (
        <div>
          <div className="field-label mb-2">Trace</div>
          <div className="flex flex-wrap gap-1.5">
            {trace.map((n, i) => (
              <span
                key={`${n}-${i}`}
                className="font-mono text-[11px] rounded-md bg-ink-50 px-2 py-1 text-ink-600"
              >
                {i + 1}. {n}
              </span>
            ))}
          </div>
        </div>
      )}

      {rescueLog.length > 0 && (
        <div>
          <div className="field-label mb-2">
            Agentic rescue log ({rescueLog.length} attempts)
          </div>
          <div className="rounded-xl border border-ink-100 overflow-hidden">
            <table className="w-full text-[12px]">
              <thead className="bg-ink-50 text-[10px] uppercase tracking-wider text-ink-500">
                <tr>
                  <th className="px-2 py-1.5 text-left">iter</th>
                  <th className="px-2 py-1.5 text-left">field</th>
                  <th className="px-2 py-1.5 text-left">method</th>
                  <th className="px-2 py-1.5 text-left">old</th>
                  <th className="px-2 py-1.5 text-left">new</th>
                  <th className="px-2 py-1.5 text-left">outcome</th>
                </tr>
              </thead>
              <tbody>
                {rescueLog.map((a, i) => (
                  <tr
                    key={`${a.field_id}-${a.iteration}-${i}`}
                    className="border-t border-ink-100"
                  >
                    <td className="px-2 py-1.5 font-mono text-[11px] text-ink-500">
                      {a.iteration}
                    </td>
                    <td className="px-2 py-1.5 font-mono text-[11px] text-ink-600 max-w-[160px] truncate">
                      {a.field_id}
                    </td>
                    <td className="px-2 py-1.5">
                      <span className="rounded-full bg-accent-100 px-2 py-0.5 text-[10px] font-semibold text-accent-600">
                        {a.method}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 max-w-[180px] truncate font-mono text-[11px] text-ink-500">
                      {a.old_text || <span className="italic">∅</span>}
                    </td>
                    <td className="px-2 py-1.5 max-w-[180px] truncate font-mono text-[11px] text-ink-800">
                      {a.new_text || <span className="italic">∅</span>}
                    </td>
                    <td className="px-2 py-1.5">
                      <span
                        className={clsx(
                          "inline-flex items-center gap-1 text-[11px] font-medium",
                          a.accepted ? "text-ok-500" : "text-ink-400",
                        )}
                      >
                        {a.accepted ? (
                          <CheckCircle2 size={12} />
                        ) : (
                          <XCircle size={12} />
                        )}
                        {a.accepted ? "accepted" : a.reason}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl border border-ink-100 bg-white p-3">
      <div className="field-label">{label}</div>
      <div className="text-ink-800 font-semibold mt-1 truncate">{value}</div>
    </div>
  );
}

function shorten(s: string) {
  return s.replace(/^lane_[abc]_/, "").replace(/_/g, " ");
}
