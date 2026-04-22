"use client";

import { useMemo, useState } from "react";
import clsx from "clsx";
import { Search, SlidersHorizontal } from "lucide-react";
import { FieldDetail } from "@/lib/types";

export interface FieldsTableProps {
  fields: FieldDetail[];
  onHover?: (id: string | null) => void;
  onSelect?: (id: string | null) => void;
  highlightId?: string | null;
  selectedId?: string | null;
}

export function FieldsTable({
  fields, onHover, onSelect, highlightId, selectedId,
}: FieldsTableProps) {
  const [query, setQuery] = useState("");
  const [hideEmpty, setHideEmpty] = useState(true);
  const [minConf, setMinConf] = useState(0);

  const list = useMemo(() => {
    const q = query.trim().toLowerCase();
    return fields.filter((f) => {
      if (hideEmpty && !String(f.value || "").trim()) return false;
      if (minConf > 0 && (f.confidence ?? 0) < minConf) return false;
      if (!q) return true;
      return (
        f.id.toLowerCase().includes(q) ||
        String(f.value || "").toLowerCase().includes(q) ||
        String(f.metadata?.field_name || "").toLowerCase().includes(q)
      );
    });
  }, [fields, query, hideEmpty, minConf]);

  return (
    // h-full + min-h-0 so we fill the parent flex column without
    // introducing our own competing scroll container.  The previous
    // layout had an inner ``max-h-[620px] overflow-auto`` around the
    // table — when the parent also scrolls, both containers compete
    // and scroll-snapping produces the "vibrates on scroll" flicker
    // the user reported on the Fields tab.  Letting the outer panel
    // own the scroll kills it.
    <div className="panel flex flex-col h-full min-h-0">
      <div className="flex flex-wrap items-center gap-3 px-4 pt-4 pb-3 border-b border-ink-100 shrink-0">
        <div className="relative min-w-[220px] flex-1">
          <Search
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-400"
            size={14}
          />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search fields…"
            className="w-full rounded-xl border border-ink-200 bg-white px-9 py-2 text-sm outline-none focus:border-accent-400"
          />
        </div>
        <label className="flex items-center gap-2 text-xs text-ink-600 select-none">
          <input
            type="checkbox"
            checked={hideEmpty}
            onChange={(e) => setHideEmpty(e.target.checked)}
            className="rounded"
          />
          Hide empty
        </label>
        <label className="flex items-center gap-2 text-xs text-ink-600">
          <SlidersHorizontal size={12} />
          Min conf
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={minConf}
            onChange={(e) => setMinConf(parseFloat(e.target.value))}
            className="w-24 accent-accent-500"
          />
          <span className="font-mono w-8 text-right text-ink-500">
            {minConf.toFixed(2)}
          </span>
        </label>
        <div className="ml-auto text-xs text-ink-500">
          {list.length}/{fields.length} shown
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-auto px-4 pb-4">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-white z-10">
            <tr className="text-left text-[11px] font-semibold uppercase tracking-wider text-ink-500">
              <th className="px-2 py-2 w-[30%]">Field</th>
              <th className="px-2 py-2">Value</th>
              <th className="px-2 py-2 w-[90px]">Conf</th>
              <th className="px-2 py-2 w-[120px]">Engine</th>
            </tr>
          </thead>
          <tbody>
            {list.map((f) => {
              const ink = f.metadata?.blank_status || "";
              const engine = f.metadata?.ocr_engine || "";
              const esc = f.metadata?.escalation || "";
              const conf = f.confidence ?? 0;
              const highlight = highlightId === f.id;
              const isSelected = selectedId === f.id;
              return (
                <tr
                  key={f.id}
                  onMouseEnter={() => onHover?.(f.id)}
                  onMouseLeave={() => onHover?.(null)}
                  onClick={() => onSelect?.(f.id)}
                  className={clsx(
                    "border-b border-ink-100 transition cursor-pointer",
                    isSelected
                      ? "bg-accent-100/70"
                      : highlight
                      ? "bg-accent-50"
                      : "hover:bg-ink-50",
                  )}
                >
                  <td className="px-2 py-2 align-top">
                    <div className="font-medium text-ink-800 text-[13px]">
                      {f.metadata?.field_name || f.id}
                    </div>
                    <div className="font-mono text-[11px] text-ink-400">
                      {f.id}
                    </div>
                  </td>
                  <td className="px-2 py-2 align-top">
                    <div
                      className={clsx(
                        "whitespace-pre-wrap break-words text-ink-800 font-mono text-[12px]",
                        !f.value && "italic text-ink-300",
                      )}
                    >
                      {f.value || "—"}
                    </div>
                  </td>
                  <td className="px-2 py-2 align-top">
                    <ConfidenceBar value={conf} />
                  </td>
                  <td className="px-2 py-2 align-top">
                    <div className="flex flex-col gap-1">
                      <span className="font-mono text-[11px] text-ink-500">
                        {prettyEngine(engine)}
                      </span>
                      {ink && (
                        <span
                          className={clsx(
                            "text-[10px] font-semibold uppercase",
                            ink === "blank" && "text-ink-400",
                            ink === "uncertain" && "text-warn-500",
                            ink === "filled" && "text-ok-500",
                          )}
                        >
                          {ink}
                        </span>
                      )}
                      {esc && esc !== "none" && (
                        <span className="text-[10px] text-accent-500">
                          {esc}
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
            {list.length === 0 && (
              <tr>
                <td
                  colSpan={4}
                  className="py-8 text-center text-sm text-ink-400"
                >
                  No fields match these filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const v = Math.max(0, Math.min(1, value));
  const color =
    v >= 0.75 ? "bg-ok-500" : v >= 0.45 ? "bg-warn-500" : "bg-bad-500";
  return (
    <div className="flex items-center gap-2 min-w-[80px]">
      <div className="h-1.5 flex-1 rounded-full bg-ink-100 overflow-hidden">
        <div
          className={clsx("h-full", color)}
          style={{ width: `${(v * 100).toFixed(0)}%` }}
        />
      </div>
      <span className="w-8 text-right font-mono text-[11px] text-ink-600">
        {v.toFixed(2)}
      </span>
    </div>
  );
}

function prettyEngine(raw: string) {
  if (!raw) return "—";
  return raw
    .replace(/_/g, " ")
    .replace(/(^|\s)\w/g, (m) => m.toUpperCase());
}
