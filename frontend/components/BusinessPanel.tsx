"use client";

import { useMemo, useState } from "react";
import { Search } from "lucide-react";

export function BusinessPanel({ business }: { business: Record<string, any> }) {
  const [q, setQ] = useState("");
  const entries = useMemo(() => {
    const all = flatten(business);
    const qq = q.trim().toLowerCase();
    return all.filter(([k, v]) => {
      if (!qq) return true;
      return (
        k.toLowerCase().includes(qq) ||
        String(v || "").toLowerCase().includes(qq)
      );
    });
  }, [business, q]);

  if (!business || Object.keys(business).length === 0) {
    return (
      <div className="panel p-5 text-sm text-ink-500">
        No business-schema mapping produced for this form.
      </div>
    );
  }

  return (
    <div className="panel p-4">
      <div className="relative mb-3">
        <Search
          size={14}
          className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-400"
        />
        <input
          placeholder="Search business keys…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          className="w-full rounded-xl border border-ink-200 bg-white px-9 py-2 text-sm outline-none focus:border-accent-400"
        />
      </div>
      <div className="max-h-[620px] overflow-auto divide-y divide-ink-100">
        {entries.map(([k, v]) => (
          <div key={k} className="grid grid-cols-[2fr_3fr] gap-2 py-2">
            <div className="font-mono text-[12px] text-ink-500 truncate">{k}</div>
            <div className="font-mono text-[12px] text-ink-800 break-words">
              {String(v) || <span className="text-ink-300 italic">—</span>}
            </div>
          </div>
        ))}
        {entries.length === 0 && (
          <div className="py-8 text-center text-sm text-ink-400">
            No matches.
          </div>
        )}
      </div>
    </div>
  );
}

function flatten(
  obj: Record<string, any>,
  prefix = "",
): [string, any][] {
  const out: [string, any][] = [];
  for (const [k, v] of Object.entries(obj || {})) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) {
      out.push(...flatten(v, key));
    } else if (Array.isArray(v)) {
      out.push([key, JSON.stringify(v)]);
    } else {
      out.push([key, v]);
    }
  }
  return out;
}
