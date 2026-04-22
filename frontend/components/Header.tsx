"use client";

import { Activity, Github, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";

export function Header() {
  const [health, setHealth] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    getHealth().then((h) => {
      if (alive) {
        setHealth(h);
        setLoading(false);
      }
    });
    const iv = setInterval(() => {
      getHealth().then((h) => alive && setHealth(h));
    }, 30_000);
    return () => {
      alive = false;
      clearInterval(iv);
    };
  }, []);

  const ok = !!health && health.status === "healthy";

  return (
    <header className="sticky top-0 z-20 border-b border-ink-100 bg-white/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent-500 text-white shadow-sm">
            <Zap size={18} strokeWidth={2.5} />
          </div>
          <div>
            <div className="font-semibold tracking-tight leading-none">
              Doc2Data
            </div>
            <div className="text-xs text-ink-500 mt-0.5">
              Agentic form extraction · LangGraph
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <HealthPill ok={ok} loading={loading} health={health} />
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="btn-ghost"
            aria-label="Repository"
          >
            <Github size={16} />
          </a>
        </div>
      </div>
    </header>
  );
}

function HealthPill({
  ok,
  loading,
  health,
}: {
  ok: boolean;
  loading: boolean;
  health: any;
}) {
  if (loading) {
    return (
      <span className="chip-muted">
        <span className="h-1.5 w-1.5 rounded-full bg-ink-300 animate-pulseDot" />
        checking…
      </span>
    );
  }
  if (!ok) {
    return (
      <span className="chip-bad">
        <span className="h-1.5 w-1.5 rounded-full bg-bad-500" />
        backend offline
      </span>
    );
  }
  const caps = health?.capabilities || {};
  return (
    <span className="chip-ok" title={JSON.stringify(caps)}>
      <Activity size={12} />
      backend online
    </span>
  );
}
