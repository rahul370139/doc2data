"use client";

import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import {
  CornerDownLeft,
  Loader2,
  MessageSquare,
  Sparkles,
  User,
} from "lucide-react";

import { FieldDetail } from "@/lib/types";

interface QueryPanelProps {
  extractedFields: Record<string, string>;
  fieldDetails: FieldDetail[];
  formType: string;
  /** When the user clicks a source chip, highlight that field in the PDF. */
  onSelectField?: (id: string) => void;
}

type Turn =
  | { role: "user"; text: string }
  | { role: "assistant"; text: string; sources: string[]; error?: boolean };

// A handful of prompts that showcase what the assistant can do even on a
// brand-new session.  Kept short on purpose so they fit in a single row on
// desktop without wrapping.
const STARTER_PROMPTS = [
  "Summarize this claim in 2 sentences.",
  "What is the patient's full address?",
  "List all diagnosis codes.",
  "Who is the billing provider and their NPI?",
  "Are any required fields missing?",
];

export function QueryPanel({
  extractedFields,
  fieldDetails,
  formType,
  onSelectField,
}: QueryPanelProps) {
  const [history, setHistory] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Build a quick lookup so we can show the extracted value next to each
  // source-field chip.  Keeps the UI informative without the user hunting
  // through the fields tab.
  const fieldValueById: Record<string, string> = {};
  for (const fd of fieldDetails || []) {
    if (fd?.id) fieldValueById[fd.id] = String(fd.value || "");
  }

  // Non-empty extracted values gives a sense of how much the assistant has
  // to work with.  Useful cue when the extraction clearly missed a lot.
  const nonEmpty = Object.values(extractedFields || {}).filter(
    (v) => v && String(v).trim(),
  ).length;

  const disabled =
    loading || !input.trim() || Object.keys(extractedFields || {}).length === 0;

  // Scroll to the newest turn whenever the history grows.  Depend on
  // history.length only — depending on `loading` causes the effect to
  // fire twice per ask (loading=true → loading=false), which on a
  // nested-overflow page can race the browser's scroll layout pass and
  // produce a vibrate-on-scroll feel.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [history.length]);

  const ask = async (q: string) => {
    const question = q.trim();
    if (!question) return;
    setHistory((h) => [...h, { role: "user", text: question }]);
    setInput("");
    setLoading(true);
    try {
      const resp = await fetch("/api/backend/chat/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: question,
          extracted_fields: extractedFields,
          field_details: fieldDetails,
          form_type: formType || "cms-1500",
        }),
      });
      if (!resp.ok) {
        const text = await resp.text();
        setHistory((h) => [
          ...h,
          {
            role: "assistant",
            text: `Backend returned ${resp.status}. ${text.slice(0, 240)}`,
            sources: [],
            error: true,
          },
        ]);
      } else {
        const data = await resp.json();
        setHistory((h) => [
          ...h,
          {
            role: "assistant",
            text: data?.answer || "(empty response)",
            sources: Array.isArray(data?.source_fields) ? data.source_fields : [],
          },
        ]);
      }
    } catch (e: any) {
      setHistory((h) => [
        ...h,
        {
          role: "assistant",
          text: `Could not reach SLM. ${String(e?.message || e).slice(0, 240)}`,
          sources: [],
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Cmd/Ctrl+Enter submits, plain Enter adds a newline.  We copy the
    // convention used by most modern chat UIs so users don't have to think.
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      if (!disabled) ask(input);
    }
  };

  return (
    // h-full so we fill the parent (which is already min-h-0 +
    // overflow-auto on the workspace).  Hard-coded min/max heights
    // here used to fight the parent's scroll container and produced
    // a vibrate-on-scroll feel on the Ask tab — single scroll
    // container is the calmest layout.
    <div className="panel flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-ink-100 px-4 py-3">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-50 text-accent-600">
          <MessageSquare size={14} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-ink-800">
            Ask the extraction
          </div>
          <div className="text-[11px] text-ink-500">
            Natural-language query over{" "}
            <span className="font-mono">{nonEmpty}</span> populated fields ·{" "}
            grounded in the JSON, not hallucinated
          </div>
        </div>
        {history.length > 0 && (
          <button
            className="btn-ghost text-[11px]"
            onClick={() => setHistory([])}
          >
            Clear
          </button>
        )}
      </div>

      {/* Transcript */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-auto px-4 py-4 space-y-4"
      >
        {history.length === 0 && (
          <EmptyState
            onPick={(p) => {
              setInput(p);
              ask(p);
            }}
          />
        )}
        {history.map((turn, i) => (
          <TurnBubble
            key={i}
            turn={turn}
            valueById={fieldValueById}
            onSelectField={onSelectField}
          />
        ))}
        {loading && (
          <div className="flex items-center gap-2 text-sm text-ink-500">
            <Loader2 size={14} className="animate-spin" /> Thinking…
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="border-t border-ink-100 px-4 py-3">
        <div className="relative">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            rows={2}
            placeholder={
              nonEmpty === 0
                ? "Upload and extract a document first…"
                : "Ask a question about this form…"
            }
            className="w-full resize-none rounded-xl border border-ink-200 bg-white px-3 py-2 pr-24 text-sm outline-none focus:border-accent-400"
            disabled={nonEmpty === 0 || loading}
          />
          <button
            onClick={() => ask(input)}
            disabled={disabled}
            className={clsx(
              "absolute right-2 bottom-2 btn-primary px-3 py-1.5 text-[12px]",
            )}
          >
            {loading ? (
              <Loader2 size={12} className="animate-spin" />
            ) : (
              <CornerDownLeft size={12} />
            )}
            Ask
          </button>
        </div>
        <div className="mt-1.5 flex items-center gap-2 text-[11px] text-ink-400">
          <span className="kbd">⌘</span>
          <span>+</span>
          <span className="kbd">↵</span>
          <span>to send · answers cite source field IDs</span>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (p: string) => void }) {
  return (
    <div className="space-y-3">
      <div className="rounded-2xl border border-dashed border-ink-200 bg-ink-50/40 p-4">
        <div className="flex items-center gap-2 text-sm font-medium text-ink-700">
          <Sparkles size={14} className="text-accent-500" />
          Ask anything about this form
        </div>
        <p className="mt-1.5 text-[12px] leading-relaxed text-ink-500">
          The assistant only reads the JSON extracted from your document —{" "}
          it will cite the exact field IDs it used and say “Not found in
          document” when something wasn’t extracted.
        </p>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {STARTER_PROMPTS.map((p) => (
          <button
            key={p}
            onClick={() => onPick(p)}
            className="rounded-full border border-ink-200 bg-white px-3 py-1 text-[12px] text-ink-600 hover:border-accent-300 hover:text-accent-600"
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}

function TurnBubble({
  turn,
  valueById,
  onSelectField,
}: {
  turn: Turn;
  valueById: Record<string, string>;
  onSelectField?: (id: string) => void;
}) {
  if (turn.role === "user") {
    return (
      <div className="flex items-start gap-2 justify-end">
        <div className="rounded-2xl rounded-tr-sm bg-accent-500 px-3 py-2 text-sm text-white max-w-[85%] whitespace-pre-wrap">
          {turn.text}
        </div>
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-ink-100 text-ink-600">
          <User size={14} />
        </div>
      </div>
    );
  }
  return (
    <div className="flex items-start gap-2">
      <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-50 text-accent-600">
        <Sparkles size={14} />
      </div>
      <div
        className={clsx(
          "rounded-2xl rounded-tl-sm px-3 py-2 text-sm max-w-[85%]",
          turn.error
            ? "bg-bad-100/60 text-bad-500 border border-bad-100"
            : "bg-ink-50 text-ink-800",
        )}
      >
        <div className="whitespace-pre-wrap">{turn.text}</div>
        {turn.sources && turn.sources.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {turn.sources.map((id) => (
              <button
                key={id}
                onClick={() => onSelectField?.(id)}
                className="group inline-flex items-center gap-1 rounded-full border border-ink-200 bg-white px-2 py-0.5 text-[11px] font-mono text-ink-600 hover:border-accent-300 hover:text-accent-600"
                title={
                  valueById[id]
                    ? `${id} → ${valueById[id]}`
                    : `Highlight ${id} in the document`
                }
              >
                <span className="text-accent-500 group-hover:text-accent-600">
                  §
                </span>
                {id}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
