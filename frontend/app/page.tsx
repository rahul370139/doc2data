"use client";

import { useCallback, useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  Clock,
  FileText,
  Play,
  Rewind,
  Sparkles,
  Workflow,
  X,
} from "lucide-react";
import clsx from "clsx";

import { Header } from "@/components/Header";
import { UploadZone } from "@/components/UploadZone";
import { ProgressStream } from "@/components/ProgressStream";
import { FieldsTable } from "@/components/FieldsTable";
import { BusinessPanel } from "@/components/BusinessPanel";
import { ValidationPanel } from "@/components/ValidationPanel";
import { DebugPanel } from "@/components/DebugPanel";
import { PdfViewer } from "@/components/PdfViewer";
import { QueryPanel } from "@/components/QueryPanel";
import { streamGraph, PipelineMethod } from "@/lib/api";
import { ExtractionResult, NodeEvent } from "@/lib/types";

type TabKey = "fields" | "business" | "validation" | "debug" | "query";

const METHOD_OPTIONS: Array<{
  value: PipelineMethod;
  label: string;
  hint: string;
}> = [
  {
    value: "cms1500",
    label: "CMS-1500",
    hint:
      "Skip form detection — force the CMS-1500 pipeline (auto-picks " +
      "widgets for fillable PDFs, template alignment + per-field OCR " +
      "for scans).  Use this when you know the form is a CMS-1500 " +
      "claim; shaves ~200-500ms off cold start.",
  },
  {
    value: "general",
    label: "General",
    hint:
      "Run the full form-detection agent, then auto-route based on " +
      "the detected form type.  Use this for unknown or mixed-form " +
      "batches (e.g. a queue with CMS-1500 + UB-04 + custom forms).",
  },
];

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [enableVlm, setEnableVlm] = useState(false);
  const [enableTables, setEnableTables] = useState(true);
  // GOT-OCR 2.0 is a third-opinion rescue engine (stepfun-ai).  Free
  // on DGX where its weights are pre-cached; disabled by default on
  // machines without a GPU or without the HF weights.
  const [enableGotOcr, setEnableGotOcr] = useState(true);
  const [method, setMethod] = useState<PipelineMethod>("cms1500");

  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<NodeEvent[]>([]);
  const [result, setResult] = useState<ExtractionResult | null>(null);
  const [error, setError] = useState<string>("");
  const [requestId, setRequestId] = useState<string>("");
  const [controller, setController] = useState<AbortController | null>(null);
  const [tab, setTab] = useState<TabKey>("fields");
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const lane = useMemo(
    () =>
      result?.lane ||
      events.find((e) => e.lane)?.lane ||
      undefined,
    [events, result],
  );
  const planReason = useMemo(
    () =>
      result?.plan_reason ||
      events.find((e) => e.plan_reason)?.plan_reason ||
      "",
    [events, result],
  );

  const reset = useCallback(() => {
    setEvents([]);
    setResult(null);
    setError("");
    setRequestId("");
    setTab("fields");
    setHoverId(null);
    setSelectedId(null);
  }, []);

  const handleRun = useCallback(async () => {
    if (!file || running) return;
    reset();
    setRunning(true);
    const ac = new AbortController();
    setController(ac);

    try {
      await streamGraph({
        file,
        enableVlm,
        enableTables,
        enableGotOcr,
        useOcrV2: true,
        method,
        signal: ac.signal,
        onStart: (p) => setRequestId(p.request_id),
        onNode: (evt) => setEvents((prev) => [...prev, evt]),
        onComplete: (r) => {
          setResult(r);
          setRunning(false);
        },
        onError: (msg) => {
          setError(msg);
          setRunning(false);
        },
      });
    } catch (e: any) {
      if (e?.name !== "AbortError") {
        setError(String(e?.message || e));
      }
      setRunning(false);
    }
  }, [file, enableVlm, enableTables, enableGotOcr, method, running, reset]);

  const handleCancel = useCallback(() => {
    controller?.abort();
    setRunning(false);
  }, [controller]);

  const fields = useMemo(() => result?.field_details || [], [result]);
  const stats = useMemo(() => {
    const totalFields = fields.length;
    const filled = fields.filter((f) => String(f.value || "").trim()).length;
    const highConf = fields.filter((f) => (f.confidence || 0) >= 0.75).length;
    const errors = result?.validation?.errors?.length || 0;
    const total = result?.debug?.total_latency_sec ?? 0;
    return { totalFields, filled, highConf, errors, total };
  }, [fields, result]);

  const showWorkspace = running || !!result || !!error;

  return (
    <>
      <Header />
      <main className="mx-auto max-w-[1600px] px-4 py-6 space-y-5">
        {/* HERO (only when nothing is running and no result) */}
        {!showWorkspace && (
          <section className="panel p-8 md:p-10">
            <div className="grid items-center gap-8 md:grid-cols-[1.3fr_1fr]">
              <div className="space-y-4">
                <div className="inline-flex items-center gap-2 rounded-full bg-accent-100 px-3 py-1 text-xs font-semibold text-accent-600">
                  <Sparkles size={12} /> LangGraph · batched Florence-2 ·
                  adaptive blank detection
                </div>
                <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-ink-800">
                  Extract healthcare forms
                  <br className="hidden md:block" />
                  in <span className="text-accent-500">seconds</span>, not minutes.
                </h1>
                <p className="text-ink-600 max-w-xl">
                  Agentic pipeline that plans a lane (widgets, digital text,
                  or scan), runs batched OCR with adaptive blank detection,
                  reflects on validation errors, and escalates the bad
                  fields through a field-type-aware ladder — VLM →
                  Florence-2 raw → GOT-OCR 2.0 → Florence-2 aggressive →
                  SLM normalization — until they validate.
                </p>
                <ul className="grid gap-2 text-sm text-ink-600 md:grid-cols-2">
                  <Bullet>86 CMS-1500 fields per page</Bullet>
                  <Bullet>Structural blank detection (adaptive)</Bullet>
                  <Bullet>One batched Florence-2 call</Bullet>
                  <Bullet>4-step agentic rescue ladder</Bullet>
                </ul>
              </div>
              <UploadZone
                file={file}
                onFile={setFile}
                disabled={running}
              />
            </div>

            <div className="mt-6 flex flex-wrap items-center gap-3">
              <MethodPicker value={method} onChange={setMethod} />
              <ToggleChip
                label="VLM rescue"
                active={enableVlm}
                onToggle={() => setEnableVlm((x) => !x)}
                help="Enable Ollama VLM for low-confidence rescue (adds latency)"
              />
              <ToggleChip
                label="Table extraction"
                active={enableTables}
                onToggle={() => setEnableTables((x) => !x)}
                help="Use VLM for Box 24 service-line tables"
              />
              <ToggleChip
                label="GOT-OCR rescue"
                active={enableGotOcr}
                onToggle={() => setEnableGotOcr((x) => !x)}
                help="Use GOT-OCR 2.0 as a third-opinion rescue engine (diverse architecture vs Florence-2 & VLM)"
              />
              <div className="ml-auto flex items-center gap-2">
                {file && (
                  <button className="btn-secondary" onClick={() => setFile(null)}>
                    <Rewind size={14} /> Clear
                  </button>
                )}
                <button
                  className="btn-primary"
                  disabled={!file}
                  onClick={handleRun}
                >
                  <Play size={14} /> Run pipeline
                </button>
              </div>
            </div>
          </section>
        )}

        {/* WORKSPACE (PDF | results) */}
        {showWorkspace && (
          <section className="space-y-4">
            {/* Toolbar */}
            <div className="panel px-4 py-3 flex items-center gap-3 flex-wrap">
              <div className="flex items-center gap-2 min-w-0">
                <FileText size={16} className="text-accent-500 shrink-0" />
                <div className="min-w-0">
                  <div className="font-semibold text-ink-800 truncate max-w-[420px]">
                    {file?.name || "Pipeline run"}
                  </div>
                  {requestId && (
                    <div className="font-mono text-[10px] text-ink-400">
                      req {requestId.slice(0, 12)}
                    </div>
                  )}
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2 ml-auto">
                <MethodPicker value={method} onChange={setMethod} />
                {result && <MiniStats stats={stats} result={result} />}
                <button
                  className="btn-secondary"
                  onClick={() => {
                    reset();
                    setFile(null);
                  }}
                >
                  <Rewind size={14} /> New run
                </button>
              </div>
            </div>

            {/* Progress stream — shown while running, collapsed once done */}
            <ProgressStream
              active={running}
              events={events}
              lane={lane}
              planReason={planReason}
              error={error}
              onCancel={handleCancel}
            />

            {/* Main split layout */}
            <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] items-stretch">
              {/* Left: PDF viewer */}
              <div className="min-h-[720px]">
                <PdfViewer
                  file={file}
                  fields={fields}
                  pageWidth={result?.page_width}
                  pageHeight={result?.page_height}
                  originalWidth={result?.original_width}
                  originalHeight={result?.original_height}
                  alignedImage={result?.aligned_image_b64}
                  originalImage={result?.original_image_b64}
                  hasOriginalQuads={result?.has_original_quads}
                  highlightId={hoverId}
                  selectedId={selectedId}
                  onHoverField={setHoverId}
                  onSelectField={(id) =>
                    setSelectedId((prev) => (prev === id ? null : id))
                  }
                />
              </div>

              {/* Right: results tabs */}
              <div className="min-h-[720px] flex flex-col">
                {result ? (
                  <>
                    <SelectedFieldCard
                      field={
                        fields.find(
                          (f) => f.id === (selectedId || hoverId),
                        ) || null
                      }
                      validation={result.validation}
                      onClear={() => setSelectedId(null)}
                    />
                    <TabBar tab={tab} onChange={setTab} />
                    {/* Tabs that own their own internal scroll container
                        (Fields, Ask) are mounted with a non-scrolling
                        flex wrapper so we don't stack two scroll
                        parents — that double-scroll was the source of
                        the vibrate-on-scroll bug in the Fields tab.
                        Simpler tabs (Business / Validation / Debug)
                        are plain lists so the outer scroll works fine. */}
                    <div
                      className={
                        tab === "query" || tab === "fields"
                          ? "mt-3 flex-1 min-h-0 flex flex-col"
                          : "mt-3 flex-1 min-h-0 overflow-auto"
                      }
                    >
                      {tab === "fields" && (
                        <FieldsTable
                          fields={fields}
                          highlightId={hoverId || selectedId}
                          onHover={setHoverId}
                          onSelect={(id) =>
                            setSelectedId((prev) =>
                              prev === id ? null : id,
                            )
                          }
                          selectedId={selectedId}
                        />
                      )}
                      {tab === "business" && (
                        <BusinessPanel
                          business={result.business_fields || {}}
                        />
                      )}
                      {tab === "validation" && (
                        <ValidationPanel report={result.validation} />
                      )}
                      {tab === "debug" && <DebugPanel result={result} />}
                      {tab === "query" && (
                        <QueryPanel
                          extractedFields={result.extracted_fields || {}}
                          fieldDetails={result.field_details || []}
                          formType={result.form_type || "cms-1500"}
                          onSelectField={(id: string) => setSelectedId(id)}
                        />
                      )}
                    </div>
                  </>
                ) : (
                  <div className="panel flex-1 flex flex-col items-center justify-center text-ink-400 gap-2">
                    <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent-500 border-t-transparent" />
                    <div className="text-sm">Waiting for extraction…</div>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        <footer className="pt-4 pb-10 text-center text-xs text-ink-400">
          Doc2Data · LangGraph orchestrator · Florence-2 · PaddleOCR · TrOCR · MiniCPM-V
        </footer>
      </main>
    </>
  );
}

function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex items-start gap-2">
      <Check className="mt-0.5 text-ok-500 shrink-0" size={14} />
      <span>{children}</span>
    </li>
  );
}

function ToggleChip({
  label,
  active,
  onToggle,
  help,
}: {
  label: string;
  active: boolean;
  onToggle: () => void;
  help?: string;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      title={help}
      className={clsx(
        "rounded-full border px-3 py-1.5 text-xs font-medium transition",
        active
          ? "border-accent-500 bg-accent-100 text-accent-600"
          : "border-ink-200 bg-white text-ink-600 hover:border-ink-300",
      )}
    >
      <span
        className={clsx(
          "inline-block mr-1.5 h-1.5 w-1.5 rounded-full",
          active ? "bg-accent-500" : "bg-ink-300",
        )}
      />
      {label}
    </button>
  );
}

function MethodPicker({
  value,
  onChange,
}: {
  value: PipelineMethod;
  onChange: (v: PipelineMethod) => void;
}) {
  const selected = METHOD_OPTIONS.find((o) => o.value === value) ?? METHOD_OPTIONS[0];
  return (
    <label
      className="flex items-center gap-2 rounded-full border border-ink-200 bg-white px-2 py-1 text-xs text-ink-600"
      title={selected.hint}
    >
      <Workflow size={12} className="text-accent-500" />
      <span className="text-[11px] text-ink-500">Method</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as PipelineMethod)}
        className="bg-transparent text-xs font-medium text-ink-800 outline-none"
      >
        {METHOD_OPTIONS.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function MiniStats({
  stats,
  result,
}: {
  stats: {
    totalFields: number;
    filled: number;
    highConf: number;
    errors: number;
    total: number;
  };
  result: ExtractionResult;
}) {
  const ok = stats.errors === 0;
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <Pill color="accent" icon={<Check size={11} />}>
        {stats.filled}/{stats.totalFields} filled
      </Pill>
      <Pill
        color={ok ? "ok" : "bad"}
        icon={ok ? <Check size={11} /> : <AlertTriangle size={11} />}
      >
        {ok ? "Validated" : `${stats.errors} errors`}
      </Pill>
      <Pill color="muted" icon={<Workflow size={11} />}>
        Lane {result.lane || "—"}
      </Pill>
      <Pill color="muted" icon={<Clock size={11} />}>
        {stats.total.toFixed(1)}s
      </Pill>
    </div>
  );
}

function Pill({
  color,
  icon,
  children,
}: {
  color: "accent" | "ok" | "bad" | "muted";
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  const cls =
    color === "accent"
      ? "bg-accent-100 text-accent-600"
      : color === "ok"
      ? "bg-ok-100 text-ok-500"
      : color === "bad"
      ? "bg-bad-100 text-bad-500"
      : "bg-ink-100 text-ink-600";
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-medium",
        cls,
      )}
    >
      {icon}
      {children}
    </span>
  );
}

function SelectedFieldCard({
  field,
  validation,
  onClear,
}: {
  field: ReturnType<typeof Array.prototype.find> | null;
  validation: ExtractionResult["validation"];
  onClear: () => void;
}) {
  // Pinned height: the FieldsTable (and every other tab) sits BELOW this
  // card in a flex column.  If the card resizes when you hover rows (no
  // field → 1-liner, field → tall with value + pills), the table below
  // shrinks/grows and the scroll offset jumps — THAT is the "vibrate on
  // scroll" bug the user keeps seeing on the Fields tab.  Fixing it at
  // ~132px (h-32 + borders) keeps the layout stable across hover
  // transitions.  Inner overflow is hidden with a truncated value so
  // long content can't push the card taller either.
  const f: any = field;
  const err = (validation?.errors || []).find(
    (e: any) => f && e.field_id === f.id,
  );
  const engine = f?.metadata?.ocr_engine || "—";
  const method = f?.metadata?.rescue_method || "";
  const isEmpty = !field;
  return (
    <div className="panel h-[132px] shrink-0 overflow-hidden p-4">
      {isEmpty ? (
        <div className="flex h-full items-center justify-center text-center text-xs text-ink-400 italic px-6">
          Hover a bounding box on the PDF (left) or a row in the Fields
          table to inspect that field here.
        </div>
      ) : (
        <div className="flex h-full flex-col gap-1.5">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <div className="text-[10px] uppercase tracking-wider font-semibold text-ink-500">
                {f.metadata?.field_type || "field"}
              </div>
              <div className="font-semibold text-ink-800 truncate text-[13px] leading-tight">
                {f.metadata?.field_name || f.id}
              </div>
              <div className="font-mono text-[10px] text-ink-400 truncate">
                {f.id}
              </div>
            </div>
            <button
              className="rounded-md p-1 text-ink-500 hover:bg-ink-100 shrink-0"
              onClick={onClear}
              title="Clear selection"
            >
              <X size={14} />
            </button>
          </div>
          <div className="rounded-md bg-ink-50 px-2 py-1 font-mono text-[12px] text-ink-800 truncate">
            {f.value || <span className="italic text-ink-400">— empty —</span>}
          </div>
          <div className="flex flex-wrap items-center gap-1.5 text-[10px] overflow-hidden">
            <Pill
              color={
                f.confidence >= 0.75
                  ? "ok"
                  : f.confidence >= 0.45
                  ? "muted"
                  : "bad"
              }
            >
              conf {Number(f.confidence || 0).toFixed(2)}
            </Pill>
            <Pill color="muted">{engine}</Pill>
            {method && <Pill color="accent">rescue: {method}</Pill>}
            {err && (
              <Pill color="bad" icon={<AlertTriangle size={11} />}>
                {String(err.message || "").slice(0, 50)}
              </Pill>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function TabBar({
  tab,
  onChange,
}: {
  tab: TabKey;
  onChange: (t: TabKey) => void;
}) {
  const tabs: { id: TabKey; label: string }[] = [
    { id: "fields", label: "Fields" },
    { id: "query", label: "Ask" },
    { id: "business", label: "Business schema" },
    { id: "validation", label: "Validation" },
    { id: "debug", label: "Debug" },
  ];
  return (
    <div className="flex gap-1 border-b border-ink-200 mt-2">
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={clsx(
            "px-4 py-2 text-sm font-medium transition relative",
            tab === t.id
              ? "text-accent-500"
              : "text-ink-500 hover:text-ink-800",
          )}
        >
          {t.label}
          {tab === t.id && (
            <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-accent-500" />
          )}
        </button>
      ))}
    </div>
  );
}
