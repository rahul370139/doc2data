"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, ImageOff, Minus, Plus, ZoomIn } from "lucide-react";

import { FieldDetail } from "@/lib/types";

interface PdfViewerProps {
  file: File | null;
  fields: FieldDetail[];
  /** Aligned (template-space) page size reported by the pipeline.
   *  Only used as a fallback when ``bbox_original_quad`` is missing. */
  pageWidth?: number;
  pageHeight?: number;
  /** Original (pre-warp) page size — matches the PDF rasterised by the
   *  backend, which is the same content pdf.js renders, just at a
   *  different scale.  Used to scale ``bbox_original_quad`` onto the
   *  client-rendered viewport. */
  originalWidth?: number;
  originalHeight?: number;
  /** Optional backend-rendered images.  We DO NOT display these — the
   *  pdf.js render of the user's own file is always the background.
   *  These props are kept for backwards compat with the response shape. */
  alignedImage?: string;
  originalImage?: string;
  hasOriginalQuads?: boolean;
  highlightId?: string | null;
  onHoverField?: (id: string | null) => void;
  onSelectField?: (id: string | null) => void;
  selectedId?: string | null;
}

/**
 * Single-view PDF viewer: renders the user's uploaded PDF with pdf.js
 * (the exact page they uploaded — no warp, no red-channel drop,
 * nothing) and draws axis-aligned bounding-rects on top.
 *
 * Coordinate story:
 *   - The backend rasterises the same PDF at ~300dpi to run the
 *     pipeline.  Its page size is ``(originalWidth, originalHeight)``.
 *   - For each field it emits ``bbox_original_quad`` — the aligned-
 *     space bbox projected through the inverse homography so the 4
 *     corners land in ORIGINAL image coords.
 *   - pdf.js renders the same page at scale 1.6 (different pixel size)
 *     so we rescale the quad by ``pdfjs_size / original_size``.
 *   - We draw the axis-aligned bounding rect of the quad.  Position
 *     follows the homography; shape is clean (no perspective tilt).
 */
export function PdfViewer({
  file,
  fields,
  pageWidth,
  pageHeight,
  originalWidth,
  originalHeight,
  hasOriginalQuads,
  highlightId,
  onHoverField,
  onSelectField,
  selectedId,
}: PdfViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [pageSize, setPageSize] = useState<{ w: number; h: number } | null>(
    null,
  );
  const [zoom, setZoom] = useState(1);
  const [err, setErr] = useState<string>("");
  const [rendering, setRendering] = useState(false);
  const [isImage, setIsImage] = useState(false);
  const [imageUrl, setImageUrl] = useState<string>("");

  const renderVersion = useRef(0);

  useEffect(() => {
    setErr("");
    setPageSize(null);
    setIsImage(false);
    setImageUrl("");
    if (!file) return;

    // Non-PDF uploads (image files) render via <img>.
    if (!file.type.includes("pdf")) {
      setIsImage(true);
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      const img = new Image();
      img.onload = () =>
        setPageSize({ w: img.naturalWidth, h: img.naturalHeight });
      img.onerror = () => setErr("Could not decode image");
      img.src = url;
      return () => URL.revokeObjectURL(url);
    }

    // PDF: render page 1 of the user's own file with pdf.js.  This
    // guarantees we show EXACTLY the bytes the user uploaded — no
    // backend rasterisation, no warp, no red-channel munging.
    let cancelled = false;
    renderVersion.current += 1;
    const myVersion = renderVersion.current;

    (async () => {
      setRendering(true);
      try {
        const pdfjs: any = await import("pdfjs-dist");
        pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
        const buf = await file.arrayBuffer();
        const loading = pdfjs.getDocument({ data: buf, isEvalSupported: false });
        const doc = await loading.promise;
        if (cancelled || myVersion !== renderVersion.current) return;
        const page = await doc.getPage(1);
        const viewport = page.getViewport({ scale: 1.6 });
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        await page.render({ canvasContext: ctx, viewport }).promise;
        if (cancelled || myVersion !== renderVersion.current) return;
        setPageSize({ w: viewport.width, h: viewport.height });
      } catch (e: any) {
        setErr(
          `Could not render PDF (${String(e?.message || e).slice(0, 120)})`,
        );
      } finally {
        if (myVersion === renderVersion.current) setRendering(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [file]);

  // Reference space for bbox coordinates.
  //   If quads are available, they're already in ORIGINAL pixel space,
  //   so ``originalWidth/originalHeight`` is the right denominator.
  //   Otherwise fall back to aligned-space dims — the overlay will be
  //   approximate for a skewed upload but at least on the same canvas.
  const useQuads = !!hasOriginalQuads;
  const refWidth = useQuads
    ? originalWidth || pageWidth || 1
    : pageWidth || originalWidth || 1;
  const refHeight = useQuads
    ? originalHeight || pageHeight || 1
    : pageHeight || originalHeight || 1;

  const rectFor = useMemo(() => {
    return (f: FieldDetail): [number, number, number, number] | null => {
      if (useQuads && f.bbox_original_quad) {
        const xs = f.bbox_original_quad.map((p) => p[0]);
        const ys = f.bbox_original_quad.map((p) => p[1]);
        return [
          Math.min(...xs),
          Math.min(...ys),
          Math.max(...xs),
          Math.max(...ys),
        ];
      }
      if (!f.bbox) return null;
      const [x0, y0, x1, y1] = f.bbox;
      // No quads → aligned-space bbox.  If we know both original and
      // aligned page dims we can apply a uniform scale so at least the
      // aspect ratio is correct; otherwise pass through and accept the
      // approximation.
      if (
        pageWidth &&
        pageHeight &&
        originalWidth &&
        originalHeight &&
        !useQuads
      ) {
        const sx = originalWidth / pageWidth;
        const sy = originalHeight / pageHeight;
        return [x0 * sx, y0 * sy, x1 * sx, y1 * sy];
      }
      return [x0, y0, x1, y1];
    };
  }, [useQuads, pageWidth, pageHeight, originalWidth, originalHeight]);

  useEffect(() => {
    if (!selectedId || !pageSize || !scrollRef.current || !refWidth || !refHeight)
      return;
    const field = fields.find((f) => f.id === selectedId);
    if (!field) return;
    const rect = rectFor(field);
    if (!rect) return;
    const sy = (pageSize.h / refHeight) * zoom;
    const yMid = (rect[1] + rect[3]) / 2;
    const topPx = yMid * sy - scrollRef.current.clientHeight / 2;
    scrollRef.current.scrollTo({ top: Math.max(0, topPx), behavior: "smooth" });
  }, [selectedId, pageSize, refWidth, refHeight, fields, zoom, rectFor]);

  // Only draw boxes for fields that actually carry a value.  The
  // uncertain-blank overlay used to double the visual noise on a
  // scanned form because 30-50 boxes would flicker in regardless of
  // whether OCR got anything.  The user wanted a cleaner view — keep
  // the box only when there's a value the user can verify against.
  const overlayFields = useMemo(
    () => fields.filter((f) => {
      const val = (f.value || "").trim();
      if (!val) return false;
      // Don't overlay fields we ourselves flagged as "probably blank"
      // after the rescue ladder: blank_status markers like
      // "cleared_as_template_leak" mean we already know this is a
      // false positive.
      const status = (f.metadata?.blank_status || "").toLowerCase();
      if (status.startsWith("cleared") || status === "blank") return false;
      return true;
    }),
    [fields],
  );

  const isEmpty = !file;
  const bboxScaleX = pageSize && refWidth ? pageSize.w / refWidth : 1;
  const bboxScaleY = pageSize && refHeight ? pageSize.h / refHeight : 1;

  return (
    <div className="panel flex flex-col h-full overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-ink-100 bg-ink-50/50">
        <div className="flex items-center gap-1.5 text-[12px] font-semibold uppercase tracking-wider text-ink-500">
          <ZoomIn size={12} /> Document
        </div>
        {pageSize && (
          <span
            className="rounded-md border border-ink-200 bg-white px-2 py-0.5 text-[10px] font-medium text-ink-500"
            title={
              useQuads
                ? "Bounding boxes projected back to the original page via inverse homography."
                : "No homography available — boxes scaled from aligned coords."
            }
          >
            {useQuads ? "original · homography-aligned" : "original · scaled"}
          </span>
        )}
        <div className="ml-auto flex items-center gap-1">
          <button
            className="rounded-md p-1 text-ink-500 hover:bg-ink-100"
            disabled={!pageSize}
            onClick={() => setZoom((z) => Math.max(0.5, +(z - 0.1).toFixed(2)))}
            title="Zoom out"
          >
            <Minus size={13} />
          </button>
          <span className="font-mono text-[11px] w-10 text-center text-ink-600">
            {(zoom * 100).toFixed(0)}%
          </span>
          <button
            className="rounded-md p-1 text-ink-500 hover:bg-ink-100"
            disabled={!pageSize}
            onClick={() => setZoom((z) => Math.min(2.5, +(z + 0.1).toFixed(2)))}
            title="Zoom in"
          >
            <Plus size={13} />
          </button>
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-auto bg-ink-50">
        <div className="relative w-full flex justify-center p-4">
          {isEmpty && (
            <div className="flex flex-col items-center justify-center py-24 text-ink-400">
              <ImageOff size={28} className="mb-2" />
              <div className="text-sm">No document uploaded yet</div>
            </div>
          )}
          {err && (
            <div className="flex items-center gap-2 rounded-lg bg-bad-100 px-3 py-2 text-sm text-bad-500">
              <AlertTriangle size={14} />
              {err}
            </div>
          )}
          {!err && !isEmpty && (
            <div
              className="relative origin-top"
              style={{
                width: pageSize ? `${pageSize.w * zoom}px` : undefined,
                height: pageSize ? `${pageSize.h * zoom}px` : undefined,
              }}
            >
              {isImage ? (
                imageUrl && (
                  <img
                    src={imageUrl}
                    alt="Uploaded page"
                    className="block rounded shadow-md"
                    style={{
                      width: `${(pageSize?.w || 0) * zoom}px`,
                      height: `${(pageSize?.h || 0) * zoom}px`,
                    }}
                  />
                )
              ) : (
                <canvas
                  ref={canvasRef}
                  className="block rounded shadow-md"
                  style={{
                    width: pageSize ? `${pageSize.w * zoom}px` : "auto",
                    height: pageSize ? `${pageSize.h * zoom}px` : "auto",
                  }}
                />
              )}

              {pageSize && (
                <svg
                  className="absolute inset-0 pointer-events-none"
                  width={pageSize.w * zoom}
                  height={pageSize.h * zoom}
                  viewBox={`0 0 ${pageSize.w} ${pageSize.h}`}
                  style={{ width: pageSize.w * zoom, height: pageSize.h * zoom }}
                >
                  {overlayFields.map((f) => {
                    const rect = rectFor(f);
                    if (!rect) return null;
                    const [x0, y0, x1, y1] = rect;
                    const rx = x0 * bboxScaleX;
                    const ry = y0 * bboxScaleY;
                    const rw = Math.max(2, (x1 - x0) * bboxScaleX);
                    const rh = Math.max(2, (y1 - y0) * bboxScaleY);

                    const isSelected = selectedId === f.id;
                    const isHover = highlightId === f.id;
                    const conf = f.confidence ?? 0;
                    const hasError = f.metadata?.validation_error;
                    const stroke = hasError
                      ? "#ef4444"
                      : conf >= 0.75
                      ? "#10b981"
                      : conf >= 0.45
                      ? "#f59e0b"
                      : "#ef4444";
                    const fill = isSelected
                      ? "rgba(94, 92, 230, 0.25)"
                      : isHover
                      ? "rgba(94, 92, 230, 0.12)"
                      : "transparent";
                    return (
                      <g
                        key={f.id}
                        className="pointer-events-auto cursor-pointer"
                        onMouseEnter={() => onHoverField?.(f.id)}
                        onMouseLeave={() => onHoverField?.(null)}
                        onClick={() => onSelectField?.(f.id)}
                      >
                        <rect
                          x={rx}
                          y={ry}
                          width={rw}
                          height={rh}
                          fill={fill}
                          stroke={stroke}
                          strokeWidth={isSelected ? 3 : 2}
                          rx={3}
                        />
                        {(isSelected || isHover) && (
                          <g>
                            <rect
                              x={rx}
                              y={Math.max(0, ry - 22)}
                              height={20}
                              width={Math.max(100, f.id.length * 7)}
                              fill="#1f2937"
                              rx={4}
                            />
                            <text
                              x={rx + 6}
                              y={Math.max(14, ry - 7)}
                              fill="white"
                              fontSize={11}
                              fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
                            >
                              {f.id}
                            </text>
                          </g>
                        )}
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>
          )}

          {rendering && !pageSize && (
            <div className="absolute inset-0 flex items-center justify-center bg-white/60">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-accent-500 border-t-transparent" />
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between px-3 py-1.5 border-t border-ink-100 bg-ink-50/50 text-[11px] text-ink-500">
        <span>{overlayFields.length} bounding boxes overlaid</span>
        <span className="font-mono">
          {pageSize ? `${pageSize.w}×${pageSize.h}` : "—"}
        </span>
      </div>
    </div>
  );
}
