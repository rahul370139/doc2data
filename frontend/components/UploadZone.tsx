"use client";

import { useCallback, useRef, useState } from "react";
import { FileText, UploadCloud, X } from "lucide-react";
import clsx from "clsx";

export interface UploadZoneProps {
  file: File | null;
  onFile: (f: File | null) => void;
  disabled?: boolean;
}

export function UploadZone({ file, onFile, disabled }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setDrag(false);
      if (disabled) return;
      const f = e.dataTransfer.files?.[0];
      if (f && /\.(pdf|png|jpe?g|tiff?)$/i.test(f.name)) onFile(f);
    },
    [onFile, disabled],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      className={clsx(
        "panel relative flex flex-col items-center justify-center gap-3 px-6 py-10 text-center transition",
        "border-2 border-dashed",
        drag
          ? "border-accent-500 bg-accent-50"
          : "border-ink-200 hover:border-accent-300",
        disabled && "opacity-60 cursor-not-allowed",
      )}
    >
      {!file ? (
        <>
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-100 text-accent-500">
            <UploadCloud size={22} />
          </div>
          <div className="space-y-1">
            <div className="font-semibold text-ink-800">
              Drop a CMS-1500 or UB-04 PDF
            </div>
            <div className="text-sm text-ink-500">
              or click to browse · PDF, PNG, JPG, TIFF
            </div>
          </div>
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="btn-primary mt-2"
            disabled={disabled}
          >
            Choose file
          </button>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onFile(f);
              e.currentTarget.value = "";
            }}
          />
        </>
      ) : (
        <div className="flex w-full items-center justify-between rounded-xl bg-ink-50 px-4 py-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-ink-600 border border-ink-200">
              <FileText size={18} />
            </div>
            <div className="min-w-0 text-left">
              <div className="truncate font-medium text-ink-800">
                {file.name}
              </div>
              <div className="text-xs text-ink-500">
                {(file.size / 1024 / 1024).toFixed(2)} MB · {file.type || "pdf"}
              </div>
            </div>
          </div>
          <button
            type="button"
            className="btn-ghost"
            onClick={() => onFile(null)}
            disabled={disabled}
            aria-label="Remove"
          >
            <X size={16} />
          </button>
        </div>
      )}
    </div>
  );
}
