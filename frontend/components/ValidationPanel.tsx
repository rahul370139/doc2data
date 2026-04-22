"use client";

import { AlertCircle, CheckCircle2, Info } from "lucide-react";
import { ValidationReport } from "@/lib/types";

export function ValidationPanel({ report }: { report?: ValidationReport }) {
  const errors = report?.errors || [];
  const warnings = report?.warnings || [];
  const notes = report?.qa_notes || [];

  if (!errors.length && !warnings.length && !notes.length) {
    return (
      <div className="panel p-5 flex items-center gap-3 text-ok-500">
        <CheckCircle2 size={18} />
        <div>
          <div className="font-medium text-ink-800">No validation issues</div>
          <div className="text-xs text-ink-500">
            All typed validators passed (NPI, dates, phones, ZIP, money, tax ID).
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="panel p-5 space-y-4">
      <div className="flex items-center gap-2">
        <div className="font-semibold text-ink-800">Validation</div>
        <div className="flex gap-1.5">
          {errors.length > 0 && (
            <span className="chip-bad">{errors.length} errors</span>
          )}
          {warnings.length > 0 && (
            <span className="chip-warn">{warnings.length} warnings</span>
          )}
          {notes.length > 0 && (
            <span className="chip-muted">{notes.length} notes</span>
          )}
        </div>
      </div>

      {errors.length > 0 && (
        <IssueGroup
          title="Errors"
          issues={errors}
          color="bad"
          Icon={AlertCircle}
        />
      )}
      {warnings.length > 0 && (
        <IssueGroup
          title="Warnings"
          issues={warnings}
          color="warn"
          Icon={AlertCircle}
        />
      )}
      {notes.length > 0 && (
        <IssueGroup title="QA notes" issues={notes} color="muted" Icon={Info} />
      )}
    </div>
  );
}

function IssueGroup({
  title,
  issues,
  color,
  Icon,
}: {
  title: string;
  issues: any[];
  color: "bad" | "warn" | "muted";
  Icon: any;
}) {
  const pillClass =
    color === "bad" ? "chip-bad" : color === "warn" ? "chip-warn" : "chip-muted";
  return (
    <div>
      <div className="field-label mb-2">{title}</div>
      <ul className="space-y-2">
        {issues.map((issue, i) => (
          <li
            key={i}
            className="flex items-start gap-2 rounded-xl border border-ink-100 bg-white p-3 text-sm"
          >
            <Icon
              className={
                color === "bad"
                  ? "text-bad-500 mt-0.5"
                  : color === "warn"
                  ? "text-warn-500 mt-0.5"
                  : "text-ink-500 mt-0.5"
              }
              size={14}
            />
            <div className="flex-1 min-w-0">
              <div className="font-medium text-ink-800">
                {issue.field_id ? (
                  <span className="font-mono text-[12px]">{issue.field_id}</span>
                ) : (
                  "(general)"
                )}
              </div>
              <div className="text-ink-600 mt-0.5">
                {issue.message || issue.error || JSON.stringify(issue)}
              </div>
            </div>
            {issue.severity && <span className={pillClass}>{issue.severity}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
