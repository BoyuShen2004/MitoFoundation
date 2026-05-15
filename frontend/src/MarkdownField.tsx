import { useEffect, useRef, useState } from "react";
import { MarkdownPreview } from "./MarkdownPreview";

type Props = {
  id: string;
  label: React.ReactNode;
  value: string;
  onChange: (next: string) => void;
  rows?: number;
  placeholder?: string;
  /** Remount key: change when external data reload should exit edit mode */
  resetKey?: string;
  labelClassName?: string;
  previewWrapClassName?: string;
  textareaClassName?: string;
};

/**
 * GitHub-style rendered markdown by default; Edit shows raw source with Cancel / Save.
 */
export function MarkdownField({
  id,
  label,
  value,
  onChange,
  rows = 6,
  placeholder = "",
  resetKey = "",
  labelClassName = "studio-label",
  previewWrapClassName = "markdown-field-preview markdown-field-preview--studio",
  textareaClassName = "field-input studio-field studio-textarea",
}: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setEditing(false);
  }, [resetKey]);

  useEffect(() => {
    if (!editing) {
      setDraft(value);
    }
  }, [value, editing]);

  useEffect(() => {
    if (editing) {
      taRef.current?.focus();
    }
  }, [editing]);

  const startEdit = () => {
    setDraft(value);
    setEditing(true);
  };

  const cancel = () => {
    setDraft(value);
    setEditing(false);
  };

  const save = () => {
    onChange(draft);
    setEditing(false);
  };

  const labelEl =
    typeof label === "string" ? (
      <label className={labelClassName} htmlFor={editing ? id : undefined}>
        {label}
      </label>
    ) : (
      label
    );

  return (
    <div className="markdown-field" data-reset-key={resetKey || undefined}>
      {labelEl}
      {editing ? (
        <>
          <textarea
            ref={taRef}
            id={id}
            className={textareaClassName}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={placeholder}
            rows={rows}
            spellCheck={false}
          />
          <div className="markdown-field-actions">
            <button type="button" className="ghost" onClick={cancel}>
              Cancel
            </button>
            <button type="button" className="primary" onClick={save}>
              Save
            </button>
          </div>
        </>
      ) : (
        <>
          <div className={previewWrapClassName} role="region" aria-label="Markdown preview">
            <MarkdownPreview markdown={value} emptyLabel="(empty — click Edit to add)" />
          </div>
          <div className="markdown-field-toolbar">
            <button type="button" className="ghost markdown-field-edit" onClick={startEdit}>
              Edit
            </button>
          </div>
        </>
      )}
    </div>
  );
}
