import { useCallback, useEffect, useState } from "react";
import {
  getAgentSkillDocument,
  getAgentSkillIndex,
  patchAgentSkillSlug,
  postCreateAgentSkill,
  putAgentSkillDocument,
  type AgentSkillMeta,
} from "./api";

type SkillKind = "chat" | "orchestration";

const SLUG_RE = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$/;

/** Folder slug: lowercase, spaces → underscores, non-alphanumeric → underscore, trimmed. */
export function skillNameToFolderSlug(raw: string): string {
  let s = raw.trim().toLowerCase();
  s = s.replace(/\s+/g, "_");
  s = s.replace(/[^a-z0-9_]/g, "_");
  s = s.replace(/_+/g, "_");
  s = s.replace(/^_+|_+$/g, "");
  return s;
}

export function parseSkillMd(text: string): { meta: Record<string, string>; body: string } {
  const t = text.replace(/\r\n/g, "\n").trim();
  if (!t.startsWith("---")) return { meta: {}, body: t };
  const m = t.match(/^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/);
  if (!m) return { meta: {}, body: t };
  const meta: Record<string, string> = {};
  for (const line of m[1].split("\n")) {
    const idx = line.indexOf(":");
    if (idx === -1) continue;
    const k = line.slice(0, idx).trim();
    const v = line.slice(idx + 1).trim();
    if (k) meta[k] = v;
  }
  return { meta, body: m[2].trimEnd() };
}

export function buildSkillMd(meta: Record<string, string>, body: string): string {
  const preferred = ["id", "title", "label"] as const;
  const keys: string[] = [];
  const seen = new Set<string>();
  for (const k of preferred) {
    const v = (meta[k] ?? "").trim();
    if (v) {
      keys.push(k);
      seen.add(k);
    }
  }
  for (const k of Object.keys(meta).sort()) {
    if (seen.has(k)) continue;
    const v = (meta[k] ?? "").trim();
    if (!v) continue;
    keys.push(k);
  }
  const lines = ["---", ...keys.map((k) => `${k}: ${(meta[k] ?? "").trim()}`), "---", "", body.trim()];
  return `${lines.join("\n")}\n`;
}

function SkillRow({
  kind,
  meta,
  onError,
  onReload,
}: {
  kind: SkillKind;
  meta: AgentSkillMeta;
  onError: (msg: string) => void;
  onReload: () => Promise<void>;
}) {
  const [document, setDocument] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editSlug, setEditSlug] = useState("");
  const [editLabel, setEditLabel] = useState("");
  const [editTitle, setEditTitle] = useState("");
  const [editId, setEditId] = useState("");
  const [editBody, setEditBody] = useState("");
  const [extraMeta, setExtraMeta] = useState<Record<string, string>>({});

  const loadDoc = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getAgentSkillDocument(kind, meta.slug);
      setDocument(res.document);
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [kind, meta.slug, onError]);

  const startEdit = () => {
    if (document === null) return;
    const { meta: fm, body } = parseSkillMd(document);
    const extra = { ...fm };
    delete extra.id;
    delete extra.title;
    delete extra.label;
    setExtraMeta(extra);
    setEditSlug(meta.slug);
    setEditLabel((fm.label || fm.title || meta.label).trim());
    setEditTitle((fm.title || meta.title).trim());
    setEditId((fm.id || meta.id).trim());
    setEditBody(body);
    setEditing(true);
  };

  const cancelEdit = () => {
    setEditing(false);
    setEditSlug("");
    setEditLabel("");
    setEditTitle("");
    setEditId("");
    setEditBody("");
    setExtraMeta({});
  };

  const saveEdit = async () => {
    const slugTrim = editSlug.trim();
    if (!SLUG_RE.test(slugTrim)) {
      onError("Folder slug must start with a letter or number and use only letters, numbers, underscores, or hyphens.");
      return;
    }
    const lab = editLabel.trim();
    if (!lab) {
      onError("Display name (label) is required.");
      return;
    }
    try {
      const nextMeta: Record<string, string> = {
        ...extraMeta,
        id: editId.trim() || slugTrim,
        title: editTitle.trim() || lab,
        label: lab,
      };
      const built = buildSkillMd(nextMeta, editBody);
      if (slugTrim !== meta.slug) {
        await patchAgentSkillSlug(kind, meta.slug, slugTrim);
      }
      await putAgentSkillDocument(kind, slugTrim, built);
      setDocument(built);
      setEditing(false);
      setEditSlug("");
      setEditLabel("");
      setEditTitle("");
      setEditId("");
      setEditBody("");
      setExtraMeta({});
      await onReload();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <details
      className="agent-skill-details"
      onToggle={(e) => {
        const el = e.currentTarget;
        if (el.open && document === null && !loading) void loadDoc();
      }}
    >
      <summary className="agent-skill-summary">{meta.label}</summary>
      <div className="agent-skill-body">
        <div className="agent-skill-toolbar">
          {!editing ? (
            <button
              type="button"
              className="ghost agent-skill-edit-btn"
              disabled={loading || document === null}
              onClick={() => startEdit()}
            >
              Edit
            </button>
          ) : (
            <>
              <button type="button" className="primary agent-skill-save-btn" onClick={() => void saveEdit()}>
                Save
              </button>
              <button type="button" className="ghost agent-skill-cancel-btn" onClick={() => cancelEdit()}>
                Cancel
              </button>
            </>
          )}
        </div>
        <p className="agent-skill-meta muted-note">
          <code>{meta.slug}</code>
          {meta.title !== meta.label ? ` · ${meta.title}` : ""}
        </p>
        {loading ? <p className="muted-note">Loading…</p> : null}
        {!loading && document !== null && !editing ? <pre className="agent-skill-pre">{document}</pre> : null}
        {editing ? (
          <div className="agent-skill-edit-fields">
            <label className="agent-skill-field">
              <span>Folder slug (rename)</span>
              <input
                className="field-input"
                value={editSlug}
                onChange={(e) => setEditSlug(e.target.value)}
                spellCheck={false}
                autoComplete="off"
              />
            </label>
            <label className="agent-skill-field">
              <span>Display name (label)</span>
              <input
                className="field-input"
                value={editLabel}
                onChange={(e) => setEditLabel(e.target.value)}
                spellCheck={false}
                autoComplete="off"
              />
            </label>
            <label className="agent-skill-field">
              <span>Title</span>
              <input
                className="field-input"
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
                spellCheck={false}
                autoComplete="off"
              />
            </label>
            <label className="agent-skill-field">
              <span>Skill id (front matter)</span>
              <input
                className="field-input"
                value={editId}
                onChange={(e) => setEditId(e.target.value)}
                spellCheck={false}
                autoComplete="off"
              />
            </label>
            <label className="agent-skill-field agent-skill-field-body">
              <span>Body (markdown below front matter)</span>
              <textarea
                className="agent-skill-editor"
                value={editBody}
                onChange={(e) => setEditBody(e.target.value)}
                spellCheck={false}
                rows={14}
              />
            </label>
          </div>
        ) : null}
      </div>
    </details>
  );
}

function AddSkillPanel({
  onError,
  onReload,
}: {
  onError: (msg: string) => void;
  onReload: () => Promise<void>;
}) {
  const [targetKind, setTargetKind] = useState<SkillKind>("chat");
  const [skillName, setSkillName] = useState("");
  const [skillContent, setSkillContent] = useState("");
  const [busy, setBusy] = useState(false);

  const folderPreview = skillNameToFolderSlug(skillName);

  const submit = async () => {
    const nameTrim = skillName.trim();
    if (!nameTrim) {
      onError("Enter a skill name.");
      return;
    }
    const slug = skillNameToFolderSlug(skillName);
    if (!slug || !SLUG_RE.test(slug)) {
      onError(
        "Skill name must yield a valid folder name: lowercase, use spaces between words (they become underscores), letters and numbers only.",
      );
      return;
    }
    const body = skillContent.trim();
    if (!body) {
      onError("Enter skill content (markdown body).");
      return;
    }
    setBusy(true);
    try {
      await postCreateAgentSkill(targetKind, {
        slug,
        label: nameTrim,
        title: nameTrim,
        body: `${body}\n`,
      });
      setSkillName("");
      setSkillContent("");
      await onReload();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="agent-skill-add-card agent-skill-add-card-top">
      <h4 className="agent-skill-add-title">Add skill</h4>
      <fieldset className="agent-skill-kind-fieldset">
        <legend className="agent-skill-kind-legend">Save to</legend>
        <div className="agent-skill-kind-row">
          <label className="agent-skill-radio">
            <input
              type="radio"
              name="agent-skill-target-kind"
              checked={targetKind === "chat"}
              onChange={() => setTargetKind("chat")}
            />
            Chat skills
          </label>
          <label className="agent-skill-radio">
            <input
              type="radio"
              name="agent-skill-target-kind"
              checked={targetKind === "orchestration"}
              onChange={() => setTargetKind("orchestration")}
            />
            Orchestration skills
          </label>
        </div>
      </fieldset>
      <div className="agent-skill-edit-fields">
        <label className="agent-skill-field">
          <span>Skill name</span>
          <input
            className="field-input"
            value={skillName}
            onChange={(e) => setSkillName(e.target.value)}
            placeholder="e.g. My custom check (folder: my_custom_check)"
            spellCheck={false}
            autoComplete="off"
          />
          {folderPreview ? (
            <span className="muted-note agent-skill-folder-preview">
              Folder: <code>{folderPreview}</code>
            </span>
          ) : null}
        </label>
        <label className="agent-skill-field agent-skill-field-body">
          <span>Skill content</span>
          <textarea
            className="agent-skill-editor"
            value={skillContent}
            onChange={(e) => setSkillContent(e.target.value)}
            placeholder="Markdown for the skill body (shown under YAML front matter in skill.md)."
            spellCheck={false}
            rows={10}
          />
        </label>
        <button type="button" className="primary" disabled={busy} onClick={() => void submit()}>
          {busy ? "Creating…" : "Create skill"}
        </button>
      </div>
    </div>
  );
}

function SkillSection({
  title,
  kind,
  items,
  onError,
  onReload,
}: {
  title: string;
  kind: SkillKind;
  items: AgentSkillMeta[];
  onError: (msg: string) => void;
  onReload: () => Promise<void>;
}) {
  return (
    <section className="agent-skills-section">
      <h3 className="agent-skills-section-title">{title}</h3>
      {items.length === 0 ? (
        <p className="muted-note">No skills in this section yet — use Add skill above.</p>
      ) : (
        <div className="agent-skills-table">
          {items.map((m) => (
            <SkillRow key={`${kind}:${m.slug}`} kind={kind} meta={m} onError={onError} onReload={onReload} />
          ))}
        </div>
      )}
    </section>
  );
}

export function AgentSkillsSettings() {
  const [chat, setChat] = useState<AgentSkillMeta[]>([]);
  const [orch, setOrch] = useState<AgentSkillMeta[]>([]);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  const loadIndex = useCallback(async () => {
    setLoadErr(null);
    try {
      const res = await getAgentSkillIndex();
      setChat(res.chat ?? []);
      setOrch(res.orchestration ?? []);
    } catch (e) {
      setLoadErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void loadIndex();
  }, [loadIndex]);

  const onRowErr = useCallback((msg: string) => {
    setLoadErr(msg);
  }, []);

  return (
    <div className="agent-skills-settings">
      <p className="muted-note agent-skills-intro">
        Each skill is a folder with <code>skill.md</code>. Add skills at the top (folder name is derived from the skill
        name: lowercase, underscores between words). Edit existing skills to change names, body, or folder slug; saves
        apply on the next chat turn.
      </p>
      <AddSkillPanel onError={onRowErr} onReload={loadIndex} />
      {loadErr ? (
        <p className="settings-error" role="alert">
          {loadErr}
        </p>
      ) : null}
      <SkillSection title="Chat skills" kind="chat" items={chat} onError={onRowErr} onReload={loadIndex} />
      <SkillSection title="Orchestration skills" kind="orchestration" items={orch} onError={onRowErr} onReload={loadIndex} />
      <p className="muted-note">
        <button type="button" className="ghost linkish" onClick={() => void loadIndex()}>
          Reload list
        </button>
      </p>
    </div>
  );
}
