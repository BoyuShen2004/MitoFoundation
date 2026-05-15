/* global structuredClone */
const API_BASE = "/api/studio/training-yaml";

function getByPath(obj, path) {
  const parts = path.split(".");
  let cur = obj;
  for (const k of parts) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = cur[k];
  }
  return cur;
}

function setByPath(obj, path, value) {
  const parts = path.split(".");
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const k = parts[i];
    if (cur[k] == null || typeof cur[k] !== "object") cur[k] = {};
    cur = cur[k];
  }
  cur[parts[parts.length - 1]] = value;
}

function fillSelect(sel, keys, current) {
  sel.innerHTML = "";
  for (const k of keys) {
    const o = document.createElement("option");
    o.value = k;
    o.textContent = k;
    if (k === current) o.selected = true;
    sel.appendChild(o);
  }
  if (current && !keys.includes(current)) {
    const o = document.createElement("option");
    o.value = current;
    o.textContent = `${current} (custom)`;
    o.selected = true;
    sel.appendChild(o);
  }
}

function fillTtaFlip(sel, opts, current) {
  sel.innerHTML = "";
  let curStr;
  if (Array.isArray(current)) curStr = JSON.stringify(current);
  else curStr = String(current ?? "all");
  for (const row of opts) {
    const o = document.createElement("option");
    o.value = row.value;
    o.textContent = row.label;
    if (row.value === curStr) o.selected = true;
    sel.appendChild(o);
  }
  if (![...sel.options].some((o) => o.selected)) {
    const o = document.createElement("option");
    o.value = curStr;
    o.textContent = `${curStr} (custom)`;
    o.selected = true;
    sel.appendChild(o);
  }
}

function readTtaFlip(val) {
  if (typeof val === "string" && val.startsWith("[")) {
    try {
      return JSON.parse(val);
    } catch {
      return val;
    }
  }
  return val;
}

function mountStep(templateId, host) {
  const tpl = document.getElementById(templateId);
  host.appendChild(tpl.content.cloneNode(true));
}

function applyOptionsToForm(options, cfg) {
  const prof = options.profiles || {};
  const pipeline = prof.pipeline_profiles || [];
  const system = prof.system_profiles || [];
  const arch = prof.arch_profiles || [];
  const dataloader = prof.dataloader_profiles || [];
  const aug = prof.augmentation_profiles || [];
  const optim = prof.optimizer_profiles || [];

  document.querySelectorAll('[data-path="default.pipeline_profile"]').forEach((el) => {
    fillSelect(el, pipeline, getByPath(cfg, "default.pipeline_profile"));
  });
  document.querySelectorAll('[data-path="default.system.profile"]').forEach((el) => {
    fillSelect(el, system, getByPath(cfg, "default.system.profile"));
  });
  document.querySelectorAll('[data-path="default.model.arch.profile"]').forEach((el) => {
    fillSelect(el, arch, getByPath(cfg, "default.model.arch.profile"));
  });
  document.querySelectorAll('[data-path="default.data.dataloader.profile"]').forEach((el) => {
    fillSelect(el, dataloader, getByPath(cfg, "default.data.dataloader.profile"));
  });
  document.querySelectorAll('[data-path="default.data.augmentation.profile"]').forEach((el) => {
    fillSelect(el, aug, getByPath(cfg, "default.data.augmentation.profile"));
  });
  document.querySelectorAll('[data-path="train.optimization.profile"]').forEach((el) => {
    fillSelect(el, optim, getByPath(cfg, "train.optimization.profile"));
  });

  document.querySelectorAll('[data-path="default.inference.test_time_augmentation.flip_axes"]').forEach((el) => {
    fillTtaFlip(el, options.tta_flip_axes || [], getByPath(cfg, "default.inference.test_time_augmentation.flip_axes"));
  });
}

function fillForm(cfg) {
  document.querySelectorAll("[data-path]").forEach((el) => {
    const path = el.getAttribute("data-path");
    const v = getByPath(cfg, path);
    if (el.type === "checkbox" && el.hasAttribute("data-bool")) {
      el.checked = Boolean(v);
      return;
    }
    if (el.tagName === "SELECT") return;
    el.value = v === undefined || v === null ? "" : v;
  });

  document.querySelectorAll("[data-triple]").forEach((el) => {
    const base = el.getAttribute("data-triple");
    const idx = Number(el.getAttribute("data-i"));
    const arr = getByPath(cfg, base);
    el.value = Array.isArray(arr) && arr[idx] !== undefined ? arr[idx] : "";
  });

  document.querySelectorAll("[data-lines]").forEach((el) => {
    const path = el.getAttribute("data-lines");
    const v = getByPath(cfg, path);
    el.value = Array.isArray(v) ? v.join("\n") : v || "";
  });

  const metrics = getByPath(cfg, "default.inference.evaluation.metrics") || [];
  document.querySelectorAll("[data-metric]").forEach((cb) => {
    cb.checked = metrics.includes(cb.value);
  });

}

function readFormInto(cfg) {
  document.querySelectorAll("[data-path]").forEach((el) => {
    const path = el.getAttribute("data-path");
    if (el.type === "checkbox" && el.hasAttribute("data-bool")) {
      setByPath(cfg, path, el.checked);
      return;
    }
    if (el.tagName === "SELECT") {
      const raw = el.value;
      if (path.endsWith("flip_axes")) {
        setByPath(cfg, path, readTtaFlip(raw));
      } else {
        setByPath(cfg, path, raw);
      }
      return;
    }
    const raw = el.value;
    if (path.endsWith("dropout")) {
      setByPath(cfg, path, raw === "" ? null : parseFloat(raw));
    } else if (
      path.includes("n_steps_per_epoch") ||
      path.includes("max_epochs") ||
      path.includes("batch_size") ||
      path.includes("save_top_k") ||
      path.includes("loss_every") ||
      path.includes("log_every") ||
      path.includes("max_images") ||
      path.includes("num_slices")
    ) {
      setByPath(cfg, path, raw === "" ? null : Number(raw));
    } else {
      setByPath(cfg, path, raw);
    }
  });

  document.querySelectorAll("[data-triple]").forEach((el) => {
    const base = el.getAttribute("data-triple");
    const idx = Number(el.getAttribute("data-i"));
    const cur = getByPath(cfg, base);
    const arr = Array.isArray(cur) ? [...cur] : [0, 0, 0];
    arr[idx] = el.value === "" ? 0 : Number(el.value);
    setByPath(cfg, base, arr);
  });

  document.querySelectorAll("[data-lines]").forEach((el) => {
    const path = el.getAttribute("data-lines");
    const lines = el.value
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    setByPath(cfg, path, lines);
  });

  const picked = [];
  document.querySelectorAll("[data-metric]:checked").forEach((cb) => picked.push(cb.value));
  setByPath(cfg, "default.inference.evaluation.metrics", picked);
}

async function init() {
  const hint = document.getElementById("loadHint");
  const panels = document.getElementById("panels");
  const pathEl = document.getElementById("configPath");
  const msg = document.getElementById("msg");

  try {
    const [optRes, cfgRes] = await Promise.all([
      fetch(`${API_BASE}/options`),
      fetch(`${API_BASE}/config`),
    ]);
    if (!optRes.ok) throw new Error(`options ${optRes.status}`);
    if (!cfgRes.ok) throw new Error(`config ${cfgRes.status}`);
    const options = await optRes.json();
    const payload = await cfgRes.json();
    window.__options = options;
    window.__fullConfig = payload.config;
    pathEl.textContent = payload.path;

    panels.innerHTML = "";
    mountStep("tpl-step1", panels);
    mountStep("tpl-step2", panels);
    mountStep("tpl-step3", panels);

    applyOptionsToForm(options, payload.config);
    fillForm(payload.config);

    const cards = panels.querySelectorAll("section.card");
    const stepGroups = [
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9],
    ];
    cards.forEach((c, idx) => {
      if (!stepGroups[0].includes(idx)) c.classList.add("hidden");
    });

    hint.classList.add("hidden");
    panels.classList.remove("hidden");

    document.querySelectorAll(".step-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        const step = Number(btn.getAttribute("data-step"));
        document.querySelectorAll(".step-tab").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const showIdx = stepGroups[step] || [];
        cards.forEach((c, idx) => {
          if (showIdx.includes(idx)) c.classList.remove("hidden");
          else c.classList.add("hidden");
        });
      });
    });

    const syncBtn = document.getElementById("syncPreprocessedBtn");
    const syncMsg = document.getElementById("syncMsg");
    if (syncBtn && syncMsg) {
      syncBtn.addEventListener("click", async () => {
        syncMsg.textContent = "";
        syncMsg.className = "msg";
        try {
          const res = await fetch(`${API_BASE}/config`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sync_preprocessed_volumes: true }),
          });
          const body = await res.json().catch(() => ({}));
          if (!res.ok) {
            const d = body.detail;
            const errText =
              typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg || x).join("; ") : JSON.stringify(d);
            throw new Error(errText || res.statusText);
          }
          window.__fullConfig = body.config;
          applyOptionsToForm(window.__options, body.config);
          fillForm(body.config);
          const n = Array.isArray(body.basenames) ? body.basenames.length : 0;
          syncMsg.textContent = n ? `Synced ${n} paired volume(s). YAML saved.` : "Synced. YAML saved.";
          syncMsg.className = "msg ok";
        } catch (e) {
          syncMsg.textContent = e instanceof Error ? e.message : String(e);
          syncMsg.className = "msg err";
        }
      });
    }
  } catch (e) {
    hint.textContent = `Failed to load: ${e.message}`;
    console.error(e);
    return;
  }

  document.getElementById("saveBtn").addEventListener("click", async () => {
    msg.textContent = "";
    const cfg = structuredClone(window.__fullConfig);
    readFormInto(cfg);
    try {
      const res = await fetch(`${API_BASE}/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: cfg }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const d = body.detail;
        const msgErr =
          typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg || x).join("; ") : JSON.stringify(d);
        throw new Error(msgErr || res.statusText);
      }
      window.__fullConfig = cfg;
      msg.textContent = "Saved.";
      msg.className = "msg ok";
    } catch (e) {
      msg.textContent = `Save failed: ${e.message}`;
      msg.className = "msg err";
    }
  });
}

init();
