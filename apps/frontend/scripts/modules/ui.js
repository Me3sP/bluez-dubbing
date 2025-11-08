import { el } from "./dom.js";
import { state } from "./state.js";
import { utils } from "./utils.js";
import { lang } from "./language.js";

export const ui = {
  log(message) {
    const now = new Date().toLocaleTimeString();
    if (el.log) {
      el.log.textContent += `\n[${now}] ${message}`;
      el.log.scrollTop = el.log.scrollHeight;
    }
  },

  resetLog() {
    if (el.log) el.log.textContent = "Waiting…";
  },

  setStatus(text, tone = "idle") {
    if (!el.statusBadge) return;
    el.statusBadge.textContent = text;
    const colors = {
      idle: "rgba(59, 130, 246, 0.18)",
      running: "rgba(52, 211, 153, 0.2)",
      success: "rgba(34, 197, 94, 0.25)",
      error: "rgba(239, 68, 68, 0.25)",
      paused: "rgba(245, 158, 11, 0.25)"
    };
    const color = colors[tone] || colors.idle;
    el.statusBadge.style.background = color;
    el.statusBadge.style.borderColor = color;
  },

  updateSourcePreview(src, label, { keepExisting = false, objectUrl = null } = {}) {
    if (!el.sourcePreview?.video) return;

    if (!keepExisting && state.objectUrls.source) {
      URL.revokeObjectURL(state.objectUrls.source);
      state.objectUrls.source = null;
    }

    const targetSrc = objectUrl || utils.resolveUrl(src);
    if (!targetSrc) {
      el.sourcePreview.video.removeAttribute("src");
      el.sourcePreview.video.load();
      el.sourcePreview.card?.classList.add("empty");
      if (el.sourcePreview.text) el.sourcePreview.text.textContent = label || "Waiting for media…";
      return;
    }

    el.sourcePreview.video.src = targetSrc;
    el.sourcePreview.video.load();
    el.sourcePreview.card?.classList.remove("empty");
    if (el.sourcePreview.text) el.sourcePreview.text.textContent = label || "";
    if (objectUrl) state.objectUrls.source = objectUrl;
  },

  updateResultPreview(src, label, { objectUrl = null } = {}) {
    if (!el.resultPreview?.video) return;

    if (state.objectUrls.result) {
      URL.revokeObjectURL(state.objectUrls.result);
      state.objectUrls.result = null;
    }

    const targetSrc = objectUrl || utils.resolveUrl(src);
    if (!targetSrc) {
      el.resultPreview.video.removeAttribute("src");
      el.resultPreview.video.load();
      el.resultPreview.card?.classList.add("empty");
      if (el.resultPreview.text) el.resultPreview.text.textContent = label || "No output yet.";
      return;
    }

    el.resultPreview.video.src = targetSrc;
    el.resultPreview.video.load();
    el.resultPreview.card?.classList.remove("empty");
    if (el.resultPreview.text) el.resultPreview.text.textContent = label || "";
    if (objectUrl) state.objectUrls.result = objectUrl;
  },

  updateDownloadProgress({ active, progress = null, label = "" }) {
    if (!el.download?.progress) return;

    if (!active) {
      el.download.progress.hidden = true;
      el.download.bar.style.width = "0%";
      el.download.bar.style.animation = "none";
      el.download.label.hidden = !label;
      if (label) el.download.label.textContent = label;
      return;
    }

    el.download.progress.hidden = false;
    el.download.label.hidden = false;
    el.download.label.textContent = label || "Downloading…";

    if (progress !== null) {
      const pct = Math.min(100, Math.max(0, progress));
      el.download.bar.style.width = `${pct}%`;
      el.download.bar.style.animation = "none";
    } else {
      el.download.bar.style.width = "30%";
      el.download.bar.style.animation = "progressPulse 1s ease-in-out infinite alternate";
    }
  },

  updateResultSelector(payload) {
    if (!el.resultPreview?.select) return;

    const languages = Object.keys(payload.languages || {});
    el.resultPreview.select.innerHTML = "";

    if (!languages.length) {
      el.resultPreview.select.disabled = true;
      const opt = new Option(
        payload.final_video?.url ? "Default output" : "No output",
        ""
      );
      el.resultPreview.select.appendChild(opt);

      if (payload.final_video?.url) {
        this.updateResultPreview(payload.final_video.url, "Rendered output");
      } else {
        this.updateResultPreview(null, "No output yet.");
      }
      return;
    }

    languages.forEach(l => {
      el.resultPreview.select.appendChild(
        new Option(lang.toLabel(l) || l, l)
      );
    });

    const preferred = payload.default_language && languages.includes(payload.default_language)
      ? payload.default_language
      : languages[0];

    el.resultPreview.select.value = preferred;
    el.resultPreview.select.disabled = false;
    this.updateResultForLanguage(preferred);
  },

  updateResultForLanguage(langCode) {
    if (!langCode || !state.latestResult) {
      this.updateResultPreview(null, "No output yet.");
      return;
    }

    const entry = state.latestResult.languages?.[langCode];
    const label = lang.toLabel(langCode) || langCode;

    if (!entry?.final_video?.url) {
      this.updateResultPreview(null, `No preview for ${label}`);
      return;
    }

    this.updateResultPreview(entry.final_video.url, `Previewing ${label}`);
  },

  showInterrupt(disabled = true) {
    if (!el.interruptBtn) return;
    el.interruptBtn.hidden = false;
    el.interruptBtn.disabled = !!disabled;
  },

  hideInterrupt() {
    if (!el.interruptBtn) return;
    el.interruptBtn.hidden = true;
    el.interruptBtn.disabled = true;
  }
};
