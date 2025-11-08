import { API } from "./modules/api.js";
import { el } from "./modules/dom.js";
import { state } from "./modules/state.js";
import { ui } from "./modules/ui.js";
import { token } from "./modules/token.js";
import { targetLangs } from "./modules/targetLangs.js";
import { handlers } from "./modules/handlers.js";
import { transcriptionReview, alignmentReview, ttsReview } from "./modules/reviews.js";
import { theme } from "./modules/theme.js";

const initInvolveMode = () => {
  if (!el.modeToggle) return;

  const setMode = enabled => {
    state.involveMode = Boolean(enabled);
    el.modeToggle.classList.toggle("active", state.involveMode);
    el.modeToggle.setAttribute("aria-pressed", state.involveMode ? "true" : "false");
    el.modeToggle.textContent = state.involveMode ? "Involve mode: On" : "Involve mode: Off";
  };

  setMode(false);
  el.modeToggle.onclick = () => {
    setMode(!state.involveMode);
    ui.log(state.involveMode
      ? "🛠 Involve mode enabled. The pipeline will pause at different steps for manual review."
      : "🏃 Involve mode disabled. The pipeline will run uninterrupted.");
  };
};

const initReviewButtons = () => {
  if (el.transcription.apply) el.transcription.apply.onclick = () => transcriptionReview.submit(true);
  if (el.transcription.skip) el.transcription.skip.onclick = () => transcriptionReview.submit(false);
  if (el.transcription.add) {
    el.transcription.add.onclick = () => {
      const review = state.pendingReviews.transcription;
      if (!review) return;
      const segments = review.payload.segments || [];
      transcriptionReview.addSegment(segments.length - 1);
    };
  }

  if (el.alignment.apply) el.alignment.apply.onclick = () => alignmentReview.submit(true);
  if (el.alignment.skip) el.alignment.skip.onclick = () => alignmentReview.submit(false);
  if (el.alignment.add) {
    el.alignment.add.onclick = () => {
      const review = state.pendingReviews.alignment;
      if (!review) return;
      alignmentReview.addSegment(review.payload.segments.length - 1);
    };
  }

  if (el.tts.apply) el.tts.apply.onclick = () => ttsReview.submit(true);
  if (el.tts.skip) el.tts.skip.onclick = () => ttsReview.submit(false);
};

const initInitialUI = () => {
  transcriptionReview.hide();
  alignmentReview.hide();
  ttsReview.hide();
  ui.updateSourcePreview(null, "Waiting for media…");
  ui.updateResultPreview(null, "No output yet.");
  targetLangs.render();
};

const initResultSelector = () => {
  if (!el.resultPreview.select) return;
  el.resultPreview.select.onchange = () => {
    ui.updateResultForLanguage(el.resultPreview.select.value);
  };
};

const initInterruptButton = () => {
  if (!el.interruptBtn) return;
  el.interruptBtn.onclick = async () => {
    if (!state.runId) {
      ui.log("⚠️ No active run to interrupt.");
      return;
    }

    const prevLabel = el.interruptBtn.textContent;
    el.interruptBtn.disabled = true;
    el.interruptBtn.textContent = "Stopping…";

    try {
      const body = new URLSearchParams({ run_id: state.runId });
      await fetch(API.routes.stop, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body,
        keepalive: true
      });
      ui.log("⏹ Cancellation requested.");
      ui.setStatus("Stopping…", "running");
    } catch (err) {
      ui.log(`❌ Failed to interrupt: ${err.message}`);
      el.interruptBtn.disabled = false;
    } finally {
      el.interruptBtn.textContent = prevLabel;
    }
  };
};

const initMediaInputs = () => {
  if (el.fileInput) {
    el.fileInput.onchange = () => {
      if (state.uploadToken) {
        token.release();
      } else {
        token.set("");
        state.sourceDescriptor = "";
      }

      if (state.objectUrls.source) {
        URL.revokeObjectURL(state.objectUrls.source);
        state.objectUrls.source = null;
      }

      if (el.fileInput.files?.length > 0) {
        const blobUrl = URL.createObjectURL(el.fileInput.files[0]);
        ui.updateSourcePreview(blobUrl, "Local upload", { objectUrl: blobUrl });
        state.sourceDescriptor = el.fileInput.files[0].name || "";
      } else if (!el.videoLink?.value.trim()) {
        ui.updateSourcePreview(null, "Waiting for media…");
      }
    };
  }

  if (el.videoLink) {
    el.videoLink.oninput = () => {
      const trimmedValue = el.videoLink.value.trim();
      if (state.uploadToken && trimmedValue !== state.sourceDescriptor) {
        token.release();
        state.sourceDescriptor = trimmedValue;
      } else if (!trimmedValue) {
        token.set("");
        state.sourceDescriptor = "";
      } else {
        state.sourceDescriptor = trimmedValue;
      }

      if (trimmedValue && (!el.fileInput?.files || el.fileInput.files.length === 0)) {
        ui.updateSourcePreview(null, "Remote media will be downloaded on run…");
        ui.updateDownloadProgress({ active: false, label: "" });
      }
    };
  }
};

const initFormHandling = () => {
  if (el.form) {
    el.form.onsubmit = e => handlers.submitForm(e);
  }
};

const initUnloadCleanup = () => {
  window.addEventListener("beforeunload", () => {
    if (!state.uploadToken) return;
    const params = new URLSearchParams({ token: state.uploadToken });
    const blob = new Blob([params.toString()], { type: "application/x-www-form-urlencoded" });
    if (!navigator.sendBeacon || !navigator.sendBeacon(API.routes.release, blob)) {
      fetch(API.routes.release, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: params,
        keepalive: true
      }).catch(() => {});
    }
  });
};

const initApp = () => {
  theme.init();
  initInvolveMode();
  initReviewButtons();
  initInitialUI();
  initResultSelector();
  initInterruptButton();
  initMediaInputs();
  initFormHandling();
  initUnloadCleanup();

  handlers.fetchOptions().catch(err => ui.log(`⚠️ ${err.message}`));
};

initApp();
