import { API } from './api.js';
import { el } from './dom.js';
import { state } from './state.js';
import { lang } from './language.js';
import { models } from './models.js';
import { targetLangs } from './targetLangs.js';
import { ui } from './ui.js';
import { token } from './token.js';
import { results } from './results.js';
import { transcriptionReview, alignmentReview, ttsReview } from './reviews.js';

// Event Handlers
const handlers = {
  async fetchOptions() {
    const resp = await fetch(API.routes.options);
    if (!resp.ok) throw new Error("Unable to load model registry");
    state.options = await resp.json();
    this.populateSelectors(state.options);
  },

  populateSelectors(opts) {
    const asrSelect = document.getElementById("asr-model");
    const trSelect = document.getElementById("tr-model");
    const ttsSelect = document.getElementById("tts-model");
    const sepSelect = document.getElementById("sep-model");
    const translationStrategy = document.getElementById("translation-strategy");
    const dubbingStrategy = document.getElementById("dubbing-strategy");
    const subtitleStyle = document.getElementById("subtitle-style");
    const languageList = document.getElementById("language-list");
    
    state.asrModels = opts.asr_models || [];
    state.translationModels = opts.translation_models || [];
    state.ttsModels = opts.tts_models || [];
    
    const codes = new Set();
    [state.asrModels, state.translationModels, state.ttsModels].forEach(group => {
      group.forEach(model => (model.languages || []).forEach(l => codes.add(l)));
    });
    
    const initSourceCode = lang.resolve(el.sourceLang?.value || "");
    const initTargetCode = targetLangs.getPrimary() || initSourceCode;
    
    models.refresh(asrSelect, state.asrModels, initSourceCode);
    models.refresh(trSelect, state.translationModels, initTargetCode);
    models.refresh(ttsSelect, state.ttsModels, initTargetCode);
    
    opts.audio_separation_models.forEach(group => {
      const optGroup = document.createElement("optgroup");
      optGroup.label = group.architecture;
      group.models.forEach(model => {
        optGroup.appendChild(new Option(
          `${model.filename} (${model.stems.join(" + ")})`,
          model.filename
        ));
      });
      sepSelect.appendChild(optGroup);
    });
    
    opts.translation_strategies.forEach(strategy => {
      translationStrategy.appendChild(new Option(strategy, strategy));
    });
    
    opts.dubbing_strategies.forEach(strategy => {
      dubbingStrategy.appendChild(new Option(strategy, strategy));
    });
    
    opts.subtitle_styles.forEach(style => {
      ["Desktop", "Mobile"].forEach(mode => {
        const value = mode === "Mobile" ? `${style}_mobile` : style;
        subtitleStyle.appendChild(new Option(`${style} (${mode})`, value));
      });
    });
    
    const preferredOrder = ["en", "fr"];
    const languageEntries = Array.from(codes).map(code => ({
      code: code.toLowerCase(),
      display: lang.register(code)
    }));
    
    languageEntries.sort((a, b) => {
      const idxA = preferredOrder.indexOf(a.code);
      const idxB = preferredOrder.indexOf(b.code);
      if (idxA !== -1 && idxB !== -1) return idxA - idxB;
      if (idxA !== -1) return -1;
      if (idxB !== -1) return 1;
      return a.display.localeCompare(b.display);
    });
    
    const fillLanguageOptions = (filter = "") => {
      const normalized = filter.trim().toLowerCase();
      const suggestions = normalized
        ? languageEntries.filter(e =>
            e.display.toLowerCase().includes(normalized) || e.code.includes(normalized)
          )
        : languageEntries;
      
      languageList.innerHTML = "";
      suggestions.forEach(e => {
        const opt = document.createElement("option");
        opt.value = e.display;
        opt.dataset.code = e.code;
        languageList.appendChild(opt);
      });
    };
    
    fillLanguageOptions();
    targetLangs.updateModels();
    
    const updateLangSuggestions = inputEl => {
      if (inputEl) fillLanguageOptions(inputEl.value);
    };
    
    if (el.sourceLang) {
      el.sourceLang.oninput = () => {
        updateLangSuggestions(el.sourceLang);
        const sourceCode = lang.resolve(el.sourceLang.value);
        models.refresh(asrSelect, state.asrModels, sourceCode);
        targetLangs.updateModels();
      };
    }
    
    if (el.targetLangInput) {
      el.targetLangInput.oninput = () => updateLangSuggestions(el.targetLangInput);
      
      el.targetLangInput.onkeydown = e => {
        if (["Enter", "Tab", ","].includes(e.key)) {
          const pending = el.targetLangInput.value.trim();
          if (!pending) {
            if (e.key === "Enter" || e.key === ",") e.preventDefault();
            return;
          }
          e.preventDefault();
          targetLangs.add(pending);
        } else if (e.key === "Backspace" && !el.targetLangInput.value && state.targetLangs.length) {
          const last = state.targetLangs[state.targetLangs.length - 1];
          targetLangs.remove(last);
        }
      };
      
      el.targetLangInput.onblur = () => {
        if (el.targetLangInput.value.trim()) {
          targetLangs.add(el.targetLangInput.value);
        }
      };
    }
  },

  handleEvent(event) {
    if (!event?.type) return;
    
    const eventHandlers = {
      run_id: () => {
        state.runId = event.run_id || "";
        if (state.runId) {
          ui.showInterrupt(false);
          ui.log(`🆔 Run started (${state.runId})`);
        }
      },
      
      step: () => {
        if (event.event === "start") {
          ui.log(`▶️ ${event.step}…`);
        } else if (event.event === "end") {
          ui.log(`✅ ${event.step} (${event.duration.toFixed(2)}s)`);
        }
      },
      
      cancelled: () => {
        ui.log("⏹ Run cancelled.");
        ui.setStatus("Cancelled", "error");
        el.results.textContent = "Run cancelled.";
        ui.updateResultPreview(null, "Cancelled.");
        transcriptionReview.hide();
        alignmentReview.hide();
        ttsReview.hide();
        ui.hideInterrupt();
        state.runId = "";
      },
      
      result: () => {
        ui.log("🎉 Pipeline completed.");
        ui.setStatus("Done", "success");
        transcriptionReview.hide();
        alignmentReview.hide();
        ttsReview.hide();
        results.render(event.result);
        
        const tokenVal = event.result?.upload_token || "";
        token.set(tokenVal);
        state.sourceDescriptor = (event.result?.source_media || "").trim();
        if (tokenVal && el.fileInput) el.fileInput.value = "";
        
        if (event.result?.source_video?.url) {
          ui.updateSourcePreview(event.result.source_video.url, "Source (workspace)");
        }
      },
      
      status: () => {
        if (event.event === "download_start") {
          ui.log(`⬇️ Downloading remote media: ${event.url || ""}`);
          ui.updateSourcePreview(null, "Downloading media…", { keepExisting: false });
          ui.updateDownloadProgress({ active: true, label: "Downloading…" });
        } else if (event.event === "download_complete") {
          ui.log("✅ Download complete");
          el.sourcePreview.text.textContent = "Download complete. Preparing source…";
          ui.updateDownloadProgress({ active: false, label: "Download complete." });
        } else if (event.event === "download_progress") {
          if (typeof event.total === "number" && typeof event.downloaded === "number") {
            const pct = Math.round((event.downloaded / event.total) * 100);
            ui.updateDownloadProgress({ active: true, progress: pct, label: `Downloading… ${pct}%` });
          } else {
            ui.updateDownloadProgress({ active: true, progress: null, label: "Downloading…" });
          }
        } else if (event.event === "awaiting_transcription_review") {
          ui.setStatus("Awaiting transcription review", "paused");
        } else if (event.event === "awaiting_alignment_review") {
          ui.setStatus("Awaiting alignment review", "paused");
        } else if (event.event === "awaiting_tts_review") {
          ui.setStatus("Awaiting TTS review", "paused");
        } else {
          ui.log(`ℹ️ ${event.event || "status"}`);
        }
      },
      
      transcription_review: () => {
        ui.log("✏️ Awaiting transcription review.");
        ui.setStatus("Awaiting review", "paused");
        ttsReview.hide();
        transcriptionReview.show(event);
      },
      
      transcription_review_complete: () => {
        ui.log("✅ Transcription review submitted. Resuming pipeline.");
        ui.setStatus("Running", "running");
        transcriptionReview.hide();
      },
      
      alignment_review: () => {
        ui.log("✏️ Awaiting alignment review.");
        ui.setStatus("Awaiting alignment review", "paused");
        transcriptionReview.hide();
        ttsReview.hide();
        alignmentReview.show(event);
      },
      
      alignment_review_complete: () => {
        ui.log("✅ Alignment review submitted. Resuming pipeline.");
        ui.setStatus("Running", "running");
        alignmentReview.hide();
      },
      
      tts_review: () => {
        ui.log("🎧 Awaiting TTS review.");
        ui.setStatus("Awaiting TTS review", "paused");
        transcriptionReview.hide();
        alignmentReview.hide();
        ttsReview.show(event);
      },
      
      tts_review_complete: () => {
        ui.log("✅ TTS review submitted. Resuming pipeline.");
        ui.setStatus("Running", "running");
        ttsReview.hide();
      },
      
      tts_review_regenerated: () => {
        if (event.segment) ttsReview.updateSegment(event.segment);
      },
      
      source_preview: () => {
        if (event.preview?.url) {
          ui.updateSourcePreview(event.preview.url, "Source ready");
          ui.log("🎬 Source preview available.");
        }
      },
      
      error: () => {
        ui.log(`❌ Error: ${event.message || "unknown failure"}`);
        ui.setStatus("Error", "error");
        transcriptionReview.hide();
        alignmentReview.hide();
        ttsReview.hide();
        ui.hideInterrupt();
        state.runId = "";
      },
      
      complete: () => {
        ui.log("🌊 Stream closed.");
        ui.hideInterrupt();
        transcriptionReview.hide();
        alignmentReview.hide();
        ttsReview.hide();
        state.runId = "";
      }
    };
    
    const handler = eventHandlers[event.type];
    if (handler) {
      handler();
    } else {
      ui.log(`ℹ️ ${JSON.stringify(event)}`);
    }
  },

  async submitForm(e) {
    e.preventDefault();
    if (!state.options) {
      ui.log("⚠️ Model registry not loaded.");
      return;
    }
    
    transcriptionReview.hide();
    alignmentReview.hide();
    
    let linkValue = el.videoLink.value.trim();
    let hasFile = el.fileInput.files?.length > 0;
    
    if (linkValue && hasFile) {
      if (state.uploadToken) await token.release();
      el.fileInput.value = "";
      hasFile = false;
    }
    
    if (el.targetLangInput?.value.trim()) {
      targetLangs.add(el.targetLangInput.value);
    }
    
    const formData = new FormData(el.form);
    linkValue = el.videoLink.value.trim();
    const cachedToken = el.reuseToken?.value.trim() || "";
    
    if (!hasFile && !linkValue && !cachedToken) {
      ui.log("⚠️ Provide a media file or a video link.");
      ui.setStatus("Error", "error");
      return;
    }
    
    if (linkValue) {
      formData.set("video_url", linkValue);
      formData.delete("file");
    } else if (!hasFile) {
      formData.delete("video_url");
      formData.delete("file");
    } else {
      formData.delete("video_url");
    }
    
    if (cachedToken && !hasFile && !linkValue) {
      formData.set("reuse_media_token", cachedToken);
    } else {
      formData.delete("reuse_media_token");
    }
    
    const resolvedSource = lang.resolve(formData.get("source_lang"));
    formData.set("source_lang", resolvedSource);
    
    formData.delete("target_langs");
    state.targetLangs.forEach(l => formData.append("target_langs", l));
    const primaryTarget = state.targetLangs[0] || "";
    if (primaryTarget) {
      formData.set("target_lang", primaryTarget);
    } else {
      formData.delete("target_lang");
    }
    
    formData.set("audio_sep", document.getElementById("audio-sep").checked ? "true" : "false");
    formData.set("perform_vad_trimming", document.getElementById("vad-trim").checked ? "true" : "false");
    formData.set("sophisticated_dub_timing", document.getElementById("sophisticated-timing").checked ? "true" : "false");
    formData.set("persist_intermediate", document.getElementById("persist-intermediate").checked ? "true" : "false");
    formData.set("involve_mode", state.involveMode ? "true" : "false");
    
    const normalizeSpeakerField = (field, label) => {
      const raw = formData.get(field);
      if (raw === null) return { value: null, ok: true };
      
      const trimmed = String(raw).trim();
      if (!trimmed) {
        formData.delete(field);
        return { value: null, ok: true };
      }
      
      const parsed = Number(trimmed);
      if (!Number.isInteger(parsed) || parsed < 1) {
        ui.log(`⚠️ ${label} must be a positive integer.`);
        ui.setStatus("Error", "error");
        return { value: null, ok: false };
      }
      
      formData.set(field, String(parsed));
      return { value: parsed, ok: true };
    };
    
    const minHint = normalizeSpeakerField("min_speakers", "Min speakers");
    if (!minHint.ok) return;
    
    const maxHint = normalizeSpeakerField("max_speakers", "Max speakers");
    if (!maxHint.ok) return;
    
    if (minHint.value !== null && maxHint.value !== null && minHint.value > maxHint.value) {
      ui.log("⚠️ Min speakers cannot exceed max speakers.");
      ui.setStatus("Error", "error");
      return;
    }
    
    state.latestResult = null;
    if (el.resultPreview.select) {
      el.resultPreview.select.innerHTML = '<option value="">Processing…</option>';
      el.resultPreview.select.disabled = true;
    }
    
    el.results.textContent = "Processing current run…";
    state.runId = "";
    ui.showInterrupt(true);
    ui.resetLog();
    el.log.textContent = "Starting pipeline…";
    ui.setStatus("Running", "running");
    
    if (!hasFile && linkValue) {
      ui.updateSourcePreview(null, "Downloading media…");
    }
    ui.updateResultPreview(null, "Processing output…");
    el.runBtn.disabled = true;
    
    try {
      const response = await fetch(API.routes.run, {
        method: "POST",
        body: formData
      });
      
      if (!response.ok) throw new Error(`Request failed: ${response.status}`);
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";
        
        for (const block of lines) {
          const line = block.split("\n").find(l => l.startsWith("data:"));
          if (line) {
            const data = JSON.parse(line.slice(5).trim());
            this.handleEvent(data);
          }
        }
      }
      
      if (buffer.trim()) {
        const line = buffer.split("\n").find(l => l.startsWith("data:"));
        if (line) {
          const data = JSON.parse(line.slice(5).trim());
          this.handleEvent(data);
        }
      }
    } catch (err) {
      ui.log(`❌ ${err.message}`);
      ui.setStatus("Error", "error");
    } finally {
      el.runBtn.disabled = false;
      ui.hideInterrupt();
      state.runId = "";
    }
  }
};

export { handlers };
