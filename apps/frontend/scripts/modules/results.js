import { el } from './dom.js';
import { state } from './state.js';
import { ui } from './ui.js';
import { lang } from './language.js';
import { utils } from './utils.js';

// Results Rendering
const results = {
  render(payload) {
    if (!payload) {
      el.results.textContent = "Pipeline finished without a payload.";
      return;
    }
    
    state.latestResult = payload;
    const {
      languages = {},
      default_language: defaultLang,
      final_video,
      final_audio,
      speech_track,
      subtitles,
      models: usedModels,
      timings,
      workspace_id,
      source_media,
      source_video,
      available_languages = []
    } = payload;

    const renderLink = (item, label) => {
      if (!item?.url) return `<li>${label}: unavailable</li>`;
      const href = utils.resolveUrl(item.url);
      const pathLabel = item.path || href;
      return `<li>${label}: <a href="${href}" target="_blank" rel="noopener">${pathLabel}</a></li>`;
    };
    
    const renderLangSection = (langCode, data) => {
      const label = lang.toLabel(langCode) || langCode;
      const entries = [
        renderLink(data.final_video, "Final video"),
        renderLink(data.final_audio, "Dubbed audio"),
        renderLink(data.speech_track, "Speech track"),
        renderLink(data.subtitles?.aligned?.srt, "Aligned SRT"),
        renderLink(data.subtitles?.aligned?.vtt, "Aligned VTT")
      ];
      return `<li><strong>${label}</strong><ul>${entries.join("")}</ul></li>`;
    };
    
    const langsMarkup = Object.entries(languages)
      .map(([l, d]) => renderLangSection(l, d))
      .join("") || "<li>No target languages processed.</li>";
    
    const modelValue = v => {
      if (!v) return "n/a";
      if (typeof v === "string") return v;
      if (typeof v === "object") {
        return Object.entries(v).map(([k, val]) => `${k}: ${val}`).join(", ") || "n/a";
      }
      return String(v);
    };
    
    const timingsMarkup = Object.entries(timings || {})
      .map(([step, duration]) => `<li>${step}: ${duration.toFixed(2)}s</li>`)
      .join("");
    
    el.results.innerHTML = `
      <p><strong>Workspace:</strong> ${workspace_id || "n/a"}</p>
      <p><strong>Source:</strong> ${source_media || "n/a"}</p>
      <p><strong>Default language:</strong> ${defaultLang ? `${lang.toLabel(defaultLang) || defaultLang} (${defaultLang})` : "n/a"}</p>
      <p><strong>Models:</strong> ASR=${modelValue(usedModels?.asr)}, Translation=${modelValue(usedModels?.translation)}, TTS=${modelValue(usedModels?.tts)}</p>
      <p><strong>Available languages:</strong> ${available_languages.length ? available_languages.join(", ") : "n/a"}</p>
      <ul>
        ${renderLink(source_video, "Source video")}
        ${renderLink(final_video, "Default final video")}
        ${renderLink(final_audio, "Default dubbed audio")}
        ${renderLink(speech_track, "Default speech track")}
      </ul>
      <details open>
        <summary>Per-language outputs</summary>
        <ul>${langsMarkup}</ul>
      </details>
      <details>
        <summary>Subtitles (default)</summary>
        <ul>
          ${renderLink(subtitles?.original?.srt, "Original SRT")}
          ${renderLink(subtitles?.original?.vtt, "Original VTT")}
          ${renderLink(subtitles?.aligned?.srt, "Aligned SRT")}
          ${renderLink(subtitles?.aligned?.vtt, "Aligned VTT")}
        </ul>
      </details>
      <details>
        <summary>Timings</summary>
        <ul>${timingsMarkup}</ul>
      </details>
    `;
    
    ui.updateResultSelector(payload);
  }
};

export { results };
