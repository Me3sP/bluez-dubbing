import { state } from "./state.js";
import { lang } from "./language.js";
import { models } from "./models.js";
import { el } from "./dom.js";

export const targetLangs = {
  add(rawValue) {
    const code = lang.resolve(rawValue)?.toLowerCase();
    if (!code) {
      if (el.targetLangInput) {
        el.targetLangInput.value = "";
        el.targetLangInput.dispatchEvent(new Event("input", { bubbles: true }));
      }
      return;
    }

    if (!state.targetLangs.includes(code)) {
      state.targetLangs.push(code);
      this.render();
      this.updateModels();
    }

    if (el.targetLangInput) {
      el.targetLangInput.value = "";
      el.targetLangInput.dispatchEvent(new Event("input", { bubbles: true }));
    }
  },

  remove(rawValue) {
    const normalized = lang.resolve(rawValue);
    if (!normalized) return;

    const index = state.targetLangs.indexOf(normalized);
    if (index === -1) return;

    state.targetLangs.splice(index, 1);
    this.render();
    this.updateModels();
  },

  render() {
    if (!el.targetLangTags) return;
    el.targetLangTags.innerHTML = "";

    state.targetLangs.forEach(code => {
      const display = lang.register(code);
      const badge = document.createElement("span");
      badge.className = "tag-badge";
      badge.textContent = display;

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.setAttribute("aria-label", `Remove ${display}`);
      removeBtn.textContent = "×";
      removeBtn.onclick = e => {
        e.preventDefault();
        e.stopPropagation();
        this.remove(code);
      };

      const hidden = document.createElement("input");
      hidden.type = "hidden";
      hidden.name = "target_langs";
      hidden.value = code;

      badge.append(removeBtn, hidden);
      el.targetLangTags.appendChild(badge);
    });
  },

  getPrimary() {
    return state.targetLangs[0] || lang.resolve(el.targetLangInput?.value) || "";
  },

  updateModels() {
    const primary = this.getPrimary();
    const sourceCode = lang.resolve(el.sourceLang?.value || "");
    models.refresh(document.getElementById("tr-model"), state.translationModels, primary || sourceCode);
    models.refresh(document.getElementById("tts-model"), state.ttsModels, primary);
  }
};
