import { API } from "./api.js";

export const utils = {
  resolveUrl(url) {
    if (!url || /^(https?:|blob:|data:)/i.test(url)) return url;
    if (url.startsWith("/")) return `${API.origin}${url}`;

    const prefixNoSlash = API.prefix.replace(/^\//, "");
    if (prefixNoSlash && (url === prefixNoSlash || url.startsWith(`${prefixNoSlash}/`))) {
      return new URL(url, `${API.origin}/`).href;
    }
    return new URL(url, `${API.origin}${API.prefix || ""}/`).href;
  },

  clone: obj => structuredClone?.(obj) ?? JSON.parse(JSON.stringify(obj)),

  formatTime(value) {
    if (typeof value !== "number" || Number.isNaN(value)) return "—";
    const total = Math.max(0, value);
    const min = Math.floor(total / 60);
    const sec = Math.floor(total % 60);
    return `${min}:${sec.toString().padStart(2, "0")}`;
  },

  generateId: () => crypto?.randomUUID?.() ?? `seg-${Date.now()}-${Math.random().toString(16).slice(2)}`,

  markValidity(input, isValid, message = "") {
    if (!input) return;
    if (isValid) {
      input.removeAttribute("data-error");
      input.style.borderColor = "";
      input.title = "";
    } else {
      input.setAttribute("data-error", "true");
      input.style.borderColor = "#f87171";
      input.title = message || "Invalid value";
    }
  }
};
