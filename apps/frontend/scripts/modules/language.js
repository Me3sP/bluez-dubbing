const displayNames = typeof Intl !== "undefined" && Intl.DisplayNames
  ? new Intl.DisplayNames([navigator.language || "en"], { type: "language" })
  : null;

export const lang = {
  map: new Map(),

  toLabel(code) {
    if (!code) return "";
    const lower = code.toLowerCase();
    const label = displayNames?.of(lower);
    return label ? label.charAt(0).toUpperCase() + label.slice(1) : lower;
  },

  register(code) {
    if (!code) return "";
    const normalized = code.toLowerCase();
    const label = this.toLabel(normalized);
    const display = `${label} (${normalized})`;

    [normalized, label.toLowerCase(), display.toLowerCase()].forEach(key => {
      if (!this.map.has(key)) this.map.set(key, normalized);
    });

    return display;
  },

  resolve(value) {
    if (!value) return "";
    const trimmed = String(value).trim();
    if (!trimmed) return "";

    const match = trimmed.match(/\(([a-zA-Z-]+)\)\s*$/);
    if (match?.[1]) {
      const candidate = match[1].toLowerCase();
      return this.map.get(candidate) || candidate;
    }

    const key = trimmed.toLowerCase();
    return this.map.get(key) || key;
  }
};
