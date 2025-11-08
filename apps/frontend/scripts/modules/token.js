import { state } from "./state.js";
import { API } from "./api.js";
import { el } from "./dom.js";

export const token = {
  set(value = "") {
    state.uploadToken = value || "";
    if (el.reuseToken) el.reuseToken.value = state.uploadToken;
  },

  async release() {
    if (!state.uploadToken) return;

    const tokenValue = state.uploadToken;
    this.set("");
    state.sourceDescriptor = "";

    try {
      const params = new URLSearchParams({ token: tokenValue });
      const url = API.routes.release;

      let sent = false;
      if (navigator.sendBeacon) {
        const blob = new Blob([params.toString()], {
          type: "application/x-www-form-urlencoded"
        });
        sent = navigator.sendBeacon(url, blob);
      }

      if (!sent) {
        await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: params
        });
      }
    } catch (err) {
      console.warn("Failed to release cached media token", err);
    }
  }
};
