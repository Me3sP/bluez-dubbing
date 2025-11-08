const getApiBase = () => {
  const stored = (() => {
    try {
      return localStorage?.getItem("bluez-backend-base");
    } catch {
      return null;
    }
  })();

  const fallback = (() => {
    const { protocol, hostname } = window.location;
    if (hostname === "localhost" || hostname === "127.0.0.1") {
      return `${protocol}//${hostname}:8000/api`;
    }
    return `${window.location.origin.replace(/\/$/, "")}/api`;
  })();

  const apiUrl = new URL(window.BLUEZ_API_BASE || stored || fallback, window.location.origin);
  const base = apiUrl.href.replace(/\/$/, "");

  return {
    base,
    origin: apiUrl.origin,
    prefix: apiUrl.pathname.replace(/\/$/, ""),
    routes: {
      options: `${base}/options`,
      run: `${base}/jobs/run`,
      stop: `${base}/jobs/stop`,
      release: `${base}/jobs/release_media`,
      file: `${base}/jobs/file`,
      reviewTranscription: `${base}/jobs/transcription_review`,
      reviewAlignment: `${base}/jobs/alignment_review`,
      reviewTTS: `${base}/jobs/tts_review`,
      regenerateTTS: `${base}/jobs/tts_review/regenerate`
    }
  };
};

export const API = getApiBase();
export { getApiBase };
