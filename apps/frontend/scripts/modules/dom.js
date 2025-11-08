export const el = {
  log: document.getElementById("log"),
  results: document.getElementById("results"),
  statusBadge: document.getElementById("status-badge"),
  runBtn: document.getElementById("run-btn"),
  interruptBtn: document.getElementById("interrupt-btn"),
  form: document.getElementById("dub-form"),
  sourceLang: document.querySelector("input[name='source_lang']"),
  targetLangInput: document.getElementById("target-lang-input"),
  targetLangTags: document.getElementById("target-lang-tags"),
  videoLink: document.querySelector("input[name='video_url']"),
  fileInput: document.querySelector("input[type='file']"),
  themeToggle: document.getElementById("theme-toggle"),
  sourcePreview: {
    card: document.getElementById("source-preview-card"),
    text: document.getElementById("source-preview-text"),
    video: document.getElementById("source-video")
  },
  resultPreview: {
    card: document.getElementById("result-preview-card"),
    text: document.getElementById("result-preview-text"),
    video: document.getElementById("result-video"),
    select: document.getElementById("result-language-select")
  },
  download: {
    progress: document.getElementById("download-progress"),
    bar: document.getElementById("download-bar"),
    label: document.getElementById("download-label")
  },
  reuseToken: document.getElementById("reuse-media-token"),
  modeToggle: document.getElementById("mode-toggle-btn"),
  transcription: {
    card: document.getElementById("transcription-review"),
    body: document.getElementById("transcription-review-body"),
    status: document.getElementById("transcription-review-status"),
    apply: document.getElementById("transcription-review-apply"),
    skip: document.getElementById("transcription-review-skip"),
    add: document.getElementById("transcription-review-add")
  },
  alignment: {
    card: document.getElementById("alignment-review"),
    body: document.getElementById("alignment-review-body"),
    status: document.getElementById("alignment-review-status"),
    apply: document.getElementById("alignment-review-apply"),
    skip: document.getElementById("alignment-review-skip"),
    add: document.getElementById("alignment-review-add"),
    speakers: document.getElementById("speaker-list")
  },
  tts: {
    card: document.getElementById("tts-review"),
    body: document.getElementById("tts-review-body"),
    status: document.getElementById("tts-review-status"),
    apply: document.getElementById("tts-review-apply"),
    skip: document.getElementById("tts-review-skip")
  }
};
