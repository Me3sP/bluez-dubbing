const logEl = document.getElementById("log");
const resultsEl = document.getElementById("results");
const statusBadge = document.getElementById("status-badge");
const runBtn = document.getElementById("run-btn");
const interruptBtn = document.getElementById("interrupt-btn");
const formEl = document.getElementById("dub-form");
const sourceLangInput = document.querySelector("input[name='source_lang']");
const targetLangInput = document.getElementById("target-lang-input");
const targetLangTags = document.getElementById("target-lang-tags");
const targetLangField = document.getElementById("target-lang-field");
const videoLinkInput = document.querySelector("input[name='video_url']");
const fileInput = document.querySelector("input[type='file']");
const themeToggle = document.getElementById("theme-toggle");
const themeLabel = document.querySelector("#theme-toggle .theme-label");
const sourcePreviewCard = document.getElementById("source-preview-card");
const sourcePreviewText = document.getElementById("source-preview-text");
const sourceVideoEl = document.getElementById("source-video");
const resultPreviewCard = document.getElementById("result-preview-card");
const resultPreviewText = document.getElementById("result-preview-text");
const resultVideoEl = document.getElementById("result-video");
const resultLanguageSelect = document.getElementById("result-language-select");
const downloadProgress = document.getElementById("download-progress");
const downloadBar = document.getElementById("download-bar");
const downloadLabel = document.getElementById("download-label");
const reuseMediaTokenInput = document.getElementById("reuse-media-token");
const involveToggleBtn = document.getElementById("mode-toggle-btn");
const transcriptionReviewCard = document.getElementById("transcription-review");
const transcriptionReviewBody = document.getElementById("transcription-review-body");
const transcriptionReviewStatus = document.getElementById("transcription-review-status");
const transcriptionReviewApplyBtn = document.getElementById("transcription-review-apply");
const transcriptionReviewSkipBtn = document.getElementById("transcription-review-skip");
const transcriptionReviewAddBtn = document.getElementById("transcription-review-add");
const alignmentReviewCard = document.getElementById("alignment-review");
const alignmentReviewBody = document.getElementById("alignment-review-body");
const alignmentReviewStatus = document.getElementById("alignment-review-status");
const alignmentReviewApplyBtn = document.getElementById("alignment-review-apply");
const alignmentReviewSkipBtn = document.getElementById("alignment-review-skip");
const alignmentReviewAddBtn = document.getElementById("alignment-review-add");
const alignmentSpeakerList = document.getElementById("speaker-list");
const ttsReviewCard = document.getElementById("tts-review");
const ttsReviewBody = document.getElementById("tts-review-body");
const ttsReviewStatus = document.getElementById("tts-review-status");
const ttsReviewApplyBtn = document.getElementById("tts-review-apply");
const ttsReviewSkipBtn = document.getElementById("tts-review-skip");

let optionsCache = null;
let asrModels = [];
let translationModels = [];
let ttsModels = [];
let activeUploadToken = "";
let currentSourceDescriptor = "";
let activeRunId = "";
let selectedTargetLangs = [];
let latestResultPayload = null;
let involveMode = false;
let pendingTranscriptionReview = null;
let pendingAlignmentReview = null;
let pendingTTSReview = null;
function readStoredBackendBase() {
    try {
        return window.localStorage?.getItem("bluez-backend-base") || null;
    } catch (_err) {
        return null;
    }
}

const fallbackBackendBase = (() => {
    const { protocol, hostname } = window.location;
    if (hostname === "localhost" || hostname === "127.0.0.1") {
        return `${protocol}//${hostname}:8000/api`;
    }
    return `${window.location.origin.replace(/\/$/, "")}/api`;
})();

const apiBaseCandidate = window.BLUEZ_API_BASE || readStoredBackendBase() || fallbackBackendBase;
const apiBaseRaw = String(apiBaseCandidate || "/api");
const apiUrl = new URL(apiBaseRaw, window.location.origin);
const API_BASE = apiUrl.href.replace(/\/$/, "");
const BACKEND_ORIGIN = apiUrl.origin;
const API_PATH_PREFIX = apiUrl.pathname.replace(/\/$/, "");
const API_PATH_PREFIX_NO_SLASH = API_PATH_PREFIX.replace(/^\//, "");
const JOBS_BASE = `${API_BASE}/jobs`;
const ROUTES = {
    options: `${API_BASE}/options`,
    run: `${JOBS_BASE}/run`,
    stop: `${JOBS_BASE}/stop`,
    release: `${JOBS_BASE}/release_media`,
    file: `${JOBS_BASE}/file`,
    reviewTranscription: `${JOBS_BASE}/transcription_review`,
    reviewAlignment: `${JOBS_BASE}/alignment_review`,
    reviewTTS: `${JOBS_BASE}/tts_review`,
    regenerateTTS: `${JOBS_BASE}/tts_review/regenerate`,
};

function resolveBackendUrl(url) {
    if (!url) return "";
    if (/^(https?:|blob:|data:)/i.test(url)) {
        return url;
    }
    if (url.startsWith("/")) {
        return `${BACKEND_ORIGIN}${url}`;
    }
    const trimmed = url.replace(/^\/+/, "");
    if (API_PATH_PREFIX_NO_SLASH && (trimmed === API_PATH_PREFIX_NO_SLASH || trimmed.startsWith(`${API_PATH_PREFIX_NO_SLASH}/`))) {
        return `${BACKEND_ORIGIN}/${trimmed}`;
    }
    const joined = API_PATH_PREFIX
        ? `${API_PATH_PREFIX}/${trimmed}`
        : `/${trimmed}`;
    return `${BACKEND_ORIGIN}${joined}`;
}

const languageMap = new Map();
const languageDisplay = typeof Intl !== "undefined" && Intl.DisplayNames
    ? new Intl.DisplayNames([navigator.language || "en"], { type: "language" })
    : null;
let sourceObjectUrl = null;
let resultObjectUrl = null;

function toLanguageLabel(code) {
    if (!code) return "";
    const lower = code.toLowerCase();
    const label = languageDisplay ? languageDisplay.of(lower) : null;
    if (!label) return lower;
    return label.charAt(0).toUpperCase() + label.slice(1);
}

function registerLanguage(code) {
    if (!code) return "";
    const normalized = code.toLowerCase();
    const label = toLanguageLabel(normalized);
    const display = `${label} (${normalized})`;
    const keys = [
        normalized,
        label.toLowerCase(),
        display.toLowerCase(),
        `${label} (${normalized.toUpperCase()})`.toLowerCase(),
    ];
    keys.forEach(key => {
        if (!languageMap.has(key)) {
            languageMap.set(key, normalized);
        }
    });
    return display;
}

function resolveLanguageCode(value) {
    if (!value) return "";
    const strValue = typeof value === "string" ? value : String(value);
    const trimmed = strValue.trim();
    if (!trimmed) return "";
    const match = trimmed.match(/\(([a-zA-Z-]+)\)\s*$/);
    if (match && match[1]) {
        const candidate = match[1].toLowerCase();
        if (languageMap.has(candidate)) {
            return languageMap.get(candidate);
        }
        return candidate;
    }
    const key = trimmed.toLowerCase();
    if (languageMap.has(key)) {
        return languageMap.get(key);
    }
    return key;
}

function refreshModelSelect(select, models, languageCode) {
    if (!select) return;
    const previous = select.value;
    select.innerHTML = "";
    const autoOption = document.createElement("option");
    autoOption.value = "auto";
    autoOption.textContent = "Auto";
    select.appendChild(autoOption);

    models.forEach(model => {
        const supportsLanguage =
            !languageCode ||
            !(model.languages || []).length ||
            model.languages.includes(languageCode);
        if (!supportsLanguage) {
            return;
        }
        const option = document.createElement("option");
        option.value = model.key;
        option.textContent = model.key;
        select.appendChild(option);
    });

    if ([...select.options].some(opt => opt.value === previous)) {
        select.value = previous;
    }
}

function getPrimaryTargetLang() {
    if (selectedTargetLangs.length > 0) {
        return selectedTargetLangs[0];
    }
    if (targetLangInput) {
        return resolveLanguageCode(targetLangInput.value);
    }
    return "";
}

function renderTargetLanguageTags() {
    if (!targetLangTags) return;
    targetLangTags.innerHTML = "";
    selectedTargetLangs.forEach(code => {
        const display = registerLanguage(code);
        const badge = document.createElement("span");
        badge.className = "tag-badge";
        badge.textContent = display;
        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.setAttribute("aria-label", `Remove ${display}`);
        removeBtn.textContent = "×";
        removeBtn.addEventListener("click", event => {
            event.preventDefault();
            event.stopPropagation();
            removeTargetLanguage(code);
        });
        const hidden = document.createElement("input");
        hidden.type = "hidden";
        hidden.name = "target_langs";
        hidden.value = code;
        badge.appendChild(removeBtn);
        badge.appendChild(hidden);
        targetLangTags.appendChild(badge);
    });
}

function removeTargetLanguage(rawValue) {
    const normalized = resolveLanguageCode(rawValue);
    if (!normalized) {
        return;
    }
    const index = selectedTargetLangs.indexOf(normalized);
    if (index === -1) {
        return;
    }
    selectedTargetLangs.splice(index, 1);
    renderTargetLanguageTags();
    updateModelSelectorsForTarget();
}

function addTargetLanguage(rawValue) {
    const code = resolveLanguageCode(rawValue);
    if (!code) {
        if (targetLangInput) {
            targetLangInput.value = "";
            targetLangInput.dispatchEvent(new Event("input", { bubbles: true }));
        }
        return;
    }
    const normalized = code.toLowerCase();
    if (!selectedTargetLangs.includes(normalized)) {
        selectedTargetLangs.push(normalized);
        renderTargetLanguageTags();
        updateModelSelectorsForTarget();
    }
    if (targetLangInput) {
        targetLangInput.value = "";
        targetLangInput.dispatchEvent(new Event("input", { bubbles: true }));
    }
}

function updateModelSelectorsForTarget() {
    const primary = getPrimaryTargetLang();
    const sourceCode = resolveLanguageCode(sourceLangInput ? sourceLangInput.value : "");
    const trSelect = document.getElementById("tr-model");
    const ttsSelect = document.getElementById("tts-model");
    refreshModelSelect(trSelect, translationModels, primary || sourceCode);
    refreshModelSelect(ttsSelect, ttsModels, primary);
}

function setInvolveMode(enabled) {
    involveMode = Boolean(enabled);
    if (!involveToggleBtn) return;
    involveToggleBtn.classList.toggle("active", involveMode);
    involveToggleBtn.setAttribute("aria-pressed", involveMode ? "true" : "false");
    involveToggleBtn.textContent = involveMode ? "Involve mode: On" : "Involve mode: Off";
}

function clonePayload(payload) {
    if (typeof structuredClone === "function") {
        return structuredClone(payload);
    }
    return JSON.parse(JSON.stringify(payload));
}

function formatTimestamp(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "—";
    }
    const total = Math.max(0, value);
    const minutes = Math.floor(total / 60);
    const seconds = Math.floor(total % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function createSegmentId() {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
        return crypto.randomUUID();
    }
    return `seg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function markInputValidity(input, isValid, message = "") {
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

function hideTranscriptionReview() {
    pendingTranscriptionReview = null;
    if (transcriptionReviewBody) {
        transcriptionReviewBody.innerHTML = '<p class="review-instructions">Review and adjust each segment before continuing.</p>';
    }
    if (transcriptionReviewStatus) {
        transcriptionReviewStatus.textContent = "Pipeline paused for transcription review.";
    }
    if (transcriptionReviewApplyBtn) {
        transcriptionReviewApplyBtn.disabled = false;
    }
    if (transcriptionReviewSkipBtn) {
        transcriptionReviewSkipBtn.disabled = false;
    }
    if (transcriptionReviewAddBtn) {
        transcriptionReviewAddBtn.disabled = false;
        transcriptionReviewAddBtn.hidden = true;
    }
    if (transcriptionReviewCard) {
        transcriptionReviewCard.hidden = true;
    }
}

function showTranscriptionReview(event) {
    if (!transcriptionReviewCard || !transcriptionReviewBody) {
        return;
    }
    const rawPayload = event?.raw || {};
    const runId = event?.run_id || "";
    const durationValue = typeof event?.duration === "number" ? event.duration : null;
    const toleranceValue =
        typeof event?.tolerance === "number" && Number.isFinite(event.tolerance) ? event.tolerance : 0.25;
    let supportedLangs = Array.isArray(event?.languages)
        ? Array.from(
              new Set(
                  event.languages
                      .map(lang => (typeof lang === "string" ? lang.trim().toLowerCase() : ""))
                      .filter(Boolean),
              ),
          )
        : [];
    supportedLangs.forEach(code => registerLanguage(code));
    supportedLangs = supportedLangs.sort((a, b) => {
        const labelA = toLanguageLabel(a) || a;
        const labelB = toLanguageLabel(b) || b;
        return labelA.localeCompare(labelB);
    });
    pendingTranscriptionReview = {
        runId,
        payload: clonePayload(rawPayload),
        duration: durationValue,
        languages: supportedLangs,
        tolerance: toleranceValue,
    };

    if (!Array.isArray(pendingTranscriptionReview.payload.segments)) {
        pendingTranscriptionReview.payload.segments = [];
    }
    pendingTranscriptionReview.payload.segments.forEach(segment => {
        if (!segment.segment_id) {
            segment.segment_id = createSegmentId();
        }
        const start = Number(segment.start);
        segment.start = Number.isFinite(start) ? start : 0;
        const end = Number(segment.end);
        segment.end = Number.isFinite(end) ? end : segment.start;
        segment.lang = segment.lang || "";
        segment.text = segment.text || "";
        if (segment.lang) {
            registerLanguage(segment.lang);
        }
    });

    renderTranscriptionSegments();

    if (transcriptionReviewStatus) {
        const durationLabel =
            typeof durationValue === "number" && Number.isFinite(durationValue)
                ? ` Audio length: ${durationValue.toFixed(3)}s (±${toleranceValue.toFixed(2)} s tolerance).`
                : "";
        transcriptionReviewStatus.textContent = `Pipeline paused for transcription review.${durationLabel}`;
    }
    if (transcriptionReviewApplyBtn) {
        transcriptionReviewApplyBtn.disabled = false;
    }
    if (transcriptionReviewSkipBtn) {
        transcriptionReviewSkipBtn.disabled = false;
    }
    if (transcriptionReviewAddBtn) {
        transcriptionReviewAddBtn.disabled = false;
        transcriptionReviewAddBtn.hidden = false;
    }
    transcriptionReviewCard.hidden = false;
    transcriptionReviewCard.scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderTranscriptionSegments() {
    if (!pendingTranscriptionReview || !transcriptionReviewBody) {
        return;
    }
    const { payload, duration, languages, tolerance: toleranceRaw } = pendingTranscriptionReview;
    const tolerance = typeof toleranceRaw === "number" && Number.isFinite(toleranceRaw) ? toleranceRaw : 0;
    const segments = Array.isArray(payload.segments) ? payload.segments : [];

    transcriptionReviewBody.innerHTML = "";
    const intro = document.createElement("p");
    intro.className = "review-instructions";
    const durationLabel =
        typeof duration === "number" && Number.isFinite(duration)
            ? `Total audio duration: ${duration.toFixed(3)}s (±${tolerance.toFixed(2)} s tolerance).`
            : "Adjust segment timings, languages, and text as needed.";
    intro.textContent = `Adjust segment timings, languages, and text. ${durationLabel}`;
    transcriptionReviewBody.appendChild(intro);

    if (!segments.length) {
        const emptyNote = document.createElement("p");
        emptyNote.className = "review-instructions";
        emptyNote.textContent = "No segments available. Use “Add segment” to create one.";
        transcriptionReviewBody.appendChild(emptyNote);
        return;
    }

    segments.forEach((segment, index) => {
        const wrapper = document.createElement("div");
        wrapper.className = "review-segment";
        wrapper.dataset.segmentId = segment.segment_id || "";
        wrapper.dataset.index = String(index);

        const startValue = Number(segment.start) || 0;
        const endValue = Number(segment.end) || 0;
        const langDisplay = segment.lang ? ` · ${toLanguageLabel(segment.lang) || segment.lang}` : "";
        const meta = document.createElement("div");
        meta.className = "segment-meta";
        const preciseLabel = `${startValue.toFixed(3)}s – ${endValue.toFixed(3)}s`;
        meta.textContent = `Segment ${index + 1} (${formatTimestamp(startValue)} – ${formatTimestamp(endValue)} · ${preciseLabel})${langDisplay}`;
        wrapper.appendChild(meta);

        const startLabel = document.createElement("label");
        startLabel.textContent = "Start (s)";
        const startInput = document.createElement("input");
        startInput.type = "number";
        startInput.step = "0.001";
        startInput.min = "0";
        startInput.dataset.role = "segment-start";
        if (typeof duration === "number" && Number.isFinite(duration) && duration > 0) {
            startInput.max = String((duration + tolerance).toFixed(3));
        }
        startInput.value = Number.isFinite(startValue) ? String(startValue) : "0";
        startLabel.appendChild(startInput);

        const endLabel = document.createElement("label");
        endLabel.textContent = "End (s)";
        const endInput = document.createElement("input");
        endInput.type = "number";
        endInput.step = "0.001";
        endInput.min = "0";
        endInput.dataset.role = "segment-end";
        if (typeof duration === "number" && Number.isFinite(duration) && duration > 0) {
            endInput.max = String((duration + tolerance).toFixed(3));
        }
        endInput.value = Number.isFinite(endValue) ? String(endValue) : startInput.value;
        endLabel.appendChild(endInput);

        const langLabel = document.createElement("label");
        langLabel.textContent = "Language";
        let langControl;
        if (Array.isArray(languages) && languages.length) {
            langControl = document.createElement("select");
            langControl.dataset.role = "segment-lang";
            const autoOption = document.createElement("option");
            autoOption.value = "";
            autoOption.textContent = "Auto";
            langControl.appendChild(autoOption);
            languages.forEach(code => {
                const option = document.createElement("option");
                option.value = code;
                option.textContent = toLanguageLabel(code) || code;
                langControl.appendChild(option);
            });
            const normalized = (segment.lang || "").toLowerCase();
            langControl.value = normalized && languages.includes(normalized) ? normalized : "";
        } else {
            langControl = document.createElement("input");
            langControl.type = "text";
            langControl.placeholder = "auto";
            langControl.dataset.role = "segment-lang";
            langControl.setAttribute("list", "language-list");
            langControl.value = segment.lang || "";
        }
        langLabel.appendChild(langControl);

        const textLabel = document.createElement("label");
        textLabel.textContent = "Text";
        const textarea = document.createElement("textarea");
        textarea.dataset.role = "segment-text";
        textarea.rows = Math.max(2, Math.min(6, Math.ceil(((segment.text || "").length || 1) / 60)));
        textarea.value = segment.text || "";
        textLabel.appendChild(textarea);

        const actions = document.createElement("div");
        actions.className = "segment-actions";

        const addBeforeBtn = document.createElement("button");
        addBeforeBtn.type = "button";
        addBeforeBtn.className = "secondary";
        addBeforeBtn.textContent = "Add before";
        addBeforeBtn.addEventListener("click", () => addTranscriptionSegment(index - 1));

        const addAfterBtn = document.createElement("button");
        addAfterBtn.type = "button";
        addAfterBtn.className = "secondary";
        addAfterBtn.textContent = "Add after";
        addAfterBtn.addEventListener("click", () => addTranscriptionSegment(index));

        const deleteBtn = document.createElement("button");
        deleteBtn.type = "button";
        deleteBtn.className = "ghost";
        deleteBtn.textContent = "Delete";
        deleteBtn.addEventListener("click", () => removeTranscriptionSegment(index));

        actions.appendChild(addBeforeBtn);
        actions.appendChild(addAfterBtn);
        actions.appendChild(deleteBtn);

        wrapper.appendChild(startLabel);
        wrapper.appendChild(endLabel);
        wrapper.appendChild(langLabel);
        wrapper.appendChild(textLabel);
        wrapper.appendChild(actions);
        transcriptionReviewBody.appendChild(wrapper);
    });
}

function addTranscriptionSegment(afterIndex) {
    if (!pendingTranscriptionReview) {
        return;
    }
    const segments = Array.isArray(pendingTranscriptionReview.payload.segments)
        ? pendingTranscriptionReview.payload.segments
        : [];
    const duration = typeof pendingTranscriptionReview.duration === "number" ? pendingTranscriptionReview.duration : null;
    const tolerance =
        typeof pendingTranscriptionReview.tolerance === "number" && Number.isFinite(pendingTranscriptionReview.tolerance)
            ? pendingTranscriptionReview.tolerance
            : 0;
    const prev = afterIndex >= 0 ? segments[afterIndex] : null;
    const next = segments[afterIndex + 1] || null;
    const prevEnd = prev ? Number(prev.end ?? prev.start ?? 0) : 0;
    const nextStartRaw = next ? Number(next.start ?? prevEnd) : duration;
    const gap = Number.isFinite(nextStartRaw) ? Math.max(0, nextStartRaw - prevEnd) : 1;
    let start = Number.isFinite(prevEnd) ? prevEnd : 0;
    if (duration !== null) {
        start = Math.min(Math.max(0, start), duration + tolerance);
    }
    let end = start + (gap > 0 ? Math.min(1, gap) : 1);
    if (duration !== null) {
        end = Math.min(end, duration + tolerance);
    }
    if (end <= start) {
        end = duration !== null ? Math.min(duration + tolerance, start + 0.5) : start + 0.5;
    }
    const newSegment = {
        segment_id: createSegmentId(),
        start: Number.parseFloat(start.toFixed(3)),
        end: Number.parseFloat(end.toFixed(3)),
        text: "",
        lang: "",
        speaker_id: null,
        words: null,
    };
    segments.splice(Math.max(afterIndex + 1, 0), 0, newSegment);
    renderTranscriptionSegments();
}

function removeTranscriptionSegment(index) {
    if (!pendingTranscriptionReview) {
        return;
    }
    const segments = Array.isArray(pendingTranscriptionReview.payload.segments)
        ? pendingTranscriptionReview.payload.segments
        : [];
    if (index < 0 || index >= segments.length) {
        return;
    }
    segments.splice(index, 1);
    renderTranscriptionSegments();
}

async function submitTranscriptionReview(applyChanges) {
    if (!pendingTranscriptionReview) {
        return false;
    }
    const { runId, payload, duration } = pendingTranscriptionReview;
    if (!runId) {
        return false;
    }

    const submission = clonePayload(payload);
    if (!Array.isArray(submission.segments)) {
        submission.segments = [];
    }

    if (applyChanges && transcriptionReviewBody) {
        const baseSegments = Array.isArray(payload.segments) ? payload.segments : [];
        const durationLimit = typeof duration === "number" && Number.isFinite(duration) ? duration : null;
        const tolerance =
            typeof pendingTranscriptionReview.tolerance === "number" && Number.isFinite(pendingTranscriptionReview.tolerance)
                ? pendingTranscriptionReview.tolerance
                : 0;
        const updatedSegments = [];
        let hasError = false;
        const errorMessages = new Set();

        transcriptionReviewBody.querySelectorAll(".review-segment").forEach(segmentEl => {
            const segmentId = segmentEl.dataset.segmentId || createSegmentId();
            const startInput = segmentEl.querySelector('[data-role="segment-start"]');
            const endInput = segmentEl.querySelector('[data-role="segment-end"]');
            const langControl = segmentEl.querySelector('[data-role="segment-lang"]');
            const textArea = segmentEl.querySelector('[data-role="segment-text"]');

            const startValueRaw = startInput ? Number.parseFloat(startInput.value) : 0;
            const endValueRaw = endInput ? Number.parseFloat(endInput.value) : startValueRaw;

            const startValid =
                Number.isFinite(startValueRaw) &&
                startValueRaw >= -tolerance &&
                (durationLimit === null || startValueRaw <= durationLimit + tolerance);
            const endValid =
                Number.isFinite(endValueRaw) &&
                endValueRaw >= -tolerance &&
                (durationLimit === null || endValueRaw <= durationLimit + tolerance) &&
                Number.isFinite(startValueRaw) &&
                endValueRaw >= startValueRaw;

            let segmentHasError = false;
            if (!startValid) {
                hasError = true;
                segmentHasError = true;
                errorMessages.add(`Segment start times must stay within the audio duration (±${tolerance.toFixed(2)} s).`);
            }
            if (!endValid) {
                hasError = true;
                segmentHasError = true;
                errorMessages.add(`Segment end times must stay within the audio duration (±${tolerance.toFixed(2)} s) and ≥ start.`);
            }
            markInputValidity(
                startInput,
                startValid,
                `Start must be within audio duration (±${tolerance.toFixed(2)} s).`,
            );
            markInputValidity(
                endInput,
                endValid,
                `End must be within audio duration (±${tolerance.toFixed(2)} s) and ≥ start.`,
            );

            const textValue = textArea ? textArea.value.trim() : "";
            const langRaw = langControl ? langControl.value.toString().trim() : "";
            const resolvedLang = resolveLanguageCode(langRaw) || langRaw;
            const langValue = resolvedLang ? resolvedLang.toLowerCase() : "";

            const reference = baseSegments.find(seg => seg.segment_id === segmentId) || {};

            if (!segmentHasError) {
                const clampedStart =
                    durationLimit === null
                        ? Math.max(0, startValueRaw)
                        : Math.min(Math.max(0, startValueRaw), durationLimit + tolerance);
                const clampedEnd =
                    durationLimit === null
                        ? Math.max(clampedStart, endValueRaw)
                        : Math.min(Math.max(clampedStart, endValueRaw), durationLimit + tolerance);
                updatedSegments.push({
                    segment_id: segmentId,
                    start: Number.parseFloat((clampedStart || 0).toFixed(3)),
                    end: Number.parseFloat((clampedEnd || clampedStart || 0).toFixed(3)),
                    text: textValue,
                    lang: langValue,
                    speaker_id: reference.speaker_id || null,
                    words: null,
                });
            }
        });

        if (hasError) {
            if (transcriptionReviewStatus) {
                transcriptionReviewStatus.textContent = `Fix highlighted segment issues: ${Array.from(errorMessages).join(" ")}`;
            }
            if (transcriptionReviewApplyBtn) {
                transcriptionReviewApplyBtn.disabled = false;
            }
            if (transcriptionReviewSkipBtn) {
                transcriptionReviewSkipBtn.disabled = false;
            }
            if (transcriptionReviewAddBtn) {
                transcriptionReviewAddBtn.disabled = false;
            }
            return false;
        }

        updatedSegments.sort((a, b) => a.start - b.start);
        submission.segments = updatedSegments;
    }

    if (transcriptionReviewApplyBtn) {
        transcriptionReviewApplyBtn.disabled = true;
    }
    if (transcriptionReviewSkipBtn) {
        transcriptionReviewSkipBtn.disabled = true;
    }
    if (transcriptionReviewAddBtn) {
        transcriptionReviewAddBtn.disabled = true;
    }
    if (transcriptionReviewStatus) {
        transcriptionReviewStatus.textContent = "Submitting review…";
    }

    try {
        const response = await fetch(ROUTES.reviewTranscription, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ run_id: runId, transcription: submission }),
        });
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }
        appendLog(applyChanges ? "📝 Submitted transcription corrections." : "👍 Approved transcription without changes.");
        hideTranscriptionReview();
        setStatus("Running", "running");
        return true;
    } catch (err) {
        if (transcriptionReviewStatus) {
            transcriptionReviewStatus.textContent = `Failed to submit review: ${err.message}`;
        }
        if (transcriptionReviewApplyBtn) {
            transcriptionReviewApplyBtn.disabled = false;
        }
        if (transcriptionReviewSkipBtn) {
            transcriptionReviewSkipBtn.disabled = false;
        }
        if (transcriptionReviewAddBtn) {
            transcriptionReviewAddBtn.disabled = false;
        }
        appendLog(`❌ Failed to submit transcription review: ${err.message}`);
        return false;
    }
}

function collectSpeakersFromSegments(segments) {
    const set = new Set();
    (segments || []).forEach(segment => {
        if (segment && segment.speaker_id) {
            set.add(segment.speaker_id);
        }
    });
    return Array.from(set).sort();
}

function normalizeSpeakerList(...lists) {
    const set = new Set();
    lists.flat().forEach(value => {
        if (Array.isArray(value)) {
            value.forEach(item => {
                const trimmed = typeof item === "string" ? item.trim() : "";
                if (trimmed) {
                    set.add(trimmed);
                }
            });
        } else {
            const trimmed = typeof value === "string" ? value.trim() : "";
            if (trimmed) {
                set.add(trimmed);
            }
        }
    });
    return Array.from(set).sort();
}

function updateAlignmentSpeakerOptions(speakers) {
    if (!alignmentSpeakerList) return;
    alignmentSpeakerList.innerHTML = "";
    (speakers || []).forEach(name => {
        const option = document.createElement("option");
        option.value = name;
        alignmentSpeakerList.appendChild(option);
    });
}

function refreshAlignmentSpeakerList(extra = []) {
    if (!pendingAlignmentReview) return;
    const fromSegments = collectSpeakersFromSegments(pendingAlignmentReview.payload?.segments || []);
    const current = pendingAlignmentReview.speakers || [];
    const normalized = normalizeSpeakerList(current, fromSegments, extra);
    pendingAlignmentReview.speakers = normalized;
    updateAlignmentSpeakerOptions(normalized);
}

function addAlignmentSpeakerName(name) {
    if (!pendingAlignmentReview) return;
    const value = (name || "").trim();
    if (!value) {
        return;
    }
    refreshAlignmentSpeakerList([value]);
    renderAlignmentReview();
}

function hideAlignmentReview() {
    pendingAlignmentReview = null;
    if (alignmentReviewBody) {
        alignmentReviewBody.innerHTML = '<p class="review-instructions">Adjust segment timings, text, and speakers. You can add, delete, or merge segments as needed.</p>';
    }
    if (alignmentReviewStatus) {
        alignmentReviewStatus.textContent = "Pipeline paused for alignment review.";
    }
    if (alignmentReviewApplyBtn) {
        alignmentReviewApplyBtn.disabled = false;
    }
    if (alignmentReviewSkipBtn) {
        alignmentReviewSkipBtn.disabled = false;
    }
    if (alignmentReviewAddBtn) {
        alignmentReviewAddBtn.disabled = false;
    }
    if (alignmentReviewCard) {
        alignmentReviewCard.hidden = true;
    }
    if (alignmentSpeakerList) {
        alignmentSpeakerList.innerHTML = "";
    }
}

function renderAlignmentReview() {
    if (!pendingAlignmentReview || !alignmentReviewBody) {
        return;
    }

    const { payload } = pendingAlignmentReview;
    const segments = Array.isArray(payload.segments) ? payload.segments : [];

    alignmentReviewBody.innerHTML = "";

    const intro = document.createElement("p");
    intro.className = "review-instructions";
    intro.textContent = "Adjust segment timings, text, and speakers. You can add, delete, or merge segments as needed.";
    alignmentReviewBody.appendChild(intro);

    const speakerControls = document.createElement("div");
    speakerControls.className = "review-speaker-controls";
    const speakerInput = document.createElement("input");
    speakerInput.type = "text";
    speakerInput.placeholder = "Add new speaker label";
    speakerInput.setAttribute("list", "speaker-list");
    const speakerAddBtn = document.createElement("button");
    speakerAddBtn.type = "button";
    speakerAddBtn.className = "secondary";
    speakerAddBtn.textContent = "Add speaker";
    speakerAddBtn.addEventListener("click", () => {
        addAlignmentSpeakerName(speakerInput.value);
        speakerInput.value = "";
    });
    speakerInput.addEventListener("keydown", event => {
        if (event.key === "Enter") {
            event.preventDefault();
            addAlignmentSpeakerName(speakerInput.value);
            speakerInput.value = "";
        }
    });
    speakerControls.appendChild(speakerInput);
    speakerControls.appendChild(speakerAddBtn);
    alignmentReviewBody.appendChild(speakerControls);

    if (!segments.length) {
        const notice = document.createElement("p");
        notice.className = "review-instructions";
        notice.textContent = "No aligned segments available. Add segments below if needed.";
        alignmentReviewBody.appendChild(notice);
        return;
    }

    segments.forEach((segment, index) => {
        const wrapper = document.createElement("div");
        wrapper.className = "review-segment alignment";
        wrapper.dataset.index = String(index);

        const meta = document.createElement("div");
        meta.className = "segment-meta";
        meta.textContent = `Segment ${index + 1} (${formatTimestamp(segment.start)} – ${formatTimestamp(segment.end)})`;

        const timingRow = document.createElement("div");
        timingRow.className = "flex";

        const startLabel = document.createElement("label");
        startLabel.textContent = "Start";
        const startInput = document.createElement("input");
        startInput.type = "number";
        startInput.step = "0.01";
        startInput.value = segment.start ?? 0;
        startInput.addEventListener("change", () => {
            updateAlignmentSegmentTime(index, "start", Number(startInput.value), meta);
        });
        startLabel.appendChild(startInput);

        const endLabel = document.createElement("label");
        endLabel.textContent = "End";
        const endInput = document.createElement("input");
        endInput.type = "number";
        endInput.step = "0.01";
        endInput.value = segment.end ?? segment.start ?? 0;
        endInput.addEventListener("change", () => {
            updateAlignmentSegmentTime(index, "end", Number(endInput.value), meta);
        });
        endLabel.appendChild(endInput);

        const speakerLabel = document.createElement("label");
        speakerLabel.textContent = "Speaker";
        const speakerSelect = document.createElement("select");
        speakerSelect.className = "speaker-select";
        const addOption = (value, label = value) => {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = label;
            speakerSelect.appendChild(option);
        };
        addOption("", "Unassigned");
        const knownSpeakers = pendingAlignmentReview.speakers || [];
        knownSpeakers.forEach(name => addOption(name));
        if (segment.speaker_id && !knownSpeakers.includes(segment.speaker_id)) {
            addOption(segment.speaker_id);
        }
        speakerSelect.value = segment.speaker_id || "";
        speakerSelect.addEventListener("change", () => {
            updateAlignmentSegmentSpeaker(index, speakerSelect.value);
        });
        speakerLabel.appendChild(speakerSelect);

        timingRow.appendChild(startLabel);
        timingRow.appendChild(endLabel);
        timingRow.appendChild(speakerLabel);

        const textArea = document.createElement("textarea");
        textArea.value = segment.text || "";
        textArea.addEventListener("input", () => {
            updateAlignmentSegmentText(index, textArea.value);
        });

        const controls = document.createElement("div");
        controls.className = "review-actions";

        const deleteBtn = document.createElement("button");
        deleteBtn.type = "button";
        deleteBtn.className = "secondary";
        deleteBtn.textContent = "Delete segment";
        deleteBtn.disabled = segments.length <= 1;
        deleteBtn.addEventListener("click", () => removeAlignmentSegment(index));

        const mergePrevBtn = document.createElement("button");
        mergePrevBtn.type = "button";
        mergePrevBtn.className = "secondary";
        mergePrevBtn.textContent = "Merge with previous";
        mergePrevBtn.disabled = index === 0;
        mergePrevBtn.addEventListener("click", () => mergeAlignmentSegment(index - 1));

        const mergeBtn = document.createElement("button");
        mergeBtn.type = "button";
        mergeBtn.className = "secondary";
        mergeBtn.textContent = "Merge with next";
        mergeBtn.disabled = index >= segments.length - 1;
        mergeBtn.addEventListener("click", () => mergeAlignmentSegment(index));

        const addAfterBtn = document.createElement("button");
        addAfterBtn.type = "button";
        addAfterBtn.className = "secondary";
        addAfterBtn.textContent = "Add after";
        addAfterBtn.addEventListener("click", () => addAlignmentSegment(index));

        controls.appendChild(deleteBtn);
        controls.appendChild(mergePrevBtn);
        controls.appendChild(mergeBtn);
        controls.appendChild(addAfterBtn);

        wrapper.appendChild(meta);
        wrapper.appendChild(timingRow);
        wrapper.appendChild(textArea);
        wrapper.appendChild(controls);

        alignmentReviewBody.appendChild(wrapper);
    });
}

function ensureAlignmentReviewInitialized() {
    if (!pendingAlignmentReview) {
        pendingAlignmentReview = {
            runId: "",
            payload: { segments: [] },
            original: { segments: [] },
            speakers: [],
        };
    }
}

function addAlignmentSegment(afterIndex) {
    ensureAlignmentReviewInitialized();
    const segments = pendingAlignmentReview.payload.segments;
    const insertIndex = Number.isInteger(afterIndex) && afterIndex >= -1 ? afterIndex + 1 : segments.length;
    const previous = afterIndex >= 0 ? segments[afterIndex] : segments[segments.length - 1];
    const next = segments[afterIndex + 1];
    const baseStart = typeof previous?.end === "number"
        ? previous.end
        : typeof previous?.start === "number"
            ? previous.start
            : 0;
    const nextStart = typeof next?.start === "number" ? next.start : baseStart + 1;
    const speakerList = pendingAlignmentReview?.speakers || [];
    const defaultSpeaker = segments[afterIndex]?.speaker_id
        || segments[Math.max(0, segments.length - 1)]?.speaker_id
        || speakerList[speakerList.length - 1]
        || speakerList[0]
        || "";

    const newSegment = {
        start: baseStart,
        end: Math.max(baseStart, nextStart),
        text: "",
        speaker_id: defaultSpeaker,
    };
    segments.splice(insertIndex, 0, newSegment);
    refreshAlignmentSpeakerList([defaultSpeaker]);
    renderAlignmentReview();
}

function removeAlignmentSegment(index) {
    if (!pendingAlignmentReview) return;
    const segments = pendingAlignmentReview.payload.segments;
    if (!Array.isArray(segments) || segments.length <= 1) {
        return;
    }
    segments.splice(index, 1);
    refreshAlignmentSpeakerList();
    renderAlignmentReview();
}

function mergeAlignmentSegment(index) {
    if (!pendingAlignmentReview) return;
    const segments = pendingAlignmentReview.payload.segments;
    if (!Array.isArray(segments) || index < 0 || index >= segments.length - 1) {
        return;
    }
    const current = segments[index];
    const next = segments[index + 1];
    current.text = [current.text || "", next.text || ""].filter(Boolean).join(" ").trim();
    if (typeof next.end === "number") {
        current.end = next.end;
    }
    if (!current.speaker_id && next.speaker_id) {
        current.speaker_id = next.speaker_id;
    }
    current.words = null;
    segments.splice(index + 1, 1);
    refreshAlignmentSpeakerList();
    renderAlignmentReview();
}

function updateAlignmentSegmentTime(index, field, value, metaEl) {
    if (!pendingAlignmentReview) return;
    const segments = pendingAlignmentReview.payload.segments;
    const segment = segments?.[index];
    if (!segment || Number.isNaN(value)) {
        return;
    }
    segment[field] = value;
    if (field === "start" && typeof segment.end === "number" && segment.end < value) {
        segment.end = value;
    }
    if (field === "end" && typeof segment.start === "number" && segment.start > value) {
        segment.start = value;
    }
    if (metaEl) {
        metaEl.textContent = `Segment ${index + 1} (${formatTimestamp(segment.start)} – ${formatTimestamp(segment.end)})`;
    }
}

function updateAlignmentSegmentSpeaker(index, speaker) {
    if (!pendingAlignmentReview) return;
    const segments = pendingAlignmentReview.payload.segments;
    const segment = segments?.[index];
    if (!segment) return;
    const normalized = speaker.trim();
    segment.speaker_id = normalized || null;
    refreshAlignmentSpeakerList([normalized]);
}

function updateAlignmentSegmentText(index, text) {
    if (!pendingAlignmentReview) return;
    const segments = pendingAlignmentReview.payload.segments;
    const segment = segments?.[index];
    if (!segment) return;
    segment.text = text;
}

function showAlignmentReview(event) {
    if (!alignmentReviewCard || !alignmentReviewBody) {
        return;
    }
    const alignedPayload = event?.aligned || {};
    const speakers = Array.isArray(event?.speakers) ? event.speakers : collectSpeakersFromSegments(alignedPayload.segments);
    pendingAlignmentReview = {
        runId: event?.run_id || "",
        payload: clonePayload(alignedPayload),
        original: clonePayload(alignedPayload),
        speakers: Array.from(new Set(speakers)).sort(),
    };

    if (!Array.isArray(pendingAlignmentReview.payload.segments)) {
        pendingAlignmentReview.payload.segments = [];
    }
    if (!Array.isArray(pendingAlignmentReview.original.segments)) {
        pendingAlignmentReview.original.segments = [];
    }

    refreshAlignmentSpeakerList();
    renderAlignmentReview();

    if (alignmentReviewStatus) {
        alignmentReviewStatus.textContent = "Pipeline paused for alignment review.";
    }
    if (alignmentReviewApplyBtn) {
        alignmentReviewApplyBtn.disabled = false;
    }
    if (alignmentReviewSkipBtn) {
        alignmentReviewSkipBtn.disabled = false;
    }
    if (alignmentReviewAddBtn) {
        alignmentReviewAddBtn.disabled = false;
    }
    alignmentReviewCard.hidden = false;
    alignmentReviewCard.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function submitAlignmentReview(applyChanges) {
    if (!pendingAlignmentReview) {
        return false;
    }
    const { runId, payload } = pendingAlignmentReview;
    if (!runId) {
        return false;
    }

    const base = applyChanges ? payload : pendingAlignmentReview.original;
    const submission = clonePayload(base);
    if (applyChanges && Array.isArray(submission.segments)) {
        submission.segments.forEach(segment => {
            segment.words = null;
        });
    }
    if (Array.isArray(submission.segments)) {
        submission.segments.sort((a, b) => {
            const startA = typeof a.start === "number" ? a.start : 0;
            const startB = typeof b.start === "number" ? b.start : 0;
            return startA - startB;
        });
        submission.segments.forEach(segment => {
            if (segment) {
                if (typeof segment.start !== "number") {
                    const parsed = Number(segment.start);
                    segment.start = Number.isNaN(parsed) ? 0 : parsed;
                }
                if (typeof segment.end !== "number") {
                    const parsed = Number(segment.end);
                    segment.end = Number.isNaN(parsed) ? segment.start : parsed;
                }
                if (segment.end < segment.start) {
                    segment.end = segment.start;
                }
                if (typeof segment.text !== "string") {
                    segment.text = segment.text ? String(segment.text) : "";
                }
                if (typeof segment.speaker_id === "string" && !segment.speaker_id.trim()) {
                    segment.speaker_id = null;
                }
                if (segment.speaker_id === "") {
                    segment.speaker_id = null;
                }
                if (segment.speaker_id === undefined) {
                    segment.speaker_id = null;
                }
            }
        });
    }

    if (alignmentReviewApplyBtn) {
        alignmentReviewApplyBtn.disabled = true;
    }
    if (alignmentReviewSkipBtn) {
        alignmentReviewSkipBtn.disabled = true;
    }
    if (alignmentReviewAddBtn) {
        alignmentReviewAddBtn.disabled = true;
    }
    if (alignmentReviewStatus) {
        alignmentReviewStatus.textContent = "Submitting alignment review…";
    }

    try {
        const response = await fetch(ROUTES.reviewAlignment, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ run_id: runId, alignment: submission }),
        });
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }
        appendLog(applyChanges ? "🪄 Applied alignment adjustments." : "👍 Alignment accepted without changes.");
        hideAlignmentReview();
        setStatus("Running", "running");
        return true;
    } catch (err) {
        if (alignmentReviewStatus) {
            alignmentReviewStatus.textContent = `Failed to submit alignment review: ${err.message}`;
        }
        if (alignmentReviewApplyBtn) {
            alignmentReviewApplyBtn.disabled = false;
        }
        if (alignmentReviewSkipBtn) {
            alignmentReviewSkipBtn.disabled = false;
        }
        if (alignmentReviewAddBtn) {
            alignmentReviewAddBtn.disabled = false;
        }
        appendLog(`❌ Failed to submit alignment review: ${err.message}`);
        return false;
    }
}

function hideTTSReview() {
    pendingTTSReview = null;
    if (ttsReviewBody) {
        ttsReviewBody.innerHTML = '<p class="review-instructions">Listen to each synthesized segment, adjust text or language, and regenerate as needed.</p>';
    }
    if (ttsReviewStatus) {
        ttsReviewStatus.textContent = "Pipeline paused for TTS review.";
    }
    if (ttsReviewApplyBtn) {
        ttsReviewApplyBtn.disabled = false;
    }
    if (ttsReviewSkipBtn) {
        ttsReviewSkipBtn.disabled = false;
    }
    if (ttsReviewCard) {
        ttsReviewCard.hidden = true;
    }
}

function showTTSReview(event) {
    if (!ttsReviewCard || !ttsReviewBody) {
        return;
    }
    const runId = event?.run_id || "";
    const language = event?.language || "";
    const ttsModel = event?.tts_model || "";
    let supportedLangs = Array.isArray(event?.languages)
        ? Array.from(
            new Set(
                event.languages
                    .map(lang => (typeof lang === "string" ? lang.trim().toLowerCase() : ""))
                    .filter(Boolean),
            ),
        )
        : [];
    supportedLangs.forEach(code => registerLanguage(code));
    supportedLangs = supportedLangs.sort((a, b) => {
        const labelA = toLanguageLabel(a) || a;
        const labelB = toLanguageLabel(b) || b;
        return labelA.localeCompare(labelB);
    });
    const segments = Array.isArray(event?.segments) ? [...event.segments] : [];
    segments.sort((a, b) => {
        const startA = typeof a.start === "number" ? a.start : 0;
        const startB = typeof b.start === "number" ? b.start : 0;
        return startA - startB;
    });

    pendingTTSReview = {
        runId,
        language,
        ttsModel,
        languages: supportedLangs,
        segmentRefs: new Map(),
    };

    ttsReviewBody.innerHTML = "";

    if (!segments.length) {
        const emptyNote = document.createElement("p");
        emptyNote.className = "review-instructions";
        emptyNote.textContent = "No synthesized segments available for review.";
        ttsReviewBody.appendChild(emptyNote);
    } else {
        segments.forEach((segment, index) => {
            const segmentId = segment.segment_id || `seg-${index}`;
            const wrapper = document.createElement("div");
            wrapper.className = "review-segment";
            wrapper.dataset.segmentId = segmentId;

            const header = document.createElement("div");
            header.className = "segment-meta";
            const rangeLabel = `${formatTimestamp(segment.start)} – ${formatTimestamp(segment.end)}`;
            const langLabel = segment.lang ? (toLanguageLabel(segment.lang) || segment.lang) : "auto";
            header.textContent = `Segment ${index + 1} (${rangeLabel}) · ${langLabel}`;

            const audioContainer = document.createElement("div");
            audioContainer.className = "segment-audio";
            let audioEl = null;
            if (segment.audio && segment.audio.url) {
                audioEl = document.createElement("audio");
                audioEl.controls = true;
                audioEl.preload = "metadata";
                audioEl.src = resolveBackendUrl(segment.audio.url);
                audioContainer.appendChild(audioEl);
            } else {
                const note = document.createElement("p");
                note.className = "review-instructions";
                note.textContent = "Audio preview unavailable.";
                audioContainer.appendChild(note);
            }

            const textLabel = document.createElement("label");
            textLabel.textContent = "Text";
            const textarea = document.createElement("textarea");
            textarea.dataset.segmentId = segmentId;
            textarea.value = segment.text || "";
            textarea.rows = Math.max(2, Math.min(6, Math.ceil((textarea.value.length || 1) / 60)));
            textLabel.appendChild(textarea);

            const controls = document.createElement("div");
            controls.className = "segment-controls";

            const langLabelEl = document.createElement("label");
            langLabelEl.textContent = "Language";
            let langControl;
            if (supportedLangs.length) {
                langControl = document.createElement("select");
                langControl.dataset.segmentId = segmentId;
                langControl.dataset.role = "segment-lang";
                const autoOption = document.createElement("option");
                autoOption.value = "";
                autoOption.textContent = "Auto";
                langControl.appendChild(autoOption);
                supportedLangs.forEach(code => {
                    const opt = document.createElement("option");
                    opt.value = code;
                    opt.textContent = toLanguageLabel(code) || code;
                    langControl.appendChild(opt);
                });
                const normalizedLang = (segment.lang || "").toLowerCase();
                if (normalizedLang && supportedLangs.includes(normalizedLang)) {
                    langControl.value = normalizedLang;
                } else {
                    langControl.value = "";
                }
            } else {
                langControl = document.createElement("input");
                langControl.type = "text";
                langControl.dataset.segmentId = segmentId;
                langControl.dataset.role = "segment-lang";
                langControl.placeholder = "auto";
                langControl.value = segment.lang || "";
                langControl.setAttribute("list", "language-list");
            }
            langLabelEl.appendChild(langControl);

            const regenBtn = document.createElement("button");
            regenBtn.type = "button";
            regenBtn.className = "secondary";
            regenBtn.textContent = "Regenerate audio";
            regenBtn.dataset.segmentId = segmentId;
            regenBtn.addEventListener("click", () => {
                regenerateTTSReviewSegment(segmentId);
            });

            const statusSpan = document.createElement("span");
            statusSpan.className = "segment-status";
            statusSpan.textContent = "Ready.";

            controls.appendChild(langLabelEl);
            controls.appendChild(regenBtn);
            controls.appendChild(statusSpan);

            wrapper.appendChild(header);
            wrapper.appendChild(audioContainer);
            wrapper.appendChild(textLabel);
            wrapper.appendChild(controls);
            ttsReviewBody.appendChild(wrapper);

            pendingTTSReview.segmentRefs.set(segmentId, {
                wrapper,
                audioEl,
                textArea: textarea,
                langControl,
                regenerateBtn: regenBtn,
                statusLabel: statusSpan,
            });
        });
    }

    if (ttsReviewStatus) {
        const label = language ? toLanguageLabel(language) || language : "";
        const modelLabel = ttsModel ? ` · Model: ${ttsModel}` : "";
        ttsReviewStatus.textContent = label
            ? `Pipeline paused for TTS review (${label})${modelLabel}.`
            : `Pipeline paused for TTS review${modelLabel}.`;
    }
    if (ttsReviewApplyBtn) {
        ttsReviewApplyBtn.disabled = false;
    }
    if (ttsReviewSkipBtn) {
        ttsReviewSkipBtn.disabled = false;
    }
    ttsReviewCard.hidden = false;
    ttsReviewCard.scrollIntoView({ behavior: "smooth", block: "center" });
}

function updateTTSReviewSegment(segment) {
    if (!pendingTTSReview || !segment || !segment.segment_id) {
        return;
    }
    const refs = pendingTTSReview.segmentRefs?.get(segment.segment_id);
    if (!refs) {
        return;
    }
    if (typeof segment.text === "string" && refs.textArea) {
        refs.textArea.value = segment.text;
    }
    if (refs.langControl) {
        const nextLang = (segment.lang || "").toLowerCase();
        if (refs.langControl.tagName === "SELECT") {
            let options = Array.from(refs.langControl.options).map(opt => opt.value);
            if (nextLang && !options.includes(nextLang)) {
                const opt = document.createElement("option");
                opt.value = nextLang;
                opt.textContent = toLanguageLabel(nextLang) || nextLang;
                refs.langControl.appendChild(opt);
                options = Array.from(refs.langControl.options).map(opt => opt.value);
            }
            if (nextLang && options.includes(nextLang)) {
                refs.langControl.value = nextLang;
            } else {
                refs.langControl.value = "";
            }
        } else {
            refs.langControl.value = segment.lang || "";
        }
    }
    if (refs.audioEl && segment.audio && segment.audio.url) {
        const nextSrc = resolveBackendUrl(segment.audio.url);
        if (nextSrc) {
            refs.audioEl.src = nextSrc;
            refs.audioEl.load();
        }
    }
    if (refs.statusLabel) {
        refs.statusLabel.textContent = "Audio regenerated.";
    }
}

async function regenerateTTSReviewSegment(segmentId) {
    if (!pendingTTSReview || !segmentId) {
        return;
    }
    const refs = pendingTTSReview.segmentRefs?.get(segmentId);
    if (!refs) {
        return;
    }

    const textValue = refs.textArea ? refs.textArea.value.trim() : "";
    const langRaw = refs.langControl ? refs.langControl.value.toString().trim() : "";
    const langValue = resolveLanguageCode(langRaw) || langRaw;

    if (refs.regenerateBtn) {
        refs.regenerateBtn.disabled = true;
    }
    if (refs.statusLabel) {
        refs.statusLabel.textContent = "Regenerating…";
    }

    try {
        const response = await fetch(ROUTES.regenerateTTS, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                run_id: pendingTTSReview.runId,
                language: pendingTTSReview.language || null,
                segment_id: segmentId,
                text: textValue,
                lang: langValue || null,
            }),
        });
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }
        const data = await response.json();
        if (data.segment) {
            updateTTSReviewSegment(data.segment);
        }
        appendLog(`🔄 Regenerated TTS segment ${segmentId}.`);
    } catch (err) {
        if (refs.statusLabel) {
            refs.statusLabel.textContent = `Failed: ${err.message}`;
        }
        appendLog(`❌ Failed to regenerate TTS segment ${segmentId}: ${err.message}`);
    } finally {
        if (refs.regenerateBtn) {
            refs.regenerateBtn.disabled = false;
        }
    }
}

async function submitTTSReview(applyChanges) {
    if (!pendingTTSReview) {
        return false;
    }
    const { runId, language } = pendingTTSReview;
    if (!runId) {
        return false;
    }

    const segments = [];
    if (ttsReviewBody) {
        ttsReviewBody.querySelectorAll(".review-segment").forEach(segmentEl => {
            const segmentId = segmentEl.dataset.segmentId;
            if (!segmentId) {
                return;
            }
            const textArea = segmentEl.querySelector(`textarea[data-segment-id="${segmentId}"]`);
            const langControl = segmentEl.querySelector(`[data-role="segment-lang"]`);
            const textValue = textArea ? textArea.value.trim() : "";
            const langInputRaw = langControl ? langControl.value.toString().trim() : "";
            const resolvedLang = resolveLanguageCode(langInputRaw) || langInputRaw;
            segments.push({
                segment_id: segmentId,
                text: textValue,
                lang: resolvedLang || null,
            });
        });
    }

    if (ttsReviewApplyBtn) {
        ttsReviewApplyBtn.disabled = true;
    }
    if (ttsReviewSkipBtn) {
        ttsReviewSkipBtn.disabled = true;
    }
    if (ttsReviewStatus) {
        ttsReviewStatus.textContent = "Submitting TTS review…";
    }

    try {
        const response = await fetch(ROUTES.reviewTTS, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                run_id: runId,
                language: language || null,
                segments,
            }),
        });
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }
        appendLog(applyChanges ? "🎧 Applied TTS adjustments." : "👍 Approved TTS segments without changes.");
        hideTTSReview();
        setStatus("Running", "running");
        return true;
    } catch (err) {
        if (ttsReviewStatus) {
            ttsReviewStatus.textContent = `Failed to submit TTS review: ${err.message}`;
        }
        if (ttsReviewApplyBtn) {
            ttsReviewApplyBtn.disabled = false;
        }
        if (ttsReviewSkipBtn) {
            ttsReviewSkipBtn.disabled = false;
        }
        appendLog(`❌ Failed to submit TTS review: ${err.message}`);
        return false;
    }
}

function updateSourcePreview(src, label, { keepExisting = false, objectUrl = null } = {}) {
    if (!keepExisting && sourceObjectUrl) {
        URL.revokeObjectURL(sourceObjectUrl);
        sourceObjectUrl = null;
    }
    const targetSrc = objectUrl || resolveBackendUrl(src);
    if (!targetSrc) {
        sourceVideoEl.removeAttribute("src");
        sourceVideoEl.load();
        sourcePreviewCard.classList.add("empty");
        sourcePreviewText.textContent = label || "Waiting for media…";
        return;
    }

    sourceVideoEl.src = targetSrc;
    sourceVideoEl.load();
    sourcePreviewCard.classList.remove("empty");
    sourcePreviewText.textContent = label || "";
    if (objectUrl) {
        sourceObjectUrl = objectUrl;
    }
}

function updateDownloadProgress({ active, progress = null, label = "" }) {
    if (!downloadProgress) return;
    if (!active) {
        downloadProgress.hidden = true;
        downloadBar.style.width = "0%";
        downloadBar.style.animation = "none";
        if (label) {
            downloadLabel.textContent = label;
            downloadLabel.hidden = false;
        } else {
            downloadLabel.hidden = true;
        }
        return;
    }
    downloadProgress.hidden = false;
        downloadLabel.hidden = false;
        downloadLabel.textContent = label || "Downloading…";
        if (progress !== null) {
            const pct = Math.min(100, Math.max(0, progress));
            downloadBar.style.width = `${pct}%`;
            downloadBar.style.animation = "none";
        } else {
            downloadBar.style.width = "30%";
            downloadBar.style.animation = "progressPulse 1s ease-in-out infinite alternate";
        }
}

function updateResultPreview(src, label, { objectUrl = null } = {}) {
    if (resultObjectUrl) {
        URL.revokeObjectURL(resultObjectUrl);
        resultObjectUrl = null;
    }
    const targetSrc = objectUrl || resolveBackendUrl(src);
    if (!targetSrc) {
        resultVideoEl.removeAttribute("src");
        resultVideoEl.load();
        resultPreviewCard.classList.add("empty");
        resultPreviewText.textContent = label || "No output yet.";
        return;
    }

    resultVideoEl.src = targetSrc;
    resultVideoEl.load();
    resultPreviewCard.classList.remove("empty");
    resultPreviewText.textContent = label || "";
    if (objectUrl) {
        resultObjectUrl = objectUrl;
    }
}

function updateResultPreviewForLanguage(lang) {
    if (!lang || !latestResultPayload) {
        updateResultPreview(null, "No output yet.");
        return;
    }
    const languages = latestResultPayload.languages || {};
    const entry = languages[lang];
    const label = toLanguageLabel(lang) || lang;
    if (!entry || !entry.final_video || !entry.final_video.url) {
        updateResultPreview(null, `No preview for ${label}`);
        return;
    }
    updateResultPreview(entry.final_video.url, `Previewing ${label}`);
}

function updateResultLanguageSelector(payload) {
    if (!resultLanguageSelect) return;
    const languages = Object.keys(payload.languages || {});
    resultLanguageSelect.innerHTML = "";
    if (!languages.length) {
        resultLanguageSelect.disabled = true;
        const placeholder = document.createElement("option");
        placeholder.value = "";
        const hasDefaultVideo = payload.final_video && payload.final_video.url;
        placeholder.textContent = hasDefaultVideo ? "Default output" : "No output";
        resultLanguageSelect.appendChild(placeholder);
        if (hasDefaultVideo) {
            updateResultPreview(payload.final_video.url, "Rendered output");
        } else {
            updateResultPreview(null, "No output yet.");
        }
        return;
    }
    languages.forEach(lang => {
        const option = document.createElement("option");
        option.value = lang;
        option.textContent = toLanguageLabel(lang) || lang;
        resultLanguageSelect.appendChild(option);
    });
    const preferred = payload.default_language && languages.includes(payload.default_language)
        ? payload.default_language
        : languages[0];
    resultLanguageSelect.value = preferred;
    resultLanguageSelect.disabled = false;
    updateResultPreviewForLanguage(preferred);
}

function setActiveUploadToken(token = "") {
    activeUploadToken = token || "";
    if (reuseMediaTokenInput) {
        reuseMediaTokenInput.value = activeUploadToken;
    }
}

function showInterruptButton(disabled = true) {
    if (!interruptBtn) return;
    interruptBtn.hidden = false;
    interruptBtn.disabled = !!disabled;
    interruptBtn.textContent = "Interrupt run";
}

function hideInterruptButton() {
    if (!interruptBtn) return;
    interruptBtn.hidden = true;
    interruptBtn.disabled = true;
    interruptBtn.textContent = "Interrupt run";
}

async function releaseActiveUploadToken() {
    if (!activeUploadToken) return;
    const token = activeUploadToken;
    setActiveUploadToken("");
    currentSourceDescriptor = "";
    try {
        const params = new URLSearchParams({ token });
        const url = ROUTES.release;
        let sent = false;
        if (navigator.sendBeacon) {
            const blob = new Blob([params.toString()], { type: "application/x-www-form-urlencoded" });
            sent = navigator.sendBeacon(url, blob);
        }
        if (!sent) {
            await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: params,
            });
        }
    } catch (err) {
        console.warn("Failed to release cached media token", err);
    }
}

window.addEventListener("beforeunload", () => {
    if (!activeUploadToken) return;
    const params = new URLSearchParams({ token: activeUploadToken });
    const blob = new Blob([params.toString()], { type: "application/x-www-form-urlencoded" });
    if (!navigator.sendBeacon || !navigator.sendBeacon(ROUTES.release, blob)) {
        fetch(ROUTES.release, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: params,
            keepalive: true,
        }).catch(() => {});
    }
});

const THEME_KEY = "bluez-ui-theme";
const themeIconSpan = themeToggle ? themeToggle.querySelector("span[aria-hidden='true']") : null;

function applyTheme(theme) {
    const normalized = theme === "light" ? "light" : "dark";
    document.body.dataset.theme = normalized;
    document.body.style.colorScheme = normalized;
    if (themeLabel) {
        themeLabel.textContent = normalized === "dark" ? "Dark" : "Light";
    }
    if (themeIconSpan) {
        themeIconSpan.textContent = normalized === "dark" ? "🌙" : "☀️";
    }
}

if (themeToggle) {
    
    applyTheme(document.body.dataset.theme);

    themeToggle.addEventListener("click", () => {
        const next = document.body.dataset.theme === "dark" ? "light" : "dark";
        applyTheme(next);
        localStorage.setItem(THEME_KEY, next);
    });
} else {
    document.body.dataset.theme = "dark";
    document.body.style.colorScheme = "dark";
}

if (involveToggleBtn) {
    setInvolveMode(false);
    involveToggleBtn.addEventListener("click", () => {
        setInvolveMode(!involveMode);
        appendLog(involveMode
            ? "🛠 Involve mode enabled. The pipeline will pause at different steps for manual review."
            : "🏃 Involve mode disabled. The pipeline will run uninterrupted.");
    });
}

if (transcriptionReviewApplyBtn) {
    transcriptionReviewApplyBtn.addEventListener("click", () => {
        submitTranscriptionReview(true);
    });
}

if (transcriptionReviewSkipBtn) {
    transcriptionReviewSkipBtn.addEventListener("click", () => {
        submitTranscriptionReview(false);
    });
}

if (transcriptionReviewAddBtn) {
    transcriptionReviewAddBtn.addEventListener("click", () => {
        if (!pendingTranscriptionReview) {
            return;
        }
        const segments = Array.isArray(pendingTranscriptionReview.payload.segments)
            ? pendingTranscriptionReview.payload.segments
            : [];
        addTranscriptionSegment(segments.length - 1);
    });
}

hideTranscriptionReview();
hideAlignmentReview();
hideTTSReview();

if (ttsReviewApplyBtn) {
    ttsReviewApplyBtn.addEventListener("click", () => {
        submitTTSReview(true);
    });
}

if (ttsReviewSkipBtn) {
    ttsReviewSkipBtn.addEventListener("click", () => {
        submitTTSReview(false);
    });
}

if (alignmentReviewApplyBtn) {
    alignmentReviewApplyBtn.addEventListener("click", () => {
        submitAlignmentReview(true);
    });
}

if (alignmentReviewSkipBtn) {
    alignmentReviewSkipBtn.addEventListener("click", () => {
        submitAlignmentReview(false);
    });
}

if (alignmentReviewAddBtn) {
    alignmentReviewAddBtn.addEventListener("click", () => {
        if (!pendingAlignmentReview) {
            return;
        }
        addAlignmentSegment(pendingAlignmentReview.payload.segments.length - 1);
    });
}

updateSourcePreview(null, "Waiting for media…");
updateResultPreview(null, "No output yet.");
renderTargetLanguageTags();
if (resultLanguageSelect) {
    resultLanguageSelect.addEventListener("change", () => {
        updateResultPreviewForLanguage(resultLanguageSelect.value);
    });
}

async function fetchOptions() {
    const resp = await fetch(ROUTES.options);
    if (!resp.ok) throw new Error("Unable to load model registry");
    optionsCache = await resp.json();
    populateSelectors(optionsCache);
}

function populateSelectors(opts) {
    const asrSelect = document.getElementById("asr-model");
    const trSelect = document.getElementById("tr-model");
    const ttsSelect = document.getElementById("tts-model");
    const sepSelect = document.getElementById("sep-model");
    const translationStrategy = document.getElementById("translation-strategy");
    const dubbingStrategy = document.getElementById("dubbing-strategy");
    const subtitleStyle = document.getElementById("subtitle-style");
    const languageList = document.getElementById("language-list");

    asrModels = opts.asr_models || [];
    translationModels = opts.translation_models || [];
    ttsModels = opts.tts_models || [];

    const codes = new Set();
    [asrModels, translationModels, ttsModels].forEach(group => {
        group.forEach(model => (model.languages || []).forEach(lang => codes.add(lang)));
    });

    const initSourceCode = resolveLanguageCode(sourceLangInput ? sourceLangInput.value : "");
    const initTargetCode = getPrimaryTargetLang() || initSourceCode;
    refreshModelSelect(asrSelect, asrModels, initSourceCode);
    refreshModelSelect(trSelect, translationModels, initTargetCode);
    refreshModelSelect(ttsSelect, ttsModels, initTargetCode);

    opts.audio_separation_models.forEach(group => {
        const optGroup = document.createElement("optgroup");
        optGroup.label = group.architecture;
        group.models.forEach(model => {
            const opt = document.createElement("option");
            opt.value = model.filename;
            opt.textContent = `${model.filename} (${model.stems.join(" + ")})`;
            optGroup.appendChild(opt);
        });
        sepSelect.appendChild(optGroup);
    });

    opts.translation_strategies.forEach(strategy => {
        const option = document.createElement("option");
        option.value = strategy;
        option.textContent = strategy;
        translationStrategy.appendChild(option);
    });

    opts.dubbing_strategies.forEach(strategy => {
        const option = document.createElement("option");
        option.value = strategy;
        option.textContent = strategy;
        dubbingStrategy.appendChild(option);
    });

    opts.subtitle_styles.forEach(style => {
        ["Desktop", "Mobile"].forEach(mode => {
            const value = mode === "Mobile" ? `${style}_mobile` : style;
            const option = document.createElement("option");
            option.value = value;
            option.textContent = `${style} (${mode})`;
            subtitleStyle.appendChild(option);
        });
    });

    const preferredOrder = ["en", "fr"];
    const languageEntries = Array.from(codes).map(code => ({
        code: code.toLowerCase(),
        display: registerLanguage(code),
    }));

    languageEntries.sort((a, b) => {
        const idxA = preferredOrder.indexOf(a.code);
        const idxB = preferredOrder.indexOf(b.code);
        if (idxA !== -1 && idxB !== -1) return idxA - idxB;
        if (idxA !== -1) return -1;
        if (idxB !== -1) return 1;
        return a.display.localeCompare(b.display);
    });

    function fillLanguageOptions(filter = "") {
        const normalized = filter.trim().toLowerCase();
        const suggestions = normalized
            ? languageEntries.filter(entry =>
                  entry.display.toLowerCase().includes(normalized) || entry.code.includes(normalized)
              )
            : languageEntries;
        languageList.innerHTML = "";
        suggestions.forEach(entry => {
            const opt = document.createElement("option");
            opt.value = entry.display;
            opt.dataset.code = entry.code;
            languageList.appendChild(opt);
        });
    }

    fillLanguageOptions();
    updateModelSelectorsForTarget();

    function updateLanguageSuggestions(inputEl) {
        if (!inputEl) return;
        fillLanguageOptions(inputEl.value);
    }

    if (sourceLangInput) {
        sourceLangInput.addEventListener("input", () => {
            updateLanguageSuggestions(sourceLangInput);
            const sourceCode = resolveLanguageCode(sourceLangInput.value);
            refreshModelSelect(asrSelect, asrModels, sourceCode);
            updateModelSelectorsForTarget();
        });
    }
    if (targetLangInput) {
        targetLangInput.addEventListener("input", () => {
            updateLanguageSuggestions(targetLangInput);
        });
        targetLangInput.addEventListener("keydown", event => {
            if (["Enter", "Tab", ","].includes(event.key)) {
                const pending = targetLangInput.value.trim();
                if (!pending) {
                    if (event.key === "Enter" || event.key === ",") {
                        event.preventDefault();
                    }
                    return;
            }
                event.preventDefault();
                addTargetLanguage(pending);
            } else if (event.key === "Backspace" && !targetLangInput.value && selectedTargetLangs.length) {
                const last = selectedTargetLangs[selectedTargetLangs.length - 1];
                removeTargetLanguage(last);
            }
        });
        targetLangInput.addEventListener("blur", () => {
            if (targetLangInput.value.trim()) {
                addTargetLanguage(targetLangInput.value);
            }
        });
    }
}

function appendLog(message) {
    const now = new Date().toLocaleTimeString();
    logEl.textContent += `\n[${now}] ${message}`;
    logEl.scrollTop = logEl.scrollHeight;
}

if (interruptBtn) {
    interruptBtn.addEventListener("click", async () => {
        if (!activeRunId) {
            appendLog("⚠️ No active run to interrupt.");
            return;
        }
        const previousLabel = interruptBtn.textContent;
        interruptBtn.disabled = true;
        interruptBtn.textContent = "Stopping…";
        try {
            const body = new URLSearchParams({ run_id: activeRunId });
            await fetch(ROUTES.stop, {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body,
                keepalive: true,
            });
            appendLog("⏹ Cancellation requested." );
            setStatus("Stopping…", "running");
        } catch (err) {
            appendLog(`❌ Failed to interrupt: ${err.message}`);
            interruptBtn.disabled = false;
        } finally {
            interruptBtn.textContent = previousLabel;
        }
    });
}

fileInput.addEventListener("change", () => {
    if (activeUploadToken) {
        releaseActiveUploadToken();
    } else {
        setActiveUploadToken("");
        currentSourceDescriptor = "";
    }
    if (sourceObjectUrl) {
        URL.revokeObjectURL(sourceObjectUrl);
        sourceObjectUrl = null;
    }
    if (fileInput.files && fileInput.files.length > 0) {
        const blobUrl = URL.createObjectURL(fileInput.files[0]);
        updateSourcePreview(blobUrl, "Local upload", { objectUrl: blobUrl });
        currentSourceDescriptor = fileInput.files[0].name || "";
    } else if (!videoLinkInput.value.trim()) {
        updateSourcePreview(null, "Waiting for media…");
    }
});

videoLinkInput.addEventListener("input", () => {
    const trimmedValue = videoLinkInput.value.trim();
    if (activeUploadToken && trimmedValue !== currentSourceDescriptor) {
        releaseActiveUploadToken();
        currentSourceDescriptor = trimmedValue;
    } else if (!trimmedValue) {
        setActiveUploadToken("");
        currentSourceDescriptor = "";
    } else {
        currentSourceDescriptor = trimmedValue;
    }
    if (trimmedValue && (!fileInput.files || fileInput.files.length === 0)) {
        updateSourcePreview(null, "Remote media will be downloaded on run…");
        updateDownloadProgress({ active: false, label: "" });
    }
});

function resetLog() {
    logEl.textContent = "Waiting…";
}

function setStatus(text, tone = "idle") {
    statusBadge.textContent = text;
    const palette = {
        idle: "rgba(59, 130, 246, 0.18)",
        running: "rgba(52, 211, 153, 0.2)",
        success: "rgba(34, 197, 94, 0.25)",
        error: "rgba(239, 68, 68, 0.25)",
        paused: "rgba(245, 158, 11, 0.25)",
    };
    statusBadge.style.background = palette[tone] || palette.idle;
    statusBadge.style.borderColor = palette[tone] || palette.idle;
}

function renderResults(payload) {
    if (!payload) {
        resultsEl.textContent = "Pipeline finished without a payload.";
        return;
    }
    latestResultPayload = payload;
    const { languages = {}, default_language: defaultLanguage, final_video, final_audio, speech_track, subtitles, models, timings, workspace_id, source_media, source_video, available_languages: availableLanguages = [] } = payload;

    function renderLink(item, label) {
        if (!item || !item.url) return `<li>${label}: unavailable</li>`;
        const href = resolveBackendUrl(item.url);
        const pathLabel = item.path || href;
        return `<li>${label}: <a href="${href}" target="_blank" rel="noopener">${pathLabel}</a></li>`;
    }

    function renderLanguageSection(lang, data) {
        const label = toLanguageLabel(lang) || lang;
        const entries = [];
        entries.push(renderLink(data.final_video, "Final video"));
        entries.push(renderLink(data.final_audio, "Dubbed audio"));
        entries.push(renderLink(data.speech_track, "Speech track"));
        const alignedSubs = data.subtitles?.aligned || {};
        entries.push(renderLink(alignedSubs.srt, "Aligned SRT"));
        entries.push(renderLink(alignedSubs.vtt, "Aligned VTT"));
        return `<li><strong>${label}</strong><ul>${entries.join("")}</ul></li>`;
    }

    const languagesMarkup = Object.entries(languages)
        .map(([lang, data]) => renderLanguageSection(lang, data))
        .join("") || "<li>No target languages processed.</li>";

    const modelValue = value => {
        if (!value) return "n/a";
        if (typeof value === "string") return value;
        if (typeof value === "object") {
            return Object.entries(value)
                .map(([key, val]) => `${key}: ${val}`)
                .join(", ") || "n/a";
        }
        return String(value);
    };

    resultsEl.innerHTML = `
        <p><strong>Workspace:</strong> ${workspace_id || "n/a"}</p>
        <p><strong>Source:</strong> ${source_media || "n/a"}</p>
        <p><strong>Default language:</strong> ${defaultLanguage ? `${toLanguageLabel(defaultLanguage) || defaultLanguage} (${defaultLanguage})` : "n/a"}</p>
        <p><strong>Models:</strong> ASR=${modelValue(models?.asr)}, Translation=${modelValue(models?.translation)}, TTS=${modelValue(models?.tts)}</p>
        <p><strong>Available languages:</strong> ${availableLanguages.length ? availableLanguages.join(", ") : "n/a"}</p>
        <ul>
            ${renderLink(source_video, "Source video")}
            ${renderLink(final_video, "Default final video")}
            ${renderLink(final_audio, "Default dubbed audio")}
            ${renderLink(speech_track, "Default speech track")}
        </ul>
        <details open>
            <summary>Per-language outputs</summary>
            <ul>
                ${languagesMarkup}
            </ul>
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
            <ul>
                ${Object.entries(timings || {}).map(([step, duration]) => `<li>${step}: ${duration.toFixed(2)}s</li>`).join("")}
            </ul>
        </details>
    `;

    updateResultLanguageSelector(payload);
}

formEl.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!optionsCache) {
        appendLog("⚠️ Model registry not loaded.");
        return;
    }

    hideTranscriptionReview();
    hideAlignmentReview();

    let linkValue = videoLinkInput.value.trim();
    let hasFile = fileInput.files && fileInput.files.length > 0;

    if (linkValue && hasFile) {
        if (activeUploadToken) {
            await releaseActiveUploadToken();
        }
        fileInput.value = "";
        hasFile = false;
    }

    if (targetLangInput && targetLangInput.value.trim()) {
        addTargetLanguage(targetLangInput.value);
    }

    const formData = new FormData(formEl);
    linkValue = videoLinkInput.value.trim();
    const cachedToken = reuseMediaTokenInput ? reuseMediaTokenInput.value.trim() : "";
    if (!hasFile && !linkValue && !cachedToken) {
        appendLog("⚠️ Provide a media file or a video link.");
        setStatus("Error", "error");
        return;
    }

    if (linkValue) {
        formData.set("video_url", linkValue);
    } else {
        formData.delete("video_url");
    }

    if (linkValue) {
        formData.delete("file");
    } else if (!hasFile) {
        formData.delete("file");
    }

    if (cachedToken && !hasFile && !linkValue) {
        formData.set("reuse_media_token", cachedToken);
    } else {
        formData.delete("reuse_media_token");
    }

    const resolvedSource = resolveLanguageCode(formData.get("source_lang"));
    formData.set("source_lang", resolvedSource);

    formData.delete("target_langs");
    selectedTargetLangs.forEach(lang => formData.append("target_langs", lang));
    const primaryTarget = selectedTargetLangs[0] || "";
    if (primaryTarget) {
        formData.set("target_lang", primaryTarget);
    } else {
        formData.delete("target_lang");
    }

    formData.set("audio_sep", document.getElementById("audio-sep").checked ? "true" : "false");
    formData.set("perform_vad_trimming", document.getElementById("vad-trim").checked ? "true" : "false");
    formData.set("sophisticated_dub_timing", document.getElementById("sophisticated-timing").checked ? "true" : "false");
    formData.set("persist_intermediate", document.getElementById("persist-intermediate").checked ? "true" : "false");
    formData.set("involve_mode", involveMode ? "true" : "false");

    const normalizeSpeakerField = (field, label) => {
        const raw = formData.get(field);
        if (raw === null) {
            return { value: null, ok: true };
        }
        const trimmed = String(raw).trim();
        if (!trimmed) {
            formData.delete(field);
            return { value: null, ok: true };
        }
        const parsed = Number(trimmed);
        if (!Number.isInteger(parsed) || parsed < 1) {
            appendLog(`⚠️ ${label} must be a positive integer.`);
            setStatus("Error", "error");
            return { value: null, ok: false };
        }
        formData.set(field, String(parsed));
        return { value: parsed, ok: true };
    };

    const minHint = normalizeSpeakerField("min_speakers", "Min speakers");
    if (!minHint.ok) {
        return;
    }
    const maxHint = normalizeSpeakerField("max_speakers", "Max speakers");
    if (!maxHint.ok) {
        return;
    }
    if (minHint.value !== null && maxHint.value !== null && minHint.value > maxHint.value) {
        appendLog("⚠️ Min speakers cannot exceed max speakers.");
        setStatus("Error", "error");
        return;
    }

    latestResultPayload = null;
    availableResultLanguages = [];
    if (resultLanguageSelect) {
        resultLanguageSelect.innerHTML = '<option value="">Processing…</option>';
        resultLanguageSelect.disabled = true;
    }

    resultsEl.textContent = "Processing current run…";
    activeRunId = "";
    showInterruptButton(true);
    resetLog();
    logEl.textContent = "Starting pipeline…";
    setStatus("Running", "running");
    if (!hasFile && linkValue) {
        updateSourcePreview(null, "Downloading media…");
    }
    updateResultPreview(null, "Processing output…");
    runBtn.disabled = true;

    try {
        const response = await fetch(ROUTES.run, {
            method: "POST",
            body: formData,
        });
        if (!response.ok) {
            throw new Error(`Request failed: ${response.status}`);
        }

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
                if (!line) continue;
                const data = JSON.parse(line.slice(5).trim());
                handleEvent(data);
            }
        }

        if (buffer.trim()) {
            const line = buffer.split("\n").find(l => l.startsWith("data:"));
            if (line) {
                const data = JSON.parse(line.slice(5).trim());
                handleEvent(data);
            }
        }
    } catch (err) {
        appendLog(`❌ ${err.message}`);
        setStatus("Error", "error");
    } finally {
        runBtn.disabled = false;
        hideInterruptButton();
        activeRunId = "";
    }
});

function handleEvent(event) {
    if (!event || !event.type) return;
    switch (event.type) {
        case "run_id":
            activeRunId = event.run_id || "";
            if (activeRunId) {
                showInterruptButton(false);
                appendLog(`🆔 Run started (${activeRunId})`);
            }
            break;
        case "step":
            if (event.event === "start") {
                appendLog(`▶️  ${event.step}…`);
            } else if (event.event === "end") {
                appendLog(`✅ ${event.step} (${event.duration.toFixed(2)}s)`);
            }
            break;
        case "cancelled":
            appendLog("⏹ Run cancelled.");
            setStatus("Cancelled", "error");
            resultsEl.textContent = "Run cancelled.";
            updateResultPreview(null, "Cancelled.");
            hideTranscriptionReview();
            hideAlignmentReview();
            hideTTSReview();
            hideInterruptButton();
            activeRunId = "";
            break;
        case "result":
            appendLog("🎉 Pipeline completed.");
            setStatus("Done", "success");
            hideTranscriptionReview();
            hideAlignmentReview();
            hideTTSReview();
            renderResults(event.result);
            {
                const token = event.result?.upload_token || "";
                setActiveUploadToken(token);
                currentSourceDescriptor = (event.result?.source_media || "").trim();
                if (token && fileInput) {
                    fileInput.value = "";
                }
            }
            if (event.result?.source_video?.url) {
                updateSourcePreview(event.result.source_video.url, "Source (workspace)");
            }
            break;
        case "status":
            if (event.event === "download_start") {
                appendLog(`⬇️ Downloading remote media: ${event.url || ""}`);
                updateSourcePreview(null, "Downloading media…", { keepExisting: false });
                updateDownloadProgress({ active: true, label: "Downloading…" });
            } else if (event.event === "download_complete") {
                appendLog(`✅ Download complete`);
                sourcePreviewText.textContent = "Download complete. Preparing source…";
                updateDownloadProgress({ active: false, label: "Download complete." });
            } else if (event.event === "download_progress") {
                if (typeof event.total === "number" && typeof event.downloaded === "number") {
                    const pct = Math.round((event.downloaded / event.total) * 100);
                    updateDownloadProgress({ active: true, progress: pct, label: `Downloading… ${pct}%` });
                } else {
                    updateDownloadProgress({ active: true, progress: null, label: "Downloading…" });
                }
            } else if (event.event === "awaiting_transcription_review") {
                setStatus("Awaiting transcription review", "paused");
            } else if (event.event === "awaiting_alignment_review") {
                setStatus("Awaiting alignment review", "paused");
            } else if (event.event === "awaiting_tts_review") {
                setStatus("Awaiting TTS review", "paused");
            } else {
                appendLog(`ℹ️ ${event.event || "status"}`);
            }
            break;
        case "transcription_review":
            appendLog("✍️ Awaiting transcription review.");
            setStatus("Awaiting review", "paused");
            hideTTSReview();
            showTranscriptionReview(event);
            break;
        case "transcription_review_complete":
            appendLog("✅ Transcription review submitted. Resuming pipeline.");
            setStatus("Running", "running");
            hideTranscriptionReview();
            break;
        case "alignment_review":
            appendLog("✍️ Awaiting alignment review.");
            setStatus("Awaiting alignment review", "paused");
            hideTranscriptionReview();
            hideTTSReview();
            showAlignmentReview(event);
            break;
        case "alignment_review_complete":
            appendLog("✅ Alignment review submitted. Resuming pipeline.");
            setStatus("Running", "running");
            hideAlignmentReview();
            break;
        case "tts_review":
            appendLog("🎧 Awaiting TTS review.");
            setStatus("Awaiting TTS review", "paused");
            hideTranscriptionReview();
            hideAlignmentReview();
            showTTSReview(event);
            break;
        case "tts_review_complete":
            appendLog("✅ TTS review submitted. Resuming pipeline.");
            setStatus("Running", "running");
            hideTTSReview();
            break;
        case "tts_review_regenerated":
            if (event.segment) {
                updateTTSReviewSegment(event.segment);
            }
            break;
        case "source_preview":
            if (event.preview && event.preview.url) {
                updateSourcePreview(event.preview.url, "Source ready");
                appendLog("🎬 Source preview available.");
            }
            break;
        case "error":
            appendLog(`❌ Error: ${event.message || "unknown failure"}`);
            setStatus("Error", "error");
            hideTranscriptionReview();
            hideAlignmentReview();
            hideTTSReview();
            hideInterruptButton();
            activeRunId = "";
            break;
        case "complete":
            appendLog("🏁 Stream closed.");
            hideInterruptButton();
            hideTranscriptionReview();
            hideAlignmentReview();
            hideTTSReview();
            activeRunId = "";
            break;
        default:
            appendLog(`ℹ️ ${JSON.stringify(event)}`);
    }
}

fetchOptions().catch(err => appendLog(`⚠️ ${err.message}`));