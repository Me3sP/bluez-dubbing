<div style="
  margin: 32px auto;
  padding: 36px 42px;
  max-width: 1080px;
  border-radius: 24px;
  background: linear-gradient(135deg, #313338 0%, #232428 35%, #1e1f22 100%);
  color: #f2f3f5;
  font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
  border: 1px solid rgba(80, 82, 87, 0.45);
">
  <div style="display:flex; align-items:center; gap:14px; margin-bottom:28px;">
    <img src="assets/icon.png"
         alt="Bluez icon"
         style="height:64px; width:64px; border-radius:16px; object-fit:cover; box-shadow:0 8px 18px rgba(0,0,0,0.4);">
    <img src="assets/logo.png"
         alt="Bluez logo"
         style="height:64px;">
  </div>
  <h1 style="margin:0 0 8px; font-size:2.75rem; letter-spacing:-0.015em;">Bluez-Dubbing: Multilingual AI Dubbing Pipeline</h1>
  <p style="margin:0; font-size:1.05rem; color:#b5bac1;">
    Choose your mode, dub in any language, enjoy crystal-clear vocals.
  </p>
</div>

Bluez-Dubbing is a modular, production-ready pipeline for **automatic video dubbing** and **subtitle generation**. It leverages state-of-the-art models for ASR (Automatic Speech Recognition), translation, and TTS (Text-to-Speech), supporting advanced features like audio source separation, VAD-based trimming, sophisticated dubbing strategies, and customizable subtitle styles.

---

## 🚀 Features

- **End-to-End Dubbing:** From video/audio input to fully dubbed output with burned-in subtitles.
- **Web UI:** Upload a file or paste a YouTube/Instagram/TikTok link, watch download progress (via yt-dlp), and preview source/final videos inline.
- **Modular Services:** Pluggable ASR, translation, and TTS models—easily extend or swap components.
- **Audio Source Separation:** Isolate vocals and background for high-quality dubbing.
- **Flexible Translation Strategies:** Segment-wise or full-text translation with alignment.
- **Advanced Dubbing:** Full replacement or overlay strategies, with sophisticated timing and optional VAD trimming.
- **Subtitle Generation:** Multiple styles (e.g., Netflix, mobile) and formats (SRT, VTT, ASS).
- **Workspace Management:** All intermediate and final files organized per job with download links.
- **REST API & CLI:** FastAPI endpoints plus command-line tooling for automation.

---

## 🗂️ Project Structure

```
bluez-dubbing/
│
├── assets/                # Sample videos and audio files
├── deploy/                # Deployment scripts/configs
├── libs/
│   └── common-schemas/    # Shared Pydantic models, config, and utilities
├── models_cache/          # Downloaded model weights and configs
├── outs/                  # Output workspaces for each job
├── services/
│   ├── asr/               # ASR service (WhisperX, etc.)
│   ├── orchestrator/      # Main API and pipeline logic
│   ├── translation/       # Translation service (deep_translator, etc.)
│   └── tts/               # TTS service (Edge TTS, etc.)
└── README.md
```

---

## ⚡ Quickstart

### 1. **Clone the Repository**

```bash
git clone https://github.com/your-org/bluez-dubbing.git
cd bluez-dubbing
```

### 2. **Install Dependencies (uv)**

Install the project's virtual environments and dependencies in one pass using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

This installs all service dependencies (ASR, translation, TTS, orchestrator) into their dedicated `.venv` folders.

### 3. **Configure Environment**

- Copy `.env.example` to `.env` and provide provider keys (DeepL, Azure, etc.).
- Place required model weights in `models_cache/`.

### 4. **Run the Stack**

Launch every microservice with the bundled `Makefile` targets:

```bash
make start-ui   # start full stack + web UI
```

To run API-only mode (no UI routes):

```bash
make start-api
```

Stop everything:

```bash
make stop
```

---

## 🛠️ Usage

### **Web UI**

With `make start-ui` running, open `http://localhost:8000/ui` to access the interactive orchestrator:

- Upload local media **or** provide a public link (YouTube, Instagram, TikTok…)—downloads stream via `yt-dlp` with a live progress bar.
- Review smart suggestions for source/target languages (English/French pinned, live filtering) and pick models per stage.
- Track every pipeline step in the live log, including ASR, translation, TTS, separation, etc.
- Preview the uploaded source and the final rendered video inline.
- Download final outputs, intermediate JSON dumps, subtitle files, and speech tracks directly from the UI.

Dark/light theme toggles, Discord-inspired visuals, and responsive layouts come baked in.

### **API Example**

Submit a dubbing job via HTTP:

```bash
curl -X POST -G 'http://localhost:8000/v1/dub' \
  --data-urlencode 'video_url=/path/to/video.mp4' \
  --data-urlencode 'target_lang=fr' \
  --data-urlencode 'source_lang=en' \
  --data-urlencode 'sep_model=melband_roformer_big_beta5e.ckpt' \
  --data-urlencode 'asr_model=whisperx' \
  --data-urlencode 'tr_model=deep_translator' \
  --data-urlencode 'tts_model=edge_tts' \
  --data-urlencode 'audio_sep=true' \
  --data-urlencode 'perform_vad_trimming=true' \
  --data-urlencode 'translation_strategy=short' \
  --data-urlencode 'dubbing_strategy=full_replacement' \
  --data-urlencode 'sophisticated_dub_timing=true' \
  --data-urlencode 'subtitle_style=netflix_mobile'
```

- See `/v1/dub` endpoint in `services/orchestrator/app/main.py` for all parameters.

### **Output**

- All results (final video, audio, subtitles, intermediates) are saved in `outs/<workspace_id>/`.

### **Command-Line Interface**

The orchestrator can also be triggered directly without spinning up the HTTP server:

```bash
cd bluez-dubbing
uv run python -m services.orchestrator.cli \
  /path/to/video.mp4 \
  --target-lang fr \
  --translation-strategy default \
  --output-json ./run-result.json
```

Run `uv run python -m services.orchestrator.cli --help` to see all available flags.

Additional service-level CLIs are available for debugging individual stages:

```bash
# Automatic speech recognition (runs WhisperX worker)
uv run python -m services.asr.cli /path/to/audio.wav --output-json asr.json

# Translation (works with ASR JSON output)
uv run python -m services.translation.cli asr.json --target-lang fr --output-json translation.json

# Text-to-speech synthesis (works with translation JSON output)
uv run python -m services.tts.cli translation.json --workspace ./tts_out --output-json tts.json
```

> **Note:** The orchestrator CLI expects the ASR, translation, and TTS services to be reachable (locally or remote). The per-service CLIs above can be used when you want to run individual stages without launching the APIs.

### **Tests**

CLI behavior is covered by pytest-based unit tests. Run them from the repo root:

```bash
cd bluez-dubbing
make test
```

The `Makefile` target installs the required test dependencies via `uv` and executes pytest inside the orchestrator service.

- Registry tests (`services/*/tests/test_runner_api.py`) validate that every registered ASR/translation/TTS model executes through the corresponding worker shim—useful when you introduce new model configs.
- An integration test (`services/orchestrator/tests/test_pipeline_integration.py`) drives the orchestrator pipeline end-to-end with patched dependencies to ensure the FastAPI logic and workspace handling remain consistent.

### **Continuous Integration**

The repository ships with a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs automatically on pushes and pull requests. The pipeline:

- sets up Python 3.11 and installs `uv`;
- executes `make test`, which drives the per-service CLI tests, the worker-registry coverage (ensuring every registered ASR/translation/TTS model executes), and the orchestrator pipeline integration test.

No additional secrets are required. Ensure new code keeps the test matrix green by running `make test` locally before opening a pull request.

---

## 🧩 Supported Models

- **ASR:** [WhisperX](https://github.com/m-bain/whisperx), extendable via `services/asr/app/registry.py`
- **Translation:** [deep_translator](https://github.com/nidhaloff/deep-translator), Facebook M2M100, etc.
- **TTS:** Edge TTS, Chatterbox, and more.

See `libs/common-schemas/config/` for model configs and supported languages.

---

## 📝 Configuration

- **Model configs:** `libs/common-schemas/config/*.yaml`
- **Environment variables:** `.env` (API keys, etc.)

---

## 🧪 Testing

- Unit and integration tests are recommended for each service.
- Use FastAPI’s `/docs` for interactive API testing.

---

## 🧠 Extending

- **Add new models:** Register in the appropriate service’s `registry.py` and add a config YAML.
- **Custom pipelines:** Modify `services/orchestrator/app/main.py` for new strategies or steps.

---

## 📦 Deployment

- Use Docker, systemd, or your favorite process manager for production.
- GPU recommended for ASR and TTS.

---

## 🤝 Contributing

Pull requests and issues are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

[MIT License](LICENSE)

---

## 🙏 Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperx)
- [deep-translator](https://github.com/nidhaloff/deep-translator)
- [Edge TTS](https://github.com/rany2/edge-tts)
- And all open-source contributors!

---

**Contact:**  
For questions or support, open an issue or contact the maintainer.
