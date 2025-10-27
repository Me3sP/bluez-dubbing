<p align="center">
  <img src="assets/icon.png" alt="Project logo">
</p>


<p align="center">
  <img src="assets/logo.png" alt="My Logo"xed/>
</p>

<!-- <div style="display:flex; justify-content:flex-start;gap:2px;align-items:center;">
  <img src="assets/icon.png"
       alt="Icon"
       style="height:60px; aspect-ratio:1/1; object-fit:cover; border-radius:8px;">
  <img src="assets/logo.png"
       alt="Logo"
       style="height:60px;">
</div> -->

<!-- <div style="
  display: flex;
  flex-direction: column;    /* stack vertically */
  align-items: center;       /* center horizontally */
  width: 200px;              /* total width of both */
">
  <img src="assets/icon.png"
       alt="Icon"
       style="width:100%; aspect-ratio:1/1; object-fit:cover;">
  <img src="assets/logo.png"
       alt="Logo"
       style="width:100%; height:auto; margin-top:8px;">
</div> -->


                                                                                                    
# Bluez-Dubbing: Multilingual AI Dubbing Pipeline

Bluez-Dubbing is a modular, production-ready pipeline for **automatic video dubbing** and **subtitle generation**. It leverages state-of-the-art models for ASR (Automatic Speech Recognition), translation, and TTS (Text-to-Speech), supporting advanced features like audio source separation, VAD-based trimming, sophisticated dubbing strategies, and customizable subtitle styles.

---

## 🚀 Features

- **End-to-End Dubbing:** From video/audio input to fully dubbed output with burned-in subtitles.
- **Modular Services:** Pluggable ASR, translation, and TTS models—easily extend or swap components.
- **Audio Source Separation:** Isolate vocals and background for high-quality dubbing.
- **Flexible Translation Strategies:** Segment-wise or full-text translation with alignment.
- **Advanced Dubbing:** Full replacement or overlay strategies, with sophisticated timing.
- **Subtitle Generation:** Multiple styles (e.g., Netflix, mobile) and formats (SRT, VTT, ASS).
- **Workspace Management:** All intermediate and final files organized per job.
- **REST API:** FastAPI-based orchestrator for easy integration.

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

### 2. **Install Dependencies**

Each service is isolated. Install dependencies for each:

```bash
cd services/asr && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cd ../translation && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cd ../tts && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cd ../orchestrator && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Or use `pyproject.toml` with `pip install .` in each service.

### 3. **Configure Environment**

- Copy `.env.example` to `.env` and fill in required API keys (e.g., for DeepL, OpenAI, Azure, HuggingFace).
- Place model weights in `models_cache/` as needed.

### 4. **Run the Services**

Start each service (in separate terminals or with a process manager):

```bash
# ASR Service
cd services/asr && source .venv/bin/activate && uvicorn app.main:app --port 8001

# Translation Service
cd ../translation && source .venv/bin/activate && uvicorn app.main:app --port 8002

# TTS Service
cd ../tts && source .venv/bin/activate && uvicorn app.main:app --port 8003

# Orchestrator (API)
cd ../orchestrator && source .venv/bin/activate && uvicorn app.main:app --port 8000
```

---

## 🛠️ Usage

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
