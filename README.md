

---

# **Multilingual AI Dubbing System**

<picture> <source media="(prefers-color-scheme: dark)" srcset="apps/frontend/assets/icon.png"> <source media="(prefers-color-scheme: light)" srcset="apps/frontend/assets/icon2.png"> <img src="apps/frontend/assets/icon-light.png" width="64" height="64" alt="Bluez icon"> </picture> <picture> <source media="(prefers-color-scheme: dark)" srcset="apps/frontend/assets/logo.png"> <source media="(prefers-color-scheme: light)" srcset="apps/frontend/assets/logo2.png"> <img src="apps/frontend/assets/logo-light.png" width="400" height="64" alt="Bluez logo"> </picture>


[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![uv](https://img.shields.io/badge/Package_Manager-uv-black?logo=astral\&logoColor=white)](https://docs.astral.sh/uv/)
[![FFmpeg](https://img.shields.io/badge/Powered_by-FFmpeg-red?logo=ffmpeg)](https://ffmpeg.org/)

> 🎧 *Choose your mode, dub in any language, and enjoy crystal-clear vocals.*
---

**Bluez-Dubbing** is a modular, production-ready pipeline for **automatic video dubbing** and **subtitle generation**.
It integrates state-of-the-art models for **ASR** (Automatic Speech Recognition), **translation**, and **TTS** (Text-to-Speech), supporting features like:

* audio source separation
* VAD-based duration alignment
* sophisticated dubbing strategies
* customizable subtitle styles

---

## 🚀 Features

* **End-to-End Dubbing:** From video/audio input to fully dubbed output with burned-in subtitles.
* **Multiple Modes:** Video dubbing (with or without subtitles), audio translation, or subtitling only.
* **REST API & CLI:** FastAPI endpoints and command-line tools for automation.
* **Independent Web UI:** A dedicated app offering an intuitive experience and live progress tracking. See [Web UI](#web-ui) for details.  
* **Modular Services:** Easily plug, swap, or extend ASR, translation, and TTS models.
* **Flexible Translation:** Segment-wise or full-text translation with smart synchronization.
* **Advanced Audio Synchronization:** Multiple algorithms for seamless and natural voice replacement.
* **Subtitle Generation:** Netflix-style, bold-desktop, or mobile-optimized SRT/VTT/ASS output.

---

## 🗂️ Project Structure

```bash
bluez-dubbing/
├── apps/
│   ├── backend/
│   │   ├── cache/              # Cached audio/background/intermediate data
│   │   ├── libs/
│   │   │   └── common-schemas/ # Shared Pydantic models & utilities
│   │   ├── models_cache/       # Downloaded model weights/configs
│   │   ├── outs/               # Output workspaces per job
│   │   ├── services/
│   │   │   ├── asr/            # ASR (WhisperX, etc.)
│   │   │   ├── orchestrator/   # Main API & pipeline logic
│   │   │   ├── translation/    # Translation service
│   │   │   └── tts/            # TTS service
│   │   └── uploads/            # Uploaded media from the UI
│   └── frontend/
│       ├── assets/             # UI icons and branding
│       ├── scripts/            # JS modules for the Web UI
│       ├── styles/             # Stylesheets
│       └── index.html          # Web application entry
├── Makefile
└── README.md
```

---


## 📽️ Demo

<table>
  <tr>
    <td width="33%"  align="center" valign="top">
      <h3>Original Video</h3>
      <a href="https://github.com/user-attachments/assets/7202fa21-6d69-4c25-a9d5-077b920a3ee8" target="_blank">
        <img src="https://github.com/user-attachments/assets/53f2df3c-5efa-4a71-9f3c-cf183bb0f542" alt="Original video thumbnail" style="max-width: 600px; width: 100%; border-radius: 12px;">
      </a>
    </td>
    <td width="33%" align="center" valign="top">
      <h3>Dubbed (English) W/O Subtitles</h3>
      <a href="https://github.com/user-attachments/assets/86a6098a-e1ee-4336-a131-3f516e8da3cd" target="_blank">
        <img src="https://github.com/user-attachments/assets/53f2df3c-5efa-4a71-9f3c-cf183bb0f542" alt="Dubbed English thumbnail" style="max-width: 600px; width: 100%; border-radius: 12px;">
      </a>
    </td>
    <td width="33%" align="center" valign="top">
      <h3>Dubbed (French) With Subtitles</h3>
      <a href="https://github.com/user-attachments/assets/cc63b97d-38f2-440e-bf21-a86c84843838" target="_blank">
        <img src="https://github.com/user-attachments/assets/6f08d43d-a9d1-47d0-9084-20fd3a868a43" alt="Dubbed French thumbnail" style="max-width: 600px; width: 100%; border-radius: 12px;">
      </a>
    </td>
  </tr>
</table>


---

## ⚡ Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/bluez-dubbing.git
cd bluez-dubbing
```

### 2. Install Dependencies (via `uv`)

Ensure [`ffmpeg`](https://ffmpeg.org/download.html) and [`uv`](https://docs.astral.sh/uv/getting-started/installation/) are installed.
Linux example:

```bash
sudo apt update && sudo apt install ffmpeg -y
sudo apt install uv
```

> **Note:** Some tokenizers (e.g. `mecab-python3` for Japanese) require a JVM to be installed.

To install dependencies for any service:

```bash
cd apps/backend/services/<serviceName>
uv sync
```

Or for all at once:

```bash
make install-dep
```

This sets up `.venv` environments for each service (ASR, translation, TTS, orchestrator).

**Dependency notes:**

* If `onnx` and `ml_dtypes` conflict, run:

  ```bash
  uv lock --upgrade-package ml_dtypes==0.5.3 && uv sync
  ```
* Chatterbox pins `torch==2.6.0`. If your GPU requires a newer version:

  ```bash
  uv pip uninstall torch torchaudio
  uv pip install torch==2.8.0 torchaudio==2.8.0
  ```

  For CUDA:

  ```bash
  uv pip install torch==2.8.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu12x
  ```

  ⚠️ Don’t re-run `uv sync` afterwards, as it will downgrade again.

### 3. Configure Environment

* Copy `.env.example` → `.env`
* Set required variables (`HF_TOKEN`, `ORCHESTRATOR_ALLOWED_ORIGINS`, etc.)
* Place model weights in `models_cache/`

### 4. Run the Backend Stack

```bash
make start-api      # Launch orchestrator only
make stack-up       # Launch ASR, translation, TTS, orchestrator
make stop           # Stop all services
make restart        # Restart everything
```

### 5. Serve the Frontend UI

```bash
make start-ui
```

Default URL: `http://localhost:5173`
The UI connects to the backend at `http://localhost:8000/api`.
To change it:

```js
localStorage.setItem("bluez-backend-base", "https://your-host/api");
```

Restart or stop with:

```bash
make restart-ui
make stop
```

---

## 🛠️ Usage
> See [CONTRIBUTING](CONTRIBUTING.md) for a full explanation of parameters and tuning guidance. Defaults work for most cases, and the models automatically adjust when needed.

### Web UI

<img width="2560" height="1485" alt="Screenshot 2025-11-12 121321" src="https://github.com/user-attachments/assets/e2b96024-7bf9-4bb0-85c6-6dc70c418df9" />

After serving the frontend:

* Upload a file or paste a video link (YouTube, Instagram, TikTok…)
* Adjust model and dubbing parameters or use auto-selection and hit the run dubbing pipeline that's it!
* Watch live logs (ASR → Translation → TTS → Merge)
* Preview or download results
* Choose **Lazy Mode** (fully automatic) or **Involve Mode** (manual fine-tuning)
* Toggle “Keep Intermediate Artefacts” to retain separated tracks or transcripts

### API Example

```bash
curl -X POST -G 'http://localhost:8000/v1/dub' \
  --data-urlencode 'video_url=/path/to/video.mp4' \
  --data-urlencode 'target_work=dub' \
  --data-urlencode 'target_langs=fr' \
  --data-urlencode 'asr_model=whisperx' \
  --data-urlencode 'tr_model=deep_translator' \
  --data-urlencode 'tts_model=edge_tts' \
  --data-urlencode 'perform_vad_trimming=true' \
  --data-urlencode 'dubbing_strategy=full_replacement' \
  --data-urlencode 'sophisticated_dub_timing=true' \
  --data-urlencode 'subtitle_style=netflix_mobile' \
  --data-urlencode 'persist_intermediate=false'
```

Outputs are saved to `apps/backend/outs/<workspace_id>/`.

---

## 💻 CLI Tools

Each microservice has its own CLI for debugging or running isolated stages:

```bash
# ASR
uv run python -m services.asr.cli /path/to/audio.wav --output-json asr.json

# Translation
uv run python -m services.translation.cli asr.json --target-lang fr --output-json translation.json

# TTS
uv run python -m services.tts.cli translation.json --workspace ./tts_out --output-json tts.json
```

Run `--help` on any CLI for available flags.

---

## 🧪 Tests

Run tests via:

```bash
make test
```

Includes:

* unit tests for service CLIs
* registry validation (ensures all registered models run properly)
* end-to-end integration test for the orchestrator pipeline

---

## ⚙️ Continuous Integration

GitHub Actions workflow (`.github/workflows/ci.yml`) automatically:

* sets up Python 3.11 + `uv`
* runs `make test`
* validates model registries and pipeline integration

Ensure your PRs keep all tests green.

---

## 🧩 Supported Models

* **ASR:** [WhisperX](https://github.com/m-bain/whisperx)
* **Translation:** [deep-translator](https://github.com/nidhaloff/deep-translator), M2M100, etc.
* **TTS:** Edge TTS, Chatterbox, and more

- **ASR:** WhisperX out of the box; extend via `services/asr/app/registry.py`.
- **Translation:** `deep_translator`, M2M100, and pluggable custom translators.
- **TTS:** Edge TTS, Chatterbox, plus any custom registry entry.


See `libs/common-schemas/config/` for model configs and supported languages.

---

## 🧠 Extending

Add new models via each service’s `registry.py` and model folder see [CONTRIBUTING.md](CONTRIBUTING.md) for more details

---

## 🤝 Contributing

Contributions are welcome!
Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting PRs or issues.

---

## 📄 License

Licensed under the [Apache License 2.0](LICENSE).

---

## 🙏 Acknowledgements

Thanks to these open-source projects:

* [FFmpeg](https://github.com/FFmpeg/FFmpeg)
* [WhisperX](https://github.com/m-bain/whisperx)
* [pyannote-audio](https://github.com/pyannote/pyannote-audio)
* [deep-translator](https://github.com/nidhaloff/deep-translator)
* [Edge-TTS](https://github.com/rany2/edge-tts)
* [yt-dlp](https://github.com/yt-dlp/yt-dlp)
* [Chatterbox](https://github.com/resemble-ai/chatterbox)

---

**Contact:**
📧 [contactglobluez@gmail.com](mailto:contactglobluez@gmail.com)

---
