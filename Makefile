.PHONY: run-all stop-all

run-all:
	cd services/asr && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8001 &
	cd services/translation && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8002 &
	cd services/tts && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8003 &
	cd services/orchestrator && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &

stop-all:
	pkill -f "uvicorn app.main:app"