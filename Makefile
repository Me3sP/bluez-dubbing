.PHONY: run-all stop-all restart-all check-ports

run-all:
	@echo "Starting all services..."
	cd services/asr && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8001 &
	cd services/translation && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8002 &
	cd services/tts && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8003 &
	cd services/orchestrator && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
	@echo "All services started!"

stop-all:
	@echo "Stopping all services..."
	-pkill -9 -f "uvicorn app.main:app"
	@sleep 2
	@echo "Verifying ports are free..."
	@-fuser -k 8000/tcp 2>/dev/null || true
	@-fuser -k 8001/tcp 2>/dev/null || true
	@-fuser -k 8002/tcp 2>/dev/null || true
	@-fuser -k 8003/tcp 2>/dev/null || true
	@sleep 1
	@echo "All services stopped!"

restart-all: stop-all
	@sleep 2
	@$(MAKE) run-all

check-ports:
	@echo "Checking port status..."
	@echo "Port 8000 (orchestrator):"
	@-lsof -i :8000 || echo "  Available"
	@echo "Port 8001 (asr):"
	@-lsof -i :8001 || echo "  Available"
	@echo "Port 8002 (translation):"
	@-lsof -i :8002 || echo "  Available"
	@echo "Port 8003 (tts):"
	@-lsof -i :8003 || echo "  Available"