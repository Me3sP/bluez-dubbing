.PHONY: start-ui start-api stack-up stop restart check-ports test dev dev-up dev-down dev-restart

ROOT := $(CURDIR)
UV ?= uv
RELOAD ?= --reload
ORCH_UI ?= 1

define start_service
	@echo "▶ starting $(1) service on port $(2)"
	@cd services/$(1) && $(strip $(3)) PYTHONPATH=$(ROOT) $(UV) run uvicorn app.main:app $(RELOAD) --host 0.0.0.0 --port $(2) &
endef

define stop_port
	@-fuser -k $(1)/tcp 2>/dev/null || true
endef

stack-up:
	@echo "Starting Bluez dubbing stack (ASR + translation + TTS + orchestrator)…"
	$(call start_service,asr,8001,)
	$(call start_service,translation,8002,)
	$(call start_service,tts,8003,)
	$(call start_service,orchestrator,8000,ORCHESTRATOR_ENABLE_UI=$(ORCH_UI))
	@if [ "$(ORCH_UI)" = "1" ]; then \
		echo "All services running. Orchestrator UI ⇒ http://localhost:8000/ui"; \
	else \
		echo "All services running. Call the orchestrator API via http://localhost:8000/v1/dub"; \
	fi

start-ui:
	@$(MAKE) ORCH_UI=1 stack-up

start-api:
	@$(MAKE) ORCH_UI=0 stack-up

stop dev-down:
	@echo "Stopping Bluez dubbing stack…"
	@-pkill -f "uvicorn app.main:app" || true
	@sleep 1
	$(call stop_port,8000)
	$(call stop_port,8001)
	$(call stop_port,8002)
	$(call stop_port,8003)
	@echo "All services stopped."

restart dev-restart: stop
	@sleep 1
	@$(MAKE) stack-up

dev dev-up: start-ui

check-ports:
	@echo "Checking port status..."
	@echo "Port 8000 (orchestrator):"
	@-lsof -i :8000 || echo "  Available"
	@echo "Port 8001 (ASR):"
	@-lsof -i :8001 || echo "  Available"
	@echo "Port 8002 (translation):"
	@-lsof -i :8002 || echo "  Available"
	@echo "Port 8003 (TTS):"
	@-lsof -i :8003 || echo "  Available"

test:
	cd services/asr && $(UV) run --with pytest pytest
	cd services/translation && $(UV) run --with pytest pytest
	cd services/tts && $(UV) run --with pytest pytest
	cd services/orchestrator && $(UV) run --with pytest --with pytest-asyncio pytest
