# Simple helpers (requires: cmake, Python, Node for full stack)

.PHONY: engine-build engine-run server-venv server-run

engine-build:
	cmake -S engine -B engine/build
	cmake --build engine/build -j

engine-run: engine-build
	./engine/build/chess_engine

server-venv:
	python -m venv .venv && . .venv/bin/activate && pip install -r server/requirements.txt

server-run:
	. .venv/bin/activate && uvicorn server.app.main:app --reload --port 8000
