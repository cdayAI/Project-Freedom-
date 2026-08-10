.PHONY: setup daily test verify-ledger clean-caches dashboard command-center

PY := .venv/bin/python
PIP := .venv/bin/pip

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip >/dev/null
	$(PIP) install -e ".[dev]"
	@echo "setup complete"

daily:
	$(PY) -m alpha_forge.orchestrator.daily

test:
	$(PY) -m pytest

verify-ledger:
	$(PY) -m alpha_forge.ledger.verify

dashboard:
	cd dashboard && npm install && npm run build
	$(PY) -c "from alpha_forge.reporting.export import build_static_snapshot; print(build_static_snapshot())"

command-center:
	$(PY) -m alpha_forge.command_center.server

clean-caches:
	rm -rf data/raw/*.tmp
