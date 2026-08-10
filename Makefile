.PHONY: setup daily test verify-ledger clean-caches

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

clean-caches:
	rm -rf data/raw/*.tmp
