"""Test-suite hard guards.

1. Broker credentials never leak from .env into tests: every test starts
   with ALPACA_* stripped; tests that need credentials set fakes explicitly.
2. No test may talk to a broker API: any HTTP request to an alpaca.markets
   host fails loudly. A unit test that places a real (even paper) order is a
   defect — this guard exists because it happened once.
"""

import pytest
import requests


@pytest.fixture(autouse=True)
def _no_broker_env(monkeypatch):
    for var in ("ALPACA_BASE_URL", "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _no_broker_network(monkeypatch):
    real_request = requests.sessions.Session.request

    def guarded(self, method, url, *args, **kwargs):
        if "alpaca.markets" in str(url):
            raise AssertionError(f"test attempted a real broker API call: {method} {url}")
        return real_request(self, method, url, *args, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "request", guarded)
