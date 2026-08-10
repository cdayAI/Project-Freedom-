"""Command center: one local server for managing the whole system.

Serves the built dashboard plus live JSON endpoints:

  GET /api/data      nightly research export (same data.json the snapshot uses)
  GET /api/account   sanitized Alpaca PAPER account state (balances, positions,
                     open orders, PDT flag/day-trade count, market clock)
  GET /api/status    loop status: ledger stats, last vintage, last run outputs
  POST /api/run-daily  kicks off `make daily` in the background (one at a time)

Secrets never reach the browser: the server reads .env server-side and only
sanitized snapshots go over the wire. Binds 127.0.0.1 by default.

Run: make command-center  (http://127.0.0.1:8321)
"""

from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from alpha_forge.config import REPO_ROOT
from alpha_forge.execution.alpaca import account_snapshot_public
from alpha_forge.ledger import Ledger

DIST = REPO_ROOT / "dashboard" / "dist"
DATA_JSON = REPO_ROOT / "dashboard" / "public" / "data.json"

_daily_lock = threading.Lock()
_daily_proc: subprocess.Popen | None = None

MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
        ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png"}


def _loop_status() -> dict:
    ledger = Ledger()
    global _daily_proc
    running = _daily_proc is not None and _daily_proc.poll() is None
    last_exit = None if running or _daily_proc is None else _daily_proc.returncode
    findings = REPO_ROOT / "FINDINGS.md"
    return {
        "ledger_entries": sum(1 for _ in ledger.entries()),
        "cumulative_trials": ledger.trial_count(),
        "daily_running": running,
        "last_daily_exit_code": last_exit,
        "findings_mtime": findings.stat().st_mtime if findings.exists() else None,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode())

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/api/account":
            snap = account_snapshot_public()
            self._send_json(snap if snap is not None else {"error": "no credentials or API unreachable"})
            return
        if path == "/api/status":
            self._send_json(_loop_status())
            return
        if path == "/api/data" or path == "/data.json":
            if DATA_JSON.exists():
                self._send(200, DATA_JSON.read_bytes())
            else:
                self._send_json({"error": "no export yet — run make daily"}, 404)
            return
        # static dashboard (path-traversal-safe: resolved file must stay in DIST)
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (DIST / rel).resolve()
        if f.is_file() and DIST.resolve() in [f.parent, *f.parents]:
            self._send(200, f.read_bytes(), MIME.get(f.suffix, "application/octet-stream"))
        else:
            self._send_json({"error": "not found (run `make dashboard` first?)"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] == "/api/run-daily":
            global _daily_proc
            with _daily_lock:
                if _daily_proc is not None and _daily_proc.poll() is None:
                    self._send_json({"started": False, "reason": "already running"}, 409)
                    return
                _daily_proc = subprocess.Popen(
                    [str(REPO_ROOT / ".venv" / "bin" / "python"), "-m",
                     "alpha_forge.orchestrator.daily"],
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self._send_json({"started": True})
            return
        self._send_json({"error": "not found"}, 404)

    def log_message(self, fmt: str, *args) -> None:  # quiet
        pass


def main(host: str = "127.0.0.1", port: int = 8321) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"ALPHA FORGE command center: http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
