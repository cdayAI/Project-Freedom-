"""Command center: one local server for the whole platform.

Research + account endpoints:
  GET /api/data          nightly research export (data.json)
  GET /api/account       sanitized Alpaca PAPER account state
  GET /api/status        ledger + loop status
  POST /api/run-daily    kick the nightly loop (one at a time)

Terminal endpoints (charts, history, quotes, paper orders):
  GET /api/bars?symbol=&timeframe=1Day&limit=500   intraday/daily candles (IEX feed)
  GET /api/history?symbol=                          15y daily candles from our store
  GET /api/quotes?symbols=A,B                       latest trade/quote snapshot batch
  GET /api/stream?symbols=A,B                       SSE: quote pushes every second
  POST /api/paper-order {symbol,side,qty,type,limit_price}  PAPER order (journaled)

Secrets never reach the browser: the server reads .env server-side and only
sanitized data goes over the wire. Binds 127.0.0.1. Latency note: transport
here is local-loopback; end-to-end quote latency is set by the feed tier
(free = IEX slice; consolidated SIP is a paid Alpaca subscription).

Run: make command-center  (http://127.0.0.1:8321)
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from alpha_forge.config import REPO_ROOT, STORE_DIR
from alpha_forge.execution.alpaca import account_snapshot_public
from alpha_forge.ledger import Ledger

DIST = REPO_ROOT / "dashboard" / "dist"
DATA_JSON = REPO_ROOT / "dashboard" / "public" / "data.json"

_daily_lock = threading.Lock()
_daily_proc: subprocess.Popen | None = None

MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
        ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png"}


class QuoteCache:
    """One background poller feeds every SSE subscriber: snapshots for the
    union of watched symbols are refreshed once per second (a single batched
    REST call), so N browser tabs never multiply upstream requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._symbols: set[str] = set()
        self._quotes: dict = {}
        self._stamp = 0.0
        self._thread: threading.Thread | None = None

    def watch(self, symbols: list[str]) -> None:
        with self._lock:
            self._symbols.update(s.upper() for s in symbols if s)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def get(self) -> tuple[dict, float]:
        with self._lock:
            return dict(self._quotes), self._stamp

    def _run(self) -> None:
        from alpha_forge.execution.marketdata import MarketDataClient

        try:
            client = MarketDataClient()
        except RuntimeError:
            return
        while True:
            with self._lock:
                syms = sorted(self._symbols)
            if syms:
                try:
                    quotes = client.snapshots(syms)
                    with self._lock:
                        self._quotes.update(quotes)
                        self._stamp = time.time()
                except Exception:
                    pass  # transient upstream errors: keep last good quotes
            time.sleep(1.0)


_quote_cache = QuoteCache()


def _journal_manual_order(entry: dict) -> None:
    path = STORE_DIR / "paper" / "manual_orders.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


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
        parsed = urlparse(self.path)
        path = parsed.path
        q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        if path == "/api/account":
            snap = account_snapshot_public()
            self._send_json(snap if snap is not None else {"error": "no credentials or API unreachable"})
            return
        if path == "/api/status":
            self._send_json(_loop_status())
            return
        if path == "/api/bars":
            try:
                from alpha_forge.execution.marketdata import MarketDataClient

                bars = MarketDataClient().bars(
                    q.get("symbol", "").upper(),
                    q.get("timeframe", "1Day"),
                    int(q.get("limit", "500")),
                )
                self._send_json({"symbol": q.get("symbol", "").upper(),
                                 "timeframe": q.get("timeframe", "1Day"),
                                 "feed": "iex", "bars": bars})
            except Exception as exc:
                self._send_json({"error": str(exc)}, 502)
            return
        if path == "/api/history":
            from alpha_forge.execution.marketdata import deep_history

            self._send_json({"symbol": q.get("symbol", "").upper(),
                             "bars": deep_history(q.get("symbol", ""))})
            return
        if path == "/api/quotes":
            symbols = [s for s in q.get("symbols", "").split(",") if s]
            try:
                from alpha_forge.execution.marketdata import MarketDataClient

                self._send_json(MarketDataClient().snapshots([s.upper() for s in symbols]))
            except Exception as exc:
                self._send_json({"error": str(exc)}, 502)
            return
        if path == "/api/stream":
            symbols = [s for s in q.get("symbols", "").split(",") if s]
            _quote_cache.watch(symbols)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last_stamp = 0.0
            try:
                while True:
                    quotes, stamp = _quote_cache.get()
                    if stamp > last_stamp:
                        last_stamp = stamp
                        payload = json.dumps(
                            {s: quotes[s] for s in (x.upper() for x in symbols) if s in quotes}
                        )
                        self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
                    time.sleep(0.5)
            except (BrokenPipeError, ConnectionResetError):
                return
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
        if self.path.split("?")[0] == "/api/paper-order":
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                symbol = str(body["symbol"]).upper()
                side = str(body["side"]).upper()
                qty = float(body["qty"])
                order_type = str(body.get("type", "market")).lower()
                limit_price = body.get("limit_price")
                if side not in ("BUY", "SELL") or qty <= 0:
                    raise ValueError("side must be BUY/SELL and qty > 0")
                from alpha_forge.execution.alpaca import AlpacaClient

                client = AlpacaClient()  # refuses live endpoint by construction
                result = client.submit_order(
                    symbol=symbol, qty=qty, side=side, order_type=order_type,
                    limit_price=float(limit_price) if limit_price else None,
                )
                _journal_manual_order(
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "source": "command_center_manual",
                        "symbol": symbol, "side": side, "qty": qty,
                        "type": order_type, "limit_price": limit_price,
                        "broker_order_id": result.get("id"),
                        "status": result.get("status"),
                    }
                )
                self._send_json({"ok": True, "order": {
                    "id": result.get("id"), "status": result.get("status"),
                    "symbol": symbol, "side": side, "qty": qty}})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, 400)
            return
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
