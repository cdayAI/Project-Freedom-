import { useCallback, useEffect, useRef, useState } from "react";
import Chart, { Bar } from "./Chart";
import type { Account, Data } from "../types";

type Quote = {
  last: number | null; bid: number | null; ask: number | null;
  change_pct: number | null; day_volume: number | null; feed: string;
};

const TIMEFRAMES = [
  { label: "1m", tf: "1Min" },
  { label: "5m", tf: "5Min" },
  { label: "15m", tf: "15Min" },
  { label: "1h", tf: "1Hour" },
  { label: "1D", tf: "1Day" },
  { label: "MAX", tf: "history" },
];

export default function Terminal({ data, account }: { data: Data | null; account: Account | null }) {
  const [symbol, setSymbol] = useState("AAPL");
  const [tf, setTf] = useState("1Day");
  const [bars, setBars] = useState<Bar[]>([]);
  const [quotes, setQuotes] = useState<Record<string, Quote>>({});
  const [flash, setFlash] = useState<Record<string, "up" | "down">>({});
  const [offline, setOffline] = useState(false);
  const [search, setSearch] = useState("");
  const prevLast = useRef<Record<string, number>>({});

  const watchlist = Array.from(
    new Set([
      ...(account?.positions?.map((p) => p.symbol) ?? []),
      ...(data?.predictions?.predictions?.slice(0, 10).map((p) => p.instrument) ?? []),
      symbol,
    ])
  );

  const loadBars = useCallback((sym: string, timeframe: string) => {
    const url =
      timeframe === "history"
        ? `/api/history?symbol=${sym}`
        : `/api/bars?symbol=${sym}&timeframe=${timeframe}&limit=500`;
    fetch(url)
      .then((r) => r.json())
      .then((d) => {
        if (d.bars?.length) setBars(d.bars);
        else if (timeframe !== "history") {
          // thin intraday coverage on IEX: fall back to our deep history
          fetch(`/api/history?symbol=${sym}`).then((r) => r.json()).then((h) => setBars(h.bars ?? []));
        } else setBars([]);
      })
      .catch(() => setOffline(true));
  }, []);

  useEffect(() => { loadBars(symbol, tf); }, [symbol, tf, loadBars]);

  // live quotes over SSE; the server multiplexes one upstream poll
  useEffect(() => {
    const syms = watchlist.join(",");
    if (!syms) return;
    let es: EventSource | null = null;
    try {
      es = new EventSource(`/api/stream?symbols=${syms}`);
      es.onmessage = (ev) => {
        const q: Record<string, Quote> = JSON.parse(ev.data);
        setQuotes((old) => ({ ...old, ...q }));
        const f: Record<string, "up" | "down"> = {};
        for (const [s, v] of Object.entries(q)) {
          const prev = prevLast.current[s];
          if (v.last != null && prev != null && v.last !== prev) f[s] = v.last > prev ? "up" : "down";
          if (v.last != null) prevLast.current[s] = v.last;
        }
        if (Object.keys(f).length) {
          setFlash(f);
          setTimeout(() => setFlash({}), 650);
        }
      };
      es.onerror = () => { setOffline(true); es?.close(); };
    } catch { setOffline(true); }
    return () => es?.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [watchlist.join(",")]);

  const q = quotes[symbol];
  const chg = q?.change_pct;

  return (
    <div className="layout">
      <div className="glass panel">
        <div className="chart-head">
          <span className="sym">{symbol}</span>
          <span className="px">{q?.last != null ? q.last.toFixed(2) : "—"}</span>
          <span className={`chg ${chg == null ? "dim" : chg >= 0 ? "up" : "down"}`}>
            {chg == null ? "" : `${chg >= 0 ? "+" : ""}${chg.toFixed(2)}%`}
          </span>
          {q?.bid != null && q?.ask != null && (
            <span className="mono dim" style={{ fontSize: 12 }}>
              {q.bid.toFixed(2)} × {q.ask.toFixed(2)}
            </span>
          )}
          <span className="meta">
            {offline ? "static snapshot — run `make command-center` for live data" : `feed: IEX (free tier) · SIP available on paid plan`}
          </span>
        </div>
        <div className="tf-row">
          {TIMEFRAMES.map((t) => (
            <button key={t.tf} className={`tf ${tf === t.tf ? "active" : ""}`} onClick={() => setTf(t.tf)}>
              {t.label}
            </button>
          ))}
        </div>
        {bars.length ? (
          <Chart bars={bars} live={q?.last} />
        ) : (
          <div className="notice" style={{ padding: 40 }}>
            {offline
              ? "charts need the local server: make command-center"
              : "no bars for this symbol/timeframe (IEX coverage is thin off-hours — try 1D or MAX)"}
          </div>
        )}
      </div>

      <div className="rail">
        <div className="glass panel">
          <h2>Watchlist</h2>
          {watchlist.map((s) => {
            const wq = quotes[s];
            return (
              <div
                key={s}
                className={`watch-row ${s === symbol ? "sel" : ""} ${flash[s] ? `flash-${flash[s]}` : ""}`}
                onClick={() => setSymbol(s)}
              >
                <span className="wsym">{s}</span>
                <span>{wq?.last != null ? wq.last.toFixed(2) : "—"}</span>
                <span className={wq?.change_pct == null ? "faint" : wq.change_pct >= 0 ? "up" : "down"}>
                  {wq?.change_pct == null ? "" : `${wq.change_pct >= 0 ? "+" : ""}${wq.change_pct.toFixed(2)}%`}
                </span>
              </div>
            );
          })}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (search.trim()) { setSymbol(search.trim().toUpperCase()); setSearch(""); }
            }}
          >
            <input
              className="search"
              style={{ width: "100%", marginTop: 8 }}
              placeholder="add symbol…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </form>
        </div>

        <OrderTicket symbol={symbol} quote={q} disabled={offline} />

        <div className="glass panel">
          <h2>Positions</h2>
          {account?.positions?.length ? (
            <table className="pos-table">
              <thead><tr><th>sym</th><th>qty</th><th>value</th><th>uP/L</th></tr></thead>
              <tbody>
                {account.positions.map((p) => (
                  <tr key={p.symbol} onClick={() => setSymbol(p.symbol)} style={{ cursor: "pointer" }}>
                    <td>{p.symbol}</td>
                    <td>{p.qty}</td>
                    <td>${p.market_value.toFixed(0)}</td>
                    <td className={p.unrealized_pl >= 0 ? "up" : "down"}>${p.unrealized_pl.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="notice">no open positions</div>
          )}
          {account?.open_orders?.length ? (
            <div className="notice" style={{ marginTop: 8 }}>
              open: {account.open_orders.map((o) => `${o.side} ${o.qty} ${o.symbol}`).join(", ")}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function OrderTicket({ symbol, quote, disabled }: { symbol: string; quote?: Quote; disabled: boolean }) {
  const [side, setSide] = useState<"BUY" | "SELL">("BUY");
  const [qty, setQty] = useState("1");
  const [type, setType] = useState<"market" | "limit">("market");
  const [limit, setLimit] = useState("");
  const [msg, setMsg] = useState<{ text: string; kind: "ok" | "err" } | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = () => {
    setBusy(true);
    setMsg(null);
    fetch("/api/paper-order", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symbol, side, qty: parseFloat(qty), type,
        limit_price: type === "limit" ? parseFloat(limit) : undefined,
      }),
    })
      .then((r) => r.json())
      .then((d) =>
        setMsg(
          d.ok
            ? { text: `${d.order.status}: ${side} ${qty} ${symbol}`, kind: "ok" }
            : { text: d.error ?? "order rejected", kind: "err" }
        )
      )
      .catch((e) => setMsg({ text: String(e), kind: "err" }))
      .finally(() => setBusy(false));
  };

  const est = quote?.last != null ? (parseFloat(qty) || 0) * quote.last : null;

  return (
    <div className="glass panel ticket">
      <h2>Order — {symbol}</h2>
      <span className="paper-tag">PAPER ACCOUNT</span>
      <div className="seg">
        <button className={`buy ${side === "BUY" ? "on" : ""}`} onClick={() => setSide("BUY")}>BUY</button>
        <button className={`sell ${side === "SELL" ? "on" : ""}`} onClick={() => setSide("SELL")}>SELL</button>
      </div>
      <div className="row2">
        <input value={qty} onChange={(e) => setQty(e.target.value)} placeholder="qty" aria-label="quantity" />
        <select value={type} onChange={(e) => setType(e.target.value as "market" | "limit")} aria-label="order type">
          <option value="market">market</option>
          <option value="limit">limit</option>
        </select>
      </div>
      {type === "limit" && (
        <input value={limit} onChange={(e) => setLimit(e.target.value)} placeholder="limit price" aria-label="limit price" />
      )}
      {est != null && <div className="notice mono">≈ ${est.toFixed(2)} notional</div>}
      <button className="submit" onClick={submit} disabled={disabled || busy || !(parseFloat(qty) > 0)}>
        {busy ? "submitting…" : `${side} ${symbol}`}
      </button>
      {msg && <div className={`notice ${msg.kind}`}>{msg.text}</div>}
      {disabled && <div className="notice">order entry needs the local server</div>}
    </div>
  );
}
