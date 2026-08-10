import { useEffect, useState } from "react";

// Reads the nightly export. In the static snapshot the data is inlined as
// window.__DATA__; in dev/build it is fetched from public/data.json.
// The dashboard displays research output; it never computes research numbers.

type Data = {
  generated_utc: string;
  ledger: { entries: number; cumulative_trials: number };
  replacement_rate: {
    trailing_3m_rate: number | null;
    trailing_3m_graduated: number;
    trailing_3m_killed: number;
    healthy: boolean;
  };
  book: unknown[];
  gate_reports: {
    strategy_id: string;
    verdict: string;
    reasons: string[];
    net_vs_gross: { gross_total_return: number; net_total_return: number } | null;
    dsr: { dsr_probability: number | null; n_trials: number | null };
    pbo: number | null;
  }[];
  path_catalog: { n_multiple: number; symbol_window_hits: number; windows_with_any_path: number }[];
  confluence: { n_multiple: number; feature: string; auc: number | null; p_value: number | null; verdict: string }[];
  calibration: { horizon: number; bucket: number; n: number; realized_positive_rate: number }[];
  equity_curves: { month: string; gross: number; net: number }[];
  predictions: {
    data_vintage: string;
    predictions: {
      instrument: string;
      score: number;
      confidence: { value: number; status: string };
      last_close: number;
      historical_analogs: { symbol: string; entry_date: string }[];
    }[];
  } | null;
};

declare global {
  interface Window { __DATA__?: Data }
}

const css: Record<string, React.CSSProperties> = {
  page: { fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", background: "#0d1117", color: "#c9d1d9", minHeight: "100vh", padding: "24px", lineHeight: 1.5 },
  h1: { color: "#e6edf3", fontSize: 22, marginBottom: 4 },
  sub: { color: "#8b949e", fontSize: 12, marginBottom: 24 },
  grid: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: 16 },
  card: { background: "#161b22", border: "1px solid #30363d", borderRadius: 8, padding: 16 },
  h2: { color: "#e6edf3", fontSize: 14, marginBottom: 10, textTransform: "uppercase", letterSpacing: 1 },
  big: { fontSize: 28, color: "#e6edf3" },
  dim: { color: "#8b949e", fontSize: 12 },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 12 },
  th: { textAlign: "left", color: "#8b949e", borderBottom: "1px solid #30363d", padding: "4px 8px 4px 0" },
  td: { padding: "4px 8px 4px 0", borderBottom: "1px solid #21262d" },
  kill: { color: "#f85149" },
  pass: { color: "#3fb950" },
  flag: { color: "#d29922" },
};

function EquityCurve({ rows }: { rows: Data["equity_curves"] }) {
  if (!rows.length) return <div style={css.dim}>no gated backtest curves yet</div>;
  const w = 420, h = 160, pad = 8;
  const vals = rows.flatMap((r) => [r.gross, r.net]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x = (i: number) => pad + (i / (rows.length - 1)) * (w - 2 * pad);
  const y = (v: number) => h - pad - ((v - lo) / (hi - lo || 1)) * (h - 2 * pad);
  const path = (k: "gross" | "net") => rows.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(r[k]).toFixed(1)}`).join("");
  return (
    <div>
      <svg width={w} height={h} style={{ maxWidth: "100%" }}>
        <path d={path("gross")} fill="none" stroke="#8b949e" strokeWidth={1.5} strokeDasharray="4 3" />
        <path d={path("net")} fill="none" stroke="#58a6ff" strokeWidth={2} />
      </svg>
      <div style={css.dim}>dashed = gross, solid = net (gate 6: never shown apart)</div>
    </div>
  );
}

function verdictStyle(v: string) {
  return v === "PASS" ? css.pass : v === "KILL" ? css.kill : css.flag;
}

export default function App() {
  const [data, setData] = useState<Data | null>(window.__DATA__ ?? null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!data) {
      fetch("data.json").then((r) => r.json()).then(setData).catch((e) => setErr(String(e)));
    }
  }, [data]);
  if (err) return <div style={css.page}>failed to load data.json: {err}</div>;
  if (!data) return <div style={css.page}>loading…</div>;

  const rr = data.replacement_rate;
  const rrText = rr.trailing_3m_rate === null ? "N/A" : rr.trailing_3m_rate === Infinity ? "∞" : rr.trailing_3m_rate?.toFixed(2);

  return (
    <div style={css.page}>
      <div style={css.h1}>ALPHA FORGE</div>
      <div style={css.sub}>
        regenerated {data.generated_utc} · ledger entries {data.ledger.entries} ·{" "}
        <b>cumulative trials {data.ledger.cumulative_trials}</b>
      </div>
      <div style={css.grid}>
        <div style={css.card}>
          <div style={css.h2}>Replacement rate (north star)</div>
          <div style={css.big}>{rrText}</div>
          <div style={css.dim}>
            trailing 3m: {rr.trailing_3m_graduated} graduated / {rr.trailing_3m_killed} killed
            {rr.trailing_3m_rate !== null && !rr.healthy && (
              <div style={css.kill}>BELOW 1.0 — pipeline is losing to the graveyard</div>
            )}
          </div>
        </div>

        <div style={css.card}>
          <div style={css.h2}>Strategy book</div>
          {data.book.length === 0 ? (
            <div>
              <div style={css.big}>EMPTY</div>
              <div style={css.dim}>no strategy has passed all gates + live requirements; no growth rate is quoted from backtests</div>
            </div>
          ) : (
            <div>{data.book.length} strategies</div>
          )}
        </div>

        <div style={css.card}>
          <div style={css.h2}>N-x path catalog</div>
          <table style={css.table}>
            <thead><tr><th style={css.th}>N-x</th><th style={css.th}>symbol-window hits</th><th style={css.th}>windows w/ path</th></tr></thead>
            <tbody>
              {data.path_catalog.map((r) => (
                <tr key={r.n_multiple}>
                  <td style={css.td}>{r.n_multiple}x</td>
                  <td style={css.td}>{r.symbol_window_hits}</td>
                  <td style={css.td}>{r.windows_with_any_path}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={css.card}>
          <div style={css.h2}>Latest gated backtest — equity (net of cost)</div>
          <EquityCurve rows={data.equity_curves} />
        </div>

        <div style={css.card}>
          <div style={css.h2}>Gate outcomes</div>
          {data.gate_reports.map((g, i) => (
            <div key={i} style={{ marginBottom: 10 }}>
              <span style={verdictStyle(g.verdict)}>{g.verdict}</span> {g.strategy_id}
              <div style={css.dim}>
                DSR {g.dsr.dsr_probability?.toFixed(3)} @ N={g.dsr.n_trials} · PBO {g.pbo?.toFixed(3)}
                {g.net_vs_gross && (
                  <> · gross {(g.net_vs_gross.gross_total_return * 100).toFixed(1)}% / net {(g.net_vs_gross.net_total_return * 100).toFixed(1)}%</>
                )}
              </div>
              {g.reasons.slice(0, 3).map((r, j) => (<div key={j} style={{ ...css.dim, fontSize: 11 }}>· {r}</div>))}
            </div>
          ))}
        </div>

        <div style={css.card}>
          <div style={css.h2}>Confluence — identifiable fingerprint features</div>
          <table style={css.table}>
            <thead><tr><th style={css.th}>class</th><th style={css.th}>feature</th><th style={css.th}>AUC</th><th style={css.th}>verdict</th></tr></thead>
            <tbody>
              {data.confluence.filter((c) => c.verdict !== "UNDERPOWERED").map((c, i) => (
                <tr key={i}>
                  <td style={css.td}>{c.n_multiple}x</td>
                  <td style={css.td}>{c.feature}</td>
                  <td style={css.td}>{c.auc?.toFixed(3) ?? "—"}</td>
                  <td style={{ ...css.td, ...(c.verdict === "IDENTIFIABLE" ? css.pass : css.dim) }}>{c.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={css.card}>
          <div style={css.h2}>Prediction feed (research-only, size 0)</div>
          {data.predictions?.predictions?.length ? (
            <table style={css.table}>
              <thead><tr><th style={css.th}>symbol</th><th style={css.th}>score</th><th style={css.th}>confidence</th><th style={css.th}>analog</th></tr></thead>
              <tbody>
                {data.predictions.predictions.slice(0, 10).map((p) => (
                  <tr key={p.instrument}>
                    <td style={css.td}>{p.instrument}</td>
                    <td style={css.td}>{p.score.toFixed(2)}</td>
                    <td style={css.td}>{p.confidence.value.toFixed(2)} <span style={css.flag}>({p.confidence.status})</span></td>
                    <td style={css.td}>{p.historical_analogs[0]?.symbol} @ {p.historical_analogs[0]?.entry_date?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div style={css.dim}>no predictions emitted (requires identifiable confluence features)</div>
          )}
        </div>

        <div style={css.card}>
          <div style={css.h2}>Calibration (predicted vs realized)</div>
          {data.calibration.length ? (
            <table style={css.table}>
              <thead><tr><th style={css.th}>horizon</th><th style={css.th}>bucket</th><th style={css.th}>n</th><th style={css.th}>realized +rate</th></tr></thead>
              <tbody>
                {data.calibration.map((c, i) => (
                  <tr key={i}>
                    <td style={css.td}>{c.horizon}d</td>
                    <td style={css.td}>{c.bucket}</td>
                    <td style={css.td}>{c.n}</td>
                    <td style={css.td}>{(c.realized_positive_rate * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div style={css.dim}>no matured predictions yet — calibration is the system's fitness function and accrues from live grading only</div>
          )}
        </div>
      </div>
    </div>
  );
}
