import type { Data } from "../types";

function EquityCurve({ rows }: { rows: Data["equity_curves"] }) {
  if (!rows.length) return <div className="notice">no gated backtest curves yet</div>;
  const w = 420, h = 150, pad = 8;
  const vals = rows.flatMap((r) => [r.gross, r.net]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x = (i: number) => pad + (i / (rows.length - 1)) * (w - 2 * pad);
  const y = (v: number) => h - pad - ((v - lo) / (hi - lo || 1)) * (h - 2 * pad);
  const path = (k: "gross" | "net") =>
    rows.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(r[k]).toFixed(1)}`).join("");
  return (
    <div>
      <svg width={w} height={h} style={{ maxWidth: "100%" }}>
        <path d={path("gross")} fill="none" stroke="#8b95a8" strokeWidth={1.5} strokeDasharray="4 3" />
        <path d={path("net")} fill="none" stroke="#5eead4" strokeWidth={2} />
      </svg>
      <div className="notice">dashed = gross, solid = net (gate 6: never shown apart)</div>
    </div>
  );
}

const verdictClass = (v: string) => (v === "PASS" ? "pass" : v === "KILL" ? "kill" : "flag");

export default function Research({ data }: { data: Data }) {
  const rr = data.replacement_rate;
  const rrText =
    rr.trailing_3m_rate === null ? "N/A" : rr.trailing_3m_rate === Infinity ? "∞" : rr.trailing_3m_rate?.toFixed(2);

  return (
    <div className="rgrid">
      <div className="glass panel">
        <h2>Replacement rate (north star)</h2>
        <div className="big">{rrText}</div>
        <div className="notice">
          trailing 3m: {rr.trailing_3m_graduated} graduated / {rr.trailing_3m_killed} killed
          {rr.trailing_3m_rate !== null && !rr.healthy && (
            <div className="kill">BELOW 1.0 — pipeline is losing to the graveyard</div>
          )}
        </div>
      </div>

      <div className="glass panel">
        <h2>Strategy book</h2>
        {data.book.length === 0 ? (
          <div>
            <div className="big">EMPTY</div>
            <div className="notice">
              no strategy has passed all gates + live requirements; no growth rate is quoted from backtests
            </div>
          </div>
        ) : (
          <div>{data.book.length} strategies</div>
        )}
      </div>

      <div className="glass panel">
        <h2>N-x path catalog</h2>
        <table className="rtable">
          <thead><tr><th>N-x</th><th>symbol-window hits</th><th>windows w/ path</th></tr></thead>
          <tbody>
            {data.path_catalog.map((r) => (
              <tr key={r.n_multiple}>
                <td>{r.n_multiple}x</td>
                <td>{r.symbol_window_hits}</td>
                <td>{r.windows_with_any_path}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="glass panel">
        <h2>Latest gated backtest — equity (net of cost)</h2>
        <EquityCurve rows={data.equity_curves} />
      </div>

      <div className="glass panel">
        <h2>Gate outcomes</h2>
        {data.gate_reports.map((g, i) => (
          <div key={i} style={{ marginBottom: 10 }}>
            <span className={verdictClass(g.verdict)}>{g.verdict}</span> {g.strategy_id}
            <div className="notice">
              DSR {g.dsr.dsr_probability?.toFixed(3)} @ N={g.dsr.n_trials} · PBO {g.pbo?.toFixed(3)}
              {g.net_vs_gross && (
                <>
                  {" "}· gross {(g.net_vs_gross.gross_total_return * 100).toFixed(1)}% / net{" "}
                  {(g.net_vs_gross.net_total_return * 100).toFixed(1)}%
                </>
              )}
            </div>
            {g.reasons.slice(0, 3).map((r, j) => (
              <div key={j} className="notice" style={{ fontSize: 11 }}>· {r}</div>
            ))}
          </div>
        ))}
      </div>

      <div className="glass panel">
        <h2>Confluence — identifiable fingerprint features</h2>
        <table className="rtable">
          <thead><tr><th>class</th><th>feature</th><th>AUC</th><th>verdict</th></tr></thead>
          <tbody>
            {data.confluence
              .filter((c) => c.verdict !== "UNDERPOWERED")
              .map((c, i) => (
                <tr key={i}>
                  <td>{c.n_multiple}x</td>
                  <td>{c.feature}</td>
                  <td>{c.auc?.toFixed(3) ?? "—"}</td>
                  <td className={c.verdict === "IDENTIFIABLE" ? "pass" : "faint"}>{c.verdict}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>

      <div className="glass panel">
        <h2>Prediction feed (research-only, size 0)</h2>
        {data.predictions?.predictions?.length ? (
          <table className="rtable">
            <thead><tr><th>symbol</th><th>score</th><th>confidence</th><th>analog</th></tr></thead>
            <tbody>
              {data.predictions.predictions.slice(0, 10).map((p) => (
                <tr key={p.instrument}>
                  <td>{p.instrument}</td>
                  <td>{p.score.toFixed(2)}</td>
                  <td>
                    {p.confidence.value.toFixed(2)} <span className="flag">({p.confidence.status})</span>
                  </td>
                  <td>
                    {p.historical_analogs[0]?.symbol} @ {p.historical_analogs[0]?.entry_date?.slice(0, 10)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="notice">no predictions emitted (requires identifiable confluence features)</div>
        )}
      </div>

      <div className="glass panel">
        <h2>Calibration (predicted vs realized)</h2>
        {data.calibration.length ? (
          <table className="rtable">
            <thead><tr><th>horizon</th><th>bucket</th><th>n</th><th>realized +rate</th></tr></thead>
            <tbody>
              {data.calibration.map((c, i) => (
                <tr key={i}>
                  <td>{c.horizon}d</td>
                  <td>{c.bucket}</td>
                  <td>{c.n}</td>
                  <td>{(c.realized_positive_rate * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="notice">
            no matured predictions yet — calibration is the system's fitness function and accrues from live
            grading only
          </div>
        )}
      </div>
    </div>
  );
}
