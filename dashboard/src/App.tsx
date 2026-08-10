import { useEffect, useState } from "react";
import Research from "./components/Research";
import Terminal from "./components/Terminal";
import type { Account, Data } from "./types";

// Shell: liquid-glass chrome, top bar (brand, tabs, account strip), and the
// two surfaces — TERMINAL (charts/quotes/orders, needs the local server) and
// RESEARCH (the nightly science, works everywhere including the static
// snapshot). Snapshot mode auto-selects RESEARCH.

export default function App() {
  const isSnapshot = !!window.__DATA__;
  const [data, setData] = useState<Data | null>(window.__DATA__ ?? null);
  const [account, setAccount] = useState<Account | null>(window.__DATA__?.account ?? null);
  const [tab, setTab] = useState<"terminal" | "research">(isSnapshot ? "research" : "terminal");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!data) {
      fetch("data.json").then((r) => r.json()).then(setData).catch((e) => setErr(String(e)));
    }
  }, [data]);

  useEffect(() => {
    if (isSnapshot) return;
    const refresh = () =>
      fetch("/api/account")
        .then((r) => r.json())
        .then((a: Account) => { if (!a.error) setAccount(a); })
        .catch(() => {});
    refresh();
    const t = setInterval(refresh, 15_000);
    return () => clearInterval(t);
  }, [isSnapshot]);

  return (
    <div className="shell">
      <div className="orbs" aria-hidden>
        <div className="orb a" /><div className="orb b" /><div className="orb c" />
      </div>

      <div className="glass topbar">
        <span className="brand">ALPHA <em>FORGE</em></span>
        <nav className="tabs" aria-label="views">
          <button className={`tab ${tab === "terminal" ? "active" : ""}`} onClick={() => setTab("terminal")}>
            Terminal
          </button>
          <button className={`tab ${tab === "research" ? "active" : ""}`} onClick={() => setTab("research")}>
            Research
          </button>
        </nav>
        <div className="acct-strip">
          {account ? (
            <>
              <div className="kv">equity<b>${account.equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b></div>
              <div className="kv">buying power<b>${account.buying_power.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b></div>
              <div className="kv">day trades (5d)<b>{account.daytrade_count}{account.pattern_day_trader ? " · PDT" : ""}</b></div>
              <div className="kv">{account.mode.toLowerCase()}<b className={account.market_open ? "up" : "dim"}>{account.market_open ? "market open" : "market closed"}</b></div>
            </>
          ) : (
            <div className="kv">broker<b className="dim">not connected</b></div>
          )}
        </div>
      </div>

      {err && <div className="notice err" style={{ padding: 20 }}>failed to load data.json: {err}</div>}
      {tab === "terminal" && <Terminal data={data} account={account} />}
      {tab === "research" && (data ? <Research data={data} /> : <div className="notice" style={{ padding: 20 }}>loading…</div>)}

      <div className="notice faint" style={{ padding: "0 18px 14px", fontSize: 11 }}>
        cumulative ledgered trials: {data?.ledger.cumulative_trials ?? "—"} · paper account only — live order flow
        is doubly locked · quotes: IEX feed (free tier)
      </div>
    </div>
  );
}
