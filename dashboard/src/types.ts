export type Account = {
  mode: string;
  status: string;
  equity: number;
  cash: number;
  buying_power: number;
  pattern_day_trader: boolean;
  daytrade_count: number;
  market_open: boolean;
  next_open: string | null;
  positions: { symbol: string; qty: number; avg_entry_price: number; market_value: number; unrealized_pl: number }[];
  open_orders: { symbol: string; side: string; qty: string; type: string; status: string }[];
  error?: string;
};

export type Data = {
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
  account?: Account | null;
};

declare global {
  interface Window { __DATA__?: Data }
}
