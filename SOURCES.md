# SOURCES

Every number in the system is computed from data or cited here. Values that
could not be verified against a primary source are stored `UNVERIFIED` and
raise at use. Compiled 2026-08-10.

## Regulatory fee rates (full tables in `data/fees/*.json`, entry-level citations inside)

### SEC Section 31 (per $1M covered sales; charge-date based)
| Effective | Rate | Primary source |
|---|---|---|
| 2013-05-25 | $17.40 | [FY2013 Fee Rate Advisory #3](https://www.sec.gov/news/press/2013/2013-74.htm) |
| 2014-03-18 | $22.10 | [FY2014 Fee Rate Advisory #3](https://www.sec.gov/News/PressRelease/Detail/PressRelease/1370540783933) |
| 2015-02-14 | $18.40 | [FY2015 Fee Rate Advisory #3](https://www.sec.gov/news/pressrelease/2015-8.html) |
| 2016-02-16 | $21.80 | [FY2016 Fee Rate Advisory #3](https://www.sec.gov/news/pressrelease/2016-2.html) |
| 2017-07-04 | $23.10 | [FY2017 Fee Rate Advisory #3](https://www.sec.gov/news/press-release/2017-111) |
| 2018-05-22 | $13.00 | [FY2018 Fee Rate Advisory #3](https://www.sec.gov/news/press-release/2018-67) |
| 2019-04-16 | $20.70 | [FY2019 Fee Rate Advisory #2](https://www.sec.gov/news/press-release/2019-30) |
| 2020-02-18 | $22.10 | [FY2020 Fee Rate Advisory #2](https://www.sec.gov/news/press-release/2020-7) |
| 2021-02-25 | $5.10 | [FY2021 Fee Rate Advisory #2](https://www.sec.gov/news/press-release/2021-8) |
| 2022-05-14 | $22.90 | [FY2022 Fee Rate Advisory #2](https://www.sec.gov/news/press-release/2022-60) |
| 2023-02-27 | $8.00 | [FY2023 Fee Rate Advisory #2](https://www.sec.gov/newsroom/press-releases/2023-15) |
| 2024-05-22 | $27.80 | [FY2024 Fee Rate Advisory #2](https://www.sec.gov/rules-regulations/fee-rate-advisories/2024-2) |
| 2025-05-14 | $0.00 | [FY2025 Advisory](https://www.sec.gov/rules-regulations/fee-rate-advisories/2025-2) |
| 2026-04-04 | $20.60 | [FY2026 Advisory](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2), [Order 34-104909](https://www.sec.gov/files/rules/other/2026/34-104909.pdf) |

No-mid-year-change confirmations were fetched for FY2015/16/20/21/23; the
FY2013 rate's continuation into 2014 confirmed by FY2014 Advisories #2/#4.

### FINRA Trading Activity Fee — covered equity sales (per share, capped/trade)
| Effective | Rate / cap | Primary source |
|---|---|---|
| 2012-07-01 | $0.000119 / $5.95 | [Regulatory Notice 12-31](https://www.finra.org/rules-guidance/notices/12-31) |
| 2022-01-01 | $0.000130 / $6.49 | [SEC Rel. 34-90176 (SR-FINRA-2020-032)](https://www.sec.gov/files/rules/sro/finra/2020/34-90176.pdf) |
| 2023-01-01 | $0.000145 / $7.27 | same phase-in filing |
| 2024-01-01 | $0.000166 / $8.30 | same; confirmed in SR-FINRA-2024-019 & By-Laws Schedule A |
| 2026-01-01 | $0.000195 / $9.79 | [SR-FINRA-2024-019 Exhibit 5C](https://www.finra.org/sites/default/files/2024-11/sr-finra-2024-019.pdf) |

Rel. 34-90176 states TAF rates unchanged 2012→2021; 2025 explicitly
no-change per SR-FINRA-2024-019.

### FINRA TAF — option sales (per contract)
$0.002 (in force through 2021, window-start anchor 2014-01-01; original
effective date pre-2014, deliberately not stated) → $0.00218 (2022-01-01) →
$0.00244 (2023-01-01) → $0.00279 (2024-01-01) → $0.00329 (2026-01-01).
Sources: same two filings as above.

### NFA assessment (futures, per side)
$0.01 as of 2026-07-01; $0.02 effective 2027-07-01 —
[NFA Bylaw 1301](https://www.nfa.futures.org/rulebooksql/rules.aspx?Section=3&RuleID=BYLAW+1301)
and [NFA Assessment Fees FAQs](https://www.nfa.futures.org/faqs/members/nfa-assessment-fees.html).
Pre-2026 history: UNVERIFIED effective date (blocks historical futures backtests).

### Not yet sourced (stored as blocking-UNVERIFIED, no table on disk)
ORF per exchange per month; OCC clearing fee schedule; CME exchange+clearing
fees per product (member/non-member). Any code path touching these raises.

## Market-microstructure constants
- Minimum equity price increment $0.01 (tick floor for the spread model):
  SEC Rule 612 of Regulation NMS (17 CFR 242.612).
- T+1 settlement for US equities/options since 2024-05-28: SEC rule
  amendment, Rel. 34-96930.
- FINRA Pattern Day Trader rule (margin accounts <$25k, 3 day trades / 5
  business days): FINRA Rule 4210(f)(8)(B).
- Zero-commission US equities at the target brokers (Alpaca/Schwab/Fidelity
  retail): vendor pricing pages; recorded as broker profile input, not a
  market constant.

## Methods (primary literature)
- Deflated Sharpe Ratio: Bailey & López de Prado (2014), "The Deflated
  Sharpe Ratio", Journal of Portfolio Management 40(5), 94-107.
- PBO/CSCV: Bailey, Borwein, López de Prado & Zhu (2015), "The Probability
  of Backtest Overfitting", Journal of Computational Finance 20(4).
- Purged/embargoed splits: López de Prado (2018), Advances in Financial
  Machine Learning, ch. 7, Wiley.
- Spread estimator: Corwin & Schultz (2012), "A Simple Way to Estimate
  Bid-Ask Spreads from Daily High and Low Prices", Journal of Finance 67(2).
- Spread estimator (blend partner): Abdi & Ranaldo (2017), "A Simple
  Estimation of Bid-Ask Spreads from Daily Close, High, and Low Prices",
  Review of Financial Studies 30(12). Cost model uses max(CS, AR)/2 per
  side — deliberately conservative.
- Isotonic score calibration: Zadrozny & Elkan (2002), "Transforming
  classifier scores into accurate multiclass probability estimates", KDD;
  pool-adjacent-violators per Ayer et al. (1955), Ann. Math. Statist. 26(4).
- Permutation p-value add-one estimator: Phipson & Smyth (2010), Statistical
  Applications in Genetics and Molecular Biology 9(1), Article 39.
- Demo hypothesis (cross-sectional momentum): Jegadeesh & Titman (1993),
  "Returns to Buying Winners and Selling Losers", Journal of Finance 48(1).

## Data vendors

- SEC EDGAR (catalyst layer): official CIK/ticker map
  (sec.gov/files/company_tickers.json) and per-company filing histories
  (data.sec.gov/submissions). 8-K Item 2.02 = "Results of Operations and
  Financial Condition" per the SEC's Form 8-K item definitions — used as
  the earnings-announcement marker. Known limitation (in manifest): the
  submissions "recent" window caps at ~1,000 filings per company.
- NASDAQ Trader symbol directory (universe): nasdaqtrader.com SymDir files;
  current listings only — survivorship bias recorded in manifest.
- Cboe delayed option chains (cdn.cboe.com/api/global/delayed_quotes):
  real exchange-published delayed (~15 min) bids/asks/IV/open interest,
  archived nightly as immutable snapshots watermarked REAL_DELAYED; the
  system's own accumulating chain history (no free historical chains exist).
- Yahoo Finance v8 chart API (query1.finance.yahoo.com) daily OHLCV;
  OHLC adjusted by the vendor's adjclose/close factor; adjustment quality
  not independently verified (recorded in manifest); keyless, rate-limited.
  (Stooq was tried first and abandoned: its CSV endpoint now sits behind a
  JavaScript proof-of-work browser check this system will not script around.)
