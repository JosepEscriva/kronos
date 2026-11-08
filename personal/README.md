# Personal experiments

## Kronos buy-timing backtest

`kronos_timing_backtest.py` asks one question: for the quarterly VT / CHSPI
purchase, does waiting up to 10 trading days and buying on the day Kronos
forecasts to be cheapest beat simply buying on day 1?

- Data: 5 years of daily IBKR bars (`data/*_daily_ibkr.csv`, VT on ARCA, CHSPI on SIX), Nov 2021 to Nov 2026.
- Model: `NeoQuasar/Kronos-small` with `Kronos-Tokenizer-base`, 256 daily candles of context, 8 sampled paths averaged.
- No look-ahead in the inputs: Kronos sees only candles up to the close before the window.
- Results land in `results/` (`decisions.csv` per decision, `summary.csv` per ticker).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python personal/kronos_timing_backtest.py            # full backtest (~1 h on CPU)
python personal/kronos_timing_backtest.py --forecast # suggested day for the next window
```

Caveat: Kronos was pre-trained on market data that may overlap this period, which
would flatter it. A result of "no edge" is therefore the stronger statement.

Informational only, not financial advice.

## Results (run 2026-11-08)

Savings vs buying on day 1, in % of price (positive = Kronos bought cheaper):

| Set | Ticker | Decisions | Kronos vs day 1 | t-stat | Kronos beat day 1 | Random day vs day 1 | Perfect hindsight |
|---|---|---|---|---|---|---|---|
| Quarterly | VT | 16 | -0.83% | -1.9 | 19% | -0.36% | +1.44% |
| Quarterly | CHSPI | 16 | -0.08% | -0.1 | 19% | -0.07% | +1.39% |
| Rolling (every 5 days) | VT | 198 | -0.27% | -2.0 | 37% | -0.29% | +1.35% |
| Rolling (every 5 days) | CHSPI | 199 | +0.08% | +0.6 | 38% | -0.14% | +1.29% |

Reading:

- Kronos did not find cheaper days. On VT it was slightly worse than buying on day 1; on CHSPI it was a wash.
- Waiting itself costs a little (a random day in the window is on average 0.1-0.4% dearer than day 1), because these markets drift up.
- Kronos mostly picks day 1 or day 10, i.e. it extrapolates the recent trend; its up/down direction calls were right 38-57% of the time.
- In the April 2025 sell-off, the one window where waiting would have saved 11-13%, Kronos said "buy on day 1".
- In money terms, on a 4,000 CHF VT buy the quarterly result is about -33 CHF per quarter; on CHSPI about 0.

Conclusion: keep buying on a fixed day. Kronos adds no measurable value for timing these purchases.
