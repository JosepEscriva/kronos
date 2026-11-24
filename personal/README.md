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

## Can Kronos predict NVIDIA's next days?

`kronos_nvda_direction.py` is a walk-forward test on 2 years of NVDA daily bars
from IBKR (`data/NVDA_daily_ibkr.csv`, Nov 2024 to Nov 2026). Every 2 trading days
Kronos-small sees the previous 256 candles and draws 20 independent paths for the
next 1 or 5 days. The median path gives a direction, the share of paths ending up
gives a probability, and the 10th-90th percentile of path ends gives an 80% range.
Baselines: always "up", momentum, reversal, and an 80% range from 20-day volatility.

```bash
python personal/kronos_nvda_direction.py --horizon 5    # ~10 min on CPU
python personal/kronos_nvda_direction.py --horizon 1
python personal/kronos_nvda_direction.py --forecast --keep-last
```

### Results (run 2026-11-24)

| | 1 day ahead (n=122) | 5 days ahead (n=120) |
|---|---|---|
| Kronos direction hit rate | 45.9% (p=0.42) | 53.3% (p=0.52) |
| Always "up" | 54.1% | 50.8% |
| Momentum / reversal | 52.5% / 47.5% | 45.0% / 55.0% |
| Correlation, forecast vs actual return | -0.25 | -0.02 |
| Brier score of P(up), lower is better | 0.335 (base rate 0.248) | 0.343 (base rate 0.250) |
| Kronos 80% range actually covered | 57% (width 4.3%) | 39% (width 7.3%) |
| Volatility 80% range actually covered | 81% (width 6.0%) | 77% (width 13.4%) |

Reading:

- No direction edge. Neither hit rate is distinguishable from a coin flip, and the
  1-day forecasts were slightly *inversely* related to what happened.
- Kronos' probabilities are worse than just using the base rate: it is confidently wrong.
- Its ranges are far too narrow. An "80%" Kronos range held the price only 39-57% of
  the time; a plain volatility range held it 77-81% of the time. For "how far can
  NVDA move", use volatility (the dashboard's GARCH and options-implied ranges), not Kronos.
- 120 decisions over one year is a small sample, and Kronos may have seen this period
  in pre-training, which would flatter it. "No edge" is the robust conclusion.

Informational only, not financial advice.
