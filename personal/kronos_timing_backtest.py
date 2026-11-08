"""Does Kronos help pick a better buy day for a regular ETF purchase?

Question: each quarter Josep invests in VT and CHSPI. Instead of buying on the
first trading day of the quarter, could he wait up to N trading days and buy on
the day Kronos forecasts to be cheapest?

For every decision point the script only shows Kronos the candles up to the
close of the day *before* the buying window. It then forecasts the next N
daily closes, picks the day with the lowest forecast close, and "buys" at the
real close of that day. That price is compared with:

  * day1    - buy at the close of the first day of the window (the current plan)
  * average - the mean close over the window (what a random day gets you)
  * best    - the lowest close in the window (perfect hindsight, upper bound)

Savings are reported in % of the day-1 price; positive means cheaper than day 1.

Two sets of decision points are evaluated:
  * quarterly - the first trading day of each calendar quarter (the real plan)
  * rolling   - every `--step` trading days (many more samples, for statistics)

Usage (from the repo root):
  python personal/kronos_timing_backtest.py                  # full backtest
  python personal/kronos_timing_backtest.py --forecast       # suggested day for the next window
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

DATA_DIR = os.path.join(ROOT, "personal", "data")
OUT_DIR = os.path.join(ROOT, "personal", "results")
TICKERS = ["VT", "CHSPI"]


def load(ticker):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}_daily_ibkr.csv"))
    # IBKR stamps daily bars with the session open time; keep the date only.
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna().drop_duplicates("timestamps").sort_values("timestamps").reset_index(drop=True)
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    return df


def decision_points(df, lookback, window, mode, step):
    """Index of the first day of each buying window."""
    first = lookback
    last = len(df) - window
    if mode == "rolling":
        return list(range(first, last + 1, step))
    q = df["timestamps"].dt.to_period("Q")
    starts = df.index[q != q.shift()].tolist()
    return [i for i in starts if first <= i <= last]


def forecast_closes(predictor, df, start, lookback, window, y_timestamps, samples):
    hist = df.iloc[start - lookback:start]
    pred = predictor.predict(
        df=hist[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True),
        x_timestamp=hist["timestamps"].reset_index(drop=True),
        y_timestamp=pd.Series(y_timestamps).reset_index(drop=True),
        pred_len=window,
        T=1.0,
        top_p=0.9,
        sample_count=samples,
        verbose=False,
    )
    return pred["close"].to_numpy()


def evaluate(predictor, df, ticker, args, mode):
    rows = []
    for start in decision_points(df, args.lookback, args.window, mode, args.step):
        win = df.iloc[start:start + args.window]
        closes = win["close"].to_numpy()
        torch.manual_seed(args.seed + start)
        pred = forecast_closes(predictor, df, start, args.lookback, args.window, win["timestamps"], args.samples)
        k = int(np.argmin(pred))
        day1 = closes[0]
        rows.append({
            "ticker": ticker,
            "mode": mode,
            "window_start": win["timestamps"].iloc[0].date(),
            "kronos_day": k + 1,
            "day1": day1,
            "kronos": closes[k],
            "average": closes.mean(),
            "best": closes.min(),
            "forecast_dir_correct": np.sign(pred[-1] - df["close"].iloc[start - 1])
            == np.sign(closes[-1] - df["close"].iloc[start - 1]),
        })
    out = pd.DataFrame(rows)
    for col in ["kronos", "average", "best"]:
        out[f"save_{col}_pct"] = (out["day1"] - out[col]) / out["day1"] * 100
    out["kronos_vs_average_pct"] = (out["average"] - out["kronos"]) / out["average"] * 100
    return out


def summarize(res):
    def tstat(x):
        return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan

    g = res.groupby(["mode", "ticker"])
    summary = pd.DataFrame({
        "n": g.size(),
        "kronos_vs_day1_mean_pct": g["save_kronos_pct"].mean(),
        "kronos_vs_day1_tstat": g["save_kronos_pct"].apply(tstat),
        "kronos_beats_day1_rate": g["save_kronos_pct"].apply(lambda x: (x > 0).mean()),
        "kronos_vs_average_mean_pct": g["kronos_vs_average_pct"].mean(),
        "kronos_vs_average_tstat": g["kronos_vs_average_pct"].apply(tstat),
        "average_vs_day1_mean_pct": g["save_average_pct"].mean(),
        "best_vs_day1_mean_pct": g["save_best_pct"].mean(),
        "direction_hit_rate": g["forecast_dir_correct"].mean(),
        "mean_kronos_day": g["kronos_day"].mean(),
    })
    return summary.round(3)


def next_trading_days(last_date, n):
    return pd.bdate_range(last_date + pd.Timedelta(days=1), periods=n)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--lookback", type=int, default=256, help="daily candles shown to Kronos")
    p.add_argument("--window", type=int, default=10, help="trading days you are willing to wait")
    p.add_argument("--samples", type=int, default=8, help="forecast paths averaged per decision")
    p.add_argument("--step", type=int, default=5, help="spacing of rolling decision points")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--forecast", action="store_true", help="only print the suggested day for the next window")
    args = p.parse_args()

    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    if args.forecast:
        for t in TICKERS:
            df = load(t)
            days = next_trading_days(df["timestamps"].iloc[-1], args.window)
            torch.manual_seed(args.seed)
            pred = forecast_closes(predictor, df, len(df), args.lookback, args.window, days, args.samples)
            k = int(np.argmin(pred))
            print(f"{t}: last close {df['close'].iloc[-1]:.2f} on {df['timestamps'].iloc[-1].date()}; "
                  f"lowest forecast close {pred[k]:.2f} on {days[k].date()} (day {k + 1} of {args.window})")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for t in TICKERS:
        df = load(t)
        for mode in ["quarterly", "rolling"]:
            print(f"{t} {mode} ...", flush=True)
            frames.append(evaluate(predictor, df, t, args, mode))
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(os.path.join(OUT_DIR, "decisions.csv"), index=False)
    summary = summarize(res)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.T)


if __name__ == "__main__":
    main()
