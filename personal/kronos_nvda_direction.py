"""Can Kronos predict NVIDIA's next days?

Walk-forward test on NVDA daily bars from IBKR. At each decision point Kronos
sees only the `--lookback` candles up to that day's close and forecasts the
next `--horizon` days as `--samples` independent paths. From those paths we
take:

  * direction   - up if the median path ends above today's close
  * p_up        - share of paths that end above today's close
  * range       - 10th-90th percentile of path end prices (an 80% interval)

and score them against what really happened, next to simple baselines:

  * always_up      - always say "up" (NVDA mostly rose; the bar to beat)
  * momentum       - same direction as the last `--horizon` days
  * reversal       - opposite of the last `--horizon` days
  * vol_interval   - 80% interval from the last 20 days' volatility (normal)

Usage (from the repo root):
  python personal/kronos_nvda_direction.py              # backtest, ~15 min on CPU
  python personal/kronos_nvda_direction.py --forecast   # forecast from the latest close
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

DATA = os.path.join(ROOT, "personal", "data", "NVDA_daily_ibkr.csv")
OUT_DIR = os.path.join(ROOT, "personal", "results")
COLS = ["open", "high", "low", "close", "volume", "amount"]


def load(drop_partial_last):
    df = pd.read_csv(DATA)
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna().drop_duplicates("timestamps").sort_values("timestamps").reset_index(drop=True)
    if drop_partial_last:
        df = df.iloc[:-1].reset_index(drop=True)
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    return df


def sample_paths(predictor, hist, future_days, horizon, samples):
    """Return an array (samples, horizon) of forecast closes, one row per independent path."""
    x = hist[COLS].reset_index(drop=True)
    xt = hist["timestamps"].reset_index(drop=True)
    yt = pd.Series(future_days).reset_index(drop=True)
    preds = predictor.predict_batch(
        df_list=[x] * samples, x_timestamp_list=[xt] * samples, y_timestamp_list=[yt] * samples,
        pred_len=horizon, T=1.0, top_p=0.9, sample_count=1, verbose=False,
    )
    return np.stack([p["close"].to_numpy() for p in preds])


def binom_p(k, n, p=0.5):
    """Two-sided binomial test p-value."""
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(x for x in pmf if x <= pmf[k] + 1e-12))


def backtest(predictor, df, args):
    rows = []
    last_start = len(df) - args.horizon
    points = list(range(args.lookback, last_start + 1, args.step))
    for n, i in enumerate(points):
        hist = df.iloc[i - args.lookback:i]          # up to and including day i-1
        today = hist["close"].iloc[-1]
        fut = df.iloc[i:i + args.horizon]
        torch.manual_seed(args.seed + i)
        paths = sample_paths(predictor, hist, fut["timestamps"], args.horizon, args.samples)
        end = paths[:, -1]
        actual = fut["close"].iloc[-1]
        past = df["close"].iloc[i - 1] / df["close"].iloc[i - 1 - args.horizon] - 1
        r20 = np.log(df["close"].iloc[i - 21:i]).diff().dropna()
        vol_h = r20.std(ddof=1) * math.sqrt(args.horizon)
        rows.append({
            "date": hist["timestamps"].iloc[-1].date(),
            "close": today,
            "actual_close": actual,
            "actual_ret_pct": (actual / today - 1) * 100,
            "kronos_median_ret_pct": (np.median(end) / today - 1) * 100,
            "kronos_p_up": float((end > today).mean()),
            "kronos_lo80": float(np.percentile(end, 10)),
            "kronos_hi80": float(np.percentile(end, 90)),
            "vol_lo80": today * math.exp(-1.2816 * vol_h),
            "vol_hi80": today * math.exp(1.2816 * vol_h),
            "past_ret_pct": past * 100,
        })
        if n % 10 == 0:
            print(f"  {n + 1}/{len(points)} {rows[-1]['date']}", flush=True)
    res = pd.DataFrame(rows)
    up = res["actual_ret_pct"] > 0
    res["kronos_hit"] = (res["kronos_median_ret_pct"] > 0) == up
    res["always_up_hit"] = up
    res["momentum_hit"] = (res["past_ret_pct"] > 0) == up
    res["reversal_hit"] = (res["past_ret_pct"] <= 0) == up
    res["kronos_in80"] = res["actual_close"].between(res["kronos_lo80"], res["kronos_hi80"])
    res["vol_in80"] = res["actual_close"].between(res["vol_lo80"], res["vol_hi80"])
    return res


def summarize(res, horizon):
    n = len(res)
    out = []
    for name in ["kronos", "always_up", "momentum", "reversal"]:
        k = int(res[f"{name}_hit"].sum())
        out.append({"metric": f"{name} direction hit rate", "value": round(k / n, 3),
                    "detail": f"{k}/{n}, p={binom_p(k, n):.2f} vs coin flip"})
    # Brier score of Kronos' up-probability vs a constant base rate (lower is better)
    up = (res["actual_ret_pct"] > 0).astype(float)
    brier = ((res["kronos_p_up"] - up) ** 2).mean()
    brier_base = ((up.mean() - up) ** 2).mean()
    out.append({"metric": "kronos Brier score (p_up)", "value": round(brier, 3), "detail": f"base rate only: {brier_base:.3f}"})
    corr = res["kronos_median_ret_pct"].corr(res["actual_ret_pct"])
    out.append({"metric": "corr(forecast return, actual return)", "value": round(corr, 3), "detail": f"{horizon}-day returns, n={n}"})
    for name in ["kronos", "vol"]:
        cov = res[f"{name}_in80"].mean()
        width = ((res[f"{name}_hi80"] - res[f"{name}_lo80"]) / res["close"] * 100).mean()
        out.append({"metric": f"{name} 80% range coverage", "value": round(cov, 3), "detail": f"average width {width:.1f}% of price"})
    # Long when Kronos says up, else cash; non-overlapping only if step >= horizon
    strat = np.where(res["kronos_median_ret_pct"] > 0, res["actual_ret_pct"], 0.0)
    out.append({"metric": "avg return when kronos says up, %", "value": round(res.loc[res["kronos_median_ret_pct"] > 0, "actual_ret_pct"].mean(), 3),
                "detail": f"vs when it says down: {res.loc[res['kronos_median_ret_pct'] <= 0, 'actual_ret_pct'].mean():.3f}"})
    out.append({"metric": "avg return following kronos, %", "value": round(strat.mean(), 3),
                "detail": f"always holding: {res['actual_ret_pct'].mean():.3f}"})
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--lookback", type=int, default=256)
    p.add_argument("--horizon", type=int, default=5, help="trading days ahead to score")
    p.add_argument("--samples", type=int, default=20, help="independent forecast paths per decision")
    p.add_argument("--step", type=int, default=2, help="trading days between decision points")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-last", action="store_true", help="keep the last bar (drop it if it was fetched intraday)")
    p.add_argument("--forecast", action="store_true")
    args = p.parse_args()

    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    df = load(drop_partial_last=not args.keep_last)

    if args.forecast:
        hist = df.iloc[-args.lookback:]
        days = pd.bdate_range(hist["timestamps"].iloc[-1] + pd.Timedelta(days=1), periods=args.horizon)
        torch.manual_seed(args.seed)
        paths = sample_paths(predictor, hist, days, args.horizon, args.samples)
        today = hist["close"].iloc[-1]
        for d, col in zip(days, paths.T):
            print(f"{d.date()}  median {np.median(col):.2f}  80% {np.percentile(col, 10):.2f}-{np.percentile(col, 90):.2f}  "
                  f"P(up vs {today:.2f}) {(col > today).mean():.0%}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"NVDA walk-forward: {len(df)} bars, horizon {args.horizon}, step {args.step}, {args.samples} paths", flush=True)
    res = backtest(predictor, df, args)
    res.to_csv(os.path.join(OUT_DIR, f"nvda_direction_h{args.horizon}.csv"), index=False)
    summary = summarize(res, args.horizon)
    summary.to_csv(os.path.join(OUT_DIR, f"nvda_direction_h{args.horizon}_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
