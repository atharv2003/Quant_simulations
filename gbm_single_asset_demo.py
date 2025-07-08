"""
gbm_single_asset.py
-------------------
Monte-Carlo GBM simulator for one equity.

Features
• μ/σ auto-calibrated from yfinance (adjusted close)
• Vectorised NumPy with antithetic variance-reduction
• Risk metrics: mean ± 95 % CI, VaR, CVaR, probability of loss
• CLI flags for ticker, path count, and look-back window
• Saves the fan plot (PNG) and final-price distribution (CSV)
"""

from datetime import datetime, timedelta
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ─────────── default parameters ─────────── #
TICKER       = "GOOG"
LOOKBACK_YRS = 3
PATHS        = 40_000              # forced even for antithetic
STEPS        = 252                 # trading days
SEED         = 42
START_CAP    = 100_000             # rescale so t0 = this $
PLOT_SAMPLE  = 400                 # paths in fan plot
OUT_DIR      = Path("outputs")
# ─────────────────────────────────────────── #


# ─────────── GBM simulator (re-usable) ─────────── #
def simulate_gbm_single(
    ticker: str,
    paths: int = PATHS,
    lookback_years: int = LOOKBACK_YRS,
    steps: int = STEPS,
    seed: int | None = SEED,
) -> dict:
    """Run a single-asset GBM Monte-Carlo and return results."""
    end   = datetime.today()
    start = end - timedelta(days=365 * lookback_years)

    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )["Close"]

    # ── data validation / flatten ──
    if data.empty:
        raise RuntimeError(f"No price returned for {ticker}")

    if data.ndim != 1:              # catches MultiIndex or DataFrame
        data = data.squeeze()

    if data.isna().all():
        raise RuntimeError("Price series is empty after squeeze")

    log_r = np.log(data).diff().dropna()
    mu    = float(log_r.mean() * 252)
    sigma = float(log_r.std(ddof=0) * np.sqrt(252))
    S0    = float(data.iloc[-1])

    # ── antithetic Monte-Carlo ──
    if paths % 2:           # force even
        paths -= 1
        print("Adjusted PATHS to even number for antithetic sampling:", paths)

    dt   = 1 / steps
    rng  = np.random.default_rng(seed)
    half = paths // 2
    Z    = rng.standard_normal((half, steps))
    Z    = np.vstack([Z, -Z])             # antithetic

    drift = (mu - 0.5 * sigma**2) * dt
    incr  = drift + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(incr, axis=1)
    log_paths = np.hstack([np.zeros((paths, 1)), log_paths])
    price_paths = S0 * np.exp(log_paths)

    # rescale to START_CAP at t0
    price_paths *= START_CAP / S0
    final = price_paths[:, -1]

    # risk metrics
    mean  = final.mean()
    ci95  = 1.96 * final.std(ddof=1) / np.sqrt(paths)
    var5  = np.percentile(final, 5)
    cvar5 = final[final <= var5].mean()
    p_loss = (final < START_CAP).mean() * 100

    return {
        "price_paths": price_paths,
        "final": final,
        "mu": mu,
        "sigma": sigma,
        "S0": S0,
        "mean": mean,
        "ci95": ci95,
        "var5": var5,
        "cvar5": cvar5,
        "p_loss": p_loss,
        "paths": paths,
        "steps": steps,
    }


# ─────────── CLI / entry point ─────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-asset GBM simulator")
    parser.add_argument("-t", "--ticker", default=TICKER,
                        help="Equity ticker (default: GOOG)")
    parser.add_argument("-p", "--paths", type=int, default=PATHS,
                        help="Number of Monte-Carlo paths (default: 40 000)")
    parser.add_argument("-y", "--years", type=int, default=LOOKBACK_YRS,
                        help="Look-back window in years (default: 3)")
    args = parser.parse_args()

    TICKER       = args.ticker.upper()
    PATHS        = args.paths
    LOOKBACK_YRS = args.years

    OUT_DIR.mkdir(exist_ok=True)

    res = simulate_gbm_single(
        TICKER, paths=PATHS, lookback_years=LOOKBACK_YRS
    )

    # ─────────── plotting ─────────── #
    price_paths, final = res["price_paths"], res["final"]
    var5, cvar5 = res["var5"], res["cvar5"]
    mean, ci95  = res["mean"], res["ci95"]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    samp = np.random.choice(res["paths"],
                            size=min(PLOT_SAMPLE, res["paths"]),
                            replace=False)
    ax[0].plot(price_paths[samp].T, lw=0.6, alpha=0.7)
    ax[0].set(
        title=f"{TICKER} GBM Price Fan",
        xlabel="Trading Day",
        ylabel="Value ($)",
    )
    ax[0].grid(True)

    ax[1].hist(final, bins=120, edgecolor="black")
    ax[1].axvline(var5, c="red", ls="--", label=f"VaR 95% = ${var5:,.0f}")
    ax[1].axvline(START_CAP, c="orange", ls="--",
                  label=f"Start = ${START_CAP:,}")
    ax[1].set(
        title="Distribution of Final Values",
        xlabel="Value ($)",
        ylabel="Frequency",
    )
    ax[1].grid(True)
    ax[1].legend()

    info = (f"Mean: ${mean:,.0f} ±{ci95:,.0f} (95 % CI)\n"
            f"CVaR 5 %: ${cvar5:,.0f}\n"
            f"P(loss): {res['p_loss']:.2f}%")
    ax[1].text(
        0.97, 0.95, info,
        transform=ax[1].transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.6),
    )

    plt.tight_layout()
    png_path = OUT_DIR / f"{TICKER}_gbm_sim.png"
    csv_path = OUT_DIR / f"{TICKER}_final_values.csv"
    plt.savefig(png_path, dpi=300)
    pd.Series(final).to_csv(csv_path, index=False)

print(f"Plot saved to {png_path.resolve()}")
print(f"Final values saved to {csv_path.resolve()}")
