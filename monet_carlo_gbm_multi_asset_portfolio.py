# gbm_multi_asset.py
# ------------------
# Vectorised Monte-Carlo GBM simulator for a **weighted equity portfolio**.
#
# – μ, σ and the full covariance matrix Σ are calibrated from historical
#   log-returns downloaded with *yfinance* (auto-adjusted Close prices).
# – Correlation handled via Cholesky decomposition; variance reduced with
#   antithetic sampling; fully NumPy-vectorised → fast even for 100 k paths.
# – CLI flags let you change tickers, weights, number of paths and look-back
#   window without touching the source.
# – Outputs: PNG fan-plot + CSV of final-year portfolio values in ./outputs/
# – Risk metrics: mean ± 95 % CI, VaR, CVaR (Expected Shortfall) and probability
#   of loss versus a user-defined initial capital.
#
# Quick-start
# -----------
# $ pip install -r requirements.txt
# $ python multi_asset_gbm.py \
#       -t AAPL MSFT NVDA \
#       -w 0.4 0.3 0.3 \
#       -p 100000
#

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------
# Default parameters (override via CLI)
# ---------------------------------------------------------------------
DEFAULT_TICKERS    = ["AAPL", "GOOG", "NVDA"]
DEFAULT_WEIGHTS    = np.array([0.4, 0.3, 0.3])
LOOKBACK_YEARS     = 5
DEFAULT_PATHS      = 50_000
STEPS_PER_YEAR     = 252
SEED               = 42
INITIAL_PORTFOLIO  = 100_000
SAMPLE_PLOT_PATHS  = 500
OUT_DIR            = Path("outputs")

# ---------------------------------------------------------------------
# Historical calibration
# ---------------------------------------------------------------------
def get_hist_params(
    tickers: list[str],
    lookback_years: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (S0, mu, sigma, covariance) all annualised.

    S0      : last adjusted close for each ticker
    mu      : vector of annualised drift (mean log-return)
    sigma   : vector of annualised vol
    cov     : annualised covariance matrix
    """
    end   = datetime.today()
    start = end - timedelta(days=365 * lookback_years)

    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )["Close"]  # DataFrame, cols = tickers

    if data.empty or data.isna().all().any():
        raise RuntimeError("Price download failed for at least one ticker")

    log_r = np.log(data).diff().dropna()
    mu    = log_r.mean().to_numpy() * 252
    cov   = log_r.cov().to_numpy()   * 252
    sigma = np.sqrt(np.diag(cov))
    S0    = data.iloc[-1].to_numpy()

    return S0, mu, sigma, cov

# ---------------------------------------------------------------------
# GBM Monte-Carlo simulation
# ---------------------------------------------------------------------
def simulate_portfolio(
    S0: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    paths: int   = DEFAULT_PATHS,
    steps: int   = STEPS_PER_YEAR,
    seed:  int | None = SEED,
) -> np.ndarray:
    """
    Generate Monte-Carlo price paths and aggregate to portfolio value.

    Returns
    -------
    port_paths : ndarray (paths, steps + 1)
    """
    if paths % 2:                # antithetic needs an even count
        paths -= 1
        print("Adjusted paths ->", paths)

    n   = len(S0)
    dt  = 1 / steps
    L   = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)

    Z = rng.standard_normal((paths // 2, steps, n))
    Z = np.vstack([Z, -Z])                    # antithetic
    shocks = Z @ L.T * np.sqrt(dt)

    drift = (mu - 0.5 * np.diag(cov)) * dt
    log_paths = np.cumsum(drift + shocks, axis=1)
    log_paths = np.insert(log_paths, 0, 0.0, axis=1)
    price_paths = S0 * np.exp(log_paths)      # (P, N+1, n)

    port_paths = price_paths @ weights        # (P, N+1)
    port_paths *= INITIAL_PORTFOLIO / port_paths[:, 0:1]

    return port_paths

# ---------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------
def risk_summary(final: np.ndarray) -> dict[str, float]:
    mean   = final.mean()
    std    = final.std(ddof=1)
    ci95   = 1.96 * std / np.sqrt(len(final))
    var95  = np.percentile(final, 5)
    cvar95 = final[final <= var95].mean()
    p_loss = (final < INITIAL_PORTFOLIO).mean() * 100
    return dict(mean=mean, ci95=ci95, var=var95, cvar=cvar95, p_loss=p_loss)

# ---------------------------------------------------------------------
# High-level run wrapper (importable)
# ---------------------------------------------------------------------
def run(
    tickers: list[str],
    weights: np.ndarray,
    paths:   int = DEFAULT_PATHS,
    years:   int = LOOKBACK_YEARS,
) -> dict[str, object]:
    S0, mu, _, cov = get_hist_params(tickers, years)
    port_paths = simulate_portfolio(S0, mu, cov, weights, paths=paths)
    final = port_paths[:, -1]
    stats = risk_summary(final)
    return dict(S0=S0, paths=port_paths, final=final, stats=stats)

# ---------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-asset GBM portfolio simulator")
    p.add_argument("-t", "--tickers", nargs="+", default=DEFAULT_TICKERS,
                   help="Tickers, space-separated (default: %(default)s)")
    p.add_argument("-w", "--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS,
                   help="Weights (same length as tickers)")
    p.add_argument("-p", "--paths", type=int, default=DEFAULT_PATHS,
                   help="Monte-Carlo paths (default: %(default)s)")
    p.add_argument("-y", "--years", type=int, default=LOOKBACK_YEARS,
                   help="Look-back window in years (default: %(default)s)")
    return p.parse_args()

# ---------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------
if __name__ == "__main__":

    args = parse_cli()
    tickers = [t.upper() for t in args.tickers]
    weights = np.array(args.weights, dtype=float)
    weights /= weights.sum()                        # re-normalise
    paths = args.paths
    years = args.years

    OUT_DIR.mkdir(exist_ok=True)

    res = run(tickers, weights, paths=paths, years=years)
    port_paths, final, stats, S0 = \
        res["paths"], res["final"], res["stats"], res["S0"]

    # ---------------- plotting ------------------
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sample_idx = np.random.choice(
        port_paths.shape[0],
        size=min(SAMPLE_PLOT_PATHS, port_paths.shape[0]),
        replace=False
    )
    ax[0].plot(port_paths[sample_idx].T, lw=0.6, alpha=0.7)
    ax[0].set(title="Monte-Carlo Portfolio Paths",
              xlabel="Trading Day", ylabel="Value ($)")
    ax[0].grid(True)

    weights_pct = weights * 100
    box_txt = "\n".join(f"{t}: {w:.1f}% @ ${p:.2f}"
                        for t, w, p in zip(tickers, weights_pct, S0))
    ax[0].text(0.02, 0.98, box_txt,
               transform=ax[0].transAxes, va="top", ha="left",
               bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))

    ax[1].hist(final, bins=100, edgecolor="black")
    ax[1].axvline(stats["var"], c="red", ls="--",
                  label=f"VaR 95 % = ${stats['var']:,.0f}")
    ax[1].axvline(INITIAL_PORTFOLIO, c="orange", ls="--",
                  label=f"Start = ${INITIAL_PORTFOLIO:,}")
    ax[1].set(title="Distribution of Final Portfolio Values",
              xlabel="Value ($)", ylabel="Frequency")
    ax[1].grid(True)
    ax[1].legend()

    info_txt = (f"Mean: ${stats['mean']:,.0f} ±{stats['ci95']:,.0f}\n"
                f"CVaR 5 %: ${stats['cvar']:,.0f}\n"
                f"P(loss): {stats['p_loss']:.2f}%")
    ax[1].text(0.97, 0.95, info_txt,
               transform=ax[1].transAxes, ha="right", va="top",
               bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))

    plt.tight_layout()

    png_path = OUT_DIR / "gbm_portfolio.png"
    csv_path = OUT_DIR / "gbm_final_values.csv"
    plt.savefig(png_path, dpi=300)
    pd.Series(final, name="FinalPortfolioValue").to_csv(csv_path, index=False)

    print(f"Plot saved ➜ {png_path.resolve()}")
    print(f"CSV  saved ➜ {csv_path.resolve()}")