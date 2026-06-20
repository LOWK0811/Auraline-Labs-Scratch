# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from src.data_handler import get_price_data
from src.portfolio import Portfolio
from src.monte_carlo import MonteCarloSimulator


# ======================================================================
# SECTION 2: LOGGING SETUP
# ======================================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 3: LOAD DATA AND BUILD PORTFOLIOS
# ======================================================================
tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "TSLA", "SPY"]
START   = "2021-01-01"
END     = "2026-06-01"

prices = {}
for t in tickers:
    data = get_price_data(t, START, END)
    if data is not None:
        prices[t] = data["Close"]

portfolio = Portfolio(prices, starting_cash=100000)

equal_w   = portfolio.equal_weight()
risk_par_w = portfolio.risk_parity()
diversified_w = {"AAPL": 0.15, "MSFT": 0.15, "NVDA": 0.10,
                 "JPM": 0.15, "XOM": 0.15, "JNJ": 0.15,
                 "TSLA": 0.05, "SPY": 0.10}

strategies = {
    "Equal Weight":   equal_w,
    "Risk Parity":    risk_par_w,
    "Diversified":    diversified_w,
}

# Run simulations and collect results
all_results  = {}
all_portfolios = {}

for name, weights in strategies.items():
    pv = portfolio.simulate(weights, rebalance_frequency="monthly")
    all_portfolios[name] = pv

    mc = MonteCarloSimulator(pv, n_simulations=5000, block_size=20)
    paths   = mc.generate_paths()
    results = mc.analyze(paths)

    historical_return = (pv[-1] / pv[0] - 1) * 100
    mc.print_report(results, name, historical_return)
    all_results[name] = (results, mc, pv)


# ======================================================================
# SECTION 4: VISUALIZATION — 3-PANEL MONTE CARLO CHART
# ======================================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.patch.set_facecolor("#060d1f")

strategy_names = list(strategies.keys())
colors = ["#00d4aa", "#1a6eff", "#ffd166"]

for col, (name, color) in enumerate(zip(strategy_names, colors)):
    results, mc, hist_pv = all_results[name]
    paths = results["_paths"]
    final_returns = results["_final_returns"] * 100
    max_dds = results["_max_drawdowns"]

    # ── Row 1: Simulated paths fan ──
    ax = axes[0][col]
    ax.set_facecolor("#060d1f")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1a3357")
    ax.spines["bottom"].set_color("#1a3357")
    ax.tick_params(colors="#7b9bc0", labelsize=7)
    ax.grid(True, color="#0d1b35", linewidth=0.5, linestyle="--")

    # Draw 200 random paths in faded color
    for path in paths[np.random.choice(len(paths), 200, replace=False)]:
        ax.plot(path, color=color, alpha=0.03, linewidth=0.5)

    # Percentile bands
    p5  = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    ax.fill_between(range(len(p5)),  p5,  p95,
                    alpha=0.15, color=color)
    ax.fill_between(range(len(p25)), p25, p75,
                    alpha=0.25, color=color)
    ax.plot(p5,          color=color, linewidth=0.8,
            linestyle="--", alpha=0.6)
    ax.plot(p95,         color=color, linewidth=0.8,
            linestyle="--", alpha=0.6)
    ax.plot(np.median(paths, axis=0),
            color=color, linewidth=1.5, label="Median")
    ax.plot(hist_pv, color="#ffffff", linewidth=1.2,
            linestyle="-", alpha=0.8, label="Historical")

    ax.set_title(name, color="#e8f0fe",
                 fontsize=9, fontfamily="monospace")
    ax.yaxis.set_major_formatter(
        mticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend(fontsize=6, facecolor="#0f2040",
              edgecolor="#1a3357", labelcolor="#e8f0fe")

    if col == 0:
        ax.set_ylabel("Portfolio Value",
                      color="#7b9bc0", fontsize=8)

    # ── Row 2: Return distribution histogram ──
    ax2 = axes[1][col]
    ax2.set_facecolor("#060d1f")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#1a3357")
    ax2.spines["bottom"].set_color("#1a3357")
    ax2.tick_params(colors="#7b9bc0", labelsize=7)
    ax2.grid(True, color="#0d1b35", linewidth=0.5, linestyle="--")

    ax2.hist(final_returns, bins=60, color=color,
             alpha=0.7, edgecolor="none")
    ax2.axvline(x=0, color="#ff4d6a", linewidth=1,
                linestyle="--", label="Break-even")
    ax2.axvline(x=np.mean(final_returns),
                color="#ffffff", linewidth=1,
                linestyle="-", label="Mean")

    historical_ret = (hist_pv[-1] / hist_pv[0] - 1) * 100
    ax2.axvline(x=historical_ret,
                color="#ffd166", linewidth=1,
                linestyle=":", label="Historical")

    prob_loss = (final_returns < 0).mean() * 100
    ax2.set_title(f"Return Distribution\n"
                  f"P(loss) = {prob_loss:.1f}%",
                  color="#7b9bc0", fontsize=8)
    ax2.set_xlabel("Final Return (%)", color="#7b9bc0", fontsize=7)
    ax2.legend(fontsize=6, facecolor="#0f2040",
               edgecolor="#1a3357", labelcolor="#e8f0fe")

    if col == 0:
        ax2.set_ylabel("Frequency",
                       color="#7b9bc0", fontsize=8)

    # ── Row 3: Max drawdown distribution ──
    ax3 = axes[2][col]
    ax3.set_facecolor("#060d1f")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_color("#1a3357")
    ax3.spines["bottom"].set_color("#1a3357")
    ax3.tick_params(colors="#7b9bc0", labelsize=7)
    ax3.grid(True, color="#0d1b35", linewidth=0.5, linestyle="--")

    ax3.hist(max_dds, bins=60, color="#ff4d6a",
             alpha=0.7, edgecolor="none")
    ax3.axvline(x=results["mean_max_dd"],
                color="#ffffff", linewidth=1,
                linestyle="-", label="Mean DD")
    ax3.axvline(x=results["p5_max_dd"],
                color="#ffd166", linewidth=1,
                linestyle="--", label="5th pct (worst 5%)")

    ax3.set_title(f"Max Drawdown Distribution\n"
                  f"Mean DD = {results['mean_max_dd']:.1f}%",
                  color="#7b9bc0", fontsize=8)
    ax3.set_xlabel("Max Drawdown (%)", color="#7b9bc0", fontsize=7)
    ax3.legend(fontsize=6, facecolor="#0f2040",
               edgecolor="#1a3357", labelcolor="#e8f0fe")

    if col == 0:
        ax3.set_ylabel("Frequency",
                       color="#7b9bc0", fontsize=8)

fig.suptitle(
    "Aureline Labs — Monte Carlo Simulation\n"
    "5,000 Paths · Block Bootstrap · 8 Assets · 2021–2026",
    color="#e8f0fe", fontsize=11,
    fontfamily="monospace", y=1.01
)
plt.tight_layout()
plt.savefig("data/monte_carlo_analysis.png",
            dpi=150, bbox_inches="tight",
            facecolor="#060d1f")
plt.show()
logger.info("Chart saved to data/monte_carlo_analysis.png")