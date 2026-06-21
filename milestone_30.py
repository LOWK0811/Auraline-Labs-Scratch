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
from src.options import BlackScholes, build_options_chain


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
# SECTION 3: ESTIMATE HISTORICAL VOLATILITY FROM REAL DATA
# ======================================================================
data  = get_price_data("AAPL", "2025-01-01", "2026-06-01")
close = data["Close"]

# Annualized volatility from last 20 trading days
daily_returns = close.pct_change().dropna()
hist_vol_20d  = daily_returns.tail(20).std() * np.sqrt(252)
hist_vol_60d  = daily_returns.tail(60).std() * np.sqrt(252)

S = float(close.iloc[-1])   # latest AAPL price

print(f"\n{'='*55}")
print(f"  AURELINE LABS — OPTIONS PRICING MODULE")
print(f"{'='*55}")
print(f"  Asset:           AAPL")
print(f"  Current Price:   ${S:.2f}")
print(f"  20-day Hist Vol: {hist_vol_20d:.2%}")
print(f"  60-day Hist Vol: {hist_vol_60d:.2%}")
print(f"{'='*55}")


# ======================================================================
# SECTION 4: PRICE SPECIFIC OPTIONS
# ======================================================================
K     = round(S)      # at-the-money strike
T     = 0.25          # 3 months to expiry
r     = 0.05          # 5% risk-free rate (US Treasury)
sigma = hist_vol_20d  # use recent realized vol as estimate

bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)

print(f"\n  AT-THE-MONEY OPTION PRICING")
print(f"  Strike: ${K} | Expiry: {T*12:.0f} months | "
      f"Vol: {sigma:.2%}")
print(f"  {'-'*50}")
print(f"  Call Price: ${bs.call_price():.2f}")
print(f"  Put Price:  ${bs.put_price():.2f}")

# Verify put-call parity: C - P = S - K*e^(-rT)
parity_lhs = bs.call_price() - bs.put_price()
parity_rhs = S - K * np.exp(-r * T)
print(f"\n  PUT-CALL PARITY CHECK")
print(f"  C - P = {parity_lhs:.4f}")
print(f"  S - K·e^(-rT) = {parity_rhs:.4f}")
print(f"  Difference: {abs(parity_lhs - parity_rhs):.8f} "
      f"(should be ~0)")

print(f"\n  CALL GREEKS")
call_greeks = bs.all_greeks("call")
for greek, value in call_greeks.items():
    print(f"  {greek:<8}: {value:>10.4f}")

print(f"\n  PUT GREEKS")
put_greeks = bs.all_greeks("put")
for greek, value in put_greeks.items():
    print(f"  {greek:<8}: {value:>10.4f}")


# ======================================================================
# SECTION 5: IMPLIED VOLATILITY DEMO
# ======================================================================
print(f"\n  IMPLIED VOLATILITY DEMO")
print(f"  If the market prices the call at $15.00...")
market_price = 15.0
iv = bs.implied_volatility(market_price, "call")
if iv:
    print(f"  Implied vol: {iv:.2%}")
    print(f"  Historical vol: {sigma:.2%}")
    gap = iv - sigma
    print(f"  Vol premium: {gap:+.2%} "
          f"({'market expects MORE volatility' if gap > 0 else 'market expects LESS volatility'})")


# ======================================================================
# SECTION 6: BUILD OPTIONS CHAIN
# ======================================================================
print(f"\n  OPTIONS CHAIN (AAPL · 3-month expiry)")
chain = build_options_chain(S, T, r, sigma, n_strikes=11)
print(chain.to_string(index=False))


# ======================================================================
# SECTION 7: VISUALIZE — 4-PANEL OPTIONS ANALYSIS
# ======================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#060d1f")

strikes = np.linspace(S * 0.6, S * 1.4, 100)
colors  = {"call": "#00d4aa", "put": "#ff4d6a"}

def style_ax(ax):
    ax.set_facecolor("#060d1f")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1a3357")
    ax.spines["bottom"].set_color("#1a3357")
    ax.tick_params(colors="#7b9bc0", labelsize=8)
    ax.grid(True, color="#0d1b35", linewidth=0.8, linestyle="--")


# Panel 1: Option prices vs strike
ax1 = axes[0][0]
style_ax(ax1)
call_prices = [BlackScholes(S, k, T, r, sigma).call_price()
               for k in strikes]
put_prices  = [BlackScholes(S, k, T, r, sigma).put_price()
               for k in strikes]
ax1.plot(strikes, call_prices, color=colors["call"],
         linewidth=1.5, label="Call")
ax1.plot(strikes, put_prices,  color=colors["put"],
         linewidth=1.5, label="Put")
ax1.axvline(x=S, color="#7b9bc0", linestyle="--",
            linewidth=0.8, label=f"Current ${S:.0f}")
ax1.set_title("Option Price vs Strike",
              color="#e8f0fe", fontsize=9,
              fontfamily="monospace")
ax1.set_xlabel("Strike Price", color="#7b9bc0", fontsize=8)
ax1.set_ylabel("Option Price ($)", color="#7b9bc0", fontsize=8)
ax1.legend(fontsize=8, facecolor="#0f2040",
           edgecolor="#1a3357", labelcolor="#e8f0fe")


# Panel 2: Delta vs strike
ax2 = axes[0][1]
style_ax(ax2)
call_deltas = [BlackScholes(S, k, T, r, sigma).delta("call")
               for k in strikes]
put_deltas  = [BlackScholes(S, k, T, r, sigma).delta("put")
               for k in strikes]
ax2.plot(strikes, call_deltas, color=colors["call"],
         linewidth=1.5, label="Call Delta")
ax2.plot(strikes, put_deltas,  color=colors["put"],
         linewidth=1.5, label="Put Delta")
ax2.axvline(x=S, color="#7b9bc0", linestyle="--",
            linewidth=0.8)
ax2.axhline(y=0, color="#1a3357", linewidth=0.8)
ax2.axhline(y=0.5, color="#1a3357", linewidth=0.5,
            linestyle=":")
ax2.set_title("Delta vs Strike",
              color="#e8f0fe", fontsize=9,
              fontfamily="monospace")
ax2.set_xlabel("Strike Price", color="#7b9bc0", fontsize=8)
ax2.set_ylabel("Delta", color="#7b9bc0", fontsize=8)
ax2.legend(fontsize=8, facecolor="#0f2040",
           edgecolor="#1a3357", labelcolor="#e8f0fe")


# Panel 3: Option price vs time to expiry
ax3 = axes[1][0]
style_ax(ax3)
times      = np.linspace(0.01, 2.0, 100)
call_by_t  = [BlackScholes(S, K, t, r, sigma).call_price()
              for t in times]
put_by_t   = [BlackScholes(S, K, t, r, sigma).put_price()
              for t in times]
ax3.plot(times * 12, call_by_t, color=colors["call"],
         linewidth=1.5, label="Call")
ax3.plot(times * 12, put_by_t,  color=colors["put"],
         linewidth=1.5, label="Put")
ax3.axvline(x=T * 12, color="#7b9bc0", linestyle="--",
            linewidth=0.8, label=f"Current ({T*12:.0f}mo)")
ax3.set_title("Option Price vs Time to Expiry\n"
              "(ATM option — time value decay)",
              color="#e8f0fe", fontsize=9,
              fontfamily="monospace")
ax3.set_xlabel("Months to Expiry", color="#7b9bc0", fontsize=8)
ax3.set_ylabel("Option Price ($)", color="#7b9bc0", fontsize=8)
ax3.legend(fontsize=8, facecolor="#0f2040",
           edgecolor="#1a3357", labelcolor="#e8f0fe")


# Panel 4: Option price vs volatility (the vol smile setup)
ax4 = axes[1][1]
style_ax(ax4)
vols      = np.linspace(0.05, 1.0, 100)
call_by_v = [BlackScholes(S, K, T, r, v).call_price()
             for v in vols]
put_by_v  = [BlackScholes(S, K, T, r, v).put_price()
             for v in vols]
ax4.plot(vols * 100, call_by_v, color=colors["call"],
         linewidth=1.5, label="Call")
ax4.plot(vols * 100, put_by_v,  color=colors["put"],
         linewidth=1.5, label="Put")
ax4.axvline(x=sigma * 100, color="#7b9bc0", linestyle="--",
            linewidth=0.8, label=f"Current vol ({sigma:.0%})")
ax4.set_title("Option Price vs Volatility",
              color="#e8f0fe", fontsize=9,
              fontfamily="monospace")
ax4.set_xlabel("Volatility (%)", color="#7b9bc0", fontsize=8)
ax4.set_ylabel("Option Price ($)", color="#7b9bc0", fontsize=8)
ax4.legend(fontsize=8, facecolor="#0f2040",
           edgecolor="#1a3357", labelcolor="#e8f0fe")

fig.suptitle(
    f"Aureline Labs — Black-Scholes Options Pricing\n"
    f"AAPL · S=${S:.2f} · K=${K} · "
    f"T={T*12:.0f}mo · σ={sigma:.1%} · r={r:.1%}",
    color="#e8f0fe", fontsize=11,
    fontfamily="monospace", y=1.01
)
plt.tight_layout()
plt.savefig("data/options_analysis.png",
            dpi=150, bbox_inches="tight",
            facecolor="#060d1f")
plt.show()
logger.info("Chart saved to data/options_analysis.png")