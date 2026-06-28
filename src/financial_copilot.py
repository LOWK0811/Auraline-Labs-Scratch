# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: CONCEPT KNOWLEDGE BASE
# ======================================================================
# Each concept has: keywords, a beginner explanation,
# a professional explanation, and follow-up suggestions.

CONCEPTS = {
    "inflation": {
        "keywords": ["inflation", "cpi", "prices rising",
                     "cost of living", "purchasing power"],
        "beginner": (
            "**Inflation** is when the prices of everyday things — "
            "food, gas, rent — go up over time. If inflation is 3%, "
            "something that cost ₱100 last year now costs ₱103.\n\n"
            "For investors, inflation matters because it erodes the value "
            "of cash. ₱10,000 sitting in a bank account loses purchasing "
            "power every year if your interest rate is lower than inflation.\n\n"
            "**What to watch:** The BSP (Bangko Sentral ng Pilipinas) "
            "targets 2-4% inflation. When inflation is too high, they "
            "raise interest rates to cool it down — which can hurt stocks."
        ),
        "professional": (
            "**Inflation** is the rate of change in a general price index "
            "(typically CPI). It affects asset pricing through multiple channels:\n\n"
            "• **Real rates:** Nominal rate minus inflation. Positive real rates "
            "support currency strength and can compress equity multiples.\n"
            "• **Discount rates:** Higher inflation → higher discount rates → "
            "lower present value of future cash flows → P/E compression.\n"
            "• **Sector rotation:** High inflation benefits real assets (energy, "
            "commodities, REITs). Growth stocks with long-duration earnings are "
            "most exposed to inflation-driven multiple compression.\n"
            "• **Philippine context:** BSP targets 2-4%. At the current "
            "3.1%, real rates are positive — generally supportive for the peso."
        ),
        "followups": [
            "How does inflation affect interest rates?",
            "Which stocks benefit from inflation?",
            "What is the current Philippine inflation rate?"
        ]
    },

    "interest_rates": {
        "keywords": ["interest rate", "bsp rate", "fed rate",
                     "rate hike", "rate cut", "monetary policy",
                     "central bank", "yield"],
        "beginner": (
            "**Interest rates** are the cost of borrowing money. "
            "When you take a bank loan, the interest rate is what "
            "you pay the bank for lending you money.\n\n"
            "Central banks like the BSP set a key interest rate that "
            "affects all other rates in the economy:\n\n"
            "• **High rates:** Borrowing is expensive → companies invest "
            "less → economy slows → stocks often fall.\n"
            "• **Low rates:** Borrowing is cheap → companies expand → "
            "economy grows → stocks often rise.\n\n"
            "**Right now:** The BSP rate is 6.0%, which is considered "
            "restrictive — designed to keep inflation under control."
        ),
        "professional": (
            "**Interest rates** are the primary transmission mechanism "
            "of monetary policy. Key relationships:\n\n"
            "• **Duration risk:** Long-duration assets (growth stocks, "
            "bonds) are most sensitive to rate changes via discounted "
            "cash flow mechanics. A 100bps rate rise compresses a "
            "30-year bond more than a 2-year note.\n"
            "• **Bank NIM:** Rising rates expand net interest margins "
            "for banks — bullish for financials in early rate cycles.\n"
            "• **Yield curve:** Inverted curve (short > long rates) "
            "historically precedes recessions by 12-18 months.\n"
            "• **BSP current stance:** 6.0% key rate with 3.1% CPI "
            "gives +2.9% real rate — meaningfully restrictive."
        ),
        "followups": [
            "How do interest rates affect bank stocks?",
            "What does an inverted yield curve mean?",
            "How does this affect my Philippine stock portfolio?"
        ]
    },

    "pe_ratio": {
        "keywords": ["pe ratio", "p/e", "price to earnings",
                     "valuation", "expensive stock", "cheap stock",
                     "multiple", "priced high"],
        "beginner": (
            "**P/E Ratio** (Price-to-Earnings) is one of the most common "
            "ways to judge if a stock is expensive or cheap.\n\n"
            "**Formula:** P/E = Stock Price ÷ Earnings per Share\n\n"
            "**Simple way to think about it:** If a stock has a P/E of 20, "
            "you're paying ₱20 for every ₱1 the company earns per year. "
            "It would take 20 years to 'earn back' your investment "
            "at current earnings.\n\n"
            "**Generally:**\n"
            "• P/E below 15 → potentially cheap\n"
            "• P/E 15-25 → fairly valued\n"
            "• P/E above 30 → expensive (but may be justified by growth)\n\n"
            "Compare the P/E to similar companies in the same industry — "
            "a P/E of 30 might be normal for tech but expensive for a bank."
        ),
        "professional": (
            "**P/E ratio** = Price / EPS. Key variants:\n\n"
            "• **Trailing P/E:** Based on last 12 months' actual EPS\n"
            "• **Forward P/E:** Based on next 12 months' consensus EPS estimate "
            "— more forward-looking but subject to estimate risk\n"
            "• **Normalized P/E:** Smooths earnings over a cycle (e.g. "
            "Shiller CAPE uses 10-year average inflation-adjusted earnings)\n\n"
            "**Limitations:** P/E is meaningless for companies with negative "
            "earnings. High-growth companies justify high P/E through PEG "
            "(P/E ÷ growth rate). Sector comparisons are essential — "
            "PSEi banks trade at ~8-12x vs US tech at 25-40x."
        ),
        "followups": [
            "What is a good P/E ratio for Philippine stocks?",
            "What is the P/E ratio of AAPL right now?",
            "What is the difference between P/E and PEG ratio?"
        ]
    },

    "sharpe_ratio": {
        "keywords": ["sharpe", "risk adjusted", "return per risk",
                     "sharpe ratio", "risk reward"],
        "beginner": (
            "**Sharpe Ratio** measures how much return you got "
            "compared to the risk you took.\n\n"
            "**Simple version:** It's like comparing two employees — "
            "one who consistently delivers good work (high Sharpe) "
            "vs one who sometimes does great but sometimes fails "
            "completely (low Sharpe). You'd usually prefer the consistent one.\n\n"
            "**Reading the number:**\n"
            "• Below 0 → losing money on a risk-adjusted basis\n"
            "• 0-0.5 → weak\n"
            "• 0.5-1.0 → acceptable\n"
            "• 1.0-2.0 → good\n"
            "• Above 2.0 → excellent (rare in practice)\n\n"
            "The Aureline Labs ML strategy on AAPL achieved a Sharpe "
            "of 3.073 — which would be exceptional if it holds out-of-sample."
        ),
        "professional": (
            "**Sharpe Ratio** = (Portfolio Return − Risk-Free Rate) / "
            "Portfolio Standard Deviation × √252 (annualized).\n\n"
            "**Key considerations:**\n"
            "• Uses standard deviation as the risk proxy — penalizes "
            "both upside and downside volatility equally\n"
            "• **Sortino ratio** (downside deviation only) is often more "
            "appropriate for asymmetric return distributions\n"
            "• High Sharpe in backtests often reflects overfitting — "
            "walk-forward validation is essential\n"
            "• Hedge fund benchmarks: >1.0 is decent, >2.0 is strong, "
            ">3.0 is exceptional and should trigger model scrutiny\n"
            "• Our Aureline Labs ML strategy achieved Sharpe 3.073 on "
            "AAPL walk-forward — driven by regime filtering avoiding "
            "the -33% buy-and-hold drawdown."
        ),
        "followups": [
            "What is max drawdown?",
            "How does the Aureline Labs strategy compare to buy and hold?",
            "What is the difference between Sharpe and Sortino ratio?"
        ]
    },

    "rsi": {
        "keywords": ["rsi", "relative strength", "overbought",
                     "oversold", "momentum indicator", "rsi below 30",
                     "rsi above 70"],
        "beginner": (
            "**RSI (Relative Strength Index)** is a number from 0 to 100 "
            "that tells you if a stock has been moving too fast.\n\n"
            "**Reading it:**\n"
            "• **RSI above 70:** The stock has been rising very fast — "
            "it may be 'overbought' and due for a pullback\n"
            "• **RSI below 30:** The stock has been falling very fast — "
            "it may be 'oversold' and could bounce\n"
            "• **RSI 30-70:** Normal range — no strong signal\n\n"
            "**Important:** RSI tells you about momentum, not fundamentals. "
            "A stock can stay overbought (RSI > 70) for months during a "
            "strong bull run, or stay oversold during a prolonged selloff. "
            "It's a clue, not a guarantee."
        ),
        "professional": (
            "**RSI** = 100 − 100/(1 + RS), where RS = "
            "Average gain / Average loss over N periods (typically 14).\n\n"
            "**Advanced interpretation:**\n"
            "• **RSI divergence:** Price makes new high but RSI doesn't → "
            "bearish divergence, potential reversal signal\n"
            "• **RSI failure swing:** RSI breaks below a prior RSI trough "
            "without price confirming — often precedes breakdown\n"
            "• **Regime-conditional RSI:** RSI < 30 in a BULL_TRENDING regime "
            "is a stronger buy signal than RSI < 30 in HIGH_VOLATILITY\n"
            "• In Aureline Labs experiments, RSI (mom_rsi_14) ranked as the "
            "second most important feature after mom_return_20d in the "
            "mean-reversion experiment (EXP-54F78E, ROC-AUC 0.5674)"
        ),
        "followups": [
            "What is the current RSI for AAPL?",
            "What other technical indicators does Aureline Labs use?",
            "What is moving average and how does it work?"
        ]
    },

    "diversification": {
        "keywords": ["diversify", "diversification", "don't put eggs",
                     "spread risk", "portfolio allocation",
                     "correlation", "different stocks"],
        "beginner": (
            "**Diversification** means spreading your money across "
            "different investments so that if one falls, the others "
            "might hold up.\n\n"
            "**Classic example:** If you put all your money in one "
            "restaurant and it closes, you lose everything. But if "
            "you invested in 10 different restaurants across different "
            "cities, one closing doesn't ruin you.\n\n"
            "**In Aureline Labs' analysis (Milestone 28):**\n"
            "• AAPL and SPY had a correlation of 0.747 — they move "
            "together, so owning both gives little diversification\n"
            "• XOM (energy) and JNJ (healthcare) had very low "
            "correlation with tech stocks — they're real diversifiers\n\n"
            "**Key insight:** True diversification isn't just owning "
            "many stocks — it's owning stocks that don't move together."
        ),
        "professional": (
            "**Portfolio diversification** is quantified through the "
            "**Diversification Ratio (DR)** = weighted average "
            "individual volatility / portfolio volatility.\n\n"
            "From Aureline Labs Portfolio Analysis:\n"
            "• Equal-weight (1/N) portfolio: DR = 1.503\n"
            "• Risk parity: DR = 1.555\n"
            "• Concentrated tech (AAPL/MSFT/NVDA): DR = 1.187\n\n"
            "Key findings:\n"
            "• AAPL↔SPY correlation: 0.747 (redundant risk)\n"
            "• XOM↔MSFT correlation: 0.053 (genuine diversifier)\n"
            "• JNJ↔NVDA correlation: -0.092 (slight hedge)\n"
            "• Equal-weight Sharpe: 0.930 vs concentrated tech: 0.723\n"
            "**Academic reference:** DeMiguel et al. (2009) showed 1/N "
            "allocation outperforms optimized portfolios out-of-sample."
        ),
        "followups": [
            "What is correlation between stocks?",
            "What is risk parity?",
            "How should I build a diversified Philippine portfolio?"
        ]
    },

    "market_regime": {
        "keywords": ["regime", "bull market", "bear market",
                     "market condition", "bull trending",
                     "high volatility", "sideways market"],
        "beginner": (
            "A **market regime** describes what kind of environment "
            "the market is in right now — like weather for investing.\n\n"
            "**Aureline Labs detects 4 regimes:**\n\n"
            "📈 **Bull Trending:** Prices are rising steadily. "
            "Trend-following strategies work well. Good time to be invested.\n\n"
            "📉 **Bear Trending:** Prices are falling. High risk. "
            "Aureline Labs sits out during this regime.\n\n"
            "⚠️ **High Volatility:** Prices are swinging wildly up "
            "and down. Very unpredictable. Our system avoids trading.\n\n"
            "➡️ **Sideways:** No clear direction. Harder to make money "
            "from trends. Mean-reversion strategies may work better.\n\n"
            "**Key finding from our research:** By only trading in Bull "
            "Trending and Sideways regimes, the Aureline Labs strategy "
            "improved its Sharpe ratio from 0.587 to 0.888."
        ),
        "professional": (
            "**Market regime detection** classifies the current market "
            "environment to condition strategy selection.\n\n"
            "**Aureline Labs Regime Detector (Milestone 27):**\n"
            "Inputs: SMA(200), realized vol vs historical median, "
            "trend strength (ADX proxy), 50-day price momentum\n\n"
            "**Classification logic (priority order):**\n"
            "1. HIGH_VOLATILITY: realized vol > 75th percentile of 252d\n"
            "2. BEAR_TRENDING: price < SMA200 AND momentum < -2%\n"
            "3. BULL_TRENDING: price > SMA200 AND momentum > 2% AND "
            "trend strength > 0.5\n"
            "4. SIDEWAYS: otherwise\n\n"
            "**AAPL 2021-2026 distribution:**\n"
            "Bull: 34.6% · Sideways: 32.9% · "
            "High Vol: 24.1% · Bear: 8.3%\n\n"
            "**Strategy impact:** Filtering to Bull+Sideways improved "
            "Sharpe from 0.587→0.888 and reduced max drawdown "
            "from -11.17%→-6.56% on AAPL."
        ),
        "followups": [
            "What is the current regime for AAPL?",
            "How does Aureline Labs use regime detection in trading?",
            "What is the Sharpe ratio?"
        ]
    },

    "options": {
        "keywords": ["option", "call option", "put option",
                     "black scholes", "derivative", "hedge",
                     "implied volatility", "strike price"],
        "beginner": (
            "An **option** is a contract that gives you the right — "
            "but not the obligation — to buy or sell a stock at a "
            "specific price before a specific date.\n\n"
            "**Two types:**\n"
            "• **Call option:** Right to BUY at a set price. "
            "You profit if the stock goes UP.\n"
            "• **Put option:** Right to SELL at a set price. "
            "You profit if the stock goes DOWN.\n\n"
            "**Real example from Aureline Labs:**\n"
            "AAPL is at $283.78. A 3-month call option at the same "
            "price costs $20.82. If AAPL rises to $320, your call "
            "is worth ~$36 — a profit of ~$15 on a $20.82 investment.\n\n"
            "**Risk:** If AAPL stays flat or falls, the option expires "
            "worthless and you lose the $20.82 premium."
        ),
        "professional": (
            "**Options** are derivatives priced using the "
            "Black-Scholes model (Milestone 30):\n\n"
            "**Call price:** C = S·N(d₁) − K·e^(-rT)·N(d₂)\n"
            "**Put price:** P = K·e^(-rT)·N(-d₂) − S·N(-d₁)\n\n"
            "Where d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)\n\n"
            "**Greeks from Aureline Labs (AAPL ATM, 3m):**\n"
            "• Delta: 0.5752 — option moves $0.575 per $1 stock move\n"
            "• Gamma: 0.0145 — delta sensitivity to stock price\n"
            "• Theta: -$0.081/day — daily time decay\n"
            "• Vega: $0.611/1% vol — vol sensitivity\n\n"
            "**Volatility risk premium:** When market call priced at $15 "
            "vs model price of $12.81, implied vol = 20.93% vs "
            "historical 17.35% — a +3.58% risk premium that option "
            "sellers systematically earn over time."
        ),
        "followups": [
            "What is implied volatility?",
            "What are options Greeks?",
            "How does Aureline Labs price options?"
        ]
    },

    "aureline_performance": {
        "keywords": ["aureline", "how did aureline do",
                     "your strategy", "platform performance",
                     "research results", "what have you found",
                     "experiment results", "best strategy"],
        "beginner": (
            "**What Aureline Labs has discovered so far:**\n\n"
            "We've run 13 experiments testing different strategies. "
            "Here are the most important findings:\n\n"
            "1. **Momentum works:** Stocks that have been going up "
            "for the past month tend to keep going up — the "
            "20-day return is the single most useful signal we've found.\n\n"
            "2. **Timing matters more than the strategy:** Avoiding "
            "volatile market conditions improved our results more than "
            "any strategy change.\n\n"
            "3. **Simple often beats complex:** A straightforward "
            "equal-weight portfolio (Sharpe 0.930) beat more "
            "complicated strategies.\n\n"
            "4. **More features ≠ better AI:** Our best ML model "
            "used only 6 features. Adding more made it worse."
        ),
        "professional": (
            "**Aureline Labs Research Summary (13 experiments):**\n\n"
            "**Key findings:**\n"
            "• mom_return_20d is the dominant feature across all ML "
            "experiments (appeared as top feature in 5/8 experiments)\n"
            "• Models with ≤7 features: mean AUC 0.5438 vs >7 features: "
            "0.4833 — curse of dimensionality confirmed in low-data regime\n"
            "• Best ML experiment (EXP-54F78E): ROC-AUC 0.5674, "
            "Sharpe 1.22 using 6 mean-reversion features\n"
            "• Regime-conditional ML: reduced max drawdown 69% vs global ML\n"
            "• Bull+Sideways regime filter: Sharpe 0.888 vs unfiltered 0.587\n"
            "• SMA strategy beat buy-and-hold on 0/8 tickers "
            "(2021-2026 bull market period)\n"
            "• Equal-weight portfolio: Sharpe 0.930, max DD -17.54%, "
            "beat concentrated tech on risk-adjusted basis"
        ),
        "followups": [
            "What is the best strategy Aureline Labs has found?",
            "What is the Sharpe ratio of the ML strategy?",
            "What is the curse of dimensionality?"
        ]
    },
}


# ======================================================================
# SECTION 3: LIVE DATA QUERIES
# ======================================================================
def get_live_context(ticker, db=None):
    """
    Pulls live price and sentiment data for a ticker
    from the Aureline Labs database.
    """
    try:
        from src.data_handler import get_price_data
        from src.regime_detector import RegimeDetector

        today = datetime.now().strftime("%Y-%m-%d")
        data  = get_price_data(ticker, "2021-01-01", today)

        if data is None or len(data) < 20:
            return None

        close = data["Close"]
        price = float(close.iloc[-1])
        ret1d = float(close.pct_change(1).iloc[-1]) * 100
        ret20 = float(close.pct_change(20).iloc[-1]) * 100

        dr    = close.pct_change().dropna()
        rsi_d = close.diff()
        gain  = rsi_d.clip(lower=0).rolling(14).mean()
        loss  = (-rsi_d.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, float("inf"))
        rsi   = float((100 - 100/(1+rs)).iloc[-1])

        try:
            det    = RegimeDetector()
            reg_df = det.detect(data)
            regime = reg_df["regime"].iloc[-1]
        except Exception:
            regime = "UNKNOWN"

        result = {
            "ticker": ticker,
            "price":  round(price, 2),
            "ret_1d": round(ret1d, 2),
            "ret_20d":round(ret20, 2),
            "rsi":    round(rsi, 1),
            "regime": regime,
        }

        # Add news sentiment if db available
        if db:
            try:
                news = db.get_recent_news(ticker=ticker, limit=10)
                if news:
                    scores = [n.get("impact_score", 0) for n in news]
                    avg    = sum(scores) / len(scores)
                    result["news_sentiment"] = round(avg, 3)
                    result["news_count"]     = len(news)
            except Exception:
                pass

        return result

    except Exception as e:
        logger.debug(f"Live context error for {ticker}: {e}")
        return None


# ======================================================================
# SECTION 4: QUERY ROUTER
# ======================================================================
def route_query(question):
    """
    Identifies what the question is about by matching
    against concept keywords and ticker patterns.
    Returns: (query_type, matched_concept, ticker_mentioned)
    """
    q_lower = question.lower()

    # Check for ticker mentions
    ticker_match = None
    common_tickers = ["aapl", "msft", "nvda", "tsla", "jpm",
                      "xom", "jnj", "spy", "phi", "bdouy",
                      "bphly", "jbfcf", "psei"]
    for t in common_tickers:
        if t in q_lower:
            ticker_match = t.upper()
            if ticker_match == "PSEI":
                ticker_match = "PSEI.PS"
            break

    # Also scan for uppercase tickers
    if not ticker_match:
        matches = re.findall(r'\b([A-Z]{2,5})\b', question)
        if matches:
            ticker_match = matches[0]

    # Match concept
    best_concept = None
    best_score   = 0
    for concept, config in CONCEPTS.items():
        score = sum(1 for kw in config["keywords"]
                    if kw in q_lower)
        if score > best_score:
            best_score   = score
            best_concept = concept

    # Determine query type
    if ticker_match and best_concept:
        query_type = "ticker_concept"
    elif ticker_match:
        query_type = "ticker_lookup"
    elif best_concept and best_score > 0:
        query_type = "concept"
    else:
        query_type = "unknown"

    return query_type, best_concept, ticker_match


# ======================================================================
# SECTION 5: RESPONSE GENERATOR
# ======================================================================
class FinancialCopilot:
    """
    Aureline Labs Financial Copilot.
    Answers financial questions using:
    1. Built-in concept knowledge base
    2. Live data from the database
    3. Research findings from Aureline Labs experiments
    """

    def __init__(self, db=None, is_beginner=True):
        self.db         = db
        self.is_beginner = is_beginner
        logger.info(f"Financial Copilot initialized "
                   f"({'Beginner' if is_beginner else 'Professional'} mode)")


    def answer(self, question):
        """
        Generates a response to a financial question.
        Returns: (response_text, followups, sources)
        """
        if not question or len(question.strip()) < 3:
            return ("Please ask me a question about finance, "
                    "investing, or any of the stocks in our watchlist.",
                    [], [])

        query_type, concept, ticker = route_query(question)
        mode = "beginner" if self.is_beginner else "professional"

        # ── Ticker lookup ──
        if query_type == "ticker_lookup" and ticker:
            live = get_live_context(ticker, self.db)
            if live:
                sig_text = {
                    "BUY":   "our system rates it as a potential **BUY**",
                    "AVOID": "our system currently says to **AVOID**",
                    "HOLD":  "our system says **HOLD** — no strong signal",
                }.get(self._signal(live), "no clear signal")

                if self.is_beginner:
                    response = (
                        f"**{ticker}** is currently trading at "
                        f"**${live['price']:.2f}**.\n\n"
                        f"In the past month, it has moved "
                        f"**{live['ret_20d']:+.1f}%**. "
                        f"The momentum indicator (RSI) is at "
                        f"**{live['rsi']:.0f}** "
                        f"({'oversold — may bounce' if live['rsi'] < 35 else 'overbought — may pull back' if live['rsi'] > 70 else 'normal range'}). "
                        f"The market for this stock is in a "
                        f"**{live['regime'].replace('_',' ').title()}** "
                        f"environment, and {sig_text}."
                    )
                else:
                    response = (
                        f"**{ticker}** · ${live['price']:.2f} · "
                        f"20d: {live['ret_20d']:+.1f}% · "
                        f"RSI: {live['rsi']:.1f} · "
                        f"Regime: {live['regime']}\n\n"
                        f"Signal: {self._signal(live)}"
                    )
                if live.get("news_sentiment") is not None:
                    sent  = live["news_sentiment"]
                    label = ("positive" if sent > 0.1
                             else "negative" if sent < -0.1
                             else "neutral")
                    response += (
                        f"\n\nNews sentiment: **{label}** "
                        f"({live.get('news_count',0)} recent articles, "
                        f"avg score {sent:+.3f})"
                    )
                followups = [
                    f"What is the P/E ratio of {ticker}?",
                    f"Show me the bull and bear case for {ticker}",
                    "What is market regime?",
                ]
                return response, followups, ["Aureline Labs live data"]
            else:
                return (f"I don't have live data for **{ticker}** "
                        f"in my database right now. Try running "
                        f"the Research Agent to update it.",
                        [], [])

        # ── Concept explanation ──
        if concept and query_type in ["concept", "ticker_concept"]:
            config   = CONCEPTS[concept]
            response = config[mode]

            # Attach live data if ticker also mentioned
            if ticker:
                live = get_live_context(ticker, self.db)
                if live:
                    response += (
                        f"\n\n**{ticker} right now:** "
                        f"${live['price']:.2f} · "
                        f"RSI {live['rsi']:.1f} · "
                        f"{live['regime']}"
                    )

            followups = config.get("followups", [])
            return response, followups, ["Aureline Labs knowledge base"]

        # ── Unknown query ──
        suggestions = [
            "What is inflation?",
            "How does the P/E ratio work?",
            "What is the current signal for AAPL?",
            "Explain the Sharpe ratio",
            "What is market regime detection?",
            "How has Aureline Labs performed?",
        ]

        if self.is_beginner:
            response = (
                "I can help you understand financial concepts and "
                "look up data on stocks we track. Here are some "
                "things you can ask me:"
            )
        else:
            response = (
                "Query not matched. The copilot covers financial "
                "concepts, ticker lookups, and Aureline Labs "
                "research findings. Try:"
            )

        return response, suggestions, []

    def _signal(self, live):
        rsi    = live.get("rsi", 50)
        regime = live.get("regime", "")
        ret20  = live.get("ret_20d", 0)
        if regime == "BULL_TRENDING" and rsi < 70:
            return "BUY"
        elif regime in ["BEAR_TRENDING", "HIGH_VOLATILITY"]:
            return "AVOID"
        else:
            return "HOLD"