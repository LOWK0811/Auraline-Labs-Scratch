# Aureline Labs — Inaugural Research Summary

**Aureline Labs — Quantitative Research & Intelligence Platform**
*Ateneo de Manila University · Applied Mathematics · Mathematical Finance*
*Generated: 2026-06-21*

---

## Executive Summary

This report summarizes 13 quantitative
research experiments conducted on the Aureline Labs platform. Research
spanned 3 assets
(AAPL, MSFT, NVDA) using
4 strategy frameworks
(ML_RandomForest, SMA_20, SMA_50, SMA_10).

| Metric | Value |
|--------|-------|
| Total Experiments | 13 |
| Assets Studied | AAPL, MSFT, NVDA |
| Strategies Tested | ML_RandomForest, SMA_20, SMA_50, SMA_10 |
| Beat Buy & Hold | 0/13 (0.0%) |
| Mean ROC-AUC | 0.5130 |
| Best ROC-AUC | 0.5674 |
| Mean Sharpe | -0.441 |
| Best Sharpe | 1.220 |

---

## Key Research Findings

### Finding 1: Strongest Predictive Signal Found

Experiment EXP-54F78E (ML_RandomForest on AAPL) achieved the highest ROC-AUC of 0.5674. Features used: mom_return_1d, mom_return_5d, mom_return_20d...

### Finding 2: Dominant Feature Category

Across 8 ML experiments, momentum features consistently ranked highest in feature importance. 'mom_return_20d' appeared as the top feature most frequently.

### Finding 3: Curse of Dimensionality Confirmed

Models with ≤7 features averaged ROC-AUC of 0.5438, while models with >7 features averaged 0.4833. In a low-data regime (<700 training rows), feature parsimony consistently outperforms comprehensive feature sets.

### Finding 4: SMA vs ML Strategy Comparison

SMA strategies averaged Sharpe 0.262 across 5 experiments. ML strategies averaged -0.881 across 8 experiments. Both strategy types show regime-dependent performance.


---

## Experiment Registry

| ID | Ticker | Strategy | Date | ROC-AUC | Sharpe | Return | Beat B&H |
|----|--------|----------|------|---------|--------|--------|----------|
| EXP-760624 | AAPL | SMA_20 | 2026-06-20 | N/A | 0.587 | N/A | N/A |
| EXP-A95FDC | MSFT | SMA_20 | 2026-06-20 | N/A | -0.415 | N/A | N/A |
| EXP-AD9AA9 | NVDA | SMA_20 | 2026-06-20 | N/A | 0.739 | N/A | N/A |
| EXP-616342 | AAPL | SMA_10 | 2026-06-20 | N/A | 0.262 | N/A | N/A |
| EXP-475A20 | AAPL | SMA_50 | 2026-06-20 | N/A | 0.137 | N/A | N/A |
| EXP-6F9682 | AAPL | ML_RandomForest | 2026-06-20 | 0.4828 | -1.577 | -9.98% | No |
| EXP-5BA75C | AAPL | ML_RandomForest | 2026-06-20 | 0.545 | -0.569 | 2.72% | No |
| EXP-11E402 | AAPL | ML_RandomForest | 2026-06-20 | 0.4828 | -1.577 | -9.98% | No |
| EXP-81CF97 | AAPL | ML_RandomForest | 2026-06-20 | 0.545 | -0.569 | 2.72% | No |
| EXP-934BA2 | AAPL | ML_RandomForest | 2026-06-20 | 0.5143 | -1.308 | -9.56% | No |
| EXP-54F78E | AAPL | ML_RandomForest | 2026-06-21 | 0.5674 | 1.22 | 29.54% | No |
| EXP-33815F | NVDA | ML_RandomForest | 2026-06-21 | 0.4843 | -1.362 | -8.93% | No |
| EXP-19729B | AAPL | ML_RandomForest | 2026-06-21 | 0.5471 | -1.307 | -12.38% | No |

---

## Best Performing Experiments (by ROC-AUC)

**EXP-54F78E** — ML_RandomForest on AAPL
- Hypothesis: *Assets that have experienced significant negative price momentum show statistica...*
- ROC-AUC: 0.5674 | Sharpe: 1.22 | Return: 29.54%
- Conclusion: The Random Forest model demonstrated meaningful predictive edge on AAPL with a ROC-AUC of 0.5674 on the held-out test set. The three most important fe...

**EXP-19729B** — ML_RandomForest on AAPL
- Hypothesis: *Assets with positive price momentum over multiple timeframes continue to outperf...*
- ROC-AUC: 0.5471 | Sharpe: -1.307 | Return: -12.38%
- Conclusion: The Random Forest model demonstrated modest but real predictive edge on AAPL with a ROC-AUC of 0.5471 on the held-out test set. The three most importa...

**EXP-5BA75C** — ML_RandomForest on AAPL
- Hypothesis: *Momentum features alone (multi-window returns, z-score, RSI) are sufficient to g...*
- ROC-AUC: 0.545 | Sharpe: -0.569 | Return: 2.72%
- Conclusion: The Random Forest model demonstrated modest but real predictive edge on AAPL with a ROC-AUC of 0.545 on the held-out test set. The three most importan...


---

## Experiments Requiring Further Investigation

**EXP-AD9AA9** — SMA_20 on NVDA
- Hypothesis: *Does SMA(20) capture the AI-driven momentum in NVDA despite high volatility caus...*
- ROC-AUC: N/A | Sharpe: 0.739 | Return: N/A
- Conclusion: The SMA(20) strategy on NVDA produced a Sharpe of 0.739 and CAGR of 12.76% over the study period. Strategy underperformed buy-and-hold. ATR-based posi...

**EXP-616342** — SMA_10 on AAPL
- Hypothesis: *Does a faster SMA(10) window improve responsiveness on AAPL at the cost of more ...*
- ROC-AUC: N/A | Sharpe: 0.262 | Return: N/A
- Conclusion: The SMA(10) strategy on AAPL produced a Sharpe of 0.262 and CAGR of 6.97% over the study period. Strategy underperformed buy-and-hold. ATR-based posit...

**EXP-475A20** — SMA_50 on AAPL
- Hypothesis: *Does a slower SMA(50) window reduce whipsaw and improve Sharpe on AAPL at the co...*
- ROC-AUC: N/A | Sharpe: 0.137 | Return: N/A
- Conclusion: The SMA(50) strategy on AAPL produced a Sharpe of 0.137 and CAGR of 5.92% over the study period. Strategy underperformed buy-and-hold. ATR-based posit...


---

## Research Themes

Most frequently investigated concepts:

- `aapl`: 7 experiments
- `sma`: 5 experiments
- `trend-following`: 5 experiments
- `atr-sizing`: 5 experiments
- `ml`: 5 experiments

---

## Methodology Notes

All experiments follow the Aureline Labs research protocol:

1. **Hypothesis formulation** before examining results
2. **Time-based train/test split** to prevent look-ahead bias
3. **Walk-forward validation** on held-out test sets only
4. **Transaction friction** of 0.1% per trade applied throughout
5. **ATR-based position sizing** at 1% risk per trade
6. **Automatic logging** to experiment registry with unique IDs

---

## Platform Architecture

The Aureline Labs research platform comprises:

- **Data Pipeline**: yfinance + Parquet caching (`src/data_handler.py`)
- **Feature Factory**: 27 features across 5 modules (`src/features/`)
- **Research Engine**: Automated ML experiment runner (`src/research_journal.py`)
- **Experiment Tracker**: Registry + Markdown reports (`src/experiment_tracker.py`)
- **Risk Infrastructure**: ATR sizing, circuit breakers (`src/risk.py`)
- **Portfolio Engine**: Correlation, allocation, simulation (`src/portfolio.py`)
- **Monte Carlo**: Block bootstrap simulation (`src/monte_carlo.py`)
- **Options Pricing**: Black-Scholes + Greeks (`src/options.py`)
- **AI Assistant**: Natural language hypothesis interpreter (`src/ai_assistant.py`)

---

## Disclaimer

*This document is produced by the Aureline Labs automated research
platform for educational and research purposes only. Nothing in this
report constitutes financial advice or a recommendation to buy, sell,
or hold any security. All results are based on historical data and
past performance does not predict future results.*

---

*Aureline Labs v1.0 · Quantitative Research & Intelligence Platform*
*Ateneo de Manila University · BS Applied Mathematics · Mathematical Finance*
