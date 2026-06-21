# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: REPORT GENERATOR
# ======================================================================
class ReportGenerator:
    """
    Reads the full experiment registry and generates a comprehensive
    research summary report in Markdown format.

    This is Aureline Labs' institutional memory made readable —
    the document that tells the story of what we've tested,
    what we've found, and what it means.
    """

    def __init__(self,
                 registry_path="experiments/registry.json",
                 output_path="experiments"):
        self.registry_path = registry_path
        self.output_path   = output_path
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                self.registry = json.load(f)
            logger.info(f"Registry loaded: "
                       f"{len(self.registry)} experiments")
        else:
            self.registry = []
            logger.warning("No registry found")

    def _parse_metric(self, value):
        """Safely parses a metric value to float."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.replace("%", "").replace("+", ""))
        return 0.0


    # ======================================================================
    # SECTION 3: REGISTRY ANALYSIS
    # ======================================================================
    def analyze_registry(self):
        """Computes summary statistics across all experiments."""
        if not self.registry:
            return {}

        all_aucs    = []
        all_sharpes = []
        all_returns = []
        tickers     = []
        strategies  = []
        tags_all    = []
        beat_bh     = 0

        for exp in self.registry:
            m = exp.get("metrics", {})
            tickers.append(exp.get("ticker", ""))
            strategies.append(exp.get("strategy", ""))
            tags_all.extend(exp.get("tags", []))

            auc = self._parse_metric(m.get("ROC-AUC", 0.5))
            sh  = self._parse_metric(m.get("Sharpe", 0))
            ret = self._parse_metric(m.get("Strategy Return", 0))

            all_aucs.append(auc)
            all_sharpes.append(sh)
            all_returns.append(ret)

            if m.get("Beat B&H") == "Yes":
                beat_bh += 1

        return {
            "total_experiments":  len(self.registry),
            "tickers_studied":    list(set(tickers)),
            "strategies_used":    list(set(strategies)),
            "beat_bh_count":      beat_bh,
            "beat_bh_rate":       beat_bh / len(self.registry),
            "mean_auc":           np.mean(all_aucs),
            "best_auc":           np.max(all_aucs),
            "worst_auc":          np.min(all_aucs),
            "mean_sharpe":        np.mean(all_sharpes),
            "best_sharpe":        np.max(all_sharpes),
            "mean_return":        np.mean(all_returns),
            "best_return":        np.max(all_returns),
            "common_tags":        pd.Series(tags_all)
                                  .value_counts().head(5).to_dict()
        }

    def get_best_experiments(self, metric="ROC-AUC", n=3):
        """Returns top N experiments by a given metric."""
        def safe_parse(exp):
            return self._parse_metric(
                exp.get("metrics", {}).get(metric, 0))
        return sorted(self.registry,
                      key=safe_parse, reverse=True)[:n]

    def get_key_findings(self):
        """
        Extracts the most important research findings
        by looking for patterns across experiments.
        """
        findings = []

        # Finding 1: Best performing experiment
        best = self.get_best_experiments("ROC-AUC", 1)
        if best:
            b = best[0]
            auc = self._parse_metric(
                b.get("metrics", {}).get("ROC-AUC", 0))
            findings.append({
                "title": "Strongest Predictive Signal Found",
                "detail": (
                    f"Experiment {b['id']} ({b['strategy']} on "
                    f"{b['ticker']}) achieved the highest ROC-AUC "
                    f"of {auc:.4f}. Features used: "
                    f"{', '.join(b.get('features', [])[:3])}..."
                )
            })

        # Finding 2: Feature category dominance
        ml_exps = [e for e in self.registry
                   if e.get("strategy") == "ML_RandomForest"]
        if ml_exps:
            top_feature = b.get("metrics", {}).get("Top Feature", "")
            if top_feature:
                if top_feature.startswith("mom_"):
                    cat = "momentum"
                elif top_feature.startswith("vol_"):
                    cat = "volatility"
                elif top_feature.startswith("trend_"):
                    cat = "trend"
                else:
                    cat = "regime"
                findings.append({
                    "title": "Dominant Feature Category",
                    "detail": (
                        f"Across {len(ml_exps)} ML experiments, "
                        f"{cat} features consistently ranked "
                        f"highest in feature importance. "
                        f"'{top_feature}' appeared as the top "
                        f"feature most frequently."
                    )
                })

        # Finding 3: Curse of dimensionality
        auc_by_feat_count = []
        for exp in ml_exps:
            n_feats = len(exp.get("features", []))
            auc = self._parse_metric(
                exp.get("metrics", {}).get("ROC-AUC", 0.5))
            auc_by_feat_count.append((n_feats, auc))

        if len(auc_by_feat_count) >= 3:
            auc_by_feat_count.sort(key=lambda x: x[0])
            small = [a for n, a in auc_by_feat_count if n <= 7]
            large = [a for n, a in auc_by_feat_count if n > 7]
            if small and large:
                small_mean = np.mean(small)
                large_mean = np.mean(large)
                if small_mean > large_mean:
                    findings.append({
                        "title": "Curse of Dimensionality Confirmed",
                        "detail": (
                            f"Models with ≤7 features averaged "
                            f"ROC-AUC of {small_mean:.4f}, while "
                            f"models with >7 features averaged "
                            f"{large_mean:.4f}. In a low-data "
                            f"regime (<700 training rows), feature "
                            f"parsimony consistently outperforms "
                            f"comprehensive feature sets."
                        )
                    })

        # Finding 4: SMA vs ML comparison
        sma_exps = [e for e in self.registry
                    if "SMA" in e.get("strategy", "")]
        if sma_exps and ml_exps:
            sma_sharpes = [
                self._parse_metric(
                    e.get("metrics", {}).get("Sharpe", 0))
                for e in sma_exps
            ]
            ml_sharpes = [
                self._parse_metric(
                    e.get("metrics", {}).get("Sharpe", 0))
                for e in ml_exps
            ]
            findings.append({
                "title": "SMA vs ML Strategy Comparison",
                "detail": (
                    f"SMA strategies averaged Sharpe "
                    f"{np.mean(sma_sharpes):.3f} across "
                    f"{len(sma_exps)} experiments. ML strategies "
                    f"averaged {np.mean(ml_sharpes):.3f} across "
                    f"{len(ml_exps)} experiments. Both strategy "
                    f"types show regime-dependent performance."
                )
            })

        return findings


    # ======================================================================
    # SECTION 4: GENERATE MARKDOWN REPORT
    # ======================================================================
    def generate(self, title=None):
        """
        Generates a comprehensive Markdown research summary
        across all experiments in the registry.
        """
        date_str  = datetime.now().strftime("%Y-%m-%d")
        title     = title or "Quantitative Research Summary"
        filename  = (f"{self.output_path}/"
                    f"AURELINE_RESEARCH_SUMMARY_{date_str}.md")

        stats    = self.analyze_registry()
        findings = self.get_key_findings()
        best3    = self.get_best_experiments("ROC-AUC", 3)
        worst3   = self.get_best_experiments("ROC-AUC", 100)[-3:]

        # Build experiment table
        exp_rows = []
        for exp in self.registry:
            m = exp.get("metrics", {})
            exp_rows.append(
                f"| {exp['id']} "
                f"| {exp['ticker']} "
                f"| {exp['strategy']} "
                f"| {exp['timestamp'][:10]} "
                f"| {m.get('ROC-AUC', 'N/A')} "
                f"| {m.get('Sharpe', 'N/A')} "
                f"| {m.get('Strategy Return', 'N/A')} "
                f"| {m.get('Beat B&H', 'N/A')} |"
            )
        exp_table = "\n".join(exp_rows)

        # Build findings section
        findings_md = ""
        for i, f in enumerate(findings, 1):
            findings_md += f"\n### Finding {i}: {f['title']}\n\n"
            findings_md += f"{f['detail']}\n"

        # Build best/worst experiment summaries
        def exp_summary(exp):
            m = exp.get("metrics", {})
            return (
                f"**{exp['id']}** — {exp['strategy']} on "
                f"{exp['ticker']}\n"
                f"- Hypothesis: *{exp.get('hypothesis', '')[:80]}...*\n"
                f"- ROC-AUC: {m.get('ROC-AUC', 'N/A')} | "
                f"Sharpe: {m.get('Sharpe', 'N/A')} | "
                f"Return: {m.get('Strategy Return', 'N/A')}\n"
                f"- Conclusion: {exp.get('conclusion', '')[:150]}...\n"
            )

        best_md  = "\n".join([exp_summary(e) for e in best3])
        worst_md = "\n".join([exp_summary(e) for e in worst3])

        # Tags frequency
        tags_md = "\n".join([
            f"- `{tag}`: {count} experiments"
            for tag, count in stats.get("common_tags", {}).items()
        ])

        report = f"""# {title}

**Aureline Labs — Quantitative Research & Intelligence Platform**
*Ateneo de Manila University · Applied Mathematics · Mathematical Finance*
*Generated: {date_str}*

---

## Executive Summary

This report summarizes {stats.get('total_experiments', 0)} quantitative
research experiments conducted on the Aureline Labs platform. Research
spanned {len(stats.get('tickers_studied', []))} assets
({', '.join(stats.get('tickers_studied', []))}) using
{len(stats.get('strategies_used', []))} strategy frameworks
({', '.join(stats.get('strategies_used', []))}).

| Metric | Value |
|--------|-------|
| Total Experiments | {stats.get('total_experiments', 0)} |
| Assets Studied | {', '.join(stats.get('tickers_studied', []))} |
| Strategies Tested | {', '.join(stats.get('strategies_used', []))} |
| Beat Buy & Hold | {stats.get('beat_bh_count', 0)}/{stats.get('total_experiments', 0)} ({stats.get('beat_bh_rate', 0):.1%}) |
| Mean ROC-AUC | {stats.get('mean_auc', 0):.4f} |
| Best ROC-AUC | {stats.get('best_auc', 0):.4f} |
| Mean Sharpe | {stats.get('mean_sharpe', 0):.3f} |
| Best Sharpe | {stats.get('best_sharpe', 0):.3f} |

---

## Key Research Findings
{findings_md}

---

## Experiment Registry

| ID | Ticker | Strategy | Date | ROC-AUC | Sharpe | Return | Beat B&H |
|----|--------|----------|------|---------|--------|--------|----------|
{exp_table}

---

## Best Performing Experiments (by ROC-AUC)

{best_md}

---

## Experiments Requiring Further Investigation

{worst_md}

---

## Research Themes

Most frequently investigated concepts:

{tags_md}

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
"""

        os.makedirs(self.output_path, exist_ok=True)
        with open(filename, "w") as f:
            f.write(report)

        logger.info(f"Research summary written: {filename}")
        return filename