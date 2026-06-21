# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
from src.report_generator import ReportGenerator


# ======================================================================
# SECTION 2: LOGGING SETUP
# ======================================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 3: GENERATE THE RESEARCH SUMMARY
# ======================================================================
generator = ReportGenerator()
stats     = generator.analyze_registry()
findings  = generator.get_key_findings()

print(f"\n{'='*60}")
print(f"  AURELINE LABS — RESEARCH SUMMARY GENERATOR")
print(f"{'='*60}")
print(f"\n  Registry size:    {stats['total_experiments']} experiments")
print(f"  Tickers studied:  {', '.join(stats['tickers_studied'])}")
print(f"  Strategies used:  {', '.join(stats['strategies_used'])}")
print(f"  Mean ROC-AUC:     {stats['mean_auc']:.4f}")
print(f"  Best ROC-AUC:     {stats['best_auc']:.4f}")
print(f"  Mean Sharpe:      {stats['mean_sharpe']:.3f}")
print(f"  Beat B&H:         "
      f"{stats['beat_bh_count']}/{stats['total_experiments']} "
      f"({stats['beat_bh_rate']:.1%})")

print(f"\n  KEY FINDINGS:")
for i, finding in enumerate(findings, 1):
    print(f"\n  {i}. {finding['title']}")
    print(f"     {finding['detail'][:120]}...")

filename = generator.generate(
    title="Aureline Labs — Inaugural Research Summary"
)

print(f"\n  Report generated: {filename}")
print(f"\n  Open this file in VS Code to read the full report.")
print(f"  It can also be converted to PDF for institutional sharing.")
print(f"\n{'='*60}")