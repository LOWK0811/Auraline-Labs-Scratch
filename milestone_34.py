# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import json
import logging
import pandas as pd
from src.database import Database
from src.data_handler import get_price_data
from src.experiment_tracker import ExperimentTracker


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
# SECTION 3: INITIALIZE DATABASE
# ======================================================================
db = Database()

print(f"\n{'='*55}")
print(f"  AURELINE LABS — DATABASE LAYER")
print(f"{'='*55}")


# ======================================================================
# SECTION 4: MIGRATE EXPERIMENT REGISTRY TO DATABASE
# ======================================================================
print(f"\n  Migrating experiment registry to SQLite...")

tracker = ExperimentTracker()
migrated = 0

for exp in tracker.registry:
    db.insert_experiment(exp)
    migrated += 1

print(f"  Migrated {migrated} experiments from JSON registry")


# ======================================================================
# SECTION 5: MIGRATE PRICE DATA TO DATABASE
# ======================================================================
print(f"\n  Migrating cached price data to SQLite...")

tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ",
           "TSLA", "SPY"]
price_rows = 0

for ticker in tickers:
    data = get_price_data(ticker, "2021-01-01", "2026-06-01")
    if data is not None:
        db.insert_prices(ticker, data)
        price_rows += len(data)

print(f"  Migrated {price_rows:,} price rows "
      f"across {len(tickers)} tickers")


# ======================================================================
# SECTION 6: ADD SAMPLE COMPANY PROFILES
# ======================================================================
print(f"\n  Adding company profiles...")

companies = [
    ("AAPL",  "Apple Inc.",
     "Technology", "Consumer Electronics", "US",
     "Apple designs, manufactures, and sells smartphones, "
     "computers, tablets, and related software and services. "
     "The iPhone is its primary revenue driver."),
    ("MSFT",  "Microsoft Corporation",
     "Technology", "Software", "US",
     "Microsoft develops and licenses software, hardware, "
     "and cloud services. Azure cloud and Office 365 are "
     "its fastest-growing segments."),
    ("NVDA",  "NVIDIA Corporation",
     "Technology", "Semiconductors", "US",
     "NVIDIA designs GPUs for gaming, professional "
     "visualization, data centers, and AI. The AI boom "
     "made data center its largest business segment."),
    ("JPM",   "JPMorgan Chase & Co.",
     "Financial Services", "Banks", "US",
     "JPMorgan is the largest US bank by assets, "
     "providing investment banking, consumer banking, "
     "financial transaction processing, and asset management."),
    ("XOM",   "Exxon Mobil Corporation",
     "Energy", "Oil & Gas", "US",
     "ExxonMobil is one of the world's largest publicly "
     "traded oil and gas companies, involved in exploration, "
     "production, refining, and marketing."),
]

for ticker, name, sector, industry, country, desc in companies:
    db.insert_company(ticker, name, sector, industry,
                      country, desc)

print(f"  Added {len(companies)} company profiles")


# ======================================================================
# SECTION 7: SIMULATE NEWS EVENTS
# ======================================================================
print(f"\n  Adding sample news events...")

news_events = [
    ("Fed holds rates at 4.25-4.50%, signals patience",
     "Reuters", "2026-06-11",
     "The Federal Reserve kept interest rates unchanged, "
     "citing solid economic growth and continued progress "
     "toward its 2% inflation target.",
     ["SPY", "JPM"], "positive", 0.4),
    ("NVIDIA reports record data center revenue of $18.4B",
     "Bloomberg", "2026-05-28",
     "NVIDIA's quarterly revenue surged 69% year-over-year "
     "driven by unprecedented AI chip demand from cloud "
     "hyperscalers and enterprise customers.",
     ["NVDA"], "positive", 0.8),
    ("Apple unveils next-generation iPhone with on-device AI",
     "TechCrunch", "2026-06-09",
     "Apple's annual developer conference revealed deep AI "
     "integration across iOS, with features powered entirely "
     "by on-device processing.",
     ["AAPL"], "positive", 0.6),
    ("Oil prices fall 4% on rising US inventory data",
     "Reuters", "2026-06-15",
     "Crude oil prices dropped sharply after weekly inventory "
     "data showed a larger-than-expected build, raising "
     "concerns about demand softness.",
     ["XOM"], "negative", -0.5),
    ("BSP holds key rate at 6.0% amid easing inflation",
     "Philippine Daily Inquirer", "2026-06-12",
     "Bangko Sentral ng Pilipinas kept its benchmark "
     "interest rate steady as headline inflation eased "
     "to 3.1%, within the 2-4% target band.",
     [], "neutral", 0.1),
]

for headline, source, date, summary, tickers, impact, score in \
        news_events:
    db.insert_news(headline, source, date, "",
                   summary, tickers, impact, score)

print(f"  Added {len(news_events)} news events")


# ======================================================================
# SECTION 8: TEST AGENT MEMORY SYSTEM
# ======================================================================
print(f"\n  Testing agent memory system...")

db.set_memory("research_agent", "last_run",
              "2026-06-22T03:00:00")
db.set_memory("research_agent", "tickers_covered",
              ["AAPL","MSFT","NVDA","JPM","XOM"])
db.set_memory("quant_agent", "best_experiment",
              {"id": "EXP-54F78E", "auc": 0.5674,
               "sharpe": 1.22})
db.set_memory("executive_agent", "report_count", 1)

research_memory = db.get_all_memory("research_agent")
print(f"  Research Agent memory: "
      f"{list(research_memory.keys())}")


# ======================================================================
# SECTION 9: DEMONSTRATE SQL QUERYING POWER
# ======================================================================
print(f"\n  Demonstrating SQL query capabilities...")

# Query 1: All AAPL experiments
aapl_exps = db.get_experiments(ticker="AAPL")
print(f"\n  AAPL experiments in database: {len(aapl_exps)}")

# Query 2: Best experiments by Sharpe
good_exps = db.get_experiments(min_sharpe=0.5)
print(f"  Experiments with Sharpe > 0.5: {len(good_exps)}")
for exp in good_exps[:3]:
    print(f"    {exp['id']} | {exp['ticker']} | "
          f"Sharpe: {exp['metrics'].get('Sharpe','N/A')}")

# Query 3: Latest prices for AAPL
aapl_prices = db.get_prices("AAPL", start="2026-05-01")
print(f"\n  AAPL prices since May 2026: "
      f"{len(aapl_prices)} rows")
if aapl_prices:
    latest = aapl_prices[-1]
    print(f"  Latest: {latest['date']} | "
          f"Close: ${latest['close']:.2f}")

# Query 4: Positive news events
positive_news = [n for n in db.get_recent_news(limit=10)
                 if n["market_impact"] == "positive"]
print(f"\n  Positive news events: {len(positive_news)}")
for news in positive_news[:2]:
    print(f"    [{news['published_at'][:10]}] "
          f"{news['headline'][:55]}...")

# Query 5: NVDA-specific news
nvda_news = db.get_recent_news(ticker="NVDA")
print(f"\n  NVDA-related news: {len(nvda_news)}")


# ======================================================================
# SECTION 10: DATABASE STATISTICS
# ======================================================================
stats = db.stats()

print(f"\n{'='*55}")
print(f"  AURELINE LABS DATABASE — FINAL STATISTICS")
print(f"{'='*55}")
for table, count in stats.items():
    print(f"  {table:<20} {count:>8,} records")

db_size = os.path.getsize("aureline_labs.db") / 1024
print(f"\n  Database file: aureline_labs.db")
print(f"  File size:     {db_size:.1f} KB")
print(f"{'='*55}")

db.close()