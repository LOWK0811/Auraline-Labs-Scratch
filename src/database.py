# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = "aureline_labs.db"


# ======================================================================
# SECTION 2: DATABASE MANAGER
# ======================================================================
class Database:
    """
    Aureline Labs persistent SQLite database.

    Tables:
    - experiments   : Research experiment registry
    - prices        : Daily OHLCV price data
    - companies     : Company profiles and metadata
    - news          : News events and market impact assessments
    - reports       : Generated research reports
    - agent_memory  : Agent state and memory for multi-agent system

    Design principle: every piece of data Aureline Labs generates
    or collects should eventually flow through this database so
    agents can query, join, and reason across all of it.
    """

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn    = sqlite3.connect(db_path,
                                       check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # access columns by name
        self._create_tables()
        logger.info(f"Database connected: {db_path}")


    # ======================================================================
    # SECTION 3: TABLE CREATION
    # ======================================================================
    def _create_tables(self):
        """Creates all tables if they don't already exist."""
        cursor = self.conn.cursor()

        # ── Experiments table ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id              TEXT PRIMARY KEY,
                timestamp       TEXT NOT NULL,
                hypothesis      TEXT,
                ticker          TEXT,
                start_date      TEXT,
                end_date        TEXT,
                strategy        TEXT,
                parameters      TEXT,  -- JSON string
                features        TEXT,  -- JSON string
                metrics         TEXT,  -- JSON string
                conclusion      TEXT,
                tags            TEXT,  -- JSON string
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Prices table ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                date        TEXT NOT NULL,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      INTEGER,
                UNIQUE(ticker, date)
            )
        """)

        # ── Companies table ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                ticker          TEXT PRIMARY KEY,
                name            TEXT,
                sector          TEXT,
                industry        TEXT,
                country         TEXT,
                description     TEXT,
                website         TEXT,
                market_cap      REAL,
                profile_json    TEXT,  -- full profile as JSON
                last_updated    TEXT
            )
        """)

        # ── News table ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                headline        TEXT NOT NULL,
                source          TEXT,
                published_at    TEXT,
                url             TEXT,
                summary         TEXT,
                tickers_mentioned TEXT,  -- JSON list
                market_impact   TEXT,    -- 'positive'/'negative'/'neutral'
                impact_score    REAL,    -- -1.0 to +1.0
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Reports table ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                title           TEXT NOT NULL,
                report_type     TEXT,  -- 'daily_brief'/'experiment'/'company'
                content         TEXT,
                tickers_covered TEXT,  -- JSON list
                generated_at    TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Agent memory table ──
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_memory (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name  TEXT NOT NULL,
                memory_key  TEXT NOT NULL,
                memory_value TEXT,
                updated_at  TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(agent_name, memory_key)
            )
        """)

        # ── Indexes for common queries ──
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
            ON prices(ticker, date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_ticker
            ON experiments(ticker)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_published
            ON news(published_at)
        """)

        self.conn.commit()
        logger.info("All tables verified/created")


    # ======================================================================
    # SECTION 4: EXPERIMENT OPERATIONS
    # ======================================================================
    def insert_experiment(self, experiment_dict):
        """Inserts or replaces an experiment record."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO experiments
            (id, timestamp, hypothesis, ticker, start_date,
             end_date, strategy, parameters, features,
             metrics, conclusion, tags)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            experiment_dict["id"],
            experiment_dict["timestamp"],
            experiment_dict.get("hypothesis", ""),
            experiment_dict.get("ticker", ""),
            experiment_dict.get("start", ""),
            experiment_dict.get("end", ""),
            experiment_dict.get("strategy", ""),
            json.dumps(experiment_dict.get("parameters", {})),
            json.dumps(experiment_dict.get("features", [])),
            json.dumps(experiment_dict.get("metrics", {})),
            experiment_dict.get("conclusion", ""),
            json.dumps(experiment_dict.get("tags", []))
        ))
        self.conn.commit()

    def get_experiments(self, ticker=None, strategy=None,
                        min_sharpe=None, limit=50):
        """
        Queries experiments with optional filters.
        This is the kind of query that's trivial in SQL
        but painful in JSON.
        """
        cursor = self.conn.cursor()
        query  = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        if strategy:
            query += " AND strategy LIKE ?"
            params.append(f"%{strategy}%")

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            exp = dict(row)
            exp["parameters"] = json.loads(exp["parameters"] or "{}")
            exp["features"]   = json.loads(exp["features"]   or "[]")
            exp["metrics"]    = json.loads(exp["metrics"]    or "{}")
            exp["tags"]       = json.loads(exp["tags"]       or "[]")

            # Apply min_sharpe filter in Python (JSON field)
            if min_sharpe is not None:
                sharpe = exp["metrics"].get("Sharpe", 0)
                try:
                    if float(str(sharpe)) < min_sharpe:
                        continue
                except (ValueError, TypeError):
                    continue

            results.append(exp)

        return results


    # ======================================================================
    # SECTION 5: PRICE DATA OPERATIONS
    # ======================================================================
    def insert_prices(self, ticker, df):
        """
        Bulk-inserts OHLCV price data for a ticker.
        Uses INSERT OR IGNORE to skip duplicates.
        """
        cursor  = self.conn.cursor()
        records = []

        for date, row in df.iterrows():
            date_str = date.strftime("%Y-%m-%d") \
                       if hasattr(date, "strftime") else str(date)[:10]
            records.append((
                ticker.upper(), date_str,
                float(row.get("Open",   0)),
                float(row.get("High",   0)),
                float(row.get("Low",    0)),
                float(row.get("Close",  0)),
                int(row.get("Volume",   0))
            ))

        cursor.executemany("""
            INSERT OR IGNORE INTO prices
            (ticker, date, open, high, low, close, volume)
            VALUES (?,?,?,?,?,?,?)
        """, records)
        self.conn.commit()
        logger.info(f"Inserted {len(records)} price rows for {ticker}")

    def get_prices(self, ticker, start=None, end=None):
        """Retrieves price data as a list of dicts."""
        cursor = self.conn.cursor()
        query  = "SELECT * FROM prices WHERE ticker = ?"
        params = [ticker.upper()]

        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)

        query += " ORDER BY date ASC"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


    # ======================================================================
    # SECTION 6: COMPANY OPERATIONS
    # ======================================================================
    def insert_company(self, ticker, name, sector="",
                        industry="", country="PH",
                        description="", profile_json=None):
        """Inserts or updates a company profile."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO companies
            (ticker, name, sector, industry, country,
             description, profile_json, last_updated)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            ticker.upper(), name, sector, industry,
            country, description,
            json.dumps(profile_json or {}),
            datetime.now().isoformat()
        ))
        self.conn.commit()

    def get_company(self, ticker):
        """Retrieves a company profile."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM companies WHERE ticker = ?",
            (ticker.upper(),)
        )
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result["profile_json"] = json.loads(
                result["profile_json"] or "{}")
            return result
        return None


    # ======================================================================
    # SECTION 7: NEWS OPERATIONS
    # ======================================================================
    def insert_news(self, headline, source="", published_at=None,
                    url="", summary="", tickers=None,
                    market_impact="neutral", impact_score=0.0):
        """Inserts a news event."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO news
            (headline, source, published_at, url, summary,
             tickers_mentioned, market_impact, impact_score)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            headline, source,
            published_at or datetime.now().isoformat(),
            url, summary,
            json.dumps(tickers or []),
            market_impact, impact_score
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_recent_news(self, ticker=None, days=7, limit=20):
        """Retrieves recent news, optionally filtered by ticker."""
        cursor = self.conn.cursor()
        if ticker:
            cursor.execute("""
                SELECT * FROM news
                WHERE tickers_mentioned LIKE ?
                ORDER BY published_at DESC
                LIMIT ?
            """, (f'%"{ticker}"%', limit))
        else:
            cursor.execute("""
                SELECT * FROM news
                ORDER BY published_at DESC
                LIMIT ?
            """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


    # ======================================================================
    # SECTION 8: AGENT MEMORY OPERATIONS
    # ======================================================================
    def set_memory(self, agent_name, key, value):
        """Stores a key-value memory entry for an agent."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO agent_memory
            (agent_name, memory_key, memory_value, updated_at)
            VALUES (?,?,?,?)
        """, (agent_name, key,
              json.dumps(value) if not isinstance(value, str)
              else value,
              datetime.now().isoformat()))
        self.conn.commit()

    def get_memory(self, agent_name, key, default=None):
        """Retrieves a memory entry for an agent."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT memory_value FROM agent_memory
            WHERE agent_name = ? AND memory_key = ?
        """, (agent_name, key))
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row["memory_value"])
            except (json.JSONDecodeError, TypeError):
                return row["memory_value"]
        return default

    def get_all_memory(self, agent_name):
        """Retrieves all memory for a given agent."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT memory_key, memory_value, updated_at
            FROM agent_memory
            WHERE agent_name = ?
            ORDER BY updated_at DESC
        """, (agent_name,))
        memory = {}
        for row in cursor.fetchall():
            try:
                memory[row["memory_key"]] = json.loads(
                    row["memory_value"])
            except (json.JSONDecodeError, TypeError):
                memory[row["memory_key"]] = row["memory_value"]
        return memory


    # ======================================================================
    # SECTION 9: REPORTS OPERATIONS
    # ======================================================================
    def insert_report(self, title, content, report_type="general",
                      tickers=None):
        """Stores a generated report."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO reports
            (title, report_type, content, tickers_covered)
            VALUES (?,?,?,?)
        """, (title, report_type, content,
              json.dumps(tickers or [])))
        self.conn.commit()
        return cursor.lastrowid

    def get_reports(self, report_type=None, limit=10):
        """Retrieves recent reports."""
        cursor = self.conn.cursor()
        if report_type:
            cursor.execute("""
                SELECT id, title, report_type,
                       tickers_covered, generated_at
                FROM reports WHERE report_type = ?
                ORDER BY generated_at DESC LIMIT ?
            """, (report_type, limit))
        else:
            cursor.execute("""
                SELECT id, title, report_type,
                       tickers_covered, generated_at
                FROM reports
                ORDER BY generated_at DESC LIMIT ?
            """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


    # ======================================================================
    # SECTION 10: DATABASE STATISTICS
    # ======================================================================
    def stats(self):
        """Returns a summary of what's in the database."""
        cursor = self.conn.cursor()
        tables = ["experiments", "prices", "companies",
                  "news", "reports", "agent_memory"]
        result = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            result[table] = cursor.fetchone()[0]
        return result

    def close(self):
        """Closes the database connection."""
        self.conn.close()
        logger.info("Database connection closed")