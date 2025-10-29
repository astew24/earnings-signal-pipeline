"""
db.py — PostgreSQL time-series schema for the earnings signal pipeline.

Tables:
  filings         — raw 8-K filing metadata from EDGAR
  commentary      — extracted management sentences per filing
  tone_scores     — FinBERT sentiment scores per sentence
  ohlcv           — intraday OHLCV bars from Polygon
  signal_results  — OLS regression results per ticker/earnings date
"""

import os
from datetime import date, datetime
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DSN = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/earnings_signal")


def get_conn():
    """Open a new psycopg2 connection. Caller is responsible for closing it."""
    return psycopg2.connect(DSN)


DDL = """
-- 8-K filing metadata
CREATE TABLE IF NOT EXISTS filings (
    id              SERIAL PRIMARY KEY,
    ticker          TEXT        NOT NULL,
    cik             TEXT        NOT NULL,
    accession_no    TEXT        NOT NULL UNIQUE,
    filed_at        TIMESTAMPTZ NOT NULL,
    period_of_report DATE,
    form_type       TEXT        DEFAULT '8-K',
    filing_url      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker);
CREATE INDEX IF NOT EXISTS idx_filings_filed_at ON filings(filed_at);

-- Extracted sentences from management commentary sections
CREATE TABLE IF NOT EXISTS commentary (
    id              SERIAL PRIMARY KEY,
    filing_id       INTEGER     NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    ticker          TEXT        NOT NULL,
    sentence_idx    INTEGER     NOT NULL,
    sentence_text   TEXT        NOT NULL,
    section         TEXT,           -- e.g. 'MD&A', 'Results of Operations'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_commentary_filing ON commentary(filing_id);

-- FinBERT tone scores per sentence
CREATE TABLE IF NOT EXISTS tone_scores (
    id              SERIAL PRIMARY KEY,
    commentary_id   INTEGER     NOT NULL REFERENCES commentary(id) ON DELETE CASCADE,
    ticker          TEXT        NOT NULL,
    filed_at        TIMESTAMPTZ NOT NULL,
    positive_score  REAL        NOT NULL,
    negative_score  REAL        NOT NULL,
    neutral_score   REAL        NOT NULL,
    net_tone        REAL        NOT NULL GENERATED ALWAYS AS (positive_score - negative_score) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tone_ticker      ON tone_scores(ticker);
CREATE INDEX IF NOT EXISTS idx_tone_filed_at    ON tone_scores(filed_at);

-- Intraday OHLCV bars (1-minute) from Polygon.io
CREATE TABLE IF NOT EXISTS ohlcv (
    id          BIGSERIAL   PRIMARY KEY,
    ticker      TEXT        NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    open        REAL        NOT NULL,
    high        REAL        NOT NULL,
    low         REAL        NOT NULL,
    close       REAL        NOT NULL,
    volume      BIGINT      NOT NULL,
    UNIQUE (ticker, ts)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_ts ON ohlcv(ticker, ts);

-- OLS regression results per ticker/event window
CREATE TABLE IF NOT EXISTS signal_results (
    id              SERIAL PRIMARY KEY,
    ticker          TEXT        NOT NULL,
    period_start    DATE        NOT NULL,
    period_end      DATE        NOT NULL,
    n_events        INTEGER     NOT NULL,
    beta            REAL,           -- tone coefficient
    alpha           REAL,           -- intercept
    r_squared       REAL,
    p_value         REAL,
    std_err         REAL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (ticker, period_start, period_end)
);
"""


def init_schema():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
        conn.commit()
    print("[db] Schema initialised.")


def upsert_filing(record: dict) -> int:
    sql = """
    INSERT INTO filings (ticker, cik, accession_no, filed_at, period_of_report, form_type, filing_url)
    VALUES (%(ticker)s, %(cik)s, %(accession_no)s, %(filed_at)s, %(period_of_report)s,
            %(form_type)s, %(filing_url)s)
    ON CONFLICT (accession_no) DO UPDATE SET filing_url = EXCLUDED.filing_url
    RETURNING id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, record)
            row = cur.fetchone()
        conn.commit()
    return row[0]


def insert_commentary(records: list[dict]):
    sql = """
    INSERT INTO commentary (filing_id, ticker, sentence_idx, sentence_text, section)
    VALUES (%(filing_id)s, %(ticker)s, %(sentence_idx)s, %(sentence_text)s, %(section)s)
    RETURNING id
    """
    ids = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in records:
                cur.execute(sql, r)
                ids.append(cur.fetchone()[0])
        conn.commit()
    return ids


def insert_tone_scores(records: list[dict]):
    sql = """
    INSERT INTO tone_scores (commentary_id, ticker, filed_at, positive_score, negative_score, neutral_score)
    VALUES (%(commentary_id)s, %(ticker)s, %(filed_at)s, %(positive_score)s,
            %(negative_score)s, %(neutral_score)s)
    ON CONFLICT DO NOTHING
    """
    with get_conn() as conn:
        psycopg2.extras.execute_batch(conn.cursor(), sql, records)
        conn.commit()


def upsert_ohlcv(records: list[dict]):
    sql = """
    INSERT INTO ohlcv (ticker, ts, open, high, low, close, volume)
    VALUES (%(ticker)s, %(ts)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
    ON CONFLICT (ticker, ts) DO NOTHING
    """
    with get_conn() as conn:
        psycopg2.extras.execute_batch(conn.cursor(), sql, records)
        conn.commit()


def get_tone_and_returns(ticker: Optional[str] = None) -> list[dict]:
    """
    Join aggregate net tone per filing with the next-day open-to-close return.
    Used as input to OLS regression.
    """
    where = "AND f.ticker = %(ticker)s" if ticker else ""
    sql = f"""
    WITH filing_tone AS (
        SELECT
            ts.ticker,
            f.period_of_report                AS event_date,
            f.filed_at,
            AVG(ts.net_tone)                  AS avg_net_tone,
            COUNT(*)                          AS n_sentences
        FROM tone_scores ts
        JOIN commentary c  ON c.id = ts.commentary_id
        JOIN filings    f  ON f.id = c.filing_id
        {where}
        GROUP BY ts.ticker, f.period_of_report, f.filed_at
    ),
    next_day_returns AS (
        SELECT
            o_open.ticker,
            DATE(o_open.ts AT TIME ZONE 'America/New_York') AS trade_date,
            (o_close.close - o_open.open) / NULLIF(o_open.open, 0) AS daily_return
        FROM (
            SELECT ticker, DATE(ts AT TIME ZONE 'America/New_York') AS d,
                   FIRST_VALUE(open) OVER (
                       PARTITION BY ticker, DATE(ts AT TIME ZONE 'America/New_York')
                       ORDER BY ts
                   ) AS open
            FROM ohlcv
        ) o_open
        JOIN (
            SELECT ticker, DATE(ts AT TIME ZONE 'America/New_York') AS d,
                   LAST_VALUE(close) OVER (
                       PARTITION BY ticker, DATE(ts AT TIME ZONE 'America/New_York')
                       ORDER BY ts
                       ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                   ) AS close
            FROM ohlcv
        ) o_close ON o_open.ticker = o_close.ticker AND o_open.d = o_close.d
        GROUP BY o_open.ticker, trade_date, daily_return
    )
    SELECT
        ft.ticker,
        ft.event_date,
        ft.avg_net_tone,
        ft.n_sentences,
        ndr.daily_return
    FROM filing_tone ft
    LEFT JOIN next_day_returns ndr
        ON ft.ticker = ndr.ticker
        AND ndr.trade_date = ft.event_date + INTERVAL '1 day'
    WHERE ndr.daily_return IS NOT NULL
    ORDER BY ft.ticker, ft.event_date
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"ticker": ticker} if ticker else {})
            return [dict(r) for r in cur.fetchall()]


def upsert_signal_result(record: dict):
    sql = """
    INSERT INTO signal_results
        (ticker, period_start, period_end, n_events, beta, alpha, r_squared, p_value, std_err)
    VALUES
        (%(ticker)s, %(period_start)s, %(period_end)s, %(n_events)s,
         %(beta)s, %(alpha)s, %(r_squared)s, %(p_value)s, %(std_err)s)
    ON CONFLICT (ticker, period_start, period_end) DO UPDATE SET
        beta=EXCLUDED.beta, alpha=EXCLUDED.alpha, r_squared=EXCLUDED.r_squared,
        p_value=EXCLUDED.p_value, std_err=EXCLUDED.std_err, created_at=NOW()
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, record)
        conn.commit()
