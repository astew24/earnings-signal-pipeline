"""
ingestion.py — SEC EDGAR 8-K filing ingestion + Polygon.io intraday OHLCV.

Steps:
  1. For each ticker, resolve the CIK via EDGAR company search.
  2. Fetch recent 8-K filings via the EDGAR full-text search API.
  3. Download and parse the 8-K full-text to extract management commentary sections.
  4. Fetch intraday 1-minute OHLCV from Polygon.io around the filing date.
  5. Persist all data to PostgreSQL via db.py.

Usage:
    python ingestion.py --tickers AAPL MSFT NVDA --days 90
    python ingestion.py --tickers AAPL --from-date 2024-01-01 --to-date 2024-06-30
"""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from dotenv import load_dotenv

import db

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
EDGAR_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "earnings-signal-pipeline research@example.com",
)

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_COMPANY_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}&forms=8-K"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FILING_URL  = "https://www.sec.gov/Archives/edgar/full-index/{year}/{quarter}/full-index.json"
EDGAR_FULLTEXT    = "https://efts.sec.gov/LATEST/search-index"

POLYGON_AGG_URL = (
    "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{from_date}/{to_date}"
)

RATE_LIMIT_DELAY = 0.12   # SEC fair-use: ≤10 req/s


def _headers():
    return {"User-Agent": EDGAR_USER_AGENT, "Accept": "application/json"}


# ---------------------------------------------------------------------------
# EDGAR — CIK lookup
# ---------------------------------------------------------------------------

def get_cik(ticker: str) -> Optional[str]:
    """Resolve ticker to CIK using EDGAR company tickers JSON."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=_headers(), timeout=20)
    r.raise_for_status()
    data = r.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            cik = str(entry["cik_str"]).zfill(10)
            print(f"[edgar] {ticker} → CIK {cik}")
            return cik
    print(f"[edgar] CIK not found for {ticker}")
    return None


# ---------------------------------------------------------------------------
# EDGAR — 8-K filings via EDGAR full-text search
# ---------------------------------------------------------------------------

def fetch_8k_filings(
    ticker: str,
    cik: str,
    from_date: date,
    to_date: date,
) -> list[dict]:
    """
    Use the EDGAR full-text search API to find 8-K filings for a CIK.
    Returns list of filing metadata dicts.
    """
    params = {
        "q": f'"{ticker}"',
        "dateRange": "custom",
        "startdt": from_date.isoformat(),
        "enddt": to_date.isoformat(),
        "forms": "8-K",
        "_source": "file_date,period_of_report,file_num,entity_name,file_type",
    }
    time.sleep(RATE_LIMIT_DELAY)
    r = requests.get(
        "https://efts.sec.gov/LATEST/search-index",
        params=params,
        headers=_headers(),
        timeout=30,
    )
    if r.status_code != 200:
        print(f"[edgar] Search HTTP {r.status_code} for {ticker}")
        return []

    hits = r.json().get("hits", {}).get("hits", [])
    filings = []
    for hit in hits:
        src = hit.get("_source", {})
        accession_no = hit.get("_id", "").replace("-", "")
        if not accession_no:
            continue
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession_no[:18].replace('/', '').replace('-', '')}/{accession_no}.txt"
        )
        period_str = src.get("period_of_report", "")
        try:
            period = date.fromisoformat(period_str) if period_str else None
        except ValueError:
            period = None

        filed_str = src.get("file_date", "")
        try:
            filed_at = datetime.fromisoformat(filed_str) if filed_str else datetime.utcnow()
        except ValueError:
            filed_at = datetime.utcnow()

        filings.append({
            "ticker": ticker,
            "cik": cik,
            "accession_no": accession_no,
            "filed_at": filed_at,
            "period_of_report": period,
            "form_type": "8-K",
            "filing_url": filing_url,
        })
    print(f"[edgar] {len(filings)} 8-K filings found for {ticker}")
    return filings


# ---------------------------------------------------------------------------
# EDGAR — Full text extraction
# ---------------------------------------------------------------------------

# Patterns that identify management commentary sections in 8-K filings
_SECTION_PATTERNS = [
    r"results of operations",
    r"management.{0,10}discussion",
    r"forward.looking",
    r"outlook",
    r"guidance",
    r"revenue.*quarter",
    r"earnings per share",
]
_SECTION_RE = re.compile("|".join(_SECTION_PATTERNS), re.IGNORECASE)

# Sentence splitter (simple punctuation-based)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def extract_commentary(html_text: str) -> list[dict]:
    """
    Extract management commentary sentences from an 8-K filing HTML/text.
    Returns list of {sentence_idx, sentence_text, section} dicts.
    """
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", " ", html_text)
    clean = re.sub(r"\s+", " ", clean)

    # Split into rough paragraphs and filter to commentary sections
    paragraphs = clean.split("\n")
    relevant_paragraphs = [p for p in paragraphs if _SECTION_RE.search(p)]
    if not relevant_paragraphs:
        # Fallback: take all paragraphs with >50 chars (likely prose)
        relevant_paragraphs = [p for p in paragraphs if len(p.strip()) > 50]

    sentences = []
    idx = 0
    for para in relevant_paragraphs[:30]:  # limit to first 30 matching paragraphs
        para_sentences = _SENT_RE.split(para.strip())
        for sent in para_sentences:
            sent = sent.strip()
            if 20 < len(sent) < 1000:  # filter noise
                sentences.append({
                    "sentence_idx": idx,
                    "sentence_text": sent,
                    "section": "management_commentary",
                })
                idx += 1
    return sentences


def fetch_filing_text(filing_url: str) -> Optional[str]:
    """Download the full text of an 8-K filing."""
    time.sleep(RATE_LIMIT_DELAY)
    try:
        r = requests.get(filing_url, headers=_headers(), timeout=45)
        # SEC fair-use policy: max 10 req/s — back off on 429
        if r.status_code == 429:
            time.sleep(10)
            r = requests.get(filing_url, headers=_headers(), timeout=45)
        if r.status_code == 200:
            return r.text
        print(f"[edgar] Filing fetch HTTP {r.status_code}: {filing_url}")
    except requests.RequestException as e:
        print(f"[edgar] Filing fetch error: {e}")
    return None


# ---------------------------------------------------------------------------
# Polygon.io — Intraday OHLCV
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    ticker: str,
    from_date: date,
    to_date: date,
) -> list[dict]:
    """
    Fetch 1-minute OHLCV bars from Polygon.io.
    Returns list of OHLCV dicts.
    """
    if not POLYGON_API_KEY:
        print("[polygon] No API key — skipping OHLCV fetch")
        return []

    url = POLYGON_AGG_URL.format(
        ticker=ticker,
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            print(f"[polygon] HTTP {r.status_code} for {ticker}")
            return []
        data = r.json()
        results = data.get("results", [])
        bars = [
            {
                "ticker": ticker,
                "ts": datetime.fromtimestamp(bar["t"] / 1000),
                "open": bar["o"],
                "high": bar["h"],
                "low": bar["l"],
                "close": bar["c"],
                "volume": int(bar["v"]),
            }
            for bar in results
        ]
        print(f"[polygon] {len(bars)} 1-min bars for {ticker} ({from_date} – {to_date})")
        return bars
    except requests.RequestException as e:
        print(f"[polygon] Request error: {e}")
        return []


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_ingestion(
    tickers: list[str],
    from_date: date,
    to_date: date,
    skip_ohlcv: bool = False,
):
    db.init_schema()

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"  Ingesting: {ticker}")
        print(f"{'='*50}")

        # 1. Resolve CIK
        cik = get_cik(ticker)
        if not cik:
            print(f"[skip] Cannot find CIK for {ticker}")
            continue

        # 2. Fetch 8-K filings
        filings = fetch_8k_filings(ticker, cik, from_date, to_date)

        # 3. For each filing: download, extract commentary, persist
        for filing in filings[:10]:   # cap at 10 per ticker for demo
            filing_id = db.upsert_filing(filing)

            text = fetch_filing_text(filing["filing_url"])
            if not text:
                continue

            sentences = extract_commentary(text)
            if not sentences:
                print(f"  [warn] No commentary extracted for {filing['accession_no']}")
                continue

            commentary_records = [
                {**s, "filing_id": filing_id, "ticker": ticker}
                for s in sentences
            ]
            com_ids = db.insert_commentary(commentary_records)
            print(f"  [edgar] {len(com_ids)} sentences extracted from {filing['accession_no']}")

        # 4. Fetch OHLCV
        if not skip_ohlcv:
            bars = fetch_ohlcv(ticker, from_date, to_date)
            if bars:
                db.upsert_ohlcv(bars)

    print("\n[ingestion] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SEC EDGAR 8-K ingestion + Polygon OHLCV")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--from-date", type=date.fromisoformat,
                        default=date.today() - timedelta(days=90))
    parser.add_argument("--to-date", type=date.fromisoformat,
                        default=date.today())
    parser.add_argument("--skip-ohlcv", action="store_true",
                        help="Skip Polygon OHLCV fetch (useful without API key)")
    args = parser.parse_args()
    run_ingestion(
        tickers=[t.upper() for t in args.tickers],
        from_date=args.from_date,
        to_date=args.to_date,
        skip_ohlcv=args.skip_ohlcv,
    )


if __name__ == "__main__":
    main()
