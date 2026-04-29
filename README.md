# Earnings Signal Pipeline

Python research pipeline for testing whether management tone in SEC 8-K filings is related to short-term stock returns.

[Project page](https://astew24.github.io/earnings-signal-pipeline/) | [SEC EDGAR search API](https://efts.sec.gov/LATEST/search-index) | [Polygon.io](https://polygon.io/)

## Overview

This project combines finance data engineering, NLP, and a simple regression workflow:

- pulls 8-K filing metadata and filing text from SEC EDGAR
- extracts management-commentary-like sentences from each filing
- scores sentence tone with FinBERT
- stores filings, extracted commentary, tone scores, OHLCV bars, and regression results in PostgreSQL
- joins average filing tone to next-day open-to-close returns from Polygon.io
- runs per-ticker OLS regressions and serves the output through a small Flask dashboard

The goal is not to claim a production trading signal. It is an exploratory pipeline for learning how unstructured financial text can be turned into data that can be tested.

## Why I Built This

I wanted a project that touched the full path from raw finance data to a testable signal:

- working with messy SEC filing text
- handling market data APIs and rate limits
- using a finance-specific NLP model instead of a generic sentiment model
- keeping intermediate data in a relational schema
- evaluating the result with a transparent statistical baseline

## How It Works

```text
SEC 8-K filings
      |
      v
text extraction and sentence filtering
      |
      v
spaCy sentence cleanup + FinBERT tone scoring
      |
      v
PostgreSQL tables for filings, commentary, tone, OHLCV, and results
      |
      v
OLS: average filing tone -> next-day open-to-close return
      |
      v
Plotly report + Flask dashboard
```

## Tech Stack

- Python, pandas, NumPy, SciPy
- FinBERT through Hugging Face Transformers
- spaCy for sentence handling
- PostgreSQL with `psycopg2`
- SEC EDGAR and Polygon.io APIs
- Flask and Plotly
- Docker Compose for the local database

## Project Structure

```text
earnings-signal-pipeline/
|-- ingestion.py         # EDGAR filing ingestion and Polygon OHLCV fetch
|-- scoring.py           # spaCy refinement and FinBERT tone scoring
|-- analysis.py          # per-ticker OLS regression and Plotly report
|-- app.py               # Flask dashboard and JSON endpoints
|-- db.py                # PostgreSQL schema and query helpers
|-- docs/index.html      # static GitHub Pages project overview
|-- docker-compose.yml   # local Postgres + optional web service
|-- Dockerfile
|-- requirements.txt
`-- .env.example
```

## How to Run

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create a local environment file:

```bash
cp .env.example .env
```

Set these values in `.env`:

```text
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/earnings_signal
POLYGON_API_KEY=...
EDGAR_USER_AGENT=your-name your-email@example.com
```

Start PostgreSQL:

```bash
docker compose up -d postgres
```

Run the pipeline:

```bash
python ingestion.py --tickers AAPL MSFT --from-date 2024-01-01 --to-date 2024-06-30
python scoring.py --batch-size 32
python analysis.py --output analysis_report.html
python app.py
```

Then open:

```text
http://localhost:5000
```

If you do not have a Polygon.io key yet, you can still test filing ingestion with:

```bash
python ingestion.py --tickers AAPL --from-date 2024-01-01 --to-date 2024-06-30 --skip-ohlcv
```

Regression analysis needs OHLCV data, so it will not produce return-based results until price data is available.

## Example Output

`analysis.py` writes:

- per-ticker regression rows into the `signal_results` table
- an optional Plotly HTML report
- terminal output showing beta, alpha, R-squared, p-value, and sample size for each ticker with enough events

The Flask app reads the same database tables and exposes:

- `/` for the dashboard
- `/results` for regression results as JSON
- `/tickers` for ingested tickers
- `/api/tone/<ticker>` for time-series tone data

## Limitations

- 8-K text extraction is heuristic and will miss or misclassify some sections.
- The return alignment is simple and should be upgraded to use an exchange calendar.
- OLS is only a baseline; it does not control for sector, market, volatility, multiple testing, or transaction costs.
- Results depend on the selected tickers, date range, filing quality, and Polygon.io data availability.
- This is research code, not a trading system.

## Next Steps

- Add tests around filing parsing, SQL joins, and return alignment.
- Add exchange-calendar-aware event windows.
- Store a small non-sensitive fixture dataset for repeatable local demos.
- Compare FinBERT tone against simpler lexical baselines.
- Add multiple-testing correction and richer controls before treating any signal as meaningful.
