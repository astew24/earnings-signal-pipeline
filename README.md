# earnings-signal-pipeline

**[Live Demo →](https://astew24.github.io/earnings-signal-pipeline/)**

## What this does

`earnings-signal-pipeline` is an end-to-end **quantitative NLP pipeline** that measures whether the tone of CEO/CFO language in earnings filings predicts short-term stock price moves. It:

1. **Ingests** SEC 8-K filings via the EDGAR full-text search API for a list of tickers
2. **Extracts** management commentary sentences using spaCy (sentence boundary detection + financial keyword filtering)
3. **Scores** each sentence for tone (positive / negative / neutral) using **FinBERT** — a BERT model pre-trained on Financial PhraseBank
4. **Joins** tone scores with intraday 1-minute OHLCV data from **Polygon.io**, computing next-day open-to-close returns for each filing date
5. **Runs OLS regression** (tone → return) per ticker and stores coefficients, R², and p-values
6. **Serves results** through a Flask web app with interactive Plotly scatter charts and a β-coefficient summary dashboard

The pipeline tests the research hypothesis: *"do more positive earnings releases predict positive next-day returns?"* — a standard question in quantitative finance academic literature.

---

## Project structure

```
earnings-signal-pipeline/
├── ingestion.py     # EDGAR 8-K fetch + Polygon.io OHLCV ingestion
├── scoring.py       # spaCy sentence extraction + FinBERT tone scoring
├── analysis.py      # OLS regression + Plotly report generation
├── db.py            # PostgreSQL schema + query helpers
├── app.py           # Flask web app (dashboard + Plotly charts)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick start

### 1. Install dependencies

```bash
git clone <repo>
cd earnings-signal-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
# Edit .env: add your POLYGON_API_KEY and EDGAR_USER_AGENT
```

### 2. Start Postgres

```bash
docker-compose up -d postgres
```

### 3. Ingest data

```bash
# Ingest AAPL and MSFT 8-K filings from the past 90 days
python ingestion.py --tickers AAPL MSFT --days 90

# Skip OHLCV if you don't have a Polygon API key
python ingestion.py --tickers AAPL --skip-ohlcv
```

### 4. Score tone

```bash
python scoring.py             # score all tickers
python scoring.py --ticker AAPL --batch-size 32
```

### 5. Run regression

```bash
python analysis.py
# → prints regression table and writes analysis_report.html
```

### 6. Start the web app

```bash
python app.py
# → http://localhost:5000
```

---

## Web dashboard

| Route | Description |
|---|---|
| `/` | KPI overview + signal results table |
| `/chart/scatter/<TICKER>` | Tone vs return scatter with OLS fit |
| `/chart/summary` | β coefficients bar chart for all tickers |
| `/api/tone/<TICKER>` | JSON tone time-series |
| `/results` | Full results as JSON |
| `/run/analysis` | POST to trigger OLS re-run |
| `/health` | Liveness check |

---

## Database schema

```
filings        — 8-K metadata (ticker, CIK, accession number, period)
commentary     — individual sentences extracted from filings
tone_scores    — FinBERT scores per sentence (positive, negative, neutral, net_tone)
ohlcv          — 1-minute bars from Polygon (ticker, ts, O/H/L/C/V)
signal_results — OLS results per ticker (β, α, R², p-value)
```

`net_tone` is a generated column: `positive_score - negative_score`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | local Postgres | Connection string |
| `POLYGON_API_KEY` | *(required)* | Polygon.io API key for OHLCV |
| `EDGAR_USER_AGENT` | example string | SEC requires contact info in UA header |
| `FINBERT_MODEL` | `ProsusAI/finbert` | HuggingFace FinBERT checkpoint |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model for sentence detection |
| `DEVICE` | `-1` (CPU) | Set `0` to use GPU for FinBERT |

---

## Modular design

Each module is independently runnable:

```bash
python ingestion.py --tickers NVDA --from-date 2024-01-01 --to-date 2024-03-31
python scoring.py --ticker NVDA --batch-size 128
python analysis.py --ticker NVDA --output nvda_report.html
```

This makes it easy to schedule (e.g. daily cron), backfill historical data, or plug in alternative tone models by replacing `scoring.py`.

---

## Research context

This pipeline implements a version of the **textual analysis** approach pioneered by Loughran & McDonald (2011), updated with transformer-based tone scoring. Academic studies consistently find that management tone in earnings announcements contains incremental information beyond quantitative EPS surprises. The FinBERT model used here was fine-tuned on Financial PhraseBank and consistently outperforms general-purpose BERT on financial sentiment tasks.
