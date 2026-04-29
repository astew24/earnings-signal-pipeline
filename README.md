# earnings-signal-pipeline

End-to-end pipeline that asks one question: does the tone of managementt
language in 8-K filings predict the stock's next-day move?

Ingest SEC 8-Ks via EDGAR full-text search, extract management
commentary with spaCy, score tone with FinBERT, join against 1-minute
Polygon bars to compute next-day open-to-close returns, then run OLS
per ticker and expose the results through a small Flask dashboard.

## Modules

| File | |
|---|---|
| `ingestion.py` | EDGAR 8-K fetch + Polygon 1-min OHLCV |
| `scoring.py`   | spaCy re-segmentation + FinBERT tone scores |
| `analysis.py`  | Per-ticker OLS, writes Plotly HTML report |
| `db.py`        | Postgres schema + the tone↔return join |
| `app.py`       | Flask dashboard (table + Plotly charts) |

Each module runs standalone — useful for backfills or swapping the tone
model without touching the rest of the pipeline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env   # set POLYGON_API_KEY, EDGAR_USER_AGENT

docker-compose up -d postgres
```

## Run

```bash
python ingestion.py --tickers AAPL MSFT --from-date 2024-01-01 --to-date 2024-06-30
python scoring.py --batch-size 32
python analysis.py --output report.html
python app.py   # http://localhost:5000
```

`--skip-ohlcv` on ingestion skips the Polygon call if you don't have a
key — the tone side of the pipeline still works, you just can't run the
regression.

## Config (.env)

| Variable | Default | |
|---|---|---|
| `DATABASE_URL` | local Postgres | |
| `POLYGON_API_KEY` | — | required for OHLCV |
| `EDGAR_USER_AGENT` | example string | SEC fair-use requires contact info |
| `FINBERT_MODEL` | `ProsusAI/finbert` | |
| `SPACY_MODEL` | `en_core_web_sm` | |
| `DEVICE` | `-1` | `0` for first GPU |

## Notes

- `net_tone` is a generated column on `tone_scores`: `positive - negative`.
- EDGAR fair-use is ~10 req/s; ingestion sleeps 120ms between calls and backs off on 429.
- Low R² is expected — tone explains a small slice of daily return variance. The question is whether β is consistently signed across tickers and whether p-values survive across the panel.
