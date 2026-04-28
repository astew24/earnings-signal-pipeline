# Earnings Signal Pipeline: Unstructured Text to Alpha

An end-to-end data pipeline designed to answer a core research question: **Does the tone of management language in SEC filings provide predictive signals for short-term market moves?**

This project demonstrates a complete Natural Language Understanding (NLU) workflow for finance, turning messy regulatory text into structured, actionable data for regression analysis and decision support.

## Project Overview
The pipeline automates the extraction, processing, and analysis of SEC 8-K filings:
1.  **Ingestion:** Fetches 8-K filings via SEC EDGAR and matches them with high-frequency (1-minute) price data from Polygon.io.
2.  **Processing:** Uses **spaCy** for intelligent document segmentation and **FinBERT** (a specialized NLP model for finance) to score management sentiment.
3.  **Analysis:** Computes next-day open-to-close returns and runs per-ticker OLS regressions to validate the signal's predictive power.
4.  **Visualization:** Serves insights through a Flask dashboard with interactive Plotly visualizations.

## Tech Stack
- **Languages:** Python (Pandas, NumPy, Scikit-learn)
- **NLP:** FinBERT (HuggingFace), spaCy
- **Data:** PostgreSQL, SEC EDGAR API, Polygon.io
- **Infrastructure:** Docker, Flask, Plotly

## Key Features
- **Intelligent Segmentation:** Goes beyond keyword matching to analyze the actual tone of management commentary.
- **High-Frequency Alignment:** Precisely joins event timestamps with minute-level price bars to isolate the market reaction.
- **Modular Design:** Each component (ingestion, scoring, analysis) runs independently, allowing for easy swaps of the NLP model or data source.
- **Robust Rate Limiting:** Implements SEC fair-use policies with intelligent back-off and 120ms sleep intervals.

## How to Run Locally

### 1. Environment Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env   # Set POLYGON_API_KEY and EDGAR_USER_AGENT
```

### 2. Infrastructure
```bash
docker-compose up -d postgres
```

### 3. Execution
```bash
# Ingest filings for specific tickers
python ingestion.py --tickers AAPL MSFT --from-date 2024-01-01 --to-date 2024-06-30

# Generate sentiment scores
python scoring.py --batch-size 32

# Run regression and generate report
python analysis.py --output report.html

# Launch the dashboard
python app.py   # http://localhost:5000
```

## What I Learned
- **Handling Unstructured Data:** Financial filings are notoriously noisy. I learned how to use spaCy to re-segment text into meaningful units before scoring.
- **Model Nuance:** Standard sentiment models often fail on financial jargon. Using FinBERT was critical for capturing the "hidden" tone in management's cautious or optimistic phrasing.
- **Data Engineering:** Managing rate limits and joining asynchronous event data with time-series bars required robust error handling and database design.

## Future Improvements
- **LLM Summarization:** Integrate GPT-4 or Claude to summarize the "why" behind significant sentiment shifts.
- **Real-time Monitoring:** Expand the pipeline to monitor the EDGAR RSS feed for real-time signal generation.
- **Multi-Factor Integration:** Combine sentiment with traditional factors like volume and volatility for a more robust predictive model.
