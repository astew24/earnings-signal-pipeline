"""
app.py — Flask web app serving earnings signal results with Plotly charts.

Routes:
  GET /                     — Dashboard overview
  GET /tickers              — JSON list of tickers in DB
  GET /results              — Signal results table (JSON)
  GET /chart/scatter/<ticker> — Interactive scatter plot for one ticker
  GET /chart/summary        — β coefficient summary bar chart
  GET /api/tone/<ticker>    — JSON tone time-series for a ticker
  POST /run/analysis        — Trigger OLS re-run (returns JSON)
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Optional

import psycopg2.extras
from flask import Flask, jsonify, render_template_string, request
from dotenv import load_dotenv

import db

load_dotenv()

app = Flask(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    import numpy as np
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# HTML templates (inline for single-file portability)
# ---------------------------------------------------------------------------

_BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Earnings Signal Pipeline</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #F8FAFC; color: #1E293B; }
    header { background: #1E293B; color: #F1F5F9; padding: 1rem 2rem;
             display: flex; align-items: center; gap: 1rem; }
    header h1 { font-size: 1.25rem; font-weight: 600; }
    nav a { color: #94A3B8; text-decoration: none; margin-left: 1.5rem; font-size: 0.9rem; }
    nav a:hover { color: #F1F5F9; }
    main { max-width: 1200px; margin: 2rem auto; padding: 0 1.5rem; }
    .card { background: #fff; border: 1px solid #E2E8F0; border-radius: 0.75rem;
            padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
    .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; color: #334155; }
    table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
    th { background: #F1F5F9; text-align: left; padding: 0.5rem 0.75rem;
         font-weight: 600; border-bottom: 1px solid #E2E8F0; }
    td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #F1F5F9; }
    tr:hover td { background: #F8FAFC; }
    .badge { display: inline-block; padding: 0.15rem 0.5rem;
             border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
    .badge-green { background: #DCFCE7; color: #16A34A; }
    .badge-red   { background: #FEE2E2; color: #DC2626; }
    .badge-gray  { background: #F1F5F9; color: #64748B; }
    .plotly-chart { width: 100%; height: 420px; }
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; }
    .kpi { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 0.5rem;
           padding: 1rem; text-align: center; }
    .kpi .value { font-size: 1.75rem; font-weight: 700; color: #3B82F6; }
    .kpi .label { font-size: 0.75rem; color: #64748B; margin-top: 0.25rem; }
  </style>
</head>
<body>
  <header>
    <h1>📈 Earnings Signal Pipeline</h1>
    <nav>
      <a href="/">Dashboard</a>
      <a href="/chart/summary">Beta Chart</a>
      <a href="/results">Results JSON</a>
    </nav>
  </header>
  <main>
    {content}
  </main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helper: fetch signal results from DB
# ---------------------------------------------------------------------------

def _get_signal_results() -> list[dict]:
    sql = """
    SELECT ticker, period_start, period_end, n_events, beta, alpha,
           r_squared, p_value, std_err, created_at
    FROM signal_results
    ORDER BY ticker
    """
    with db.get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]


def _get_tone_series(ticker: str, from_date=None, to_date=None) -> list[dict]:
    params: list = [ticker]
    date_clauses = ""
    if from_date:
        date_clauses += " AND filed_at >= %s"
        params.append(from_date)
    if to_date:
        date_clauses += " AND filed_at <= %s"
        params.append(to_date)
    sql = f"""
    SELECT filed_at, AVG(net_tone) as avg_tone, COUNT(*) as n_sentences
    FROM tone_scores
    WHERE ticker = %s{date_clauses}
    GROUP BY filed_at
    ORDER BY filed_at
    """
    with db.get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    results = _get_signal_results()

    # KPI summary
    n_tickers = len(results)
    n_events = sum(r["n_events"] for r in results) if results else 0
    sig_count = sum(1 for r in results if r["p_value"] and r["p_value"] < 0.05)

    kpis = f"""
    <div class="kpi-grid">
      <div class="kpi"><div class="value">{n_tickers}</div><div class="label">Tickers analysed</div></div>
      <div class="kpi"><div class="value">{n_events}</div><div class="label">Earnings events</div></div>
      <div class="kpi"><div class="value">{sig_count}</div><div class="label">Significant signals (p&lt;.05)</div></div>
    </div>
    """

    # Results table
    rows_html = ""
    for r in results:
        p = r["p_value"] or 1.0
        sig_badge = (
            '<span class="badge badge-green">p&lt;.05</span>' if p < 0.05
            else '<span class="badge badge-red">n.s.</span>'
        )
        beta_fmt = f"{r['beta']:.4f}" if r["beta"] else "—"
        r2_fmt = f"{r['r_squared']:.3f}" if r["r_squared"] else "—"
        rows_html += f"""
        <tr>
          <td><a href="/chart/scatter/{r['ticker']}">{r['ticker']}</a></td>
          <td>{r['period_start']}</td>
          <td>{r['period_end']}</td>
          <td>{r['n_events']}</td>
          <td>{beta_fmt}</td>
          <td>{r2_fmt}</td>
          <td>{sig_badge}</td>
        </tr>"""

    if not rows_html:
        rows_html = "<tr><td colspan='7'>No results yet. Run analysis.py first.</td></tr>"

    content = f"""
    <div class="card">
      <h2>Pipeline Overview</h2>
      {kpis}
    </div>
    <div class="card">
      <h2>Signal Results — Tone → Next-Day Return OLS</h2>
      <table>
        <thead>
          <tr>
            <th>Ticker</th><th>From</th><th>To</th><th>N Events</th>
            <th>β (tone)</th><th>R²</th><th>Significance</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    <div class="card">
      <h2>Beta Coefficients</h2>
      <a href="/chart/summary">→ View interactive chart</a>
    </div>
    """
    return _BASE_HTML.format(content=content)


@app.route("/tickers")
def list_tickers():
    sql = "SELECT DISTINCT ticker FROM filings ORDER BY ticker"
    with db.get_conn() as conn:
        with db.get_conn().cursor() as cur:
            cur.execute(sql)
            tickers = [r[0] for r in cur.fetchall()]
    return jsonify(tickers)


@app.route("/results")
def results_json():
    results = _get_signal_results()
    # Convert dates to strings for JSON
    for r in results:
        for k in ("period_start", "period_end", "created_at"):
            if r.get(k):
                r[k] = str(r[k])
    return jsonify(results)


@app.route("/api/tone/<ticker>")
def tone_series(ticker: str):
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    series = _get_tone_series(ticker.upper(), from_date=from_date, to_date=to_date)
    for r in series:
        r["filed_at"] = str(r["filed_at"])
    return jsonify(series)


@app.route("/chart/scatter/<ticker>")
def scatter_chart(ticker: str):
    if not HAS_PLOTLY:
        return "Plotly not installed", 500

    rows = db.get_tone_and_returns(ticker=ticker.upper())
    if not rows:
        content = f"<div class='card'><h2>No data for {ticker}</h2><p>Run ingestion + scoring + analysis first.</p></div>"
        return _BASE_HTML.format(content=content)

    df = pd.DataFrame(rows).dropna(subset=["avg_net_tone", "daily_return"])
    x = df["avg_net_tone"].tolist()
    y = (df["daily_return"] * 100).tolist()
    dates = df["event_date"].astype(str).tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        text=dates,
        textposition="top center",
        marker=dict(size=10, color="#3B82F6", opacity=0.8),
        hovertemplate="Date: %{text}<br>Tone: %{x:.3f}<br>Return: %{y:.2f}%",
        name="Events",
    ))
    if len(x) > 1:
        from scipy.stats import linregress
        slope, intercept, *_ = linregress(x, [v/100 for v in y])
        x_line = [min(x), max(x)]
        y_line = [(slope * xi + intercept) * 100 for xi in x_line]
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(color="#EF4444", width=2, dash="dash"),
            name="OLS fit",
        ))
    fig.update_layout(
        title=f"{ticker} — Earnings Tone vs Next-Day Return",
        xaxis_title="Avg Net Tone (FinBERT positive − negative)",
        yaxis_title="Next-Day Return (%)",
        template="plotly_white",
        height=500,
    )
    chart_json = fig.to_json()
    content = f"""
    <div class="card">
      <h2>{ticker} — Tone vs Return Scatter</h2>
      <div id="chart" class="plotly-chart"></div>
    </div>
    <script>
      Plotly.newPlot('chart', JSON.parse({json.dumps(chart_json)}).data,
                     JSON.parse({json.dumps(chart_json)}).layout);
    </script>
    """
    return _BASE_HTML.format(content=content)


@app.route("/chart/summary")
def summary_chart():
    if not HAS_PLOTLY:
        return "Plotly not installed", 500

    results = _get_signal_results()
    if not results:
        content = "<div class='card'><h2>No results yet</h2><p>Run analysis.py first.</p></div>"
        return _BASE_HTML.format(content=content)

    tickers = [r["ticker"] for r in results]
    betas = [r["beta"] or 0 for r in results]
    p_values = [r["p_value"] or 1 for r in results]
    std_errs = [r["std_err"] or 0 for r in results]
    colors = ["#22C55E" if p < 0.05 else "#94A3B8" for p in p_values]

    fig = go.Figure(go.Bar(
        x=tickers,
        y=betas,
        error_y=dict(type="data", array=[e * 1.96 for e in std_errs], visible=True),
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>β=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Tone → Return β Coefficients (green = statistically significant p<.05)",
        xaxis_title="Ticker",
        yaxis_title="β",
        template="plotly_white",
        height=450,
    )
    chart_json = fig.to_json()
    content = f"""
    <div class="card">
      <h2>Beta Coefficients Summary</h2>
      <div id="chart" class="plotly-chart"></div>
    </div>
    <script>
      Plotly.newPlot('chart', JSON.parse({json.dumps(chart_json)}).data,
                     JSON.parse({json.dumps(chart_json)}).layout);
    </script>
    """
    return _BASE_HTML.format(content=content)


@app.route("/run/analysis", methods=["POST"])
def trigger_analysis():
    """Trigger an OLS re-run via HTTP POST."""
    from analysis import run_ols
    ticker = request.json.get("ticker") if request.is_json else None
    results = run_ols(ticker=ticker)
    return jsonify({"status": "ok", "results": len(results)})


@app.route("/health")
def health():
    try:
        db.get_conn().close()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    db.init_schema()
    app.run(debug=True, host="0.0.0.0", port=5000)
