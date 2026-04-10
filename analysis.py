"""
analysis.py — OLS regression: tone → next-day price return.

For each ticker with sufficient data:
  Y = next-day open-to-close return
  X = average FinBERT net tone score of 8-K filing sentences

Outputs:
  - Regression coefficients, R², p-values
  - Persisted to signal_results table
  - Plotly scatter charts (tone vs return) + summary bar chart

Usage:
    python analysis.py                         # all tickers
    python analysis.py --ticker AAPL           # single ticker
    python analysis.py --output report.html    # write Plotly report
"""

from __future__ import annotations

import argparse
from typing import Optional

import db

try:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import scipy.stats as stats
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("[warn] numpy/pandas/plotly/scipy not installed — analysis disabled")


# ---------------------------------------------------------------------------
# OLS regression
# ---------------------------------------------------------------------------

def run_ols(ticker: Optional[str] = None, min_events: int = 5) -> list[dict]:
    """
    Pull tone + return data from DB and run OLS per ticker.
    Returns list of regression result dicts.
    """
    if not HAS_DEPS:
        return []

    rows = db.get_tone_and_returns(ticker=ticker)
    if not rows:
        print("[analysis] No joined tone+return data found. Run ingestion + scoring first.")
        return []

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["avg_net_tone", "daily_return"])
    df["daily_return"] = df["daily_return"].astype(float)
    df["avg_net_tone"] = df["avg_net_tone"].astype(float)

    results = []
    for tkr, group in df.groupby("ticker"):
        if len(group) < min_events:
            print(f"[analysis] {tkr}: only {len(group)} events — skipping (need ≥{min_events})")
            continue

        x = group["avg_net_tone"].values
        y = group["daily_return"].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        result = {
            "ticker": tkr,
            "period_start": group["event_date"].min(),
            "period_end": group["event_date"].max(),
            "n_events": len(group),
            "beta": float(slope),
            "alpha": float(intercept),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "std_err": float(std_err),
        }
        results.append(result)
        db.upsert_signal_result(result)

        sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
        print(
            f"  {tkr}: β={slope:.4f} α={intercept:.4f} "
            f"R²={r_value**2:.3f} p={p_value:.3f}{sig} n={len(group)}"
        )

    return results


# ---------------------------------------------------------------------------
# Plotly report
# ---------------------------------------------------------------------------

def build_plotly_report(results: list[dict], rows: list[dict]) -> go.Figure:
    """
    Build a multi-panel Plotly HTML report:
      Panel 1: Scatter (tone vs return) per ticker
      Panel 2: Bar chart of β coefficients with significance bands
    """
    if not HAS_DEPS or not results:
        return go.Figure()

    df_all = pd.DataFrame(rows).dropna(subset=["avg_net_tone", "daily_return"])
    tickers = [r["ticker"] for r in results]
    n_tickers = len(tickers)

    # --- Scatter plots ---
    scatter_fig = make_subplots(
        rows=max(1, (n_tickers + 1) // 2),
        cols=2,
        subplot_titles=tickers,
        shared_yaxes=False,
    )
    for i, res in enumerate(results):
        row, col = divmod(i, 2)
        tkr_df = df_all[df_all["ticker"] == res["ticker"]]
        x = tkr_df["avg_net_tone"].values
        y = tkr_df["daily_return"].values * 100  # as percent

        scatter_fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode="markers",
                marker=dict(size=8, opacity=0.7, color="#3B82F6"),
                name=res["ticker"],
                hovertemplate="Tone: %{x:.3f}<br>Return: %{y:.2f}%",
            ),
            row=row + 1, col=col + 1,
        )
        # Regression line
        if len(x) > 1:
            x_line = np.linspace(x.min(), x.max(), 50)
            y_line = res["beta"] * x_line + res["alpha"]
            scatter_fig.add_trace(
                go.Scatter(
                    x=x_line, y=y_line * 100,
                    mode="lines",
                    line=dict(color="#EF4444", width=2, dash="dash"),
                    name=f"{res['ticker']} OLS fit",
                    showlegend=False,
                ),
                row=row + 1, col=col + 1,
            )

    scatter_fig.update_layout(
        title="Tone vs Next-Day Return by Ticker",
        height=400 * max(1, (n_tickers + 1) // 2),
        template="plotly_white",
    )

    # --- Beta bar chart ---
    # Filter NaN betas — can occur with single-observation tickers
    results = [r for r in results if r['beta'] is not None and not pd.isna(r['beta'])]
    beta_fig = go.Figure()
    colors = [
        "#22C55E" if r["p_value"] < 0.05 else "#94A3B8"
        for r in results
    ]
    beta_fig.add_trace(go.Bar(
        x=[r["ticker"] for r in results],
        y=[r["beta"] for r in results],
        error_y=dict(
            type="data",
            array=[r["std_err"] * 1.96 for r in results],
            visible=True,
        ),
        marker_color=colors,
        hovertemplate="β=%{y:.4f}<br>Ticker: %{x}",
    ))
    beta_fig.update_layout(
        title="Tone → Return β Coefficients (green = p < 0.05)",
        yaxis_title="β (tone coefficient)",
        xaxis_title="Ticker",
        template="plotly_white",
    )

    # Combine into a single figure using subplots
    from plotly.subplots import make_subplots
    combined = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Tone vs Return Scatter", "Beta Coefficients"],
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
    )
    for trace in scatter_fig.data:
        combined.add_trace(trace, row=1, col=1)
    for trace in beta_fig.data:
        combined.add_trace(trace, row=2, col=1)

    combined.update_layout(
        height=900,
        title_text="Earnings Signal Pipeline — Tone-to-Return Analysis",
        template="plotly_white",
        showlegend=False,
    )
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OLS regression: tone → next-day return")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--output", default="analysis_report.html",
                        help="Output HTML file for Plotly charts")
    parser.add_argument("--min-events", type=int, default=5,
                        help="Minimum number of filing events required per ticker to run OLS (default: 5)")
    args = parser.parse_args()

    print("\n[analysis] Running OLS regression …")
    results = run_ols(ticker=args.ticker, min_events=args.min_events)

    if results and HAS_DEPS:
        rows = db.get_tone_and_returns(ticker=args.ticker)
        fig = build_plotly_report(results, rows)
        fig.write_html(args.output)
        print(f"[analysis] Report written → {args.output}")
    elif not results:
        print("[analysis] No results to plot.")


if __name__ == "__main__":
    main()
