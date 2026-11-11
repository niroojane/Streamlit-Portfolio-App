# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Plotly figures shared by every front-end (Streamlit, Gradio, BQuant widgets): dark theme,
Arial Narrow, only the highlighted series drawn at first (the others sit in the legend)."""
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

FONT = "Arial Narrow"


def style(fig: go.Figure, pct: bool = False, height: int = 420) -> go.Figure:
    fig.update_layout(plot_bgcolor="black", paper_bgcolor="black", font=dict(family=FONT, color="white", size=14),
                      height=height, hoverlabel_namelength=-1, legend_title_text='',
                      margin=dict(l=10, r=10, t=50, b=10))
    if pct:
        fig.update_layout(yaxis_tickformat=".2%")
    return fig


def empty(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=f"{title}: no data")
    return style(fig, height=200)


def line(frame: pd.DataFrame, title: str, highlight: Optional[Sequence[str]] = None, pct: bool = False,
         height: int = 420) -> go.Figure:
    if frame is None or frame.empty:
        return empty(title)
    fig = px.line(frame, title=title, render_mode='svg')
    shown = [h for h in (highlight or []) if h in frame.columns]
    if shown:
        fig.update_traces(visible="legendonly", selector=lambda t: t.name not in shown)
    return style(fig, pct, height)


def bars(frame: pd.DataFrame, title: str, pct: bool = False, barmode: str = 'group') -> go.Figure:
    if frame is None or frame.empty:
        return empty(title)
    return style(px.bar(frame, title=title, barmode=barmode), pct)


def heatmap(frame: pd.DataFrame, title: str, decimals: int = 2) -> go.Figure:
    fig = px.imshow(frame.round(decimals), title=title, color_continuous_scale='blues', text_auto=True, aspect="auto")
    fig.update_traces(xgap=2, ygap=2, textfont=dict(family=FONT, size=12))
    return style(fig, height=max(420, 28 * len(frame)))


def frontier(result: dict) -> go.Figure:
    """Efficient frontier (and constrained frontier) with each allocation of the grid"""
    fig = go.Figure()
    curve = result['frontier']
    fig.add_scatter(x=curve['Volatility'], y=curve['Returns'], mode='markers', name='Efficient Frontier',
                    marker=dict(color=curve['Sharpe Ratio'], colorscale='Blues', size=6),
                    customdata=curve[['Sharpe Ratio']],
                    hovertemplate="Vol %{x:.2%}<br>Return %{y:.2%}<br>Sharpe %{customdata[0]:.2f}<extra></extra>")
    constrained = result.get('constrained')
    if constrained is not None and not constrained.empty:
        fig.add_scatter(x=constrained['Volatility'], y=constrained['Returns'], mode='markers',
                        name='Constrained Frontier', marker=dict(color='indianred', size=4, opacity=0.7))
    indicators = result['indicators']
    for name in indicators.columns:
        highlight = name == 'Optimal Portfolio'
        fig.add_scatter(x=[indicators.loc['Volatility', name]], y=[indicators.loc['Returns', name]], mode='markers',
                        name=name, showlegend=False,
                        marker=dict(color='red' if highlight else 'orange', size=10 if highlight else 8, symbol='x'),
                        hovertemplate=f"{name}<br>Vol %{{x:.2%}}<br>Return %{{y:.2%}}<extra></extra>")
    fig.update_layout(title="Efficient Frontier", xaxis_title="Volatility", yaxis_title="Expected Return",
                      xaxis_tickformat=".2%")
    return style(fig, pct=True, height=480)


def histogram(counts: List[int], edges: List[float], var: float, title: str) -> go.Figure:
    centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
    fig = go.Figure(go.Bar(x=centers, y=counts, marker_color='steelblue', name='Simulated returns'))
    fig.add_vline(x=var, line_color='red', annotation_text='VaR')
    fig.update_layout(title=title, xaxis_tickformat=".2%", bargap=0.02)
    return style(fig)


# =============================================================================
# Notebook layouts: (left column, right column) of each view, with the Crypto App
# notebook's titles, pairing and series shown by default. Every front-end draws
# the two columns side by side.
# =============================================================================

def _in(names, wanted):
    return [n for n in wanted if n in names]


def universe_layout(prices: pd.DataFrame, levels: pd.DataFrame, benchmark: Optional[str]):
    shown = [benchmark] if benchmark else None
    return [line(prices, "Price", shown)], [line(levels, "Cumulative Performance", shown)]


def performance_layout(view: dict, frontier_result: Optional[dict], highlight: Sequence[str]):
    """Strategy: Performance | Drawdown, Portfolio Rolling Volatility | Efficient Frontier"""
    shown = _in(view['levels'].columns, highlight)
    left = [line(view['levels'], "Performance", shown),
            line(view['rolling_volatility'].dropna(how='all'), "Portfolio Rolling Volatility", shown, pct=True)]
    right = [line(view['drawdown'], "Drawdown", shown, pct=True),
             frontier(frontier_result) if frontier_result else empty("Efficient Frontier")]
    return left, right


def calendar_layout(view: dict, freq: str):
    """Strategy Return: Returns, Tracking Error | Volatility, Sharpe Ratio"""
    left = [bars(view['returns'], f"Returns by {freq}", pct=True),
            bars(view['tracking_error'], f"Tracking Error by {freq}", pct=True)]
    right = [bars(view['volatility'], f"Volatility by {freq}", pct=True),
             bars(view['sharpe_ratio'], f"Sharpe Ratio by {freq}")]
    return left, right


def return_analysis_layout(view: dict, fund: str):
    """Return Analysis: Return / Benchmark Return Contribution | Drawdown, Active Return Contribution"""
    total = ['Total Return']
    left = [line(view['contribution'], "Return Contribution", total, pct=True),
            line(view['benchmark_contribution'], "Benchmark Return Contribution", total, pct=True)]
    right = [line(view['drawdown'], "Drawdown", [fund], pct=True),
             line(view['active_contribution'], "Active Return Contribution", total, pct=True)]
    return left, right


def beta_layout(view: dict, bench: str, highlight: Sequence[str], benchmark_asset: Optional[str]):
    """Beta Analysis: Rolling Asset Beta, Historical Weights | Portfolio Rolling Beta, Beta Contribution"""
    left = [line(view['asset_beta'], "Rolling Asset Beta", [benchmark_asset] if benchmark_asset else None),
            line(view['weights'], "Historical Weights", pct=True)]
    right = [line(view['portfolio_beta'], f"Portfolio Rolling Beta vs {bench}",
                  _in(view['portfolio_beta'].columns, highlight)),
             line(view['beta_contribution'], "Beta Contribution", ['Total Beta'])]
    return left, right


def trajectory_layout(series: pd.DataFrame, decomposition: dict, highlight: Sequence[str],
                      tracking_error: bool = False):
    """Risk Trajectory / Tracking Error: Ex Ante X, Idiosyncratic | X Contribution, Correlation"""
    label = "Tracking Error" if tracking_error else "Volatility"
    left = [line(series, f"Ex Ante {label}", _in(series.columns, highlight), pct=True),
            line(decomposition['idiosyncratic_contribution'], "Idiosyncratic Contribution",
                 ['Total Idiosyncratic Vol'], pct=True)]
    right = [line(decomposition['vol_contribution'], f"{label} Contribution", ['Total Vol'], pct=True),
             line(decomposition['correlation_contribution'], "Correlation Contribution", ['Total Correlation'],
                  pct=True)]
    return left, right


def var_history_layout(view: dict, name: str, highlight: Sequence[str]):
    """Value at Risk History: Portfolios VaR, VaR History | Portfolio ES, ES History"""
    contributions = view['contributions'].get(name, {})
    left = [line(view['var'], "Portfolios Value At Risk", _in(view['var'].columns, highlight), pct=True),
            line(contributions.get('var'), "Value at Risk History", ['Portfolio'], pct=True)]
    right = [line(view['cvar'], "Portfolio Expected Shortfall", _in(view['cvar'].columns, highlight), pct=True),
             line(contributions.get('cvar'), "Expected Shortfall History", ['Portfolio'], pct=True)]
    return left, right


def pca_layout(view: dict):
    """Market Risk: Variance Explanation, Eigen Weights | Eigen Index, PCA Similarity"""
    left = [bars(view['variance_explained'].to_frame('Variance Explained'), "Variance Explanation in %", pct=True),
            bars(view['eigen_weights'].to_frame('Eigen Weights'), "Eigen Weights")]
    right = [line(view['eigen_index'], "Eigen Index"),
             line(view['similarity'], "PCA Similarity")]
    return left, right


def correlation_layout(view: dict, asset1: str, asset2: str):
    """Correlation: pair correlation, Mean Return | First principal component, Correlation Matrix"""
    left = [line(view['rolling_correlation'], f"{asset1}/{asset2} Correlation"),
            line(view['rolling_mean_returns'], "Mean Return", [asset1, asset2], pct=True)]
    right = [line(view['first_component'], "First principal component (Variance Explained in %)", pct=True),
             heatmap(view['matrix'], "Correlation Matrix")]
    return left, right


def market_drivers_layout(view: dict, highlight: Sequence[str], benchmark_asset: Optional[str]):
    """Market Drivers: Performance Comparison, Market Drivers, Market Correlation |
    Market Volatility, Market Intrinsic Volatility, Market Weights"""
    perf = view['performance']
    left = [line(perf, "Performance Comparison", _in(perf.columns, ['Market Index', *highlight])),
            line(view['drivers'], "Market Drivers", ['Market Index']),
            line(view['correlation_contribution'], "Market Correlation", ['Total Correlation'], pct=True)]
    right = [line(view['vol_contribution'], "Market Volatility", ['Total Vol'], pct=True),
             line(view['idiosyncratic_contribution'], "Market Intrinsic Volatility", ['Total Idiosyncratic Vol'],
                  pct=True),
             line(view['weights'], "Market Weights", [benchmark_asset] if benchmark_asset else None, pct=True)]
    return left, right
