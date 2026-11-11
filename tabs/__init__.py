# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Tabs of the Streamlit apps, mirroring the Risk Management notebooks.

    engine.py     UI-agnostic glue to src/ (also used by Gradio.py)
    ui.py         Streamlit helpers: context, caching, editors, charts
    data.py       data loaders (Excel workbook, Binance universe)
    universe.py   Investment Universe
    strategy.py   Strategy (Strategy | Strategy Return)
    portfolio.py  Portfolio Analysis (Return Analysis | Beta Analysis)
    risk.py       Risk Analysis (Volatility Analysis | Value at Risk)
    market.py     Market Risk (Market Risk | Correlation | Market Drivers)

An app is a loader plus a list of tabs:

    from tabs import run_app, data
    run_app("Portfolio Optimization", data.excel_loader)
"""
from typing import Callable, Sequence

FULL = ("Investment Universe", "Strategy", "Portfolio Analysis", "Risk Analysis", "Market Risk")
LIGHT = ("Investment Universe", "Strategy", "Risk Analysis")


def run_app(title: str, loader: Callable[[], bool], tabs: Sequence[str] = FULL, light: bool = False) -> None:
    """Draw the app: `loader` fills the universe (top of the first tab), the
    sidebar sets the analysis window, then each tab renders in order (the
    Strategy tab's allocation grid feeds the later tabs).

    light: no rolling backtest, risk tabs limited to Risk Contribution / Value at Risk.
    """
    import streamlit as st
    from . import market, portfolio, risk, strategy, ui, universe

    ui.setup_page(title)
    containers = dict(zip(tabs, st.tabs(list(tabs))))

    with containers[tabs[0]]:
        loaded = loader()
    data = ui.universe() if loaded else None
    if data is None:
        for name in tabs[1:]:
            with containers[name]:
                st.info("Load data in the Investment Universe tab first ⬅️")
        return

    prices, profile, exposure = data
    ctx = ui.Context(prices=prices, profile=profile, exposure=exposure, **ui.settings(prices))

    renderers = {
        "Investment Universe": lambda: universe.render(ctx),
        "Strategy": lambda: strategy.render(ctx, backtest=not light),
        "Portfolio Analysis": lambda: portfolio.render(ctx),
        "Risk Analysis": lambda: risk.render(ctx, full=not light),
        "Market Risk": lambda: market.render(ctx),
    }
    for name in tabs:
        with containers[name]:
            renderers[name]()
