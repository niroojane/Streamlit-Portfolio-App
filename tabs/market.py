# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Market Risk tab, laid out as the Crypto App notebook's:
Market Risk (PCA) | Correlation | Market Drivers (chart pairs from tabs/charts.py)"""
import streamlit as st

from . import charts, ui


def render(ctx: ui.Context) -> None:
    ui.date_slider(ctx, 'market_tab')
    sub = st.tabs(["Market Risk", "Correlation", "Market Drivers"])
    with sub[0]:
        _pca(ctx)
    with sub[1]:
        _correlation(ctx)
    with sub[2]:
        if ui.require_grid(ctx):
            _market_drivers(ctx, *ui.get_workspace(ctx))


def _pca(ctx: ui.Context) -> None:
    returns = ctx.window_returns
    n_assets = returns.shape[1]
    components = st.number_input("Number of Components", min_value=1, max_value=n_assets, value=min(5, n_assets),
                                 step=1)
    component = st.selectbox("Component", [f"PC{i + 1}" for i in range(int(components))], key='pca_component')
    closest = st.number_input("Closest to PCA", min_value=1, max_value=n_assets, value=min(5, n_assets), step=1)
    view = ui.run("PCA...", ui.cached, 'src.Workspace.pca_view', returns, int(components), component, int(closest))
    if view is not None:
        ui.pair(charts.pca_layout(view))


def _correlation(ctx: ui.Context) -> None:
    assets = list(ctx.prices.columns)
    if len(assets) < 2:
        st.info("Need at least two assets")
        return
    asset1 = st.selectbox("Asset 1", assets, index=0, key='corr_asset1')
    asset2 = st.selectbox("Asset 2", assets, index=1, key='corr_asset2')
    if asset1 == asset2:
        st.info("Select two different assets")
        return
    returns = ctx.prices.pct_change(fill_method=None).iloc[1:].loc[:ctx.end]
    view = ui.run("Correlation...", ui.cached, 'src.Workspace.correlation_view', returns, asset1, asset2,
                  ctx.window, ctx.start, ctx.end)
    if view is not None:
        ui.pair(charts.correlation_layout(view, asset1, asset2))


def _market_drivers(ctx: ui.Context, ws, key: str) -> None:
    frequency = st.selectbox("Frequency", ['Monthly', 'Quarterly', 'Yearly'], index=1, key='drivers_frequency')
    component = st.selectbox("Component", [f"PC{i + 1}" for i in range(min(5, ctx.prices.shape[1]))],
                             key='drivers_component')
    params = (key, ctx.start, ctx.end, frequency, component, ctx.window)
    if not ui.on_click("Get Market Drivers", 'market_drivers', params,
                       help="Eigen-portfolio market index re-estimated each period, its drivers and risk"):
        return
    view = ui.run("Market drivers...", ui.cached_view, 'market_drivers_view', key, ws, ctx.start, ctx.end,
                  frequency, component, ctx.window)
    if view is not None:
        ui.pair(charts.market_drivers_layout(view, ctx.profile.highlight, ctx.profile.benchmark_ticker))
