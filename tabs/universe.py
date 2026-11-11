# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Investment Universe tab: asset returns / risk over the analysis window, prices,
cumulative performance and the exposure table."""
import streamlit as st

from src.Metrics import get_asset_returns, get_asset_risk
from src.Workspace import levels
from src import Exposure
from . import charts, ui


def render(ctx: ui.Context) -> None:
    ui.date_slider(ctx, 'investment_tab')
    prices = ctx.window_prices

    ui.table(ui.cached('src.Metrics.get_asset_returns', prices), "Asset Returns")
    ui.table(ui.cached('src.Metrics.get_asset_risk', prices), "Asset Risk")

    ui.pair(charts.universe_layout(prices, levels(ctx.window_returns.reindex(prices.index).fillna(0)),
                                   ctx.profile.benchmark_ticker))

    if ctx.exposure is not None:
        ui.table(ctx.exposure, "Transparency Table")
        ui.table(Exposure.bounds(ctx.exposure), "Transparency Bounds")
    ui.table(prices, "Prices", decimals=4, height=400)
