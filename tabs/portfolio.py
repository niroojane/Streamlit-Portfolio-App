# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Portfolio Analysis tab: the notebooks' Return Analysis and Beta Analysis
(Current Portfolio tab), on any series of the workspace instead of the account's
P&L -- the P&L computation itself is not part of these apps. Same layout as the
notebook (tabs/charts.py)."""
import streamlit as st

from . import charts, ui


def _pick(ctx: ui.Context, names, key: str):
    col1, col2 = st.columns(2)
    default_fund = 'Fund' if 'Fund' in names else names[0]
    bench_label = ctx.profile.benchmark_label
    default_bench = bench_label if bench_label in names and bench_label != default_fund else \
        next(n for n in names if n != default_fund)
    fund = col1.selectbox("Fund", names, index=names.index(default_fund), key=f'{key}_fund')
    bench = col2.selectbox("Benchmark", names, index=names.index(default_bench), key=f'{key}_bench')
    return fund, bench


def render(ctx: ui.Context) -> None:
    if not ui.require_grid(ctx):
        return
    ws, key = ui.get_workspace(ctx)
    names = ws.names
    if len(names) < 2:
        st.info("Need at least two series")
        return
    ui.date_slider(ctx, 'portfolio_tab')
    sub = st.tabs(["Return Analysis", "Beta Analysis"])

    with sub[0]:
        fund, bench = _pick(ctx, names, 'return_analysis')
        if fund == bench:
            st.info("Benchmark and Fund must be different")
        else:
            view = ui.run("Return contribution...", ui.cached_view, 'return_analysis_view', key, ws, ctx.start,
                          ctx.end, fund, bench)
            if view is not None:
                ui.pair(charts.return_analysis_layout(view, fund))

    with sub[1]:
        fund, bench = _pick(ctx, names, 'beta_analysis')
        view = ui.run("Rolling betas...", ui.cached_view, 'beta_view', key, ws, ctx.start, ctx.end, fund, bench,
                      ctx.window)
        if view is not None:
            ui.pair(charts.beta_layout(view, bench, ['Fund', fund], ctx.profile.benchmark_ticker))
