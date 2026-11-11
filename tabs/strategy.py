# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Strategy tab, laid out as Risk Management/Streamlit_App.py (the Streamlit port of the
Crypto App notebook): Date slider, then the Strategy / Strategy Return sub-tabs.

Strategy:        asset returns / risk -> Constraints (+ risk limits), 'Optimize Portfolio' ->
                 Portfolio Construction (allocation grid) -> Core Strategy (strategy, benchmark,
                 rebalancing frequency, sliding window) -> Overlay, 'Get Results' ->
                 Strategy | Core | Overlay matrices, Allocation Table, Expected Returns,
                 Systematic Fund Metrics, Backtested Metrics, Charts, Time Series.
Strategy Return: calendar returns (Returns, Tracking Error | Volatility, Sharpe Ratio).
"""
import pandas as pd
import streamlit as st

from src import Exposure
from . import charts, engine, ui


def _base_grid(ctx: ui.Context) -> pd.DataFrame:
    base = st.session_state.get('grid_base')
    return base.reindex(columns=ctx.prices.columns).fillna(0.0) if base is not None else pd.DataFrame()


def _set_grid(grid: pd.DataFrame) -> None:
    st.session_state.grid_base = grid
    st.session_state.grid_version = st.session_state.get('grid_version', 0) + 1


def render(ctx: ui.Context, backtest: bool = True) -> None:
    """Draw the tab and fill ctx.grid / ctx.benchmark / ctx.constraints / ctx.risk (used by the later tabs)"""
    ui.date_slider(ctx, 'strategy_tab')
    sub = st.tabs(["Strategy", "Strategy Return"])
    with sub[0]:
        _strategy(ctx, backtest)
    with sub[1]:
        _strategy_return(ctx)


# =============================================================================
# Strategy
# =============================================================================

def _strategy(ctx: ui.Context, backtest: bool) -> None:
    """Order of Risk Management/Streamlit_App.py: asset returns / risk, Constraints, Portfolio
    Construction, Core Strategy, Overlay, Get Results, then the results"""
    ui.table(ui.cached('src.Metrics.get_asset_returns', ctx.window_prices))
    ui.table(ui.cached('src.Metrics.get_asset_risk', ctx.window_prices))

    # Constraints (weights / buckets and risk limits) and Optimize Portfolio
    st.subheader("Constraints")
    col1, col2 = st.columns(2)
    with col1:
        ctx.constraints = ui.constraint_editor(list(ctx.prices.columns), ctx.exposure)
    with col2:
        ctx.risk = ui.risk_editor()
    optimize = st.button("Optimize Portfolio", type="primary",
                         help=f"Every objective on {ctx.start.date()} → {ctx.end.date()}, with and without constraints")
    ui.show_messages(st.session_state.get('optimize_messages', []))

    # Portfolio Construction: the allocation grid
    st.subheader("Portfolio Construction")
    base = _base_grid(ctx)
    ctx.grid = _grid_editor(ctx, base) if not base.empty else base
    if base.empty:
        st.info("Click 'Optimize Portfolio' to build the allocation grid")
    if ctx.exposure is not None and not ctx.grid.empty:
        ui.table(Exposure.lookthrough(ctx.exposure, ctx.grid), "Look-through")

    # Core Strategy
    st.subheader("Core Strategy")
    strategy = st.selectbox("Strategy:", list(engine.STRATEGIES), key='core_strategy') if backtest else None
    benchmark = st.selectbox("Benchmark:", ['None'] + list(base.index), key='benchmark_allocation',
                             help="Benchmark allocation: tracking errors, Min Tracking Error / Max Information Ratio, "
                                  "beta / TE / IR constraints")
    ctx.benchmark = benchmark if benchmark in ctx.grid.index else None
    ctx.frequency = st.selectbox("Rebalancing Frequency:", engine.FREQUENCIES, key='frequency')
    ctx.window = int(st.number_input("Sliding Window Size:", min_value=20, value=ctx.window, step=1,
                                     key='rolling_window'))

    # Overlay and Get Results
    overlays, results = None, False
    if backtest:
        st.subheader("Overlay")
        overlays = ui.overlay_editor()
        results = st.button("Get Results", type="primary", disabled=ctx.grid.empty,
                            help="Re-optimize the strategy on each rebalancing period of the full history, blend the "
                                 "overlays, simulate Fund / Core / Overlay")

    # Clicks are handled once every input is drawn (the benchmark sits below the grid)
    if optimize:
        result = ui.run("Optimizing...", engine.optimize_grid, ctx.window_prices, ctx.constraints, ctx.risk,
                        ctx.exposure, ctx.bench_weights, ctx.frequency)
        if result is not None:
            grid, messages = result
            st.session_state.grid_optimized = grid
            st.session_state.optimize_messages = messages
            _set_grid(engine.with_model(grid, ctx.backtest['model']) if ctx.backtest is not None else grid)
            st.rerun()
    if results:
        result = ui.run("Running the rolling optimization...", engine.backtest, ctx.prices, strategy, ctx.profile,
                        ctx.frequency, overlays, ctx.constraints, ctx.risk, ctx.exposure, ctx.bench_weights)
        if result is not None:
            st.session_state.backtest = result
            _set_grid(engine.with_model(ctx.grid, result['model']))
            st.rerun()

    if not ctx.grid.empty:
        _results(ctx, backtest)


def _grid_editor(ctx: ui.Context, base: pd.DataFrame) -> pd.DataFrame:
    version = st.session_state.get('grid_version', 0)
    config = {c: st.column_config.NumberColumn(format="%.4f") for c in base.columns}
    edited = st.data_editor(base, key=f"grid_{version}", column_config=config, width='stretch')
    sums = edited.sum(axis=1)
    off = sums[(sums - 1).abs() > 1e-3]
    if len(off):
        st.caption("⚠️ Weights not summing to 100%: " + ", ".join(f"{k} ({v:.1%})" for k, v in off.items()))

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        name = st.text_input("New allocation", placeholder="Name", label_visibility="collapsed")
        if st.button("Add Allocation") and name:
            if name in edited.index:
                st.warning(f"'{name}' already exists")
            else:
                _set_grid(pd.concat([edited, pd.DataFrame(0.0, index=[name], columns=edited.columns)]))
                st.rerun()
    with col2:
        removed = st.multiselect("Remove", list(edited.index), label_visibility="collapsed",
                                 placeholder="Allocations to remove")
        if st.button("Remove Allocations") and removed:
            _set_grid(edited.drop(index=removed))
            st.rerun()
    with col3:
        if st.button("Clear Allocation", help="Back to the last optimized grid") \
                and 'grid_optimized' in st.session_state:
            grid = st.session_state.grid_optimized
            if ctx.backtest is not None:
                grid = engine.with_model(grid, ctx.backtest['model'])
            _set_grid(grid)
            st.rerun()
    return edited


def _highlight(ctx: ui.Context, columns) -> list:
    wanted = ctx.profile.highlight if ctx.backtest is not None else \
        ['Rebalanced Optimal Portfolio', ctx.profile.benchmark_label]
    return [c for c in wanted if c in columns] or list(columns[:2])


def _performance_view(ctx: ui.Context):
    ws, key = ui.get_workspace(ctx)
    view = ui.run("Computing performance...", ui.cached_view, 'performance_view', key, ws, ctx.start, ctx.end,
                  ctx.grid.reindex(columns=ctx.prices.columns).fillna(0.0), ctx.benchmark or '', ctx.window,
                  None, 0)
    return ws, key, view


def _results(ctx: ui.Context, backtest: bool) -> None:
    """Strategy | Core | Overlay matrices, Allocation Table, Expected Returns, Systematic Fund Metrics,
    Backtested Metrics, Charts, Time Series (Risk Management/Streamlit_App.py)"""
    result = ctx.backtest if backtest else None
    if backtest and result is None:
        st.info("Click 'Get Results' to run the rolling backtest")
    if result is not None:
        ui.show_messages(result['messages'])
        schedules = engine.strategy_schedules(result)
        labels = [('Fund', 'Strategy Matrix'), ('Core', 'Core Matrix'), ('Overlay', 'Overlay Matrix')]
        for column, (key, label) in zip(st.columns(3), labels):
            with column:
                ui.table(schedules[key].round(4) if key in schedules else None, label, height=320)

    _, _, view = _performance_view(ctx)
    if view is None:
        return
    frontier = ui.run("Efficient frontier...", ui.cached, 'tabs.engine.frontier', ctx.window_prices, ctx.grid,
                      ctx.constraints, ctx.risk, ctx.exposure, ctx.bench_weights, ctx.frequency)
    ui.table(ctx.grid, "Allocation Table")
    if frontier is not None:
        ui.table(frontier['indicators'], "Expected Returns")
        ui.show_messages(frontier['messages'])
    if result is not None:
        ui.table(result['metrics'], "Systematic Fund Metrics")
    st.subheader("Backtested Metrics")
    ui.table(view['returns_table'])
    ui.table(view['risk_table'])
    st.subheader("Charts")
    ui.pair(charts.performance_layout(view, frontier, _highlight(ctx, list(view['levels'].columns))))
    if result is not None and ctx.exposure is not None:
        ui.line(Exposure.exposure_history(ctx.exposure, engine.strategy_schedules(result)['Fund']),
                "Fund Exposure at each Rebalancing", pct=True)
    ui.table(view['levels'], "Time Series", decimals=2)


# =============================================================================
# Strategy Return
# =============================================================================

def _strategy_return(ctx: ui.Context) -> None:
    """Calendar returns of a fund vs a benchmark series (notebook's calendar_perf)"""
    if not ui.require_grid(ctx):
        return
    ws, key, view = _performance_view(ctx)
    if view is None:
        return
    names = list(view['levels'].columns)
    highlight = _highlight(ctx, names)
    col1, col2, col3 = st.columns(3)
    freq = col1.selectbox("Frequency:", ['Year', 'Month'], key='calendar_freq')
    fund = col2.selectbox("Fund:", names, index=names.index(highlight[0]), key='calendar_fund')
    bench_default = highlight[1] if len(highlight) > 1 else names[min(1, len(names) - 1)]
    bench = col3.selectbox("Benchmark:", names, index=names.index(bench_default), key='calendar_bench')
    if fund == bench:
        st.info("Benchmark and Fund must be different")
        return
    calendar = ui.run("Calendar metrics...", ui.cached_view, 'calendar_view', key, ws, ctx.start, ctx.end, fund,
                      bench, freq)
    if calendar is not None:
        ui.pair(charts.calendar_layout(calendar, freq))
