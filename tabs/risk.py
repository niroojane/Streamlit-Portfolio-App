# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Risk Analysis tab, laid out as the Crypto App notebook's:

Volatility Analysis  Risk Contribution | Risk Trajectory | Tracking Error
Value at Risk        Value at Risk (stress grids + simulations) | Value at Risk History

Chart pairs come from tabs/charts.py (same layout in every front-end).
"""
import numpy as np
import streamlit as st

from . import charts, engine, ui


def render(ctx: ui.Context, full: bool = True) -> None:
    if not ui.require_grid(ctx):
        return
    ui.date_slider(ctx, 'risk_tab')
    groups = st.tabs(["Volatility Analysis", "Value at Risk"])
    with groups[0]:
        names = ["Risk Contribution"] + (["Risk Trajectory", "Tracking Error"] if full else [])
        sub = st.tabs(names)
        with sub[0]:
            _risk_contribution(ctx)
        if full:
            ws, key = ui.get_workspace(ctx)
            with sub[1]:
                _risk_trajectory(ctx, ws, key)
            with sub[2]:
                _tracking_error(ctx, ws, key)
    with groups[1]:
        names = ["Value at Risk"] + (["Value at Risk History"] if full else [])
        sub = st.tabs(names)
        with sub[0]:
            _value_at_risk(ctx)
        if full:
            with sub[1]:
                _var_history(ctx, *ui.get_workspace(ctx))


def _default(names, preferred):
    return names.index(preferred) if preferred in names else 0


# =============================================================================
# Volatility Analysis
# =============================================================================

def _risk_contribution(ctx: ui.Context) -> None:
    names = list(ctx.grid.index)
    fund = st.selectbox("Fund", names, index=_default(names, 'Model'), key='risk_contribution_fund')
    bench = st.selectbox("Benchmark", names, index=_default(names, ctx.benchmark or names[-1]),
                         key='risk_contribution_bench')
    view = ui.run("Risk contribution...", ui.cached, 'src.Workspace.risk_contribution_view', ctx.window_prices,
                  ctx.grid.reindex(columns=ctx.prices.columns).fillna(0.0), fund, bench, ctx.frequency)
    if view is None:
        return
    ui.table(view['contribution'], "Performance and Risk Contribution")
    ui.table(view['excess'], "Excess P&L and TE Contribution")
    ui.table(view['ex_ante'], "Ex Ante Metrics")


def _risk_trajectory(ctx: ui.Context, ws, key: str) -> None:
    names = ws.names
    name = st.selectbox("Fund", names, index=_default(names, 'Fund'), key='trajectory_series')
    params = (key, ctx.start, ctx.end, ctx.window)
    if not ui.on_click("Get Risk History", 'risk_trajectory', params,
                       help="Rolling ex-ante volatility of every series and its decomposition"):
        return
    view = ui.run("Rolling ex-ante volatility...", ui.cached_view, 'risk_trajectory_view', key, ws, ctx.start,
                  ctx.end, ctx.window)
    if view is None or view['volatility'].empty:
        return
    name = name if name in view['decomposition'] else next(iter(view['decomposition']))
    ui.pair(charts.trajectory_layout(view['volatility'], view['decomposition'][name], ['Fund', name]))


def _tracking_error(ctx: ui.Context, ws, key: str) -> None:
    names = ws.names
    name = st.selectbox("Fund", names, index=_default(names, 'Fund'), key='te_series')
    bench = st.selectbox("Benchmark", names, index=_default(names, ctx.profile.benchmark_label), key='te_benchmark')
    params = (key, ctx.start, ctx.end, ctx.window, bench)
    if not ui.on_click("Get Tracking Error History", 'tracking_error', params):
        return
    view = ui.run("Rolling ex-ante tracking error...", ui.cached_view, 'tracking_error_view', key, ws, ctx.start,
                  ctx.end, bench, ctx.window)
    if view is None or view['tracking_error'].empty:
        return
    options = [n for n in view['decomposition'] if n != bench]
    if not options:
        return
    name = name if name in options else options[0]
    ui.pair(charts.trajectory_layout(view['tracking_error'].drop(columns=bench, errors='ignore'),
                                     view['decomposition'][name], ['Fund', name], tracking_error=True))


# =============================================================================
# Value at Risk
# =============================================================================

def _stress_inputs(ctx: ui.Context, vol_factor: float, mean_factor: float):
    """Correlation shock grid and returns shock table (editable, kept across reruns)"""
    assets = list(ctx.prices.columns)
    stale = 'stress_grid' not in st.session_state or list(st.session_state.stress_grid.columns) != assets
    if stale or st.session_state.pop('reset_stress', False):
        mean, grid = engine.default_stress(assets, vol_factor, mean_factor)
        st.session_state.mean_shock, st.session_state.stress_grid = mean, grid
        st.session_state.stress_version = st.session_state.get('stress_version', 0) + 1
    version = st.session_state.get('stress_version', 0)

    st.subheader("Correlation Shock")
    st.caption("Diagonal: volatility multipliers. Off-diagonal: correlation shocks (mirrored).")
    edited = st.data_editor(st.session_state.stress_grid, key=f"stress_grid_{version}", width='stretch')
    st.subheader("Returns Shock")
    mean = st.data_editor(st.session_state.mean_shock, key=f"mean_shock_{version}")
    mirrored = engine.symmetric(edited.astype(float))
    if not np.allclose(mirrored.to_numpy(), edited.to_numpy(dtype=float)):
        st.session_state.mean_shock, st.session_state.stress_grid = mean, mirrored
        st.session_state.stress_version = version + 1
        st.rerun()
    return mean['Mean Shock'].astype(float), mirrored


def _value_at_risk(ctx: ui.Context) -> None:
    names = list(ctx.grid.index)
    fund = st.selectbox("Fund", names, index=_default(names, 'Model'), key='var_fund')
    vol_factor = st.number_input("Stress Factor", min_value=0.1, value=1.0, step=0.1)
    mean_factor = st.number_input("Mean Shock Factor", value=1.0, step=0.1)
    iterations = st.number_input("Iterations", min_value=500, value=10000, step=500)
    scenarios = st.number_input("Scenarios", min_value=1, value=5, step=1)
    centile = st.number_input("Centile", min_value=0.001, max_value=0.5, value=0.05, step=0.01, format="%.3f")
    with st.expander("More settings", expanded=False):
        horizon = st.number_input("Horizon (days)", min_value=1, value=1, step=1)
        backtest_window = st.number_input("VaR backtest window (0 = none)", min_value=0, value=0, step=10)

    mean_shock, grid = _stress_inputs(ctx, vol_factor, mean_factor)
    preview = ui.run("Stressed moments...", ui.cached, 'src.Workspace.stress_preview', ctx.window_returns,
                     grid.to_numpy(), mean_shock)
    if preview is not None:
        ui.table(preview['moments'], "Mean and Vol Changes")
        st.subheader("Correlation Changes")
        col1, col2 = ui.two_columns()
        with col1:
            ui.table(preview['correlation'], "Original Correlation")
        with col2:
            ui.table(preview['stressed_correlation'], "New Correlation")

    col1, col2, _ = st.columns([1, 1, 6])
    with col2:
        if st.button("Reset Shocks", help="Diagonal = stress factor, no correlation shock, mean = mean shock factor"):
            st.session_state.reset_stress = True
            st.rerun()
    with col1:
        params = (ctx.window_prices, ctx.grid, centile, iterations, scenarios, horizon, grid, mean_shock,
                  backtest_window)
        run = ui.on_click("Run Simulation", 'value_at_risk', params,
                          help="Historical, parametric and simulated VaR / CVaR of every allocation")
    if not run:
        return
    allocations = ctx.grid.reindex(columns=ctx.prices.columns).fillna(0.0)
    results = ui.run("Simulating...", ui.cached, 'src.PortfolioAnalytics.value_at_risk', ctx.window_prices,
                     ctx.window_returns, allocations, var_centile=float(centile), iterations=int(iterations),
                     num_scenarios=int(scenarios), stress_factor=grid.to_numpy(), mean_shock=mean_shock,
                     horizon_days=int(horizon), backtest_window=int(backtest_window) or None)
    if results is None:
        return
    fund = fund if fund in results else next(iter(results))
    tables = engine.var_tables(results, fund, float(centile))
    ui.table(tables['summary'], "Summary")
    ui.table(tables.get('var_contribution'), f"VaR Results for {fund}")
    ui.table(tables.get('cvar_contribution'), f"CVaR Results for {fund}")
    if 'backtest' in tables:
        ui.line(tables['backtest'], f"Rolling VaR Backtest · {fund}", pct=True)
        ui.table(tables['kupiec'], "Kupiec Test")


def _var_history(ctx: ui.Context, ws, key: str) -> None:
    names = ws.names
    name = st.selectbox("Fund", names, index=_default(names, 'Fund'), key='var_history_series')
    method = st.selectbox("Distribution", list(engine.VAR_METHODS), index=3, key='var_history_method')
    centile = st.number_input("Centile", min_value=0.001, max_value=0.5, value=0.05, step=0.01, format="%.3f",
                              key='var_history_centile')
    with st.expander("More settings", expanded=False):
        iterations = st.number_input("Iterations", min_value=200, value=1000, step=100, key='var_history_iterations')
        vol_factor = st.number_input("Stress factor", min_value=0.1, value=1.0, step=0.1, key='var_history_stress')
    params = (key, ctx.start, ctx.end, method, ctx.window, centile, iterations, vol_factor)
    if not ui.on_click("Get Value At Risk History", 'var_history', params,
                       help="Rolling simulated VaR / CVaR of every series (one simulation per day: can be slow)"):
        return
    view = ui.run("Rolling Value at Risk...", ui.cached_view, 'var_history_view', key, ws, ctx.start, ctx.end,
                  engine.VAR_METHODS[method], ctx.window, float(centile), int(iterations), float(vol_factor), 1.0)
    if view is None:
        return
    ui.show_messages(view['warnings'])
    ui.pair(charts.var_history_layout(view, name, ['Fund', name]))
