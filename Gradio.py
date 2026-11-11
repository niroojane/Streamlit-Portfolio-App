# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Gradio version of the portfolio app, laid out as the Crypto App notebook of
Risk Management, on the same engine as the Streamlit / BQuant tabs
(tabs/engine.py, tabs/charts.py -> src/):

    Investment Universe
    Strategy             Strategy | Strategy Return
    Portfolio Analysis   Return Analysis | Beta Analysis
    Risk Analysis        Volatility Analysis (Risk Contribution | Risk Trajectory | Tracking Error)
                         Value at Risk (Value at Risk | Value at Risk History)
    Market Risk          Market Risk | Correlation | Market Drivers

The allocation grid is an editable table: every button reads its current content.

    python Gradio.py
"""
from typing import List, Optional

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from src import Exposure
from src import PortfolioAnalytics as analytics
from src import Workspace as workspace
from src.Metrics import get_asset_returns, get_asset_risk
from tabs import charts, engine

CONSTRAINT_HEADERS = ['Target', 'Sign', 'Limit']
RISK_HEADERS = ['Risk', 'Sign', 'Limit']
OVERLAY_HEADERS = ['Strategy', 'Limit']


# =============================================================================
# Helpers
# =============================================================================

def show(frame, index_name: str = '', decimals: int = 4) -> pd.DataFrame:
    """A frame for gr.Dataframe: the index becomes the first column, numbers rounded (as the notebook)"""
    if frame is None or (hasattr(frame, 'empty') and frame.empty):
        return pd.DataFrame()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    frame = frame.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame.index = [' · '.join(map(str, i)) for i in frame.index]
    if isinstance(frame.index, pd.DatetimeIndex):
        frame.index = frame.index.strftime('%Y-%m-%d')
    frame.index.name = index_name or frame.index.name or ''
    frame = frame.reset_index()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].round(decimals)
        elif frame[col].dtype == object:
            frame[col] = frame[col].map(lambda v: round(v, decimals) if isinstance(v, float) else str(v))
    frame.columns = [str(c) for c in frame.columns]
    return frame


def notes(messages) -> str:
    messages = list(dict.fromkeys(messages or []))
    return "\n".join(f"- {m}" for m in messages) if messages else ""


def grid_from(table, app: dict) -> pd.DataFrame:
    """Allocation grid from the editable table (first column = allocation names)"""
    if table is None or len(table) == 0 or 'Allocation' not in table.columns:
        return pd.DataFrame()
    grid = table.set_index('Allocation').apply(pd.to_numeric, errors='coerce').fillna(0.0)
    grid.index = grid.index.astype(str)
    grid = grid[grid.index != '']
    return grid.reindex(columns=app['prices'].columns).fillna(0.0)


def window(app: dict, start: str, end: str) -> pd.DataFrame:
    try:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
    except ValueError:
        raise gr.Error("Dates must be YYYY-MM-DD")
    prices = app['prices'].loc[start:end]
    if len(prices) < 3:
        raise gr.Error("The date range is too short")
    return prices


def need(app: dict, grid: Optional[pd.DataFrame] = None) -> None:
    if not app or 'prices' not in app:
        raise gr.Error("Upload a workbook in the Investment Universe tab first")
    if grid is not None and grid.empty:
        raise gr.Error("Click 'Optimize Portfolio' in the Strategy tab first")


_WORKSPACES = {}


def get_workspace(app: dict, grid: pd.DataFrame, frequency: str):
    schedules = engine.strategy_schedules(app.get('backtest'))
    key = joblib.hash((app['prices'], grid, app['profile'], frequency, schedules))
    if key not in _WORKSPACES:
        _WORKSPACES.clear()
        _WORKSPACES[key] = engine.build_workspace(app['prices'], grid, app['profile'], frequency, schedules)
    return _WORKSPACES[key]


def bench_weights(grid: pd.DataFrame, benchmark: Optional[str]):
    return grid.loc[benchmark] if benchmark and benchmark in grid.index else None


def run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except (ValueError, np.linalg.LinAlgError, KeyError) as exc:
        raise gr.Error(str(exc))


def flat(layout) -> list:
    """charts.*_layout (left, right) -> the figures in the order of plot_columns' components"""
    left, right = layout
    return list(left) + list(right)


def highlight(app, names) -> List[str]:
    profile = app['profile']
    wanted = profile.highlight if app.get('backtest') else ['Rebalanced Optimal Portfolio', profile.benchmark_label]
    return [n for n in wanted if n in names] or list(names[:2])


def span(app, start, end):
    prices = window(app, start, end)
    return prices.index[0], prices.index[-1]


# =============================================================================
# Callbacks
# =============================================================================

def on_upload(file, benchmark):
    if file is None:
        return {}, gr.update(choices=[], value=None), "", "", ""
    raw, exposure, messages = run(engine.read_workbook, open(file, 'rb').read())
    assets = list(raw.columns)
    benchmark = benchmark if benchmark in assets else ('BTCUSDT' if 'BTCUSDT' in assets else assets[0])
    profile = engine.multi_asset(benchmark)
    prices, more = run(engine.clean_prices, raw, profile)
    exposure = Exposure.read_exposure(exposure, list(prices.columns)) if exposure is not None else None
    app = {'prices': prices, 'exposure': exposure, 'profile': profile}
    return (app, gr.update(choices=assets, value=benchmark), prices.index[0].strftime('%Y-%m-%d'),
            prices.index[-1].strftime('%Y-%m-%d'), notes(messages + more))


def on_benchmark(app, benchmark):
    if app and 'prices' in app and benchmark:
        app = {**app, 'profile': engine.multi_asset(benchmark), 'backtest': None}
    return app


def universe_view(app, start, end):
    need(app)
    prices = window(app, start, end)
    levels = workspace.levels(prices.pct_change(fill_method=None).fillna(0))
    assets = list(prices.columns)
    targets = engine.constraint_targets(assets, app['exposure'])
    if app['exposure'] is None:
        targets = targets + ["(no transparency table: assets and 'All' only)"]
    return (show(get_asset_returns(prices)), show(get_asset_risk(prices)),
            *flat(charts.universe_layout(prices, levels, app['profile'].benchmark_ticker)),
            show(prices, 'Date'), show(app['exposure'], 'Bucket'), show(Exposure.bounds(app['exposure']), 'Bucket'),
            f"Constraint targets: {', '.join(targets)}",
            gr.update(choices=assets, value=assets[0]),
            gr.update(choices=assets, value=assets[min(1, len(assets) - 1)]))


def asset_tables(app, start, end):
    need(app)
    prices = window(app, start, end)
    return show(get_asset_returns(prices)), show(get_asset_risk(prices))


def optimize(app, start, end, constraints, risk, grid_table, benchmark, frequency):
    need(app)
    prices = window(app, start, end)
    grid, messages = run(engine.optimize_grid, prices, constraints, risk, app['exposure'],
                         bench_weights(grid_from(grid_table, app), benchmark), frequency)
    if app.get('backtest'):
        grid = engine.with_model(grid, app['backtest']['model'])
    return app, show(grid, 'Allocation', 6), show(Exposure.lookthrough(app['exposure'], grid), 'Bucket'), \
        notes(messages)


def get_results(app, constraints, risk, overlays, grid_table, benchmark, frequency, strategy):
    need(app)
    grid = grid_from(grid_table, app)
    need(app, grid)
    result = run(engine.backtest, app['prices'], strategy, app['profile'], frequency, overlays, constraints, risk,
                 app['exposure'], bench_weights(grid, benchmark))
    app = {**app, 'backtest': result}
    schedules = engine.strategy_schedules(result)
    tables = [show(schedules[k].round(4), 'Date') if k in schedules else pd.DataFrame()
              for k in ('Fund', 'Core', 'Overlay')]
    return (app, show(engine.with_model(grid, result['model']), 'Allocation', 6), notes(result['messages']),
            *tables, show(result['metrics']))


def performance(app, start, end, grid_table, benchmark, frequency, vol_window, constraints, risk):
    grid = grid_from(grid_table, app)
    need(app, grid)
    prices = window(app, start, end)
    ws = get_workspace(app, grid, frequency)
    view = run(workspace.performance_view, ws, prices.index[0], prices.index[-1], grid,
               benchmark if benchmark in grid.index else '', int(vol_window), None, 0)
    frontier = run(engine.frontier, prices, grid, constraints, risk, app['exposure'], bench_weights(grid, benchmark),
                   frequency)
    names = list(view['levels'].columns)
    return (show(view['returns_table']), show(view['risk_table']), show(frontier['indicators']),
            notes(frontier['messages']),
            *flat(charts.performance_layout(view, frontier, highlight(app, names))))


def choices(app, grid_table, frequency):
    """Refresh every Fund / Benchmark dropdown after the grid or the backtest changed"""
    grid = grid_from(grid_table, app) if app and 'prices' in app else pd.DataFrame()
    if grid.empty:
        return [gr.update()] * 14
    names = get_workspace(app, grid, frequency).names
    allocations = list(grid.index)
    shown = highlight(app, names)
    fund, bench = shown[0], (shown[-1] if len(shown) > 1 else names[-1])
    model = 'Model' if 'Model' in allocations else allocations[0]
    series = lambda value: gr.update(choices=names, value=value)
    return (gr.update(choices=['None'] + allocations),
            series(fund), series(bench),                                    # calendar
            series(fund), series(bench), series(fund), series(bench),       # return / beta analysis
            gr.update(choices=allocations, value=model), gr.update(choices=allocations, value=allocations[-1]),
            series(fund), series(fund), series(app['profile'].benchmark_label if app['profile'].benchmark_label in names
                                               else bench),
            gr.update(choices=allocations, value=model), series(fund))


def calendar(app, start, end, grid_table, frequency, freq, fund, bench):
    grid = grid_from(grid_table, app)
    need(app, grid)
    if fund == bench:
        raise gr.Error("Benchmark and Fund must be different")
    view = run(workspace.calendar_view, get_workspace(app, grid, frequency), *span(app, start, end), fund, bench, freq)
    return flat(charts.calendar_layout(view, freq))


def return_analysis(app, start, end, grid_table, frequency, fund, bench):
    grid = grid_from(grid_table, app)
    need(app, grid)
    if fund == bench:
        raise gr.Error("Benchmark and Fund must be different")
    view = run(workspace.return_analysis_view, get_workspace(app, grid, frequency), *span(app, start, end), fund,
               bench)
    return flat(charts.return_analysis_layout(view, fund))


def beta_analysis(app, start, end, grid_table, frequency, vol_window, fund, bench):
    grid = grid_from(grid_table, app)
    need(app, grid)
    view = run(workspace.beta_view, get_workspace(app, grid, frequency), *span(app, start, end), fund, bench,
               int(vol_window))
    return flat(charts.beta_layout(view, bench, ['Fund', fund], app['profile'].benchmark_ticker))


def risk_contribution(app, start, end, grid_table, frequency, fund, bench):
    grid = grid_from(grid_table, app)
    need(app, grid)
    if fund not in grid.index or bench not in grid.index:
        raise gr.Error("Pick a fund and a benchmark allocation")
    view = run(workspace.risk_contribution_view, window(app, start, end), grid, fund, bench, frequency)
    return show(view['contribution']), show(view['excess']), show(view['ex_ante'])


def risk_trajectory(app, start, end, grid_table, frequency, vol_window, name):
    grid = grid_from(grid_table, app)
    need(app, grid)
    view = run(workspace.risk_trajectory_view, get_workspace(app, grid, frequency), *span(app, start, end),
               int(vol_window))
    name = name if name in view['decomposition'] else next(iter(view['decomposition']))
    return flat(charts.trajectory_layout(view['volatility'], view['decomposition'][name], ['Fund', name]))


def tracking_error(app, start, end, grid_table, frequency, vol_window, name, bench):
    grid = grid_from(grid_table, app)
    need(app, grid)
    view = run(workspace.tracking_error_view, get_workspace(app, grid, frequency), *span(app, start, end), bench,
               int(vol_window))
    options = [n for n in view['decomposition'] if n != bench]
    if not options:
        raise gr.Error("No series long enough for the rolling window")
    name = name if name in options else options[0]
    return flat(charts.trajectory_layout(view['tracking_error'].drop(columns=bench, errors='ignore'),
                                         view['decomposition'][name], ['Fund', name], tracking_error=True))


def value_at_risk(app, start, end, grid_table, fund, vol_factor, mean_factor, iterations, scenarios, centile):
    grid = grid_from(grid_table, app)
    need(app, grid)
    prices = window(app, start, end)
    stress = np.eye(prices.shape[1]) * float(vol_factor)
    results = run(analytics.value_at_risk, prices, analytics.to_returns(prices), grid, var_centile=float(centile),
                  iterations=int(iterations), num_scenarios=int(scenarios), stress_factor=stress,
                  mean_shock=float(mean_factor))
    fund = fund if fund in results else next(iter(results))
    tables = engine.var_tables(results, fund, float(centile))
    return show(tables['summary']), show(tables.get('var_contribution')), show(tables.get('cvar_contribution'))


def var_history(app, start, end, grid_table, frequency, vol_window, name, method, centile):
    grid = grid_from(grid_table, app)
    need(app, grid)
    view = run(workspace.var_history_view, get_workspace(app, grid, frequency), *span(app, start, end),
               engine.VAR_METHODS[method], int(vol_window), float(centile), 1000, 1.0, 1.0)
    name = name if name in view['contributions'] else next(iter(view['contributions']), name)
    return flat(charts.var_history_layout(view, name, ['Fund', name]))


def pca(app, start, end, components, component, closest):
    need(app)
    returns = analytics.to_returns(window(app, start, end))
    n = int(min(max(components, 1), returns.shape[1]))
    view = run(workspace.pca_view, returns, n, component, int(min(max(closest, 1), returns.shape[1])))
    return flat(charts.pca_layout(view))


def correlation(app, start, end, vol_window, asset1, asset2):
    need(app)
    first, last = span(app, start, end)
    returns = app['prices'].pct_change(fill_method=None).iloc[1:].loc[:last]
    view = run(workspace.correlation_view, returns, asset1, asset2, int(vol_window), first, last)
    return flat(charts.correlation_layout(view, asset1, asset2))


def market_drivers(app, start, end, grid_table, frequency, vol_window, drivers_frequency, component):
    grid = grid_from(grid_table, app)
    need(app, grid)
    view = run(workspace.market_drivers_view, get_workspace(app, grid, frequency), *span(app, start, end),
               drivers_frequency, component, int(vol_window))
    return flat(charts.market_drivers_layout(view, app['profile'].highlight, app['profile'].benchmark_ticker))


# =============================================================================
# Layout
# =============================================================================

def plot_columns(left: int, right: int) -> List[gr.Plot]:
    """Two columns of plots (the notebook's HBox of two outputs); components in left-then-right order"""
    with gr.Row():
        with gr.Column():
            plots = [gr.Plot() for _ in range(left)]
        with gr.Column():
            plots += [gr.Plot() for _ in range(right)]
    return plots


def editor(headers, types, label):
    return gr.Dataframe(headers=headers, datatype=types, row_count=(1, 'dynamic'), col_count=(len(headers), 'fixed'),
                        interactive=True, label=label)


def build() -> gr.Blocks:
    css = "* { font-family: 'Arial Narrow', Arial, sans-serif; }"
    with gr.Blocks(css=css, title="Portfolio Construction", theme=gr.themes.Base()) as app_ui:
        app = gr.State({})
        gr.Markdown("# Portfolio Construction")
        with gr.Row():
            start = gr.Textbox(label="Start date (YYYY-MM-DD)")
            end = gr.Textbox(label="End date (YYYY-MM-DD)")

        # ------------------------------------------------------------------ Investment Universe
        with gr.Tab("Investment Universe"):
            with gr.Row():
                file = gr.File(label="Excel workbook (sheet 1: prices or returns, sheet 2: exposure table)",
                               file_types=['.xlsx'], type='filepath')
                benchmark_asset = gr.Dropdown([], label="Benchmark asset")
            upload_notes = gr.Markdown()
            refresh_universe = gr.Button("Refresh", variant="primary")
            u_returns = gr.Dataframe(label="Asset Returns")
            u_risk = gr.Dataframe(label="Asset Risk")
            u_plots = plot_columns(1, 1)
            u_prices = gr.Dataframe(label="Prices", max_height=400)
            u_exposure = gr.Dataframe(label="Transparency Table (optional sheet 2)")
            u_bounds = gr.Dataframe(label="Transparency Bounds")
            targets = gr.Markdown()

        # ------------------------------------------------------------------ Strategy
        with gr.Tab("Strategy"):
            with gr.Tab("Strategy"):
                refresh_assets = gr.Button("Refresh")
                s_returns = gr.Dataframe(label="Asset Returns")
                s_risk = gr.Dataframe(label="Asset Risk")
                with gr.Row():
                    strategy = gr.Dropdown(list(engine.STRATEGIES), value='Minimum Variance', label="Strategy")
                    frequency = gr.Dropdown(engine.FREQUENCIES, value='Monthly', label="Frequency")
                    benchmark = gr.Dropdown(['None'], value='None', label="Benchmark")
                    vol_window = gr.Number(value=252, precision=0, label="Vol Window")
                with gr.Row():                       # notebook: inputs | Add / Clear / Optimize
                    with gr.Column(scale=4):
                        constraints = editor(CONSTRAINT_HEADERS, ['str', 'str', 'number'],
                                             "Constraints (Target = asset, 'All' or exposure bucket; Sign ≤ ≥ =)")
                    with gr.Column(scale=1, min_width=160):
                        optimize_btn = gr.Button("Optimize Portfolio", variant="primary")
                strategy_notes = gr.Markdown()
                grid = gr.Dataframe(label="Allocation (editable; add rows to add allocations)", interactive=True,
                                    row_count=(1, 'dynamic'))
                lookthrough = gr.Dataframe(label="Look-through")
                overlays = editor(OVERLAY_HEADERS, ['str', 'number'], "Overlay (Strategy, Weight)")
                risk = editor(RISK_HEADERS, ['str', 'str', 'number'],
                              "Risk Constraints (Volatility, Beta, Tracking Error, Information Ratio)")
                results_btn = gr.Button("Get Results", variant="primary")
                results_notes = gr.Markdown()
                with gr.Row():
                    final_table = gr.Dataframe(label="Final Strategy", max_height=320)
                    core_table = gr.Dataframe(label="Core Strategy", max_height=320)
                    overlay_table = gr.Dataframe(label="Tactical Allocation", max_height=320)
                fund_metrics = gr.Dataframe(label="Systematic Fund Metrics")
                perf_btn = gr.Button("Refresh", variant="primary")
                p_returns = gr.Dataframe(label="Returns")
                p_risk = gr.Dataframe(label="Risk")
                p_expected = gr.Dataframe(label="Expected Returns")
                frontier_notes = gr.Markdown()
                perf_plots = plot_columns(2, 2)
            with gr.Tab("Strategy Return"):
                with gr.Row():
                    cal_freq = gr.Dropdown(['Year', 'Month'], value='Year', label="Frequency")
                    cal_fund = gr.Dropdown([], label="Fund")
                    cal_bench = gr.Dropdown([], label="Benchmark")
                    cal_btn = gr.Button("Update Perf", variant="primary")
                cal_plots = plot_columns(2, 2)

        # ------------------------------------------------------------------ Portfolio Analysis
        with gr.Tab("Portfolio Analysis"):
            with gr.Tab("Return Analysis"):
                with gr.Row():
                    ra_fund = gr.Dropdown([], label="Fund")
                    ra_bench = gr.Dropdown([], label="Benchmark")
                    ra_btn = gr.Button("Get Return Analysis", variant="primary")
                ra_plots = plot_columns(2, 2)
            with gr.Tab("Beta Analysis"):
                with gr.Row():
                    beta_fund = gr.Dropdown([], label="Fund")
                    beta_bench = gr.Dropdown([], label="Benchmark")
                    beta_btn = gr.Button("Get Beta Analysis", variant="primary")
                beta_plots = plot_columns(2, 2)

        # ------------------------------------------------------------------ Risk Analysis
        with gr.Tab("Risk Analysis"):
            with gr.Tab("Volatility Analysis"):
                with gr.Tab("Risk Contribution"):
                    with gr.Row():
                        rc_fund = gr.Dropdown([], label="Fund")
                        rc_bench = gr.Dropdown([], label="Benchmark")
                        rc_btn = gr.Button("Get Risk Contribution", variant="primary")
                    rc_contribution = gr.Dataframe(label="Performance and Risk Contribution")
                    rc_excess = gr.Dataframe(label="Excess P&L and TE Contribution")
                    rc_ex_ante = gr.Dataframe(label="Ex Ante Metrics")
                with gr.Tab("Risk Trajectory"):
                    with gr.Row():
                        traj_fund = gr.Dropdown([], label="Fund")
                        traj_btn = gr.Button("Get Risk History", variant="primary")
                    traj_plots = plot_columns(2, 2)
                with gr.Tab("Tracking Error"):
                    with gr.Row():
                        te_fund = gr.Dropdown([], label="Fund")
                        te_bench = gr.Dropdown([], label="Benchmark")
                        te_btn = gr.Button("Get Tracking Error History", variant="primary")
                    te_plots = plot_columns(2, 2)
            with gr.Tab("Value at Risk"):
                with gr.Tab("Value at Risk"):
                    var_fund = gr.Dropdown([], label="Fund")
                    with gr.Row():
                        vol_factor = gr.Number(value=1.0, label="Stress Factor")
                        mean_factor = gr.Number(value=1.0, label="Mean Shock Factor")
                        iterations = gr.Number(value=10000, precision=0, label="Iterations")
                        scenarios = gr.Number(value=5, precision=0, label="Scenarios")
                        centile = gr.Number(value=0.05, label="Centile")
                    var_btn = gr.Button("Run Simulation", variant="primary")
                    var_summary = gr.Dataframe(label="Summary")
                    var_results = gr.Dataframe(label="VaR Results")
                    cvar_results = gr.Dataframe(label="CVaR Results")
                with gr.Tab("Value at Risk History"):
                    with gr.Row():
                        vh_fund = gr.Dropdown([], label="Fund")
                        vh_method = gr.Dropdown(list(engine.VAR_METHODS), value='Gumbel Copula', label="Distribution")
                        vh_centile = gr.Number(value=0.05, label="Centile")
                        vh_btn = gr.Button("Get Value At Risk History", variant="primary")
                    vh_plots = plot_columns(2, 2)

        # ------------------------------------------------------------------ Market Risk
        with gr.Tab("Market Risk"):
            with gr.Tab("Market Risk"):
                with gr.Row():
                    components = gr.Number(value=5, precision=0, label="Number of Components")
                    component = gr.Dropdown([f"PC{i + 1}" for i in range(5)], value='PC1', label="Component")
                    closest = gr.Number(value=5, precision=0, label="Closest to PCA")
                    pca_btn = gr.Button("Get Market Risk", variant="primary")
                pca_plots = plot_columns(2, 2)
            with gr.Tab("Correlation"):
                with gr.Row():
                    asset1 = gr.Dropdown([], label="Asset 1")
                    asset2 = gr.Dropdown([], label="Asset 2")
                    corr_btn = gr.Button("Show Correlation", variant="primary")
                corr_plots = plot_columns(2, 2)
            with gr.Tab("Market Drivers"):
                with gr.Row():
                    drivers_frequency = gr.Dropdown(engine.FREQUENCIES, value='Quarterly', label="Frequency")
                    drivers_component = gr.Dropdown([f"PC{i + 1}" for i in range(5)], value='PC1', label="Component")
                    drivers_btn = gr.Button("Get Market Drivers", variant="primary")
                drivers_plots = plot_columns(3, 3)

        # ------------------------------------------------------------------ wiring
        universe_outputs = [u_returns, u_risk, *u_plots, u_prices, u_exposure, u_bounds, targets, asset1, asset2]
        file.change(on_upload, [file, benchmark_asset], [app, benchmark_asset, start, end, upload_notes]) \
            .then(universe_view, [app, start, end], universe_outputs) \
            .then(asset_tables, [app, start, end], [s_returns, s_risk])
        benchmark_asset.change(on_benchmark, [app, benchmark_asset], [app])
        refresh_universe.click(universe_view, [app, start, end], universe_outputs)
        refresh_assets.click(asset_tables, [app, start, end], [s_returns, s_risk])

        dropdowns = [benchmark, cal_fund, cal_bench, ra_fund, ra_bench, beta_fund, beta_bench, rc_fund, rc_bench,
                     traj_fund, te_fund, te_bench, var_fund, vh_fund]
        perf_inputs = [app, start, end, grid, benchmark, frequency, vol_window, constraints, risk]
        perf_outputs = [p_returns, p_risk, p_expected, frontier_notes, *perf_plots]
        optimize_btn.click(optimize, [app, start, end, constraints, risk, grid, benchmark, frequency],
                           [app, grid, lookthrough, strategy_notes]) \
            .then(choices, [app, grid, frequency], dropdowns)
        results_btn.click(get_results, [app, constraints, risk, overlays, grid, benchmark, frequency, strategy],
                          [app, grid, results_notes, final_table, core_table, overlay_table, fund_metrics]) \
            .then(choices, [app, grid, frequency], dropdowns) \
            .then(performance, perf_inputs, perf_outputs)
        perf_btn.click(performance, perf_inputs, perf_outputs)
        cal_btn.click(calendar, [app, start, end, grid, frequency, cal_freq, cal_fund, cal_bench], cal_plots)

        ra_btn.click(return_analysis, [app, start, end, grid, frequency, ra_fund, ra_bench], ra_plots)
        beta_btn.click(beta_analysis, [app, start, end, grid, frequency, vol_window, beta_fund, beta_bench],
                       beta_plots)
        rc_btn.click(risk_contribution, [app, start, end, grid, frequency, rc_fund, rc_bench],
                     [rc_contribution, rc_excess, rc_ex_ante])
        traj_btn.click(risk_trajectory, [app, start, end, grid, frequency, vol_window, traj_fund], traj_plots)
        te_btn.click(tracking_error, [app, start, end, grid, frequency, vol_window, te_fund, te_bench], te_plots)
        var_btn.click(value_at_risk, [app, start, end, grid, var_fund, vol_factor, mean_factor, iterations, scenarios,
                                      centile], [var_summary, var_results, cvar_results])
        vh_btn.click(var_history, [app, start, end, grid, frequency, vol_window, vh_fund, vh_method, vh_centile],
                     vh_plots)
        pca_btn.click(pca, [app, start, end, components, component, closest], pca_plots)
        corr_btn.click(correlation, [app, start, end, vol_window, asset1, asset2], corr_plots)
        drivers_btn.click(market_drivers, [app, start, end, grid, frequency, vol_window, drivers_frequency,
                                           drivers_component], drivers_plots)
    return app_ui


if __name__ == "__main__":
    build().launch()
