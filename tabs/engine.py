# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""App glue shared by every front-end (Streamlit tabs, Gradio, BQuant widgets).

No UI code here: plain functions from prices / editor tables to the inputs and
outputs of the src/ analytics, with the same steps as the Risk Management
notebooks (Optimize Portfolio -> allocation grid, Get Results -> rolling
backtest, workspace of named series -> tab views).

Editor tables are plain DataFrames:
  constraints       Target | Sign | Limit   (Target = asset, 'All' or an exposure bucket)
  risk constraints  Risk   | Sign | Limit   (Risk = Volatility, Beta, Tracking Error, Information Ratio)
  overlays          Strategy | Limit
Incomplete rows are skipped with a message instead of silently dropping the
rows after them.
"""
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src import Exposure
from src import PortfolioAnalytics as analytics
from src import Workspace as workspace
from src.Backtest import run_backtest
from src.Rebalancing import rebalanced_portfolio

# =============================================================================
# Vocabulary of the apps
# =============================================================================

STRATEGIES = {
    'Minimum Variance': 'minimum_variance',
    'Risk Parity': 'risk_parity',
    'Sharpe Ratio': 'sharpe_ratio',
    'Maximum Diversification': 'maximum_diversification',
    'Eigen Strategy': 'eigenportfolio',
    'Minimum Tracking Error': 'tracking_error',
    'Maximum Information Ratio': 'information_ratio',
}
RELATIVE = {'tracking_error', 'information_ratio'}

# Rows of the allocation grid, in the notebook's order
ALLOCATION_NAMES = {
    'sharpe_ratio': 'Optimal Portfolio',
    'minimum_variance': 'Minimum Variance',
    'maximum_diversification': 'Maximum Diversification',
    'risk_parity': 'Risk Parity',
    'eigenportfolio': 'Eigen Portfolio',
    'tracking_error': 'Minimum Tracking Error',
    'information_ratio': 'Maximum Information Ratio',
}
EQUAL_WEIGHT = 'Equal Weighted'
MODEL = 'Model'

RISK_TYPES = {
    'Volatility': 'volatility',
    'Beta': 'beta',
    'Tracking Error': 'tracking_error',
    'Information Ratio': 'information_ratio',
}
SIGNS = ['≤', '≥', '=']
FREQUENCIES = ['Monthly', 'Quarterly', 'Yearly']

VAR_METHODS = {
    'Multivariate': 'multivariate',
    'Gaussian Copula': 'gaussian_copula',
    'T-Student Copula': 't_copula',
    'Gumbel Copula': 'gumbel_copula',
    'Monte Carlo': 'monte_carlo',
}


@dataclass(frozen=True)
class Profile:
    """Market conventions of an app (as the backend's MARKET profiles)"""
    name: str
    benchmark_label: str            # name of the benchmark series ('Bitcoin', 'Benchmark')
    benchmark_ticker: Optional[str]  # asset held by it
    calendar_annualization: bool    # Multi Asset App metrics (calendar years) vs Crypto App
    excluded: Tuple[str, ...] = ()

    @property
    def benchmark(self) -> Tuple[str, str]:
        return self.benchmark_label, self.benchmark_ticker or ''

    @property
    def highlight(self) -> List[str]:
        """Series shown by default in the charts (the others start hidden)"""
        return ['Fund', self.benchmark_label]


CRYPTO = Profile('crypto', 'Bitcoin', 'BTCUSDT', calendar_annualization=False, excluded=('USDCUSDT',))


def multi_asset(benchmark_ticker: Optional[str]) -> Profile:
    """Profile of an uploaded universe; the benchmark is one of its assets"""
    return Profile('multi_asset', 'Benchmark', benchmark_ticker, calendar_annualization=True)


# =============================================================================
# Data
# =============================================================================

def _looks_like_returns(frame: pd.DataFrame) -> bool:
    values = frame.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return values.size > 0 and (values < 0).any() and np.abs(values).max() < 1


def read_workbook(file) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    """(prices, exposure table or None, messages) from an uploaded workbook.

    Sheet 1: dates x assets, prices -- or daily returns (as in the BQuant App),
             detected from negative values below 100% and compounded to prices.
    Sheet 2 (optional): exposure / transparency table, buckets x assets.
    """
    if isinstance(file, (bytes, bytearray)):
        file = BytesIO(file)
    book = pd.ExcelFile(file)
    messages = []
    frame = book.parse(book.sheet_names[0], index_col=0)
    frame.index = pd.to_datetime(frame.index, errors='coerce')
    frame = frame[frame.index.notna()].sort_index()
    frame = frame.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    frame.columns = frame.columns.astype(str)
    if frame.empty:
        raise ValueError("The first sheet has no dated numeric columns")
    if _looks_like_returns(frame):
        frame = (1 + frame.fillna(0)).cumprod() * 100
        messages.append("Sheet 1 read as daily returns and compounded to base-100 prices")

    exposure = None
    if len(book.sheet_names) > 1:
        exposure = Exposure.read_exposure(book.parse(book.sheet_names[1], index_col=0), list(frame.columns))
        if exposure is not None:
            messages.append(f"Exposure table '{book.sheet_names[1]}': {len(exposure)} buckets")
    return frame, exposure, messages


def clean_prices(prices: pd.DataFrame, profile: Profile) -> Tuple[pd.DataFrame, List[str]]:
    """Universe prices as the notebook's 'Get Prices' (src.Workspace.clean_prices)"""
    cleaned = workspace.clean_prices(prices.astype(float), excluded=profile.excluded)
    dropped = [c for c in prices.columns if c not in cleaned.columns and c not in profile.excluded]
    messages = [f"Not enough price history, excluded: {', '.join(dropped)}"] if dropped else []
    if cleaned.shape[1] == 0 or len(cleaned) < 3:
        raise ValueError("Not enough price data once incomplete assets and dates are removed")
    return cleaned, messages


# =============================================================================
# Editor tables -> constraint rows
# =============================================================================

def _complete(frame: Optional[pd.DataFrame], columns: Sequence[str]) -> Tuple[List[list], List[str]]:
    """Rows of `frame` with every column filled, and a message per incomplete row"""
    rows, messages = [], []
    if frame is None or frame.empty:
        return rows, messages
    for i, (_, row) in enumerate(frame.iterrows(), start=1):
        values = [row.get(c) for c in columns]
        filled = [v is not None and not (isinstance(v, float) and np.isnan(v)) and v != '' for v in values]
        if not any(filled):
            continue
        if not all(filled):
            messages.append(f"Row {i} ({', '.join(str(v) for v in values)}) is incomplete and was skipped")
            continue
        rows.append(values)
    return rows, messages


def constraint_rows(frame: Optional[pd.DataFrame], assets: Sequence[str],
                    exposure: Optional[pd.DataFrame] = None):
    """(weight rows, exposure rows, messages) from the constraint editor"""
    rows, messages = _complete(frame, ['Target', 'Sign', 'Limit'])
    buckets = set(exposure.index) if exposure is not None else set()
    weight, bucket = [], []
    for target, sign, limit in rows:
        try:
            limit = float(limit)
        except (TypeError, ValueError):
            messages.append(f"Limit '{limit}' of {target} is not a number, skipped")
            continue
        if sign not in SIGNS:
            messages.append(f"Sign '{sign}' of {target} is not one of {SIGNS}, skipped")
        elif target in assets or target == 'All':
            weight.append((target, sign, limit))
        elif target in buckets:
            bucket.append((target, sign, limit))
        else:
            messages.append(f"'{target}' is neither an asset nor an exposure bucket, skipped")
    return weight, bucket, messages


def risk_rows(frame: Optional[pd.DataFrame]) -> Tuple[Dict[str, Tuple[str, float]], List[str]]:
    """{'volatility' | 'beta' | ...: (sign, limit)} from the risk constraint editor (last row wins)"""
    rows, messages = _complete(frame, ['Risk', 'Sign', 'Limit'])
    out = {}
    for risk, sign, limit in rows:
        if risk not in RISK_TYPES or sign not in SIGNS:
            messages.append(f"Risk constraint ({risk}, {sign}) not recognised, skipped")
            continue
        out[RISK_TYPES[risk]] = (sign, float(limit))
    return out, messages


def overlay_rows(frame: Optional[pd.DataFrame]) -> Tuple[List[Tuple[str, float]], List[str]]:
    """[(objective, weight)] from the overlay editor"""
    rows, messages = _complete(frame, ['Strategy', 'Limit'])
    out = []
    for strategy, limit in rows:
        if strategy not in STRATEGIES:
            messages.append(f"Overlay '{strategy}' is not a strategy, skipped")
            continue
        out.append((STRATEGIES[strategy], float(limit)))
    return out, messages


def constraint_targets(assets: Sequence[str], exposure: Optional[pd.DataFrame] = None) -> List[str]:
    """Options of the constraint editor's Target column"""
    return list(assets) + ['All'] + (list(exposure.index) if exposure is not None else [])


# =============================================================================
# Strategy: optimization grid and rolling backtest
# =============================================================================

def benchmark_returns(prices: pd.DataFrame, bench_weights: pd.Series, frequency: str) -> pd.Series:
    """Realized returns of the benchmark allocation, rebalanced at `frequency` (beta constraints)"""
    return rebalanced_portfolio(prices, bench_weights.reindex(prices.columns).fillna(0.0),
                                frequency=frequency).sum(axis=1).pct_change()


def build_constraints(prices: pd.DataFrame, constraints: Optional[pd.DataFrame] = None,
                      risk: Optional[pd.DataFrame] = None, exposure: Optional[pd.DataFrame] = None,
                      bench_weights: Optional[pd.Series] = None, frequency: str = 'Monthly'):
    """SLSQP constraints on `prices`' window: weights, exposure buckets and risk limits.
    Returns (constraints, messages)."""
    returns = analytics.to_returns(prices)
    weight, bucket, messages = constraint_rows(constraints, list(prices.columns), exposure)
    risk_limits, more = risk_rows(risk)
    messages += more
    if bench_weights is None and set(risk_limits) - {'volatility'}:
        skipped = sorted(set(risk_limits) - {'volatility'})
        messages.append(f"Risk constraints {skipped} need a benchmark allocation: skipped")
        risk_limits = {k: v for k, v in risk_limits.items() if k == 'volatility'}
    bench_ret = None
    if bench_weights is not None and 'beta' in risk_limits:
        bench_ret = benchmark_returns(prices, bench_weights, frequency).reindex(returns.index)
    cons = analytics.build_constraints(returns, weight, risk_limits, bench_weights, bench_ret)
    cons += Exposure.exposure_constraints(exposure, bucket)
    return cons, messages


def optimize_grid(prices: pd.DataFrame, constraints: Optional[pd.DataFrame] = None,
                  risk: Optional[pd.DataFrame] = None, exposure: Optional[pd.DataFrame] = None,
                  bench_weights: Optional[pd.Series] = None, frequency: str = 'Monthly'):
    """'Optimize Portfolio': every objective, unconstrained and constrained, + equal weight.

    `prices` is the optimization window. Min Tracking Error / Max Information
    Ratio are added when a benchmark allocation is given. Returns (grid, messages).
    """
    returns = analytics.to_returns(prices)
    if returns.shape[0] < 3:
        raise ValueError("The date range is too short to optimize")
    if bench_weights is not None:
        bench_weights = bench_weights.reindex(returns.columns).fillna(0.0)
    cons, messages = build_constraints(prices, constraints, risk, exposure, bench_weights, frequency)

    objectives = [k for k in ALLOCATION_NAMES if k not in RELATIVE]
    if bench_weights is not None:
        objectives += sorted(RELATIVE)
    else:
        messages.append("ℹ️ Minimum Tracking Error / Maximum Information Ratio need a benchmark allocation: "
                        "pick one and optimize again")

    free, caught = analytics.optimize_allocations(returns, objectives, None, bench_weights)
    messages += caught
    if cons:
        bound, caught = analytics.optimize_allocations(returns, objectives, cons, bench_weights)
        messages += [f"Constrained: {m}" for m in caught]
    else:
        bound = free

    rows = {}
    for key in ALLOCATION_NAMES:
        if key in objectives:
            rows[ALLOCATION_NAMES[key]] = free.loc[key]
            rows[ALLOCATION_NAMES[key] + ' Constrained'] = bound.loc[key]
    rows[EQUAL_WEIGHT] = free.loc['equal_weight']
    grid = pd.DataFrame(rows).T.round(6)
    return grid, list(dict.fromkeys(messages))


def frontier(prices: pd.DataFrame, grid: pd.DataFrame, constraints: Optional[pd.DataFrame] = None,
             risk: Optional[pd.DataFrame] = None, exposure: Optional[pd.DataFrame] = None,
             bench_weights: Optional[pd.Series] = None, frequency: str = 'Monthly', points: int = 60) -> dict:
    """Efficient frontier of `prices`' window (and the constrained one), with each allocation's
    expected return / volatility / Sharpe ratio"""
    from src.Metrics import frontier_data
    cons, messages = build_constraints(prices, constraints, risk, exposure, bench_weights, frequency)
    returns = analytics.to_returns(prices)
    allocations = grid.reindex(columns=prices.columns).fillna(0.0)
    try:
        indicators, curve, constrained = frontier_data(returns, allocations, cons or None, points=points)
    except ValueError as exc:
        messages.append(f"Constrained frontier: {exc}")
        indicators, curve, constrained = frontier_data(returns, allocations, None, points=points)
    return {'indicators': indicators, 'frontier': curve, 'constrained': constrained, 'messages': messages}


def backtest(prices: pd.DataFrame, strategy: str, profile: Profile, frequency: str = 'Monthly',
             overlays: Optional[pd.DataFrame] = None, constraints: Optional[pd.DataFrame] = None,
             risk: Optional[pd.DataFrame] = None, exposure: Optional[pd.DataFrame] = None,
             bench_weights: Optional[pd.Series] = None) -> dict:
    """'Get Results': rolling optimization of `strategy` (a STRATEGIES label) + overlays on the
    full price history (src.Backtest.run_backtest). The result has an extra 'messages' list."""
    weight, bucket, messages = constraint_rows(constraints, list(prices.columns), exposure)
    risk_limits, more = risk_rows(risk)
    layers, more2 = overlay_rows(overlays)
    messages += more + more2
    if bench_weights is not None:
        bench_weights = bench_weights.reindex(prices.columns).fillna(0.0)
    elif set(risk_limits) - {'volatility'}:
        messages.append("Benchmark-relative risk constraints need a benchmark allocation: skipped")
        risk_limits = {k: v for k, v in risk_limits.items() if k == 'volatility'}

    result = run_backtest(
        prices,
        STRATEGIES[strategy],
        frequency=frequency,
        overlays=layers,
        weight_constraints=weight,
        risk_constraints=risk_limits or None,
        bench_weights=bench_weights,
        benchmark=profile.benchmark,
        calendar_annualization=profile.calendar_annualization,
        extra_constraints=Exposure.exposure_constraints(exposure, bucket),
    )
    result['messages'] = messages + result['warnings']
    return result


def with_model(grid: pd.DataFrame, model: Optional[pd.Series]) -> pd.DataFrame:
    """Allocation grid with the backtest's Model row (replaced if present)"""
    if model is None:
        return grid
    row = model.reindex(grid.columns).fillna(0.0).round(6).to_frame(MODEL).T
    return pd.concat([grid.drop(index=MODEL, errors='ignore'), row])


def strategy_schedules(result: Optional[dict]) -> Dict[str, pd.DataFrame]:
    """Fund / Core / Overlay weight schedules of a backtest (for the workspace)"""
    if not result:
        return {}
    return {name: frame for name, frame in result['schedules'].items() if frame is not None}


def build_workspace(prices: pd.DataFrame, grid: Optional[pd.DataFrame], profile: Profile,
                    frequency: str = 'Monthly', schedules: Optional[Dict[str, pd.DataFrame]] = None):
    """Workspace of every named series: Fund / Core / Overlay, Rebalanced X / Buy and Hold X,
    the benchmark -- each simulated on the full price history"""
    allocations = grid.reindex(columns=prices.columns).fillna(0.0) if grid is not None and len(grid) else None
    return workspace.build_workspace(
        prices,
        allocations=allocations,
        frequency=frequency,
        strategies=schedules or {},
        include_bitcoin=bool(profile.benchmark_ticker),
        benchmark=profile.benchmark,
    )


# =============================================================================
# Rebalancing trade (BQuant Trade, GradioLT)
# =============================================================================

def read_trade_workbook(content):
    """src.Trade.TradeBook of a holdings workbook (Holdings + exposure sheets)"""
    from src import Trade
    return Trade.read_trade_workbook(content)


def trade(book, constraints: Optional[pd.DataFrame] = None,
          custom: Optional[Dict[str, pd.Series]] = None) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Weights closest to the current holdings under the constraint editor's limits (Target =
    fund, 'All' or an exposure bucket), compared with the initial and `custom` allocations.
    Returns (tables, messages); tables = {'weights': ..., <category>: ...}."""
    from src import Trade
    weight, bucket, messages = constraint_rows(constraints, book.funds, book.exposure)
    optimized = Trade.optimize_trade(book, weight, bucket)
    return Trade.comparison(book, optimized, custom), messages


# =============================================================================
# Value at Risk inputs
# =============================================================================

def default_stress(assets: Sequence[str], vol_factor: float = 1.0, mean_factor: float = 1.0):
    """(mean shock table, stress grid) -- grid diagonal = volatility multipliers,
    off-diagonal = correlation shocks, as in the notebook's stress editor"""
    mean = pd.DataFrame({'Mean Shock': np.full(len(assets), float(mean_factor))}, index=list(assets))
    grid = pd.DataFrame(np.eye(len(assets)) * float(vol_factor), index=list(assets), columns=list(assets))
    return mean, grid


def symmetric(grid: pd.DataFrame) -> pd.DataFrame:
    """Correlation shocks mirrored across the diagonal (the larger shock of each pair wins)"""
    values = grid.to_numpy(dtype=float)
    upper = np.triu(values, 1)
    lower = np.tril(values, -1).T
    off = np.where(np.abs(upper) >= np.abs(lower), upper, lower)
    out = np.diag(np.diag(values)) + off + off.T
    return pd.DataFrame(out, index=grid.index, columns=grid.columns)


def var_tables(results: dict, fund: str, centile: float) -> Dict[str, pd.DataFrame]:
    """Tables of src.PortfolioAnalytics.value_at_risk: summary of every allocation,
    and the per-asset VaR / CVaR of `fund` for each simulation method"""
    labels = {'historical': 'Historical', 'parametric': 'Parametric',
              **{v: k for k, v in VAR_METHODS.items()}}
    summary = {}
    for name, result in results.items():
        for method, values in result['methods'].items():
            summary[(f'VaR {1 - centile:.0%}', labels[method])] = summary.get(
                (f'VaR {1 - centile:.0%}', labels[method]), {})
            summary[(f'VaR {1 - centile:.0%}', labels[method])][name] = values['var']
            summary[(f'CVaR {1 - centile:.0%}', labels[method])] = summary.get(
                (f'CVaR {1 - centile:.0%}', labels[method]), {})
            summary[(f'CVaR {1 - centile:.0%}', labels[method])][name] = values['cvar']
    summary = pd.DataFrame(summary).T
    summary.index = pd.MultiIndex.from_tuples(summary.index)

    out = {'summary': summary}
    methods = results.get(fund, {}).get('methods', {})
    var = {labels[m]: v['var_contribution'] for m, v in methods.items() if 'var_contribution' in v}
    cvar = {labels[m]: v['cvar_contribution'] for m, v in methods.items() if 'cvar_contribution' in v}
    if var:
        var_frame = pd.DataFrame(var)
        var_frame.loc['Portfolio'] = var_frame.sum()
        cvar_frame = pd.DataFrame(cvar)
        cvar_frame.loc['Portfolio'] = cvar_frame.sum()
        out['var_contribution'] = var_frame
        out['cvar_contribution'] = cvar_frame
    backtest_result = results.get(fund, {}).get('backtest')
    if backtest_result:
        out['backtest'] = backtest_result['series']
        out['kupiec'] = backtest_result['kupiec']
    return out
