"""Optimization, Value at Risk and per-asset metrics on whole allocation grids.

Used by the web app's controllers (and usable from the notebooks): synchronous,
CPU-bound and free of I/O. Every function takes price / return frames and
allocation grids and returns pandas objects.

Allocation grids follow the legacy app convention: index = allocation names,
columns = assets, values = weights.
"""
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .Metrics import (
    get_asset_returns,
    get_asset_risk,
    get_expected_metrics,
)
from .RiskMetrics import (
    RiskAnalysis,
    beta_constraint,
    build_constraint,
    compute_per_asset_betas,
    information_ratio_constraint,
    kupiec_test,
    rolling_var,
    tracking_error_constraint,
    vol_constraint,
)

ANNUALIZATION = 252

ABSOLUTE_OBJECTIVES = [
    'minimum_variance', 'sharpe_ratio', 'risk_parity', 'maximum_diversification', 'eigenportfolio',
]
RELATIVE_OBJECTIVES = ['tracking_error', 'information_ratio']

SIMULATION_METHODS = {
    'multivariate': 'multivariate_distribution',
    'gaussian_copula': 'gaussian_copula',
    't_copula': 't_copula',
    'gumbel_copula': 'gumbel_copula',
    'monte_carlo': 'monte_carlo',
}


# =============================================================================
# Helpers
# =============================================================================

def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns (the first, empty row is dropped)."""
    return prices.pct_change(fill_method=None).iloc[1:]


def allocation_frame(allocations: Dict[str, Dict[str, float]], assets: Sequence[str]) -> pd.DataFrame:
    """{name: {asset: weight}} -> allocation grid over `assets` (missing -> 0)."""
    frame = pd.DataFrame.from_dict(allocations, orient='index', dtype=float)
    return frame.reindex(columns=list(assets)).fillna(0.0)


def _record_warnings(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = func(*args, **kwargs)
    return result, [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


# =============================================================================
# Investment universe
# =============================================================================

def asset_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """Return and risk metrics per asset, with stable column names."""
    returns = get_asset_returns(prices)
    risk = get_asset_risk(prices)

    return pd.DataFrame({
        'total_return': returns.iloc[0],
        'ytd_return': returns.iloc[1],
        'annualized_return': returns.iloc[2],
        'volatility_1y_daily': risk.iloc[0],
        'volatility_3y_weekly': risk.iloc[1],
        'volatility_5y_monthly': risk.iloc[2],
        'volatility_monthly': risk.iloc[3],
        'cvar_parametric_95': risk.iloc[4],
        'max_drawdown': risk.iloc[5],
        'max_drawdown_date': risk.iloc[6],
    })


# =============================================================================
# Strategy
# =============================================================================

def build_constraints(
    returns: pd.DataFrame,
    weight_constraints: Iterable[Tuple[str, str, float]] = (),
    risk_constraints: Optional[Dict[str, Tuple[str, float]]] = None,
    bench_weights: Optional[pd.Series] = None,
    bench_returns: Optional[pd.Series] = None,
) -> List[dict]:
    """SLSQP constraints from weight bounds (asset or 'All', sign, limit) and
    risk bounds {'volatility' | 'beta' | 'tracking_error' | 'information_ratio': (sign, limit)}.

    The beta is measured against `bench_returns` when given (e.g. the benchmark's
    rebalanced series), else against the benchmark's static-weight returns."""
    rows = [list(c) for c in weight_constraints]
    constraints = build_constraint(returns, np.array(rows, dtype=object)) if rows else []

    for name, (sign, limit) in (risk_constraints or {}).items():
        if name == 'volatility':
            constraints += vol_constraint(returns.cov(), sign, limit)
            continue

        if bench_weights is None:
            raise ValueError(f"The {name} constraint needs a benchmark allocation")

        if name == 'beta':
            if bench_returns is None:
                bench_returns = RiskAnalysis(returns).portfolio(bench_weights.to_numpy())
            beta = compute_per_asset_betas(returns, bench_returns)
            constraints += beta_constraint(beta, sign, limit)
        elif name == 'tracking_error':
            constraints += tracking_error_constraint(returns.cov(), bench_weights, sign, limit)
        elif name == 'information_ratio':
            constraints += information_ratio_constraint(returns, bench_weights, sign, limit)
        else:
            raise ValueError(f"Unknown risk constraint {name!r}")

    return constraints


def optimize_allocations(
    returns: pd.DataFrame,
    objectives: Sequence[str],
    constraints: Optional[List[dict]] = None,
    bench_weights: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Optimal weights for each objective (+ the equal-weight portfolio).

    Returns (allocation grid, warnings) -- a warning is emitted when SLSQP
    did not converge and the constraints may not all hold.
    """
    portfolio = RiskAnalysis(returns)
    allocations = {}
    messages = []

    for objective in objectives:
        if objective in RELATIVE_OBJECTIVES and bench_weights is None:
            raise ValueError(f"The {objective} objective needs a benchmark allocation")

        weights, caught = _record_warnings(
            portfolio.optimize,
            objective,
            constraints=constraints or False,
            bench_weights=bench_weights if objective in RELATIVE_OBJECTIVES else None,
        )
        allocations[objective] = weights
        messages += caught

    n_assets = returns.shape[1]
    allocations['equal_weight'] = np.full(n_assets, 1 / n_assets)

    return pd.DataFrame(allocations, index=returns.columns).T, messages


def expected_metrics(returns: pd.DataFrame, allocations: pd.DataFrame) -> pd.DataFrame:
    """Expected return / volatility / Sharpe ratio (rows) of each allocation (columns)."""
    return get_expected_metrics(returns, allocations)


# =============================================================================
# Performance
# =============================================================================

def drawdowns(levels: pd.DataFrame) -> pd.DataFrame:
    return levels / levels.cummax() - 1


def rolling_volatility(levels: pd.DataFrame, window: int) -> pd.DataFrame:
    vol = levels.pct_change(fill_method=None).rolling(window).std() * np.sqrt(ANNUALIZATION)
    return vol.dropna(how='all')


# =============================================================================
# Risk metrics
# =============================================================================

def _var_row(distrib: pd.DataFrame, weights: pd.Series, var_centile: float) -> Tuple[pd.Series, pd.Series]:
    """VaR scenario (per-asset contributions + 'Portfolio') and the mean of
    the scenarios beyond it (CVaR)."""
    weighted = distrib.mul(weights, axis=1)
    weighted = weighted.loc[:, weights[weights != 0].index]
    weighted['Portfolio'] = weighted.sum(axis=1)

    var = weighted.sort_values(by='Portfolio').iloc[int(weighted.shape[0] * var_centile)]
    cvar = weighted.loc[weighted['Portfolio'] < var['Portfolio']].mean()
    return var, cvar


def value_at_risk(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    allocations: pd.DataFrame,
    var_centile: float = 0.05,
    iterations: int = 10000,
    num_scenarios: int = 1,
    stress_factor=1.0,
    mean_shock=1.0,
    horizon_days: int = 1,
    theta: float = 2.0,
    methods: Optional[Sequence[str]] = None,
    backtest_window: Optional[int] = None,
) -> Dict[str, dict]:
    """VaR / CVaR of each allocation: historical, parametric and simulated
    (multivariate normal, gaussian / t / gumbel copulas, Monte Carlo).

    Simulated figures are averaged over `num_scenarios` independent draws;
    each draw is shared by all allocations. With `backtest_window`, a rolling
    VaR backtest (historical / parametric / multivariate) and its Kupiec test
    are added.
    """
    portfolio = RiskAnalysis(returns)
    methods = list(methods or SIMULATION_METHODS)

    stress = np.asarray(stress_factor, dtype=float)
    # the gumbel copula and the parametric VaR only take per-asset / scalar shocks
    stress_vector = np.diag(stress) if stress.ndim == 2 else stress
    stress_scalar = float(stress) if stress.ndim == 0 else 1.0

    horizon = horizon_days / ANNUALIZATION
    spot = prices.ffill().iloc[-1]

    args = {
        'multivariate': (iterations, stress_factor, mean_shock),
        'gaussian_copula': (iterations, stress_factor, mean_shock),
        't_copula': (iterations, stress_factor, mean_shock),
        'gumbel_copula': (iterations, theta, stress_vector, mean_shock),
        'monte_carlo': (spot, horizon, iterations, stress_factor, mean_shock),
    }

    weights_by_name = {name: allocations.loc[name] for name in allocations.index}
    results = {name: {'methods': {}} for name in allocations.index}

    # Historical and parametric
    for name, weights in weights_by_name.items():
        w = weights.to_numpy()
        hist_var, hist_cvar = portfolio.historical_var(w, Q=var_centile * 100)
        param_var, param_cvar = portfolio.parametric_var(w, Q=1 - var_centile, stress_factor=stress_scalar)
        results[name]['methods']['historical'] = {'var': hist_var, 'cvar': hist_cvar}
        results[name]['methods']['parametric'] = {'var': param_var, 'cvar': param_cvar}

    # Simulations
    for method in methods:
        func = getattr(portfolio, SIMULATION_METHODS[method])
        var_rows = {name: [] for name in weights_by_name}
        cvar_rows = {name: [] for name in weights_by_name}
        last_distrib = None

        for _ in range(num_scenarios):
            output = func(*args[method])
            distrib = output[1] if method == 'monte_carlo' else output
            distrib = pd.DataFrame(np.asarray(distrib), columns=returns.columns)
            last_distrib = distrib

            for name, weights in weights_by_name.items():
                var, cvar = _var_row(distrib, weights, var_centile)
                var_rows[name].append(var)
                cvar_rows[name].append(cvar)

        for name, weights in weights_by_name.items():
            var = pd.DataFrame(var_rows[name]).mean()
            cvar = pd.DataFrame(cvar_rows[name]).mean()
            simulated = last_distrib.to_numpy() @ weights.to_numpy()
            counts, edges = np.histogram(simulated[np.isfinite(simulated)], bins=40)

            results[name]['methods'][method] = {
                'var': var['Portfolio'],
                'cvar': cvar['Portfolio'],
                'var_contribution': var.drop('Portfolio'),
                'cvar_contribution': cvar.drop('Portfolio'),
                'histogram': {'counts': counts.tolist(), 'bin_edges': edges.tolist()},
            }

    if backtest_window:
        clean_returns = returns.dropna()
        if clean_returns.shape[0] <= backtest_window:
            raise ValueError(f"Backtest window ({backtest_window}) must be shorter than the history "
                             f"({clean_returns.shape[0]} complete days)")
        for name, weights in weights_by_name.items():
            backtest = rolling_var(clean_returns, weights.to_numpy(), window=backtest_window, Q=var_centile * 100)
            results[name]['backtest'] = {
                'series': backtest,
                'kupiec': kupiec_test(backtest, Q=var_centile * 100),
            }

    return results
