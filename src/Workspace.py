"""Named portfolio series shared by the app tabs, and the views built on them.

The computations behind the Crypto App's `_build_weight_series_dict` /
`_build_ex_post_return_series` and the tabs that consume them
(strategy performance, calendar, return / beta analysis, risk trajectory,
tracking error, VaR history, risk contribution, stress preview, market risk).

A workspace holds, for every named series ('Historical Portfolio', 'Fund',
'Core', 'Overlay', 'Rebalanced X', 'Buy and Hold X', 'Bitcoin'), its daily
weight history and the asset returns those weights earn. Every series is
simulated on the FULL price history and only cropped to the analysis window,
so changing the window never restarts a simulation (same convention as the
notebook).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.correlation_tools import cov_nearest

from .Metrics import (
    frontier_data,
    get_monthly_metrics,
    get_portfolio_risk,
    get_yearly_metrics,
    rebalanced_metrics,
)
from .Rebalancing import (
    buy_and_hold,
    get_rebalancing_dates,
    rebalanced_book_cost,
    rebalanced_dynamic_quantities,
    rebalanced_portfolio,
)
from .RiskMetrics import (
    RiskAnalysis,
    compute_per_asset_betas,
    compute_rolling_per_asset_betas,
    drawdown_contribution,
    first_pca_over_time,
    get_ex_ante_vol_decomposition,
    get_var_contribution,
    performance_contribution,
)

HISTORICAL = 'Historical Portfolio'
BITCOIN = 'Bitcoin'
BITCOIN_SYMBOL = 'BTCUSDT'
CASH_SYMBOL = 'USDTUSDT'
EXCLUDED_SYMBOLS = ('USDCUSDT',)
MAX_MISSING_RETURNS = 30
# The notebook's Strategy tab annualizes its rolling volatility chart with √260
ROLLING_VOL_ANNUALIZATION = 260

SIMULATION_FUNCTIONS = {
    'multivariate': 'multivariate_distribution',
    'gaussian_copula': 'gaussian_copula',
    't_copula': 't_copula',
    'gumbel_copula': 'gumbel_copula',
    'monte_carlo': 'monte_carlo',
}


# =============================================================================
# Building the workspace
# =============================================================================

def clean_prices(prices: pd.DataFrame, max_missing: int = MAX_MISSING_RETURNS,
                 excluded: Sequence[str] = EXCLUDED_SYMBOLS) -> pd.DataFrame:
    """Universe prices as the notebook's 'Get Prices': stablecoins excluded,
    assets with fewer than `max_missing` missing returns, complete rows only."""
    prices = prices.drop(columns=[c for c in excluded if c in prices.columns]).sort_index()
    returns = prices.pct_change(fill_method=None)
    valid = returns.columns[returns.isna().sum() < max_missing]
    return prices[valid].dropna()


def _normalize(values: pd.DataFrame) -> pd.DataFrame:
    return values.div(values.sum(axis=1), axis=0)


def allocation_weights(prices: pd.DataFrame, weights: pd.Series, rebalanced: bool,
                       frequency: str = 'Monthly') -> pd.DataFrame:
    """Daily weights of a static allocation, buy-and-hold or rebalanced at `frequency`."""
    weights = weights.reindex(prices.columns).fillna(0.0)
    if rebalanced:
        values = rebalanced_portfolio(prices, weights, frequency=frequency)
    else:
        values = buy_and_hold(prices, weights)
    return _normalize(values)


def schedule_weights(prices: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Daily weights of a strategy rebalanced to `schedule` (index = rebalancing dates)."""
    quantities = rebalanced_dynamic_quantities(prices, schedule.sort_index())
    return _normalize(quantities * prices)


def historical_weights(positions: pd.DataFrame, cash: Sequence[str] = (CASH_SYMBOL,)) -> pd.DataFrame:
    """Weights of the real portfolio from its position values (cash stays in the
    denominator but carries no return), as the notebook's `weights_ex_post`."""
    positions = positions.sort_index()
    total = positions['Total'] if 'Total' in positions.columns else positions.sum(axis=1)
    weights = positions.drop(columns=[*cash, 'Total'], errors='ignore')
    return weights.div(total, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass
class Workspace:
    """Universe prices and the weight history of every named series"""
    prices: pd.DataFrame
    returns: pd.DataFrame
    all_returns: pd.DataFrame
    weights: Dict[str, pd.DataFrame] = field(default_factory=dict)
    asset_returns: Dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def names(self) -> List[str]:
        return list(self.weights)

    def require(self, *names: str) -> None:
        missing = [n for n in names if n not in self.weights]
        if missing:
            raise ValueError(f"Unknown series: {missing} (available: {self.names})")

    def aligned(self, name: str):
        """(weights, asset returns) of `name` on their common dates and columns"""
        weights, returns = self.weights[name], self.asset_returns[name]
        index = weights.index.intersection(returns.index)
        cols = weights.columns.intersection(returns.columns)
        return weights.loc[index, cols].fillna(0.0), returns.loc[index, cols]

    def weighted_returns(self, name: str) -> pd.DataFrame:
        """Per-asset contribution to the daily return: yesterday's weight x today's return"""
        weights, returns = self.aligned(name)
        return returns.mul(weights.shift(1)).fillna(0.0)

    def series_returns(self, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
        names = list(names) if names is not None else self.names
        return pd.DataFrame({n: self.weighted_returns(n).sum(axis=1) for n in names})


def build_workspace(
    prices: pd.DataFrame,
    allocations: Optional[pd.DataFrame] = None,
    frequency: str = 'Monthly',
    strategies: Optional[Dict[str, pd.DataFrame]] = None,
    include_bitcoin: bool = True,
    positions: Optional[pd.DataFrame] = None,
    historical_prices: Optional[pd.DataFrame] = None,
    benchmark: Tuple[str, str] = (BITCOIN, BITCOIN_SYMBOL),
    cash: Sequence[str] = (CASH_SYMBOL,),
) -> Workspace:
    """Workspace in the notebook's display order: Historical Portfolio, the
    strategies (Fund, Core, Overlay), 'Rebalanced X' / 'Buy and Hold X' for each
    allocation, then the benchmark (`benchmark` = (series name, ticker), held
    buy-and-hold: 'Bitcoin' for crypto, 'Benchmark' for multi-asset).

    `include_bitcoin` adds the benchmark series."""
    benchmark_name, benchmark_ticker = benchmark
    returns = prices.pct_change(fill_method=None)
    all_returns = returns
    ws = Workspace(prices=prices, returns=returns, all_returns=all_returns)

    if positions is not None and historical_prices is not None and not positions.empty:
        hist_returns = historical_prices.sort_index().pct_change(fill_method=None)
        extra = hist_returns.columns.difference(returns.columns)
        ws.all_returns = pd.concat([returns, hist_returns[extra]], axis=1).sort_index()
        ws.weights[HISTORICAL] = historical_weights(positions, cash)
        ws.asset_returns[HISTORICAL] = hist_returns

    for name, schedule in (strategies or {}).items():
        ws.weights[name] = schedule_weights(prices, schedule)
        ws.asset_returns[name] = returns

    if allocations is not None:
        for name in allocations.index:
            weights = allocations.loc[name]
            ws.weights['Rebalanced ' + name] = allocation_weights(prices, weights, True, frequency)
            ws.weights['Buy and Hold ' + name] = allocation_weights(prices, weights, False)
            ws.asset_returns['Rebalanced ' + name] = returns
            ws.asset_returns['Buy and Hold ' + name] = returns

    if include_bitcoin and benchmark_ticker in prices.columns:
        ws.weights[benchmark_name] = allocation_weights(prices, pd.Series({benchmark_ticker: 1.0}), False)
        ws.asset_returns[benchmark_name] = returns

    return ws


def levels(returns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative value (base 100 at the first date)"""
    returns = returns.copy()
    if not returns.empty:
        returns.iloc[0] = 0
    return (1 + returns).cumprod() * 100


def _check_window(frame: pd.DataFrame, window: int, name: Optional[str] = None) -> None:
    if len(frame) < window:
        if name:
            raise ValueError(f"{name} has {len(frame)} days in the date range, fewer than the rolling window "
                             f"({window}): pick another series, an earlier start date or a shorter window")
        raise ValueError(f"Date range ({len(frame)} days) is shorter than the rolling window ({window})")


# =============================================================================
# Strategy tab
# =============================================================================

def performance_view(
    ws: Workspace,
    start,
    end,
    allocations: pd.DataFrame,
    benchmark: Optional[str] = None,
    vol_window: int = 252,
    constraints: Optional[list] = None,
    frontier_points: int = 100,
) -> dict:
    """Cumulative performance, drawdown, rolling volatility, return / risk tables
    and efficient frontier (Strategy tab 'Refresh')."""
    rets = ws.series_returns().loc[start:end]
    if rets.empty:
        raise ValueError("No data in the selected date range")
    lv = levels(rets)

    range_prices = ws.prices.loc[start:end]
    risk = get_portfolio_risk(allocations, range_prices, lv, benchmark or '')
    result = {
        'levels': lv,
        'drawdown': (lv - lv.cummax()) / lv.cummax(),
        'rolling_volatility': lv.pct_change(fill_method=None).rolling(vol_window).std()
        * np.sqrt(ROLLING_VOL_ANNUALIZATION),
        'returns_table': rebalanced_metrics(lv),
        'risk_table': risk,
        'frontier': None,
        'warnings': [],
    }

    if len(allocations) and frontier_points:
        try:
            indicators, frontier, frontier_c = frontier_data(
                range_prices.pct_change(fill_method=None), allocations, constraints or None, points=frontier_points
            )
            result['frontier'] = {'indicators': indicators, 'frontier': frontier, 'constrained': frontier_c}
        except ValueError as exc:
            result['warnings'].append(f"Efficient frontier: {exc}")

    return result


def calendar_view(ws: Workspace, start, end, fund: str, benchmark: str, frequency: str = 'Year') -> dict:
    """Returns, volatility, tracking error and Sharpe ratio by year / month"""
    ws.require(fund, benchmark)
    lv = levels(ws.series_returns([fund, benchmark]).loc[start:end])
    func = get_yearly_metrics if frequency == 'Year' else get_monthly_metrics
    returns, vol, tracking_error, sharpe = func(lv, fund=fund, bench=benchmark)
    return {'returns': returns.T, 'volatility': vol.T, 'tracking_error': tracking_error, 'sharpe_ratio': sharpe.T}


# =============================================================================
# Current Portfolio tab
# =============================================================================

def return_analysis_view(ws: Workspace, start, end, fund: str, benchmark: str) -> dict:
    """Return contribution of the fund and the benchmark, active contribution
    and drawdown contribution (P&L > Return Analysis)."""
    ws.require(fund, benchmark)
    weighted = ws.weighted_returns(fund).loc[start:end]
    weighted_bench = ws.weighted_returns(benchmark).loc[start:end]
    if weighted.empty:
        raise ValueError("No data in the selected date range")

    contribution = performance_contribution(weighted)
    contribution['Total Return'] = contribution.sum(axis=1)
    bench_contribution = performance_contribution(weighted_bench)
    bench_contribution['Total Return'] = bench_contribution.sum(axis=1)

    lv = levels(ws.series_returns([fund]).loc[start:end])[fund]
    drawdown = pd.concat([drawdown_contribution(weighted), ((lv - lv.cummax()) / lv.cummax()).rename(fund)], axis=1)

    return {
        'contribution': contribution,
        'benchmark_contribution': bench_contribution,
        'active_contribution': contribution.subtract(bench_contribution, fill_value=0),
        'drawdown': drawdown,
    }


def beta_view(ws: Workspace, start, end, fund: str, benchmark: str, window: int = 252) -> dict:
    """Rolling asset betas, fund weights, rolling beta of every series and beta
    contribution of the fund vs. the benchmark series (P&L > Beta Analysis)."""
    ws.require(fund, benchmark)
    perf = ws.series_returns().loc[start:end]
    _check_window(perf, window)
    asset_returns = ws.asset_returns[fund]

    beta_rolling = compute_rolling_per_asset_betas(asset_returns, perf[benchmark], window)
    weights = ws.weights[fund]
    common_cols = beta_rolling.columns.intersection(weights.columns)
    weights_aligned = weights.reindex(index=beta_rolling.index, columns=common_cols).fillna(0.0)
    contribution = beta_rolling.reindex(columns=common_cols) * weights_aligned
    contribution['Total Beta'] = contribution.sum(axis=1, min_count=1)

    return {
        'asset_beta': beta_rolling.loc[start:end].dropna(how='all'),
        'weights': weights_aligned.loc[start:end],
        'portfolio_beta': compute_rolling_per_asset_betas(perf, perf[benchmark], window).dropna(how='all'),
        'beta_contribution': contribution.loc[start:end].dropna(how='all'),
    }


# =============================================================================
# Risk Analysis tab
# =============================================================================

def risk_contribution_view(prices: pd.DataFrame, allocations: pd.DataFrame, fund: str, benchmark: str,
                           frequency: str = 'Monthly') -> dict:
    """Simulated P&L and ex-ante risk contribution of the fund, excess P&L and
    tracking-error contribution vs. the benchmark, and the ex-ante volatility /
    tracking error / beta of every allocation (Volatility Analysis > Risk Contribution)."""
    for name in (fund, benchmark):
        if name not in allocations.index:
            raise ValueError(f"'{name}' is not one of the allocations")
    returns = prices.pct_change(fill_method=None).dropna()
    if returns.empty:
        raise ValueError("Not enough data to compute returns")

    portfolio = RiskAnalysis(returns)
    fund_weights = allocations.loc[fund]
    bench_weights = allocations.loc[benchmark]

    def simulated_pnl(weights):
        rebalanced = rebalanced_portfolio(prices, weights, frequency=frequency) / prices
        held = buy_and_hold(prices, weights) / prices
        pnl_held = (held * prices - rebalanced_book_cost(prices, held)).iloc[-1]
        pnl_rebalanced = (rebalanced * prices - rebalanced_book_cost(prices, rebalanced)).iloc[-1]
        return pd.DataFrame({'Profit and Loss (Buy and Hold)': pnl_held,
                             'Profit and Loss (Rebalanced)': pnl_rebalanced})

    bench_returns = portfolio.portfolio(bench_weights)
    beta = compute_per_asset_betas(returns, bench_returns)

    contribution = pd.concat([simulated_pnl(fund_weights), portfolio.var_contrib(fund_weights)[0] * 100], axis=1)
    contribution.loc['Total'] = contribution.sum(axis=0)

    active = fund_weights - bench_weights
    excess = pd.concat([
        simulated_pnl(active),
        portfolio.var_contrib(active)[0] * 100,
        (beta * fund_weights).to_frame(name='Beta Contribution'),
    ], axis=1)
    excess.loc['Total'] = excess.sum(axis=0)

    ex_ante = pd.DataFrame({
        'Vol Ex Ante': {i: portfolio.variance(allocations.loc[i]) for i in allocations.index},
        'Tracking Error Ex Ante': {i: portfolio.variance(allocations.loc[i] - bench_weights) for i in allocations.index},
        'Beta Ex Ante': {i: float(np.dot(allocations.loc[i], beta)) for i in allocations.index},
    })

    return {
        'contribution': contribution.fillna(0).sort_values(by='Vol Contribution', ascending=False),
        'excess': excess.fillna(0).sort_values(by='Vol Contribution', ascending=False),
        'ex_ante': ex_ante,
    }


def _lookback(frame: pd.DataFrame, start, end, window: int) -> pd.DataFrame:
    """`frame` up to `end`, from `window` - 1 rows before `start`: the first rolling
    window ends on `start` (when the history allows it), so a rolling value never
    depends on the date range and a narrower range is a slice of a wider one."""
    frame = frame.loc[:end]
    first = frame.index.searchsorted(pd.Timestamp(start))
    return frame.iloc[max(first - window + 1, 0):]


def _check_history(ws: Workspace, start, end, window: int) -> None:
    rows = len(_lookback(ws.returns, start, end, window))
    if rows < window:
        raise ValueError(f"Price history up to the end date ({rows} days) is shorter than the rolling window "
                         f"({window})")


def _decompositions(ws: Workspace, inputs, window: int, n_jobs: int) -> Dict[str, dict]:
    """Rolling vol / correlation / idiosyncratic decomposition of every series long
    enough for the window, {name: decomposition}"""
    decompositions = {}
    for name in ws.names:
        weights, returns = inputs(name)
        if len(weights) >= window:
            decompositions[name] = get_ex_ante_vol_decomposition(weights, returns, window, n_jobs=n_jobs)
    return decompositions


def risk_trajectory_view(ws: Workspace, start, end, window: int = 252, n_jobs: int = -1) -> dict:
    """Rolling ex-ante volatility of every series and its decomposition
    (Volatility Analysis > Risk Trajectory). The volatility is the sum of the
    contributions, so one pass per series gives both."""
    _check_history(ws, start, end, window)

    def inputs(name):
        weights, returns = ws.aligned(name)
        weights = _lookback(weights, start, end, window)
        return weights, returns.loc[weights.index]

    decompositions = _decompositions(ws, inputs, window, n_jobs)
    return {
        'volatility': pd.DataFrame({name: d['vol_contribution']['Total Vol'] for name, d in decompositions.items()}),
        'decomposition': decompositions,
    }


def _spread(ws: Workspace, name: str, benchmark: str) -> pd.DataFrame:
    """Active weights (series - benchmark) on the dates both series exist"""
    fund, bench = ws.weights[name], ws.weights[benchmark]
    cols = fund.columns.union(bench.columns)
    index = fund.index.intersection(bench.index)
    return (fund.reindex(index=index, columns=cols, fill_value=0)
            - bench.reindex(index=index, columns=cols, fill_value=0)).fillna(0.0)


def tracking_error_view(ws: Workspace, start, end, benchmark: str, window: int = 252, n_jobs: int = -1) -> dict:
    """Rolling ex-ante tracking error of every series vs. the benchmark and the
    decomposition of every spread (Volatility Analysis > Tracking Error)."""
    ws.require(benchmark)
    _check_history(ws, start, end, window)

    def inputs(name):
        spread = _lookback(_spread(ws, name, benchmark), start, end, window)
        returns = ws.all_returns.reindex(index=spread.index, columns=spread.columns)
        return spread, returns

    decompositions = _decompositions(ws, inputs, window, n_jobs)
    return {
        'tracking_error': pd.DataFrame({name: d['vol_contribution']['Total Vol'] for name, d in decompositions.items()}),
        'decomposition': decompositions,
    }


def _simulation_args(method: str, columns, last_prices: pd.Series, iterations: int, stress_factor: float,
                     mean_factor: float, theta: float = 2, horizon: float = 1 / 250):
    if method == 'gumbel_copula':
        return (iterations, theta, stress_factor, mean_factor)
    if method == 'monte_carlo':
        return (last_prices.reindex(columns), horizon, iterations, stress_factor, mean_factor)
    return (iterations, stress_factor, mean_factor)


def var_history_view(ws: Workspace, start, end, method: str = 'gumbel_copula',
                     window: int = 252, var_centile: float = 0.05, iterations: int = 10000,
                     stress_factor: float = 1.0, mean_factor: float = 1.0, n_jobs: int = -1) -> dict:
    """Rolling simulated VaR / CVaR of every series and their per-asset VaR / CVaR
    (Value at Risk > Value at Risk History)."""
    if method not in SIMULATION_FUNCTIONS:
        raise ValueError(f"Unknown VaR method {method!r}")
    func_name = SIMULATION_FUNCTIONS[method]
    _check_history(ws, start, end, window)
    last_prices = ws.prices.ffill().iloc[-1]

    def run(name):
        weights, returns = ws.aligned(name)
        weights = _lookback(weights, start, end, window)
        returns = returns.loc[weights.index]
        _check_window(weights, window, name)
        args = _simulation_args(method, weights.columns, last_prices, iterations, stress_factor, mean_factor)
        return get_var_contribution(func_name, args, returns, weights, window, var_centile, n_jobs=n_jobs)

    contributions, warnings = {}, []
    for name in ws.names:
        if name == HISTORICAL and method == 'monte_carlo':
            continue  # Monte Carlo needs spot prices of every held asset
        try:
            var_data, cvar_data = run(name)
            contributions[name] = {'var': var_data, 'cvar': cvar_data}
        except Exception as exc:  # one series failing (e.g. a delisted asset) must not hide the others
            warnings.append(str(exc) if str(exc).startswith(name) else f"{name}: {exc}")

    return {
        'var': pd.DataFrame({name: c['var']['Portfolio'] for name, c in contributions.items()}),
        'cvar': pd.DataFrame({name: c['cvar']['Portfolio'] for name, c in contributions.items()}),
        'contributions': contributions,
        'warnings': warnings,
    }


def stress_preview(returns: pd.DataFrame, stress_grid: np.ndarray, mean_shock: pd.Series) -> dict:
    """Means / volatilities and correlations before and after the stress grid
    (diagonal = volatility multipliers, off-diagonal = correlation shocks) and
    the mean shocks (Value at Risk > Value at Risk)."""
    returns = returns.dropna(how='all')
    cov = returns.cov()
    grid = np.asarray(stress_grid, dtype=float)

    stress = np.diag(np.diag(grid))
    stressed_cov = stress @ cov.to_numpy() @ stress
    stressed_std = np.sqrt(np.diag(stressed_cov))

    corr = stressed_cov / np.outer(stressed_std, stressed_std)
    corr = corr + np.tril(grid, -1) + np.tril(grid, -1).T
    corr = cov_nearest(np.clip(corr, -1, 1))

    means = returns.mean() * 250
    moments = pd.DataFrame({
        'Means': means,
        'Shocked Means': means * mean_shock.reindex(returns.columns).fillna(1.0),
        'Vol': returns.std() * np.sqrt(250),
        'Shocked Vol': pd.Series(stressed_std * np.sqrt(250), index=returns.columns),
    })
    return {
        'moments': moments,
        'correlation': returns.corr(),
        'stressed_correlation': pd.DataFrame(corr, index=returns.columns, columns=returns.columns),
    }


# =============================================================================
# Market Risk tab
# =============================================================================

def pca_view(returns: pd.DataFrame, num_components: int = 5, component: str = 'PC1', num_closest: int = 5) -> dict:
    """Variance explained, eigen weights, eigen index and the assets closest to
    it (Market Risk > Market Risk)."""
    returns = returns.dropna(how='all')
    num_components = max(1, min(num_components, returns.shape[1]))
    portfolio = RiskAnalysis(returns)
    eig_val, _, components = portfolio.pca(num_components=num_components)
    components = components.apply(np.real)
    if component not in components.columns:
        component = components.columns[0]

    eig_val = np.real(eig_val)
    weights = components[component] / components[component].sum()

    eigen_returns = pd.Series(returns.fillna(0).to_numpy() @ weights.to_numpy(), index=returns.index, name='PCA')
    eigen_returns.iloc[0] = 0

    comparison = returns.copy()
    comparison['PCA'] = eigen_returns
    distances = np.sqrt(((comparison.sub(eigen_returns, axis=0)) ** 2).sum(axis=0)).sort_values()
    similarity = comparison[distances.index[:num_closest]].copy()
    similarity.iloc[0] = 0

    return {
        'components': list(components.columns),
        'component': component,
        'variance_explained': pd.Series(eig_val / eig_val.sum(), index=components.columns),
        'eigen_weights': components[component].sort_values(ascending=False),
        'eigen_index': ((1 + eigen_returns).cumprod() * 100).to_frame('PCA'),
        'similarity': (1 + similarity).cumprod() * 100,
    }


def correlation_view(returns: pd.DataFrame, asset1: str, asset2: str, window: int = 252,
                     start=None, end=None) -> dict:
    """Rolling pair correlation, rolling mean returns, PC1's explained variance
    over time and the correlation matrix of [start, end] (Market Risk > Correlation).

    `returns` may start before `start`: the rolling charts then reach back one
    window, so they start on `start` whatever the date range."""
    for asset in (asset1, asset2):
        if asset not in returns.columns:
            raise ValueError(f"'{asset}' is not in the universe")
    if asset1 == asset2:
        raise ValueError("Select two different assets")
    start = returns.index[0] if start is None else start
    end = returns.index[-1] if end is None else end
    rolling = _lookback(returns, start, end, window)
    _check_window(rolling, window)

    return {
        'rolling_correlation': rolling[asset1].rolling(window).corr(rolling[asset2]).dropna()
        .to_frame(f"{asset1}/{asset2}"),
        'rolling_mean_returns': rolling.rolling(window).mean().dropna(how='all') * 252,
        'first_component': first_pca_over_time(rolling, window=window).rename(columns={0: 'PC1'}),
        'matrix': returns.loc[start:end].corr().round(2),
    }


def market_drivers_view(ws: Workspace, start, end, frequency: str = 'Quarterly', component: str = 'PC1',
                        window: int = 252, n_jobs: int = -1) -> dict:
    """Eigen-portfolio market index rebalanced at `frequency`, its drivers and
    risk decomposition, compared with the workspace series (Market Risk > Market Drivers).

    Like every workspace series, the index is simulated on the full price
    history and cropped, so its path does not depend on the date range. The
    drivers are the P&L since `start` and the decomposition's first window
    ends on `start`."""
    _check_history(ws, start, end, window)

    dates = get_rebalancing_dates(ws.returns, frequency=frequency)
    schedule = {}
    for i in range(len(dates) - 1):
        subset = ws.returns.loc[dates[i]:dates[i + 1]]
        if len(subset) < 2:
            continue
        _, _, components = RiskAnalysis(subset).pca(num_components=min(5, subset.shape[1]))
        if component in components.columns:
            schedule[subset.index[-1]] = np.round(np.real(components[component].to_numpy()), 6)
    if not schedule:
        raise ValueError("No valid eigen portfolio could be computed on this price history")

    weights = pd.DataFrame(schedule, index=ws.prices.columns).T
    quantities = rebalanced_dynamic_quantities(ws.prices, weights)
    market_portfolio = quantities * ws.prices
    market_index = market_portfolio.sum(axis=1)
    market_weights = market_portfolio.div(market_index, axis=0)

    comparison = pd.concat([market_index.pct_change(fill_method=None).rename('Market Index'),
                            ws.series_returns()], axis=1).loc[start:end]
    range_prices = ws.prices.loc[start:end]
    range_quantities = quantities.loc[range_prices.index]
    drivers = range_quantities * range_prices - rebalanced_book_cost(range_prices, range_quantities)
    drivers['Market Index'] = drivers.sum(axis=1)

    decomposition_weights = _lookback(market_weights, start, end, window)
    return {
        'performance': levels(comparison),
        'drivers': drivers,
        **get_ex_ante_vol_decomposition(decomposition_weights, ws.returns.loc[decomposition_weights.index], window,
                                        n_jobs=n_jobs),
        'weights': market_weights.loc[start:end],
    }
