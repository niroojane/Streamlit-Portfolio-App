# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Streamlit helpers shared by the tabs: page setup, the app context, caching,
editors, charts and messages."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import Workspace as workspace
from . import charts, engine

FONT = charts.FONT


# =============================================================================
# Page and context
# =============================================================================

def setup_page(title: str) -> None:
    st.set_page_config(page_title=title, layout="wide")
    st.markdown(
        f"""<style>html, body, .stApp, [class*="css"] {{ font-family: "{FONT}", Arial, sans-serif !important; }}</style>""",
        unsafe_allow_html=True,
    )
    st.title(title)


@dataclass
class Context:
    """Everything the tabs share during one script run"""
    prices: pd.DataFrame                    # cleaned universe prices, full history
    profile: engine.Profile
    exposure: Optional[pd.DataFrame] = None
    start: pd.Timestamp = None              # analysis window
    end: pd.Timestamp = None
    frequency: str = 'Monthly'
    window: int = 252                       # rolling window
    grid: pd.DataFrame = field(default_factory=pd.DataFrame)   # allocation grid (edited)
    benchmark: Optional[str] = None         # benchmark allocation (tracking error, relative objectives)
    constraints: Optional[pd.DataFrame] = None   # editor tables of the Strategy tab
    risk: Optional[pd.DataFrame] = None

    @property
    def window_prices(self) -> pd.DataFrame:
        return self.prices.loc[self.start:self.end]

    @property
    def window_returns(self) -> pd.DataFrame:
        return self.window_prices.pct_change(fill_method=None).iloc[1:]

    @property
    def bench_weights(self) -> Optional[pd.Series]:
        if self.benchmark and self.benchmark in self.grid.index:
            return self.grid.loc[self.benchmark]
        return None

    @property
    def backtest(self) -> Optional[dict]:
        return st.session_state.get('backtest')


DATA_KEYS = ('grid_base', 'grid_version', 'backtest', 'clicked', 'mean_shock', 'stress_grid')


def set_universe(prices: pd.DataFrame, profile: engine.Profile, exposure: Optional[pd.DataFrame] = None) -> None:
    """Store a new universe and reset everything computed on the previous one"""
    token = joblib.hash((prices, exposure, profile))
    if st.session_state.get('data_token') == token:
        return
    for key in DATA_KEYS:
        st.session_state.pop(key, None)
    st.session_state.update(data_token=token, prices=prices, profile=profile, exposure=exposure)


def universe() -> Optional[tuple]:
    if 'prices' not in st.session_state:
        return None
    return st.session_state.prices, st.session_state.profile, st.session_state.exposure


def settings(prices: pd.DataFrame) -> dict:
    """Defaults before the tabs draw: full history, frequency and rolling window of the Strategy tab"""
    window = int(st.session_state.get('rolling_window', min(252, max(20, len(prices) // 3))))
    return {'start': prices.index[0], 'end': prices.index[-1],
            'frequency': st.session_state.get('frequency', 'Monthly'), 'window': window}


def date_slider(ctx: 'Context', key: str) -> None:
    """'Date:' range slider at the top of a tab (as Risk Management/Streamlit_App.py): sets the
    window of this tab's computations. Keyed on the universe's dates, so a new universe resets it."""
    first, last = ctx.prices.index[0].to_pydatetime(), ctx.prices.index[-1].to_pydatetime()
    start, end = st.slider("Date:", min_value=first, max_value=last, value=(first, last), format="YYYY-MM-DD",
                           key=f"{key}_{first:%Y%m%d}_{last:%Y%m%d}")
    if start >= end:
        st.error("Pick a date range of at least two days")
        st.stop()
    ctx.start, ctx.end = pd.Timestamp(start), pd.Timestamp(end)


def require_data(ctx: Optional[Context]) -> bool:
    if ctx is None:
        st.info("Load data in the Investment Universe tab first ⬅️")
        return False
    return True


def require_grid(ctx: Context) -> bool:
    if ctx.grid is None or ctx.grid.empty:
        st.info("Click 'Optimize Portfolio' in the Strategy tab first")
        return False
    return True


# =============================================================================
# Caching
# =============================================================================

@st.cache_resource(show_spinner=False, max_entries=8)
def cached_workspace(prices, grid, profile, frequency, schedules):
    return engine.build_workspace(prices, grid, profile, frequency, schedules)


def get_workspace(ctx: Context):
    """(workspace, key) -- the key identifies the workspace in the cached views"""
    schedules = engine.strategy_schedules(ctx.backtest)
    ws = cached_workspace(ctx.prices, ctx.grid, ctx.profile, ctx.frequency, schedules)
    return ws, joblib.hash((ctx.prices, ctx.grid, ctx.profile, ctx.frequency, schedules))


@st.cache_data(show_spinner=False, max_entries=32)
def cached_view(view: str, ws_key: str, _ws, *args, **kwargs):
    """A src.Workspace view of the workspace `ws_key` (the workspace itself is not hashed)"""
    return getattr(workspace, view)(_ws, *args, **kwargs)


@st.cache_data(show_spinner=False, max_entries=32)
def cached(func_name: str, *args, **kwargs):
    """Cached call of an engine / src function given by 'module.function'"""
    import importlib
    module, name = func_name.rsplit('.', 1)
    return getattr(importlib.import_module(module), name)(*args, **kwargs)


def on_click(label: str, key: str, params, help: Optional[str] = None, container=None) -> bool:
    """Button whose result stays displayed on later reruns, as long as `params` don't change"""
    clicked = st.session_state.setdefault('clicked', {})
    token = joblib.hash(params)
    if (container or st).button(label, key=f"btn_{key}", help=help):
        clicked[key] = token
    if key in clicked and clicked[key] != token:
        st.caption("Settings changed: click again to update.")
        return False
    return clicked.get(key) == token


# =============================================================================
# Messages and tables
# =============================================================================

def show_messages(messages: Sequence[str]) -> None:
    for message in dict.fromkeys(messages or []):
        if message.startswith("ℹ️"):
            st.info(message[2:].strip(), icon="ℹ️")
        else:
            st.warning(message, icon="⚠️")


def run(label: str, func, *args, **kwargs):
    """Call `func`, showing invalid inputs as an error instead of a traceback"""
    try:
        with st.spinner(label, show_time=True):
            return func(*args, **kwargs)
    except (ValueError, np.linalg.LinAlgError, KeyError) as exc:
        st.error(str(exc))
        return None


def _arrow_safe(frame: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Numbers rounded; a column mixing numbers and dates (e.g. the 'Date of Max Drawdown' row)
    becomes text so st.dataframe shows it without Arrow conversion warnings"""
    frame = frame.copy()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    if isinstance(frame.index, pd.DatetimeIndex) and (frame.index == frame.index.normalize()).all():
        frame.index = frame.index.date                      # 2024-01-31, not 2024-01-31 00:00:00
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].round(decimals)
        elif frame[col].dtype == object:
            frame[col] = frame[col].map(lambda v: round(v, decimals) if isinstance(v, (float, np.floating))
                                        else v).astype(str).replace({'nan': ''})
    return frame


def table(frame: pd.DataFrame, title: Optional[str] = None, decimals: int = 4, height: Optional[int] = None) -> None:
    """st.dataframe, as Risk Management/Streamlit_App.py"""
    if title:
        st.subheader(title)
    if frame is None or (hasattr(frame, 'empty') and frame.empty):
        st.caption("No data")
        return
    if height:
        st.dataframe(_arrow_safe(frame, decimals), width='stretch', height=height)
    else:
        st.dataframe(_arrow_safe(frame, decimals), width='stretch')


# =============================================================================
# Editors
# =============================================================================

def editor(key: str, columns: Dict[str, object], help: Optional[str] = None) -> pd.DataFrame:
    """Dynamic-row editor of constraint-like rows; `columns` maps names to column configs"""
    empty = pd.DataFrame({name: pd.Series(dtype='float' if name == 'Limit' else 'object') for name in columns})
    if help:
        st.caption(help)
    return st.data_editor(empty, column_config=columns, num_rows="dynamic", key=key, width='stretch',
                          hide_index=True)


def constraint_editor(ctx_assets: Sequence[str], exposure: Optional[pd.DataFrame]) -> pd.DataFrame:
    return editor('constraints_editor', {
        'Target': st.column_config.SelectboxColumn("Asset / Bucket",
                                                   options=engine.constraint_targets(ctx_assets, exposure)),
        'Sign': st.column_config.SelectboxColumn(options=engine.SIGNS),
        'Limit': st.column_config.NumberColumn(format="%.4f"),
    }, help="'All' bounds every weight; exposure buckets bound the look-through exposure. Limits in decimals.")


def risk_editor() -> pd.DataFrame:
    return editor('risk_editor', {
        'Risk': st.column_config.SelectboxColumn(options=list(engine.RISK_TYPES)),
        'Sign': st.column_config.SelectboxColumn(options=engine.SIGNS),
        'Limit': st.column_config.NumberColumn(format="%.4f"),
    }, help="Ex-ante, annualized. Beta / Tracking Error / Information Ratio are measured against the benchmark "
            "allocation.")


def overlay_editor() -> pd.DataFrame:
    return editor('overlay_editor', {
        'Strategy': st.column_config.SelectboxColumn(options=list(engine.STRATEGIES)),
        'Limit': st.column_config.NumberColumn("Weight", format="%.2f", min_value=0.0, max_value=1.0),
    }, help="Tactical overlays blended into the core strategy with these weights (the core keeps the rest).")


# =============================================================================
# Charts
# =============================================================================

def _plot(fig) -> None:
    st.plotly_chart(fig, width='stretch')


def line(frame: pd.DataFrame, title: str, highlight: Optional[Sequence[str]] = None, pct: bool = False,
         height: int = 420) -> None:
    """Line chart; only the `highlight` series are drawn at first, the others sit in the legend"""
    if frame is None or frame.empty:
        st.caption(f"{title}: no data")
        return
    _plot(charts.line(frame, title, highlight, pct, height))


def bars(frame: pd.DataFrame, title: str, pct: bool = False, barmode: str = 'group') -> None:
    if frame is None or frame.empty:
        st.caption(f"{title}: no data")
        return
    _plot(charts.bars(frame, title, pct, barmode))


def heatmap(frame: pd.DataFrame, title: str, decimals: int = 2) -> None:
    _plot(charts.heatmap(frame, title, decimals))


def frontier_figure(frontier: dict):
    return charts.frontier(frontier)


def histogram(counts: List[int], edges: List[float], var: float, title: str) -> None:
    _plot(charts.histogram(counts, edges, var, title))


def pair(layout) -> None:
    """Draw a charts.*_layout (left figures, right figures) side by side, as the notebook's HBox"""
    left, right = layout
    col1, col2 = st.columns(2)
    with col1:
        for fig in left:
            _plot(fig)
    with col2:
        for fig in right:
            _plot(fig)


def two_columns():
    return st.columns([1, 1])
