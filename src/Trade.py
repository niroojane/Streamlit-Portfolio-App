# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Rebalancing trade optimizer (BQuant Trade / GradioLT).

From the current holdings w0 and their look-through (transparency) exposures,
find the weights closest to w0 -- min ||w - w0||^2, long-only, fully invested --
that satisfy limits on funds, on every weight ('All') and on exposure buckets
(exposure of the portfolio = w @ matrix, funds x buckets).

Workbook: a 'Holdings' sheet (Name, Portfolio Weighting %) and one sheet per
exposure category (Name, a weight column, then one column per bucket, in %).
"""
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .Exposure import exposure_constraints
from .RiskMetrics import build_constraint

HOLDINGS = 'Holdings'
WEIGHT = 'Portfolio Weighting %'
CASH = ('Cash EUR',)


@dataclass
class TradeBook:
    """Holdings and exposures of an uploaded workbook"""
    weights: pd.Series                  # initial weights (sum 1), index = fund names
    matrix: pd.DataFrame                # funds x buckets, fractions
    categories: Dict[str, List[str]]    # sheet -> its buckets (columns of `matrix`)
    sheets: Dict[str, pd.DataFrame] = field(default_factory=dict)   # raw sheets, for display
    messages: List[str] = field(default_factory=list)

    @property
    def funds(self) -> List[str]:
        return list(self.weights.index)

    @property
    def buckets(self) -> List[str]:
        return list(self.matrix.columns)

    @property
    def exposure(self) -> pd.DataFrame:
        """buckets x funds, the src.Exposure orientation"""
        return self.matrix.T


def read_trade_workbook(content, cash: Sequence[str] = CASH) -> TradeBook:
    if isinstance(content, (bytes, bytearray)):
        content = BytesIO(content)
    book = pd.ExcelFile(content)
    if HOLDINGS not in book.sheet_names:
        raise ValueError(f"The workbook needs a '{HOLDINGS}' sheet (Name, {WEIGHT})")
    sheets = {name: book.parse(name) for name in book.sheet_names}
    messages = []

    holdings = sheets[HOLDINGS].set_index('Name')
    if WEIGHT not in holdings.columns:
        raise ValueError(f"The '{HOLDINGS}' sheet needs a '{WEIGHT}' column")
    holdings = holdings.loc[~holdings.index.isin(cash)]
    weights = pd.to_numeric(holdings[WEIGHT], errors='coerce').fillna(0.0)
    if weights.sum() <= 0:
        raise ValueError("The holdings weights sum to zero")
    weights = weights / weights.sum()
    weights.index = weights.index.astype(str)

    frames, categories, seen = [], {}, set()
    for name, sheet in sheets.items():
        if name == HOLDINGS or 'Name' not in sheet.columns:
            continue
        frame = sheet.set_index('Name')
        frame.index = frame.index.astype(str)
        frame = frame.drop(columns=[c for c in frame.columns if 'Weighting %' in str(c)])
        frame = frame.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        frame = frame.loc[~frame.index.isin(cash)]
        missing = [f for f in weights.index if f not in frame.index]
        if missing:
            messages.append(f"{name}: no data for {len(missing)} holding(s), counted as 0")
        frame = frame.reindex(weights.index).fillna(0.0) / 100
        renamed = {c: (f"{c} ({name})" if c in seen else c) for c in frame.columns}
        seen.update(renamed.values())
        frame = frame.rename(columns=renamed)
        frames.append(frame)
        categories[name] = list(frame.columns)
    if not frames:
        raise ValueError("No exposure sheet found besides the holdings")
    matrix = pd.concat(frames, axis=1)
    return TradeBook(weights=weights, matrix=matrix, categories=categories, sheets=sheets, messages=messages)


def bucket_bounds(book: TradeBook) -> pd.DataFrame:
    """Lowest / highest exposure of any single fund to each bucket: the range a bucket
    limit can reach (outside it the constraint is infeasible)"""
    m = book.matrix
    return pd.DataFrame({'Lower Bound': m.min(), 'Upper Bound': m.max(), 'Name Min': m.idxmin(),
                         'Name Max': m.idxmax(), 'Current': book.weights @ m})


def optimize_trade(book: TradeBook, weight_rows: Sequence[Tuple[str, str, float]] = (),
                   bucket_rows: Sequence[Tuple[str, str, float]] = ()) -> pd.Series:
    """Weights closest to the current ones under the fund / 'All' / bucket limits"""
    w0 = book.weights.to_numpy(dtype=float)
    funds = pd.DataFrame(columns=book.funds)
    rows = [list(r) for r in weight_rows]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if rows:
        constraints += build_constraint(funds, np.array(rows, dtype=object))
    constraints += exposure_constraints(book.exposure, bucket_rows)

    result = minimize(lambda w: np.sum((w - w0) ** 2), w0, jac=lambda w: 2 * (w - w0), method='SLSQP',
                      bounds=[(0, 1)] * len(w0), constraints=constraints, options={'maxiter': 500})
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message} -- check the limits against the "
                         f"Transparency bounds")
    return pd.Series(np.clip(result.x, 0, None), index=book.funds, name='Optimized')


def comparison(book: TradeBook, optimized: Optional[pd.Series],
               custom: Optional[Dict[str, pd.Series]] = None) -> Dict[str, pd.DataFrame]:
    """Weights and exposures of Initial / Optimized / custom allocations, with their changes
    vs Initial: {'weights': funds table, <category>: buckets table}"""
    columns = {'Initial': book.weights}
    if optimized is not None:
        columns['Optimized'] = optimized
    for name, weights in (custom or {}).items():
        weights = weights.reindex(book.funds).fillna(0.0)
        columns[name] = weights / weights.sum() if weights.sum() else weights
    weights = pd.DataFrame(columns)
    for name in list(weights.columns[1:]):
        weights[f'Spread {name}' if name != 'Optimized' else 'Spread'] = weights[name] - weights['Initial']
    order = 'Spread' if 'Spread' in weights.columns else weights.columns[0]

    exposures = book.matrix.T @ weights
    out = {'weights': weights.sort_values(order, ascending=False)}
    for category, buckets in book.categories.items():
        out[category] = exposures.loc[buckets].sort_values(order, ascending=False)
    return out


def to_excel(book: TradeBook, tables: Dict[str, pd.DataFrame]) -> bytes:
    """Results workbook: weights, one sheet per category, the matrix and its bounds"""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        tables['weights'].to_excel(writer, sheet_name='Weights')
        for category in book.categories:
            if category in tables:
                tables[category].to_excel(writer, sheet_name=category[:31])
        book.matrix.to_excel(writer, sheet_name='Transparency Matrix')
        bucket_bounds(book).to_excel(writer, sheet_name='Sectors Bounds')
    return buffer.getvalue()
