# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Exposure (look-through / transparency) matrix: constraints and look-through.

An exposure table has one row per bucket (sector, asset class, country...) and
one column per asset; each cell is the share of the asset in the bucket. The
exposure of a portfolio to the buckets is `exposure @ weights`, so a bucket
constraint is linear in the weights and works with every objective.

Used by the BQuant App (the 'Transparency Table' sheet) and by the Streamlit /
Gradio apps when the uploaded workbook has a second sheet.
"""
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SIGN_TYPES = {'=': 'eq', '≥': 'ineq', '≤': 'ineq'}


def read_exposure(table: Optional[pd.DataFrame], assets: Sequence[str]) -> Optional[pd.DataFrame]:
    """Exposure table aligned on `assets` (missing assets -> 0), as fractions.

    Tables in percent (any |value| > 1.5, e.g. 35 for 35%) are divided by 100,
    as the BQuant App did with its Transparency Table.
    """
    if table is None or table.empty:
        return None
    table = table.apply(pd.to_numeric, errors='coerce')
    table = table.dropna(how='all').dropna(axis=1, how='all')
    table.index = table.index.astype(str)
    table.columns = table.columns.astype(str)
    common = [a for a in assets if a in table.columns]
    if not common:
        return None
    table = table.reindex(columns=list(assets)).fillna(0.0)
    if table.abs().to_numpy().max() > 1.5:
        table = table / 100
    return table


def exposure_constraints(exposure: Optional[pd.DataFrame],
                         rows: Iterable[Tuple[str, str, float]]) -> List[dict]:
    """SLSQP constraints from (bucket, sign, limit) rows; `exposure` columns must
    follow the optimizer's asset order."""
    if exposure is None:
        return []
    matrix = exposure.to_numpy(dtype=float)
    position = {name: i for i, name in enumerate(exposure.index)}
    constraints = []
    for bucket, sign, limit in rows:
        if bucket not in position or sign not in SIGN_TYPES:
            continue
        row = matrix[position[bucket]]
        limit = float(limit)
        if sign == '≤':
            fun = (lambda w, r=row, l=limit: l - r @ w)
        else:
            fun = (lambda w, r=row, l=limit: r @ w - l)
        constraints.append({'type': SIGN_TYPES[sign], 'fun': fun})
    return constraints


def bounds(exposure: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Reachable range of each bucket: the lowest / highest exposure of any single asset
    (a long-only, fully invested portfolio can't go outside it, so a bucket limit beyond
    it is infeasible), and the asset at each end"""
    if exposure is None or exposure.empty:
        return pd.DataFrame()
    return pd.DataFrame({'Lower Bound': exposure.min(axis=1), 'Upper Bound': exposure.max(axis=1),
                         'Name Min': exposure.idxmin(axis=1), 'Name Max': exposure.idxmax(axis=1)})


def lookthrough(exposure: Optional[pd.DataFrame], allocations: pd.DataFrame) -> pd.DataFrame:
    """Exposure of each allocation (columns) to each bucket (rows)."""
    if exposure is None or allocations is None or allocations.empty:
        return pd.DataFrame()
    weights = allocations.reindex(columns=exposure.columns).fillna(0.0)
    result = exposure @ weights.T
    result.loc['Total'] = result.sum(axis=0)
    result.columns.name = None
    return result


def exposure_history(exposure: Optional[pd.DataFrame], weights: pd.DataFrame) -> pd.DataFrame:
    """Daily exposure of a weight history (dates x assets) to each bucket."""
    if exposure is None or weights is None or weights.empty:
        return pd.DataFrame()
    aligned = weights.reindex(columns=exposure.columns).fillna(0.0)
    return aligned @ exposure.T
