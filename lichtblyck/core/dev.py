"""
Code to quickly get objects for testing.
"""

from typing import Dict
from ..tools.nits import ureg, PA_, name2unit
from .pfline.pfline____archive import PfLine
from .pfstate import PfState
import pandas as pd
import numpy as np


OK_COL_COMBOS = ["w", "q", "p", "pr", "qr", "pq", "wp", "wr"]


def get_index(freq="D", tz="Europe/Berlin", start=None) -> pd.DatetimeIndex:
    """Get index with random length and starting point (but always start at midnight)."""
    count = {"AS": 4, "QS": 4, "MS": 10, "D": 100, "H": 1000, "15T": 1000}.get(freq, 10)
    periods = np.random.randint(count, count * 3)
    if not start:
        a, m, d = np.array([2016, 1, 1]) + np.random.randint(0, 12, 3)  # each + 0..11
        start = f"{a}-{m}-{d}"
    return pd.date_range(start, freq=freq, periods=periods, tz=tz)


def get_series(i=None, name="w", min=10, max=20) -> pd.Series:
    """Get Series with index `i` and name `name`. Values between min (default: 10) and
    max (default: 20)."""
    if i is None:
        i = get_index()
    u = name2unit(name)
    return pd.Series(min + np.random.rand(len(i)) * (max - min), i, name=name).astype(
        f"pint[{u}]"
    )


def get_dataframe(
    i=None,
    columns="wp",
    min={"w": 100, "q": 1000, "p": 40, "r": 40_000},
    max={"w": 150, "q": 1500, "p": 80, "r": 120_000},
) -> pd.DataFrame:
    """Get DataFrame with index `i` and columns `columns`. Columns (e.g. `q` and `w`)
    are not made consistent."""
    if i is None:
        i = get_index()
    return pd.DataFrame(
        {col: get_series(i, col, min.get(col, 10), max.get(col, 20)) for col in columns}
    )


# Portfolio line.


def get_pfline(i=None, kind: str = "all") -> PfLine:
    """Get portfolio line, i.e. without children."""
    columns = {"q": "q", "p": "p", "all": "qr"}[kind]
    return PfLine(get_dataframe(i, columns))


# Portfolio state.


def get_pfstate(i=None) -> PfState:
    """Get portfolio state."""
    if i is None:
        i = get_index()
    offtakevolume = get_pfline(i, "q") * -2
    unsourcedprice = get_pfline(i, "p") * 2
    sourced = get_pfline(i, "all")
    return PfState(offtakevolume, unsourcedprice, sourced)


def get_pfstates(num=3, i=None) -> Dict[str, PfState]:
    """Get dictionary of portfolio states."""
    sample = ["Ludwig", "P2Heat", "B2B", "B2C Legacy", "Spot procurement", "B2B New"]
    return {name: get_pfstate(i if i else get_index()) for name in sample[:num]}
