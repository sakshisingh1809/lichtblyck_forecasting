"""
Code to quickly get objects for testing.
"""

from .pfline import PfLine
import pandas as pd
import numpy as np


OK_COL_COMBOS = ["w", "q", "p", "pr", "qr", "pq", "wp", "wr"]


def get_index(tz="Europe/Berlin", freq="D", start=None) -> pd.DatetimeIndex:
    """Get index with random length and starting point (but always start at midnight)."""
    count = {"AS": 4, "QS": 4, "MS": 10, "D": 100, "H": 1000, "15T": 1000}.get(freq, 10)
    periods = np.random.randint(count, count * 3)
    if not start:
        a, m, d = np.array([2016, 1, 1]) + np.random.randint(0, 12, 3)  # each + 0..11
        start = f"{a}-{m}-{d}"
    return pd.date_range(start, freq=freq, periods=periods, tz=tz)


def get_series(i=None, name="w", factor=1) -> pd.Series:
    """Get Series with index and certain name."""
    if i is None:
        i = get_index()
    return pd.Series((1 + np.random.rand(len(i))) * 10 * factor, i, name=name)


def get_dataframe(
    i=None, columns="wp", factors={"w": 1, "q": 1, "p": 1, "r": 1}
) -> pd.DataFrame:
    """Get DataFrame with index and certain columns. Columns (e.g. `q` and `w`) are not made consistent."""
    if i is None:
        i = get_index()
    return pd.DataFrame(
        {col: get_series(i, col, factors.get(col, 1)) for col in columns}
    )


# Portfolio line.


def get_pfline(i=None, columns: str = "wp", name: str = "test"):
    """Get portfolio line, i.e. without children."""
    return PfLine(get_dataframe(i, columns))
