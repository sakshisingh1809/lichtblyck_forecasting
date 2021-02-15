# -*- coding: utf-8 -*-

from .pfseries_pfframe import PfSeries, PfFrame
from .portfolio import SinglePf, MultiPf
import pandas as pd
import numpy as np


OK_COL_COMBOS = ["w", "q", "pr", "qr", "pq", "wp", "wr"]
OK_FREQ = ["AS", "QS", "MS", "D", "H", "15T"]


def get_index(tz="Europe/Berlin", freq="D"):
    """Get index with random length and starting point."""
    countdict = {"AS": 1, "QS": 4, "MS": 10, "D": 100, "H": 1000, "15T": 1000}
    count = countdict.get(freq, 1001)
    if count == 1001:
        count = countdict.get(freq + "S", 1001)
    periods = np.random.randint(count, count * 10)
    a, m, d = np.array([2016, 1, 1]) + np.random.randint(0, 12, 3)
    return pd.date_range(f"{a}-{m}-{d}", freq=freq, periods=periods, tz=tz)


def get_pfframe(i=None, columns="wp"):
    """Get PfFrame with index and certain columns."""
    if i is None:
        i = get_index()
    return PfFrame(np.random.rand(len(i), len(columns)), i, list(columns))


def get_pfseries(i=None, name="w"):
    """Get PfFrame with index and certain name."""
    if i is None:
        i = get_index()
    return PfSeries(np.random.rand(len(i)), i, name=name)


def get_singlepf(i=None, columns: str = "wp", name: str = "test"):
    """Get portfolio with index and certain name."""
    return SinglePf(get_pfframe(i, columns), name)


def get_multipf(i=None, columns: str = "wp", name: str = "test"):
    """Get portfolio with index and certain name."""
    return MultiPf([get_singlepf(i, columns, f"{name}/{c}") for c in range(3)], name)


def get_uniform_pfolio(i=None, levels: int = 2, name: str = "test"):
    """Get portfolio with index and certain number of levels."""
    if i is None:
        i = get_index()

    if levels == 1:  # return singlepf
        return SinglePf(get_pfframe(i), name)

    return MultiPf(
        [
            get_uniform_pfolio(i, levels - 1, f"{name}/{c}")
            for c in range(np.random.randint(1, 3))
        ],
        name,
    )
