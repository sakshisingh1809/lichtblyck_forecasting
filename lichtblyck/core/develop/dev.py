"""
Code to quickly get objects for testing.
"""

from typing import Dict
from ...tools.nits import ureg, PA_, name2unit
from ..pfline import PfLine, SinglePfLine, MultiPfLine
from ..pfstate import PfState
from . import mockup
import pandas as pd
import numpy as np


OK_COL_COMBOS = ["w", "q", "p", "pr", "qr", "qp", "wp", "wr"]


def get_index(freq="D", tz="Europe/Berlin", start=None) -> pd.DatetimeIndex:
    """Get index with random length and starting point (but always start at midnight)."""
    count = {"AS": 4, "QS": 4, "MS": 10, "D": 100, "H": 1000, "15T": 1000}.get(freq, 10)
    periods = np.random.randint(count, count * 3)
    if not start:
        a, m, d = np.array([2016, 1, 1]) + np.random.randint(0, 12, 3)  # each + 0..11
        if freq in ["15T", "H"] and tz is None:
            # make sure not to return a non-extistent time, so don't include DST-transition.
            m, d = 4, 2
            count = min(count, 4000)
        start = f"{a}-{m}-{d}"
    return pd.date_range(start, freq=freq, periods=periods, tz=tz)


def get_series(i=None, name="w", has_unit: bool = True) -> pd.Series:
    """Get Series with index `i` and name `name`. Values from mock-up functions (if in
    'wqpr') or random between 100 and 200."""
    if i is None:
        i = get_index()
    i.name = "ts_left"
    u = name2unit(name)
    if name == "w":
        # random average, and 3 random amplitudes with sum < 1
        avg = (10 + 30 * np.random.random()) * np.random.choice([1, 10, 100])
        ampls = np.random.rand(3) * np.array([0.3, 0.2, 0.1])
        return mockup.w_offtake(i, avg, *ampls, has_unit=has_unit)
    elif name == "q":
        q = get_series(i, "w", True) * i.duration
        q = q.rename("q")
        return q if has_unit else q.pint.m
    elif name == "p":
        # random average, and 3 random amplitudes with sum < 1
        avg = 50 + 100 * np.random.random()
        ampls = np.random.rand(3) * np.array([0.25, 0.04, 0.3])
        return mockup.p_marketprices(i, avg, *ampls, has_unit=has_unit)
    elif name == "r":
        r = get_series(i, "q", has_unit) * get_series(i, "p", has_unit)
        return r.rename("r")
    else:
        return pd.Series(100 + 100 * np.random.rand(len(i)), i, name=name)


def get_dataframe(i=None, columns="wp", has_unit: bool = True,) -> pd.DataFrame:
    """Get DataFrame with index `i` and columns `columns`. Columns (e.g. `q` and `w`)
    are not made consistent."""
    if i is None:
        i = get_index()
    i.name = "ts_left"
    series = {col: get_series(i, col, has_unit) for col in columns}
    return pd.DataFrame(series)


# Portfolio line.


def get_singlepfline(i=None, kind: str = "all") -> SinglePfLine:
    """Get single portfolio line, i.e. without children."""
    columns = {"q": "q", "p": "p", "all": "qr"}[kind]
    return SinglePfLine(get_dataframe(i, columns))


def get_multipfline(i=None, kind: str = "all") -> MultiPfLine:
    """Get multi portfolio line. With 1 level of 2 children."""
    if i is None:
        i = get_index()
    return MultiPfLine({"A": get_singlepfline(i, kind), "B": get_singlepfline(i, kind)})


def get_pfline(
    i=None,
    kind: str = "all",
    max_nlevels: int = 3,
    childcount: int = 2,
    prefix: str = "",
) -> PfLine:
    """Get portfolio line, without children or with children in random number of levels."""
    # Gather information.
    if i is None:
        i = get_index()
    nlevels = np.random.randint(0, max_nlevels)
    # Create single PfLine
    if nlevels == 0:
        return get_singlepfline(i, kind)
    # Gather information.
    if childcount == 2 and kind == "all" and np.random.rand() < 0.33:
        kinds = ["p", "q"]
    else:
        kinds = [kind] * childcount
    # Create multi PfLine.
    children = {}
    for c, knd in enumerate(kinds):
        name = f"part {prefix}{c}."
        children[name] = get_pfline(i, knd, max_nlevels - 1, prefix=f"{prefix}{c}.")
    return MultiPfLine(children)


# Portfolio state.


def get_pfstate(i=None) -> PfState:
    """Get portfolio state."""
    if i is None:
        i = get_index()
<<<<<<< HEAD
    wo = (-1) * get_singlepfline(i, "q").w
    pu = get_singlepfline(i, "p").p
=======
    wo = -get_singlepfline(i, "q")
    pu = get_singlepfline(i, "p")
>>>>>>> dec9123bea779b547b9483d9fe1508af046fd0de
    ws, ps = mockup.wp_sourced(wo)
    return PfState.from_series(wo=wo, pu=pu, ws=ws, ps=ps)


def get_pfstates(num=3, i=None) -> Dict[str, PfState]:
    """Get dictionary of portfolio states."""
    sample = ["Ludwig", "P2Heat", "B2B", "B2C Legacy", "Spot procurement", "B2B New"]
    return {name: get_pfstate(i if i else get_index()) for name in sample[:num]}
