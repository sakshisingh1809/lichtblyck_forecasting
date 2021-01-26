"""Testing Portfolio."""

from lichtblyck import PfFrame, PfSeries, Portfolio
import pandas as pd
import numpy as np
import pytest


def get_index(tz, freq):
    count = {"M": 10, "D": 100, "H": 1000, "15T": 1000}[freq]
    periods = np.random.randint(count, count * 10)
    shift = np.random.randint(0, 3)
    i = pd.date_range("2020-01-01", freq=freq, periods=periods, tz=tz)
    return i + shift * i.freq


def get_pfframe(i, columns):
    return PfFrame(np.random.rand(len(i), len(columns)), i, list(columns))


def get_pfseries(i):
    return PfSeries(np.random.rand(len(i)), i)


def get_own(i, columns):
    choices = [
        get_pfframe(i, columns),
        {c: get_pfseries(i) for c in columns},
        [get_pfseries(i) for c in columns],
    ]
    pick = np.random.randint(len(choices))
    return choices[pick]


def get_portfolio(i, levels: int, parentname: str = "", num: int = 0):
    """Get portfolio with given index, randomly chosing an initialisation method."""
    if levels == 0:
        hasown = True
        childcount = 0
    else:
        hasown = np.random.choice([True, False])
        childcount = np.random.randint(1, 3)
    own = get_own(i, "wp") if hasown else None
    name = parentname + "/" + str(num)
    return Portfolio(
        name,
        own,
        children=[get_portfolio(i, levels - 1, name, n) for n in range(childcount)],
    )


def test_equallength():
    pfolio = get_portfolio(get_index("Europe/Berlin", "D"), 2)
    pass
