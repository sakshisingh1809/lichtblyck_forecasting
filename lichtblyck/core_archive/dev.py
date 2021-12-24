# -*- coding: utf-8 -*-

from . import basics
from .singlepf_multipf import SinglePf, MultiPf
from .lbpf import LbPf
import pandas as pd
import numpy as np


OK_COL_COMBOS = ["w", "q", "pr", "qr", "pq", "wp", "wr"]
FREQUENCIES = basics.FREQUENCIES


def get_index(tz="Europe/Berlin", freq="D") -> pd.DatetimeIndex:
    """Get index with random length and starting point (but always start at midnight)."""
    if freq not in FREQUENCIES:
        raise ValueError(f"`freq` must be one of {','.join(FREQUENCIES)}.")
    count = {"AS": 1, "QS": 4, "MS": 10, "D": 100, "H": 1000, "15T": 1000}[freq]
    periods = np.random.randint(count, count * 10)
    a, m, d = np.array([2016, 1, 1]) + np.random.randint(0, 12, 3)  # add 0-11 to each
    return pd.date_range(f"{a}-{m}-{d}", freq=freq, periods=periods, tz=tz)


def get_series(i=None, name="w", factor=1) -> pd.Series:
    """Get Series with index and certain name."""
    if i is None:
        i = get_index()
    return pd.Series(np.random.rand(len(i)) * 10 * factor, i, name=name)


def get_dataframe(
    i=None, columns="wp", factors={"w": 1, "q": 1, "p": 1, "r": 1}
) -> pd.DataFrame:
    """Get DataFrame with index and certain columns. Columns (e.g. `q` and `w`) are not made consistent."""
    if i is None:
        i = get_index()
    return pd.DataFrame(
        {col: get_series(i, col, factors.get(col, 1)) for col in columns}
    )


# Single portfolio line.


def get_pfline(i=None, columns: str = "wp", name: str = "test"):
    """Get portfolio line, i.e. without children."""
    return SinglePf(get_dataframe(i, columns), name=name)


# Multi portfolio.


def get_multipf_standardcase(i=None, levels: int = 2, name: str = "test"):
    """Get portfolio with children. Each leaf has full info (w, p, q, r)"""
    if i is None:
        i = get_index()

    if levels == 1:
        return get_singlepf(i, "wp", name)

    return MultiPf(
        [
            get_multipf_standardcase(i, levels - 1, f"{name}/{c}")
            for c in range(np.random.randint(1, 3))
        ],
        name=name,
    )


def get_multipf_allcases(i=None, levels: int = 2, name: str = "test"):
    """Get portfolio with children. Each leaf has full or partial info (at least volume,
    possibly also price)."""
    if i is None:
        i = get_index()

    if levels == 1:
        choices = [
            *OK_COL_COMBOS[:2],
            *OK_COL_COMBOS[:2],
            *OK_COL_COMBOS,
        ]  # includes revenue in 50% of cases
        return get_singlepf(i, np.random.choice(choices), name)

    return MultiPf(
        [
            get_multipf_allcases(i, levels - 1, f"{name}/{c}")
            for c in range(np.random.randint(1, 3))
        ],
        name=name,
    )


# Lichtblick portfolio.


def get_lbpf_nosubs(i=None, own: str = "os", name: str = "test"):
    """Get lichtblick portfolio without children.
    Own timeseries as specified by 'own'."""
    if i is None:
        i = get_index()

    if "o" not in own:
        offtake = None
    else:
        offtake = SinglePf(-get_dataframe(i, "q"), name="Offtake")

    if "s" not in own:
        sourced = None
    else:
        parts = [SinglePf(get_dataframe(i, "qr"), name="Forward")]
        if np.random.rand() < 0.5:  # Sometimes only forward, sometimes also spot.
            parts[0] *= 0.8
            parts.append(SinglePf(get_dataframe(i, "qr"), name="Spot") * 0.2)
        sourced = MultiPf(parts, name="sourced")

    return LbPf(offtake=offtake, sourced=sourced, name=name)


def get_lbpf_subs_standardcase(i=None, levels: int = 2, name: str = "test"):
    """Get lichtblick portfolio with children.
    Own timeseries (offtake and sourced) only on leaf porfolios."""
    if i is None:
        i = get_index()

    if levels == 1:
        pf = get_lbpf_nosubs(i, "os", name=name)
    else:
        pf = get_lbpf_nosubs(i, "", name=name)
        for c in range(np.random.randint(1, 3)):
            pf.add_child(get_lbpf_subs_standardcase(i, levels - 1, f"{name}/{c}"))

    return pf


def get_lbpf_subs_allcases(i=None, levels: int = 2, name: str = "test"):
    """Get lichtblick portfolio with children.
    Own timeseries on all leaf porfolios and some intermediate portfolios.
    Own timeseries may have offtake, sourced, or both."""
    if i is None:
        i = get_index()

    choices = ["o", "s", "os"]
    if levels > 1:
        choices.append("")
    pf = get_lbpf_nosubs(i, np.random.choice(choices), name)
    if levels > 1:
        for c in range(np.random.randint(1, 3)):
            pf.add_child(get_lbpf_subs_allcases(i, levels - 1, f"{name}/{c}"))

    return pf
