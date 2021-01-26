# -*- coding: utf-8 -*-
"""
Convert between various ways that tlp profiles are stored / communicated. Con-
vert to function consumption[MW] = f(temperature[degC], timestamp).
"""

import functools
import datetime
import pandas as pd
import numpy as np
from typing import Callable


def series2function(tlp_s: pd.Series) -> Callable:
    """
    Convert tlp-series into tlp-function.

    Parameters
    ----------
    tlp_s : pd.Series
        Series with temperature [degC] as first index level and time-of-day
        as second index level, and consumption [MW] as values.

    Returns
    -------
    Callable
        Function that takes temperature [degC] as first argument and timestamp
        as second argument, and returns the consumption [MW].
    """
    # To find corresponding temperature.
    t_available = tlp_s.index.get_level_values(0).unique()

    @functools.lru_cache(maxsize=128)
    def corresponding_t(t):
        return t_available[np.abs(t_available - t).argmin()]

    # To find corresponding time.
    time_available = tlp_s.index.get_level_values(1).unique().sort_values()

    @functools.lru_cache(maxsize=128)
    def corresponding_time(time):
        i = np.searchsorted(time_available, time, "right") - 1
        if i < 0:
            i = 0  # use first value, even if provided time is earlier.
        return time_available[i]

    # Return function. (Cannot cache here because timestamp (not time) argument.)
    def tlp(t: float, ts: datetime.datetime) -> float:
        time = ts.time()
        return tlp_s[(corresponding_t(t), corresponding_time(time))]

    return tlp


def cached_tlp(f: Callable) -> Callable:
    def time(ts):
        return ts.time()

    def weekday(ts):
        return ts.weekday()

    def month(ts):
        return ts.month

    cache = {}

    def f_cached(t, ts):
        params = (t, time(ts), weekday(ts), month(ts))
        try:
            return cache[params]
        except KeyError:
            cache[params] = f(t, ts)
            return cache[params]

    return f_cached
