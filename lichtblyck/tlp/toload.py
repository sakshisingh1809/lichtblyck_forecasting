# -*- coding: utf-8 -*-
"""
Converting temperature load profiles into loads with help of a temperature 
timeseries.
"""

import pandas as pd
import numpy as np
import datetime
from typing import Callable, Union, Iterable
import functools


def _smooth_temperature(t: pd.Series, weights: Iterable) -> pd.Series:
    """
    Smooth a temperature timeseries with past values.

    Parameters
    ----------
    t : pd.Series
        Temperature timeseries.
    weights : Iterable
        Smoothing weights. Current temperature is weighted with first weight,
        previous temperature with second weight, one before that with third
        weight, etc.

    Returns
    -------
    pd.Series
        Smoothed temperature timeseries.
    """
    flippedweights = np.flip(weights)

    def smooth(df):
        try:  # usually, no error
            return df.wavg(flippedweights)
        except ValueError:  # in the beginning, lengths aren't equal
            return df.wavg(np.flip(weights[: len(df)]))

    return t.rolling(len(weights), min_periods=1).apply(smooth)


def w(
    t: pd.Series, tlp: Callable, weights: Iterable = None, freq: str = None
) -> pd.Series:
    """
    Calculate actual load of temperature-dependent load profile.

    Smooth temperature timeseries, resample to certain frequency, and calculate
    the load at each timestamp / temperature.

    Parameters
    ----------
    t : pd.Series
        Temperature values (in [degC]) timeseries.
    tlp : Callable
        Function to calculate the load in [MW]. Must take one input value
        (temperature in [degC]) or two input values (temperature in [degC] and
        timestamp).
    weights : Iterable, optional
        Smoothing weights, used to include previous temperatures into calculation
        of current load. The default is None.
    freq : TYPE, optional
        The frequency of the returned timeseries. If no value is provided, no
        resampling is done (i.e., the index of `t` is used as-is).

    Returns
    -------
    pd.Series
        Timeseries with the load in MW. The length of the timeseries is that of
        `t`; the resolution is `freq`.
    """
    if weights is not None:
        t = _smooth_temperature(t, weights)
    if freq is not None:
        t = t.resample(freq).asfreq()
    # TODO: performance improvement: resample t before tlp(tmpr, ts) but after tlp(tmpr).

    tlp = functools.lru_cache(maxsize=8192)(tlp)
    try:  # try temperature and timestamp
        tlp = np.vectorize(tlp)
        load = tlp(t.values, t.index)
        # load = [tlp(tmpr, ts) for tmpr, ts in zip(t.values, t.index)]
        # load = [tlp(tmpr, time) for tmpr, time in zip(t.values, t.index.time)]
    except TypeError:  # only temperature
        load = [tlp(tmpr) for tmpr in t.values]

    return pd.Series(load, t.index, name="w")
