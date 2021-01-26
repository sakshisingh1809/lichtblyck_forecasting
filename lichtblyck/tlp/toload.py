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
    tlp: Callable, t: pd.Series, weights: Iterable = None, freq: str = None
) -> pd.Series:
    """
    Calculate actual load of temperature-dependent load profile.

    Smooth temperature timeseries, resample to certain frequency, and calculate
    the load at each timestamp / temperature.

    Parameters
    ----------
    tlp : Callable
        Function to calculate the load in [MW]. Must take one input value
        (temperature in [degC]) or two input values (temperature in [degC] and
        timestamp).
    t : pd.Series
        Temperature values (in [degC]) timeseries.
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
        load = [tlp(tmpr, ts) for tmpr, ts in zip(t.values, t.index)]
    except TypeError:  # only temperature
        load = [tlp(tmpr) for tmpr in t.values]

    return pd.Series(load, t.index, name="w")


def tmpr2load(tlp: Union[pd.Series, Callable], t: pd.Series, spec) -> pd.Series:
    """
    Calculate actual electric load for temperature-dependent load profile.

    Parameters
    ----------
    tlp : pd.Series
        Standardized temperature-dependent load profile; with multilevel
        index. level 0: time-of-day timestamp. Level 1: temperature in [degC].
        Values: load in [K/h] at given time and temperature.
    t : pd.Series
        Temperature values (in [degC]). Index = date timestamp.
    spec : float
        Specific electrical load [kWh/K] with which to scale the profile.
        Describes the heating energy needed by the customer during a single
        day, per degC that the average outdoor temperature of that day is
        below a certain set reference value.

    Returns
    -------
    pd.Series
        Timeseries with the electrical load in MW. The length of the timeseries
        is that of `t`, and its resolution it that of `tlp`.

    Notes
    -----
    Temperature timeseries `t` most sensibly has daily values. Another freq
    may also be provided; in all cases, a value is valid until it's updated
    by the next-later timestamp.
    """
    t_available = tlp.index.get_level_values(1).unique()

    def nearest_available(t):
        idx = (np.abs(t_available - t)).argmin()
        return t_available[idx]

    time_available = tlp.index.get_level_values(0).unique().sort_values()
    timedelta = [
        datetime.timedelta(hours=t1.hour, minutes=t1.minute, seconds=t1.second)
        for t1 in time_available
    ]
    freq = (max(timedelta) - min(timedelta)) / (len(timedelta) - 1)
    new_timestamps = pd.date_range(
        t.index[0],
        t.index[-1] + datetime.timedelta(1),
        freq=freq,
        closed="left",
        tz="Europe/Berlin",
        name="ts_left",
    )

    # Put into correct time resolution.
    df = pd.DataFrame({"t": t})
    df["t_avail"] = df["t"].apply(nearest_available)
    # df['date'] = df.index.map(lambda ts: ts.date)
    df = df.reindex(new_timestamps).ffill()
    df["time"] = df.index.map(lambda ts: ts.time)

    # Add corresponding standardized load.
    merged = df.merge(tlp.rename("tlp"), left_on=("time", "t_avail"), right_index=True)
    merged = merged[["t", "tlp"]]
    merged.sort_index(inplace=True)

    # Convert into actual power.
    load = merged["tlp"] * spec * 0.001  # kW to MW
    load.index.freq = pd.infer_freq(load.index)
    # TODO: don't just use current temperature, but rather also yesterday's as described in standard process.
    return load.rename("w")
