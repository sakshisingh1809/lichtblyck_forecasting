# -*- coding: utf-8 -*-
"""
Plotting of tlp functions.
"""

from typing import Callable, Iterable
import portfolyo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def vs_time(tlp: Callable, t: Iterable = (-30, 30), freq: str = "15T") -> plt.Axes:
    """
    Plot tlp consumption (y) thoughout a single day (x) for various temperatures.

    Parameters
    ----------
    tlp : Callable[pd.Series]
        Function that takes temperature [degC] timeseries as argument, and
        returns the consumption [MW] timeseries.
    t : Iterable, optional
        Temperatures for which a curve must be plot: (min, max, [step=1]). Only
        distinct curves are actually drawn. The default is (-30, 30).
    freq : str
        Time resolution (frequency) in the graph. The default is '15T'.

    Returns
    ------
    plt.Axes
        Pyplot Axes object containing the plot.
    """
    ts = pd.date_range(
        "2020-01-01", "2020-01-02", freq=freq, closed="left", tz="Europe/Berlin"
    )
    y = {tmpr: tlp(pd.Series(tmpr, ts)).values for tmpr in np.arange(*t)}
    i = len(y) // 2
    df = pd.DataFrame(y, ts)
    df_draw = pd.concat(
        [
            df.iloc[:, :i].T.drop_duplicates(keep="last"),
            df.iloc[:, i:].T.drop_duplicates(keep="first"),
        ]
    ).T  # in case exact same curves are drawn multiple times.
    ax = df_draw.plot(cmap="coolwarm", figsize=(10, 6))
    # myFmt = mdates.DateFormatter('%H:%M')
    # ax.xaxis.set_major_formatter(myFmt)
    ax.legend(ncol=3)
    ax.set_ylabel("load [MW]")
    ax.set_xlabel("time of day")
    return ax


def vs_t(tlp: Callable, t: Iterable = (-30, 30)) -> plt.Axes:
    """
    Plot tlp consumption (y) as a function of the temperature (x).

    Parameters
    ----------
    tlp : Callable
        Function that takes temperature [degC] timeseries as argument and returns the
        offtake [MW].
    t : Iterable, optional. Default is (-30, 30).
        Temperatures for which the curve must be plot: (min, max, [step=1]).

    Returns
    -------
    plt.Axes
        Pyplot Axes object containing the plot.
    """
    t1 = np.linspace(*t)
    t2 = pd.Series(
        t1, pd.date_range("2020-01-01", freq="D", periods=len(t1), tz="Europe/Berlin")
    )
    y = tlp(t2).resample("D").mean()
    s = pd.Series(y.values, t2.values)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s)
    ax.legend(ncol=3)
    ax.set_ylabel("load [MW]")
    ax.set_xlabel("temperature [degC]")
    return ax
