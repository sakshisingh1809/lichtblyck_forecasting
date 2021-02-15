# -*- coding: utf-8 -*-
"""
Plotting of tlp functions.
"""
from ..core.pfseries_pfframe import PfSeries
from . import toload
from typing import Callable, Iterable
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    y = {tmpr: tlp(PfSeries(tmpr, ts)).values for tmpr in np.arange(*t)}
    i = len(y) // 2
    df = pd.DataFrame(y, ts)
    df_draw = pd.concat(
        [
            df.iloc[:, :i].T.drop_duplicates(keep="last"),
            df.iloc[:, i:].T.drop_duplicates(keep="first"),
        ]
    ).T  # in case exact same curves are drawn multiple times.
    ax = df_draw.plot(cmap="coolwarm")
    # myFmt = mdates.DateFormatter('%H:%M')
    # ax.xaxis.set_major_formatter(myFmt)
    ax.set_ylabel("load [MW]")
    ax.set_xlabel("time of day")
    return ax


def vs_t(tlp: Callable, t: Iterable = (-30, 30)) -> plt.Axes:
    """
    Plot tlp consumption (y) as a function of the temperature (x).

    Parameters
    ----------
    tlp : Callable
        Function that takes temperature [degC] as first argument (and possibly
        timestamp as second argument), and returns the consumption [MW].
    t : Iterable, optional
        Temperatures for which the curve must be plot: (min, max, [step=1]).
        The default is (-30, 30).

    Returns
    -------
    plt.Axes
        Pyplot Axes object containing the plot.
    """
    t = pd.Series(np.linspace(*t))
    y = tlp(t)
    ax = pd.Series(y.values, t.values).plot()
    ax.set_ylabel("load [MW]")
    ax.set_xlabel("temperature [degC]")
    return ax
