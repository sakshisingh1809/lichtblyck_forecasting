# -*- coding: utf-8 -*-
"""
Plotting of tlp functions.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from . import toload
from typing import Callable, Iterable


def vs_time(tlp: Callable, t: Iterable = (-30, 30), freq: str = "15T") -> plt.Axes:
    """
    Plot tlp consumption (y) thoughout a single day (x) for various temperatures.

    Parameters
    ----------
    tlp : Callable
        Function that takes temperature [degC] as first argument (and possibly 
        timestamp as second argument), and returns the consumption [MW].
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
    ts = pd.date_range("2020-01-01", "2020-01-02", freq=freq, closed="left")
    y = {tmpr: toload.w(tlp, pd.Series(tmpr, ts)).values for tmpr in np.arange(*t)}
    i = len(y) // 2
    df = pd.DataFrame(y, ts)
    df_draw = pd.concat(
        [
            df.iloc[:, :i].T.drop_duplicates(keep="last"),
            df.iloc[:, i:].T.drop_duplicates(keep="first"),
        ]
    ).T
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
    t = np.linspace(*t)
    y = [tlp(tmpr) for tmpr in t]
    ax = pd.Series(y, t).plot()
    ax.set_ylabel("load [MW]")
    ax.set_xlabel("temperature [degC]")
    return ax


#%%

# import pandas as pd
# import numpy as np
# import matplotlib.dates as mdates
# import matplotlib.pyplot as plt

# ts = pd.date_range("2020-01-01", "2020-01-02", freq="15T", closed="left")
# vals = np.random.rand(len(ts))
# myFmt = mdates.DateFormatter("%H:%M")


# ax = pd.Series(vals, ts).plot()
# plt.gca().xaxis.set_major_formatter(myFmt)


# ax.xaxis.set_major_formatter(myFmt)


# plt.plot(ts, vals)


# plt.plot(pd.Series(vals, ts))
# plt.gca().xaxis.set_major_formatter(myFmt)
