"""
Hedge an offtake profile with a price profile.
"""

from typing import  Union, Iterable, Dict
from .utils import ts_leftright, is_peak_hour
from .convert import group_function
from ..core import utils
import pandas as pd
import numpy as np


def _w_hedge(
    w: pd.Series, p: pd.Series = None, how: str = "vol", bpo: bool = False,
) -> Union[float, pd.Series]:
    """
    Make value hedge of power timeseries, for given price timeseries.

    Parameters
    ----------
        w : power timeseries.
        p : price timeseries.
            Ignored if how=='vol'.
        how : {'vol' (default), 'val'}
            Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
        bpo : bool
            Set to True to split hedge into peak and offpeak values. (Only sensible
            for timeseries with freq=='H' or shorter.)

    Returns
    -------
        If bpo==False, single float value (power in entire period).
        If bpo==True, Series with float values (`w_peak` and `w_offpeak`).

    Notes
    -----
    If the indices of `w` and `p` don't have a .freq attribute, all rows are assumed to
    be of equal duration.
    """

    if not bpo:
        try:
            duration = w.index.duration
        except (AttributeError, ValueError):
            duration = pd.Series([1] * len(w), w.index)
        if how.lower().startswith("vol"):  # volume hedge
            # solve for w_hedge: sum (w * duration) == w_hedge * sum (duration)
            return (w * duration).sum() / duration.sum()
        else:  # value hedge
            # solve for w_hedge: sum (w * duration * p) == w_hedge * sum (duration * p)
            return (w * duration * p).sum() / (duration * p).sum()
    else:
        apply_f = lambda df: _w_hedge(df["w"], df["p"], how, bpo=False)
        s = pd.DataFrame({"w": w, "p": p}).groupby(is_peak_hour).apply(apply_f)
        return s.rename(index={True: "w_peak", False: "w_offpeak"})


def hedge(
    w: pd.Series,
    p: pd.Series = None,
    freq: str = "MS",
    how: str = "vol",
    bpo: bool = False,
    keep_index: bool = True,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Make hedge of power timeseries, for given price timeseries.

    Parameters
    ----------
        w : Series
            Power timeseries with hourly or quarterhourly frequency.
        p: Series
            Price timeseries with same frequency, ignored if how=='vol'.
        freq : str
            Grouping frequency. One of {'D', 'MS' (default) 'QS', 'AS'} for day, month,
            quarter, or year values. ('D' not allowed for bpo==True, see below.)
        how : {'vol' (default), 'val'}
            Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
        bpo : bool
            Set to True to split hedge into peak and offpeak values. (Only sensible
            for input timeseries with .freq=='H' or shorter, and a value for `freq` of
            'MS' or longer.)
        keep_index: bool
            If True (default), returns timeseries with same index and frequency as `w`.
            If False, returns timeseries / DataFrame with grouping frequency `freq`.

    Returns
    -------
    Series
        Power timeseries with hedge of `w`.
    """
    if bpo and utils.freq_up_or_down(w.index.freq, "H") > 0:
        raise ValueError(
            "Split in peak and offpeak only possible for timeseries with frequencies "
            + "of one hour or shorter."
        )
    if freq not in ("D", "MS", "QS", "AS"):
        raise ValueError("Argument 'freq' must be in {'D', 'MS', 'QS', 'AS'}.")

    if p is not None:
        if p.index.freq != w.index.freq:
            raise ValueError("`w` and `p` (if provided) must have same frequency.")

    df = pd.DataFrame({"w": w, "p": p})

    if not keep_index:
        apply_f = lambda df: _w_hedge(df["w"], df["p"], how, bpo)
        return df.resample(freq).apply(apply_f)
    else:
        # bpo always False in apply_f, because group_f distinguishes peak/offpeak.
        apply_f = lambda df: _w_hedge(df["w"], df["p"], how, False)
        group_f = group_function(freq, bpo)

        vals = df.groupby(group_f).apply(apply_f)
        return w.groupby(group_f).transform(lambda gr: vals[gr.name]).rename("w_hedge")


def vola(
    df: Union[pd.Series, pd.DataFrame], window: int = 100
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate volatility in [fraction/year] from price time series/dataframe.

    Parameters
    ----------
        df: Series or Dataframe with price values indexed by (trading day) timestamps.
        window: number of observations for volatility estimate.

    Returns
    -------
        Series or Dataframe with volatility calculated with rolling window.
    """
    df = df.apply(np.log).diff()  # change as fraction
    volas = df.rolling(window).std()  # volatility per average timedelta
    av_years = df.rolling(window).apply(
        lambda s: ((s.index[-1] - s.index[0]) / window).total_seconds()
        / 3600
        / 24
        / 365.24
    )
    volas /= av_years.apply(np.sqrt)  # volatility per year
    return volas.dropna()
