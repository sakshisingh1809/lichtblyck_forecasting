from typing import  Union, Iterable, Dict
from .utils import ts_deliv, is_peak_hour
from ..core import functions
import pandas as pd
import numpy as np


def _group_function(freq: str, bpo: bool = False):
    """Function to group all rows that belong to same 'product'."""
    if freq == "MS":
        if bpo:
            return lambda ts: (ts.year, ts.month, is_peak_hour(ts))
        else:
            return lambda ts: (ts.year, ts.month)
    elif freq == "QS":
        if bpo:
            return lambda ts: (ts.year, ts.quarter, is_peak_hour(ts))
        else:
            return lambda ts: (ts.year, ts.quarter)
    elif freq == "AS":
        if bpo:
            return lambda ts: (ts.year, is_peak_hour(ts))
        else:
            return lambda ts: ts.year
    else:
        raise ValueError("Argument 'freq' must be in {'D', 'MS', 'QS', 'AS'}.")


def _p_bpo(p: pd.Series) -> pd.Series:
    """
    Aggregate price series to base, peak and offpeak prices.

    Parameters
    ----------
        p: price timeseries.

    Returns
    -------
        Series with prices. Index: p_base, p_peak, p_offpeak.

    Notes
    -----
    It is assumed that all rows have the same duration and that
    each is wholly inside either peak or offpeak hours. (Safe to assume
    if frequency is hour or quarterhour.)
    """
    grouped = p.groupby(is_peak_hour).mean()
    return pd.Series(
        {
            "p_base": p.mean(),
            "p_peak": grouped.get(True, np.nan),
            "p_offpeak": grouped.get(False, np.nan),
        }
    )


def p_bpo_wide(p: pd.Series, freq: str = "MS") -> pd.DataFrame:
    """
    Aggregate price series to base, peak and offpeak prices. Grouped by time
    interval (if specified).

    Parameters
    ----------
        p : price timeseries.
        freq : str
            Grouping frequency. One of {'D', 'MS' (default) 'QS', 'AS'} for day, month,
            quarter, or year prices.

    Returns
    -------
    pd.DataFrame
        Dataframe with base, peak and offpeak prices (as columns). Index: downsampled
        timestamps, of aggregated level.

    Notes
    -----
    It is assumed that all rows have the same duration and that
    each is wholly inside either peak or offpeak hours. (Safe to assume
    if frequency is hour or quarterhour.)
    """
    if freq not in ("D", "MS", "QS", "AS"):
        raise ValueError("Argument 'freq' must be in {'D', MS', 'QS', 'AS'}.")
    return p.resample(freq).apply(_p_bpo).unstack()


def p_bpo(p: pd.Series, freq: str = "MS") -> pd.Series:
    """
    Transform price series to peak and offpeak prices, by calculating the mean
    for each.

    Parameters
    ----------
        p : price timeseries.
        freq : str
            Grouping frequency. One of {'D', 'MS' (default) 'QS', 'AS'} for day, month,
            quarter, or year prices.

    Returns
    -------
        Price timeseries where each peak hour within the provided frequency
        has the same value. Idem for offpeak hours. Index: as original series.
        Values: mean value for time period.
    
    Notes
    -----
    It is assumed that all rows have the same duration and that
    each is wholly inside either peak or offpeak hours. (Safe to assume
    if frequency is hour or quarterhour.)
    """
    return p.groupby(_group_function(freq, True)).transform(np.mean)


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
            duration = w.duration
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
        w: power timeseries.
        p: price timeseries.
            Ignored if how=='vol'.
        freq: str
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
    pd.Series
        Power timeseries with hedge of `w`.
    """
    if bpo and functions.freq_diff(w.index.freq, "H") > 0:
        raise ValueError(
            "Split in peak and offpeak only possible for timeseries with frequencies "
            + "of one hour or shorter."
        )
    if freq not in ("D", "MS", "QS", "AS"):
        raise ValueError("Argument 'freq' must be in {'D', 'MS', 'QS', 'AS'}.")

    df = pd.DataFrame({"w": w, "p": p})

    if not keep_index:
        apply_f = lambda df: _w_hedge(df["w"], df["p"], how, bpo)
        return df.resample(freq).apply(apply_f)
    else:
        # bpo always False in apply_f, because group_f distinguishes peak/offpeak.
        apply_f = lambda df: _w_hedge(df["w"], df["p"], how, False)
        group_f = _group_function(freq, bpo)

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
