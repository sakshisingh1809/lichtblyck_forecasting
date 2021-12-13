"""
Hedge an offtake profile with a price profile.
"""

from typing import Tuple, Union, Iterable, Dict

from .utils import ts_leftright, is_peak_hour
from .convert import group_function
from ..tools.frames import trim_frame
import pandas as pd
import numpy as np


def _hedge(df: pd.DataFrame, how: str, po: bool,) -> pd.Series:
    """
    Make value hedge of power timeseries, for given price timeseries.

    Parameters
    ----------
    df : pd.DataFrame
        with 'w' [MW] and 'p' [Eur/MWh] columns.
    how : str. Of on {'vol', 'val'}
        Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
    po : bool
        Set to True to split hedge into peak and offpeak values. (Only sensible
        for timeseries with freq=='H' or shorter.)

    Returns
    -------
    pd.Series 
        With float values or quantities.
        If bpo==False, Series with index ['w', 'p'] (power and price in entire period).
        If bpo==True, Series with multiindex ['peak', 'offpeak'] x ['w', 'p'] (power and
        price, split between peak and offpeak intervals in the period.)

    Notes
    -----
    If the index of `df` doesn't have a .duration attribute, all rows are assumed to be 
    of equal duration.
    """

    if not po:
        try:
            # Use magnitude only, so that return value is float (if w and p are float
            # series) or quantity (if w and p are Quantities.)
            df["dur"] = df.index.duration.pint.m
        except (AttributeError, ValueError):
            df["dur"] = 1

        # Get single power and price values
        p_hedge = (df.p * df.dur).sum() / df.dur.sum()
        if how.lower().startswith("vol"):  # volume hedge
            # solve for w_hedge: sum (w * duration) == w_hedge * sum (duration)
            w_hedge = (df.w * df.dur).sum() / df.dur.sum()
        else:  # value hedge
            # solve for w_hedge: sum (w * duration * p) == w_hedge * sum (duration * p)
            w_hedge = (df.w * df.dur * df.p).sum() / (df.dur * df.p).sum()
        return pd.Series({"w": w_hedge, "p": p_hedge})
    else:
        apply_f = lambda df: _hedge(df, how, po=False)
        s = df.groupby(is_peak_hour).apply(apply_f)
        return s.rename(index={True: "peak", False: "offpeak"}).stack()


def hedge(
    w: pd.Series, p: pd.Series, how: str = "val", freq: str = "MS", po: bool = None,
) -> Tuple[pd.Series]:
    """
    Make hedge of power timeseries, for given price timeseries.

    Parameters
    ----------
    w : Series
        Power timeseries with hourly or quarterhourly frequency.
    p: Series
        Price timeseries with same frequency.
    how : str, optional (Default: 'val')
        Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
    freq : str, optional (Default: 'MS')
        Frequency of hedging products. E.g. 'QS' to hedge with quarter products. One 
        of {'D', 'MS', 'QS', 'AS'}.
    po : bool, optional
        Type of hedging products. Set to True to split hedge into peak and offpeak. 
        (Default: split if volume timeseries has hourly values or shorter and hedging
        products have monthly frequency or longer.)

    Returns
    -------
    Tuple[pd.Series]
        Power timeseries and price timeseries with hedge of `w`.
    """
    if w.index.freq not in ["15T", "H", "D"]:
        raise ValueError("Can only hedge a timeseries with daily (or shorter) values.")
    if p.index.freq != w.index.freq:
        raise ValueError(
            f"`w` and `p` must have same frequency (got: {w.index.freq} and {p.index.freq})."
        )
    if freq not in ("D", "MS", "QS", "AS"):
        raise ValueError(
            f"Frequency of hedging products must be one of {'D', 'MS', 'QS', 'AS'} (got: {freq})."
        )
    if po is None:  # default: peak/offpeak if possible
        po = w.index.freq in ["15T", "H"] and freq != "D"
    if po and not (w.index.freq in ["15T", "H"] and freq != "D"):
        raise ValueError(
            "Split into peak and offpeak only possible when (a) hedging with monthly (or longer) products, and (b) if volume timeseries has hourly (or shorter) values."
        )

    # Handle possible units.
    win, wunits = (w.pint.magnitude, w.pint.units) if hasattr(w, "pint") else (w, None)
    pin, punits = (p.pint.magnitude, p.pint.units) if hasattr(p, "pint") else (p, None)

    # Only keep full periods of overlapping timestamps.
    i = win.index.intersection(pin.index)
    df = trim_frame(pd.DataFrame({"w": win, "p": pin}).loc[i, :], freq)

    # Do actual hedge.
    group_f = group_function(freq, po)
    vals = df.groupby(group_f).apply(lambda df: _hedge(df, how, False))
    vals.index = pd.MultiIndex.from_tuples(vals.index)
    for c in ['w', 'p']:
        df[c] = df[c].groupby(group_f).transform(lambda gr: vals.loc[gr.name, c])

    # Handle possible units.
    if wunits or punits:
        df = df.astype({'w': f'pint[{wunits}]', 'p': f'pint[{punits}]'})
    
    return df['w'], df['p']



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
