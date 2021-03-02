"""Functions to work with pandas dataframes."""

from .pfseries_pfframe import FREQUENCIES
from typing import Iterable
import pandas as pd
import numpy as np


def add_header(df: pd.DataFrame, header) -> pd.DataFrame:
    """Add column level onto top, with value `header`."""
    return pd.concat([df], keys=[header], axis=1)


def concat(dfs: Iterable, axis: int = 0, *args, **kwargs) -> pd.DataFrame:
    """
    Wrapper for `pandas.concat`; concatenate pandas objects even if they have
    unequal number of levels on concatenation axis.

    Levels containing empty strings are added from below (when concatenating along
    columns) or right (when concateniting along rows) to match the maximum number
    found in the dataframes.

    Parameters
    ----------
    dfs : Iterable
        Dataframes that must be concatenated.
    axis : int, optional
        Axis along which concatenation must take place. The default is 0.

    Returns
    -------
    pd.DataFrame
        Concatenated Dataframe.

    Notes
    -----
    Any arguments and kwarguments are passed onto the `pandas.concat` function.

    See also
    --------
    pandas.concat
    """

    def index(df):
        return df.columns if axis == 1 else df.index

    def add_levels(df):
        need = want - index(df).nlevels
        if need > 0:
            df = pd.concat([df], keys=[("",) * need], axis=axis)  # prepend empty levels
            for i in range(want - need):  # move empty levels to bottom
                df = df.swaplevel(i, i + need, axis=axis)
        return df

    want = np.max([index(df).nlevels for df in dfs])
    dfs = [add_levels(df) for df in dfs]
    return pd.concat(dfs, axis=axis, *args, **kwargs)


def _aggpf(df: pd.DataFrame) -> pd.Series:
    """
    Aggregation function for PfFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with (at least) 2 of the following columns: (w or q), p, r.

    Returns
    -------
    pd.Series
        The aggregated series with the aggregated values for q, r, w and p.
    """
    duration = df.duration.sum()
    q = df.q.sum(skipna=False)
    r = df.r.sum(skipna=False)
    return pd.Series({"q": q, "r": r, "w": q / duration, "p": r / q})


def changefreq_summable(
    fr: pd.core.frame.NDFrame, freq: str = "MS"
) -> pd.core.frame.NDFrame:
    """
    Resample and aggregate DataFrame or Series with time-summable quantities.

    Parameters
    ----------
    freq : str, optional
        The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
        for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
        quarterhour. The default is 'MS'.

    Notes
    -----
    A 'time-summable' quantity is one that can be summed to get to an aggregate
    value, like revenue [Eur] or energy [MWh]. Prices [Eur/MWh] and powers [MW] 
    are not time-summable.
    
    Returns
    -------
    DataFrame or Series
    """

    # Some resampling labels are right-bound by default. Change to make left-bound.
    if freq == "M" or freq == "A" or freq == "Q":
        freq += "S"
    if freq not in FREQUENCIES:
        raise ValueError(f"Parameter `freq` must be one of {','.join(FREQUENCIES)}.")

    # # Don't resample, just aggregate.
    # if freq is None:
    #     duration = spf.duration.sum()
    #     q = spf.q.sum(skipna=False)
    #     r = spf.r.sum(skipna=False)
    #     # Must return all data, because time information is lost.
    #     return pd.Series({"w": q / duration, "q": q, "p": r / q, "r": r})
    #     # i = pd.date_range(start = spf.index[0], end=spf.ts_right[-1], periods=1, tz=spf.index.tz)
    #     # return DataFrame([[q/duration, q, r/q, r]], i, columns=list('wqpr'))

    # Empty frame.
    if len(fr) == 0:
        return fr.resample(freq).mean()  # empty frame.

    down_or_up = FREQUENCIES.index(freq) - FREQUENCIES.index(fr.index.freq)

    # Nothing more needed; portfolio already in desired frequency.
    if down_or_up == 0:
        return fr

    # Must downsample.
    elif down_or_up < 0:
        newfr = fr.resample(freq).sum()
        # Discard rows in new pd.DataFrame that are only partially present in original pd.DataFrame.
        return newfr[(newfr.index >= fr.index[0]) & (newfr.ts_right <= fr.ts_right[-1])]

    # Must upsample.
    else:
        # Workaround to avoid missing final values:
        fr = fr.copy()
        # first, add additional row...
        fr.loc[fr.ts_right[-1], :] = None
        # ... then do upsampling ...
        newfr = fr.resample(freq).asfreq().fillna(0)  # sum correct, but must...
        newfr = newfr.resample(fr.index.freq).transform(np.mean)  # ...distribute
        # ... and then remove final row.
        return newfr.iloc[:-1]


def freq_diff(freq1, freq2) -> float:
    """
    Compare two frequencies to determine which is larger.

    Args:
        freq1, freq2 : frequencies to compare.

    Returns:
        1 (0, -1) if freq1 denotes a longer (equal, shorter) time period than freq2.

    Notes
    -----
    Arbitrarily using a time point as anchor to calculate the length of the time period
    from. May have influence on the ratio (duration of a month, quarter, year etc are
    influenced by this), but, for most common frequencies, not on which is larger.
    """
    common_ts = pd.Timestamp("2020-01-01")
    ts1 = common_ts + pd.tseries.frequencies.to_offset(freq1)
    ts2 = common_ts + pd.tseries.frequencies.to_offset(freq2)
    if ts1 > ts2:
        return 1
    elif ts1 < ts2:
        return -1
    return 0
