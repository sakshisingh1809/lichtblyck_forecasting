"""Functions to work with pandas dataframes."""

from .pfseries_pfframe import FREQUENCIES
from pandas.core.frame import NDFrame
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


# def _aggpf(df: pd.DataFrame) -> pd.Series:
#     """
#     Aggregation function for PfFrames.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Dataframe with (at least) 2 of the following columns: (w or q), p, r.

#     Returns
#     -------
#     pd.Series
#         The aggregated series with the aggregated values for q, r, w and p.
#     """
#     duration = df.duration.sum()
#     q = df.q.sum(skipna=False)
#     r = df.r.sum(skipna=False)
#     return pd.Series({"q": q, "r": r, "w": q / duration, "p": r / q})


def _changefreq_general(fr: NDFrame, freq: str = "MS", is_summable: bool = True):
    """Change frequency of a Series or DataFrame, depending on the type of data it 
    contains."""

    # Some resampling labels are right-bound by default. Change to make left-bound.
    if freq in ["M", "A", "Q"]:
        freq += "S"
    if freq not in FREQUENCIES:
        raise ValueError(f"Parameter `freq` must be one of {','.join(FREQUENCIES)}.")

    # Empty frame.
    if len(fr) == 0:
        return fr.resample(freq).mean()  # empty frame.

    up_or_down = freq_up_or_down(fr.index.freq, freq)

    # Nothing more needed; portfolio already in desired frequency.
    if up_or_down == 0:
        return fr

    # Must downsample.
    elif up_or_down == -1:
        if is_summable:
            # Downsampling is easiest for summable frames: simply sum child values.
            fr2 = fr.resample(freq).sum()
            # Discard rows in new frame that are only partially present in original.
            return fr2[(fr2.index >= fr.index[0]) & (fr2.ts_right <= fr.ts_right[-1])]
        else:
            # For averagable frames: first make summable.
            summable = fr.mul(fr.duration, axis=0)
            summable2 = _changefreq_general(summable, freq, True)
            fr2 = summable2.div(summable2.duration, axis=0)
            return fr2

    # Must upsample.
    else:
        if not is_summable:
            # Upsampling is easiest for averagable frames: simply duplicate parent value.
            # Workaround to avoid missing final values:
            fr = fr.copy()
            # first, add additional row...
            try:  # fails if series
                fr.loc[fr.ts_right[-1], :] = None
            except TypeError:
                fr.loc[fr.ts_right[-1]] = None
            # ... then do upsampling ...
            fr2 = fr.resample(freq).asfreq().ffill()  # duplicate value
            # ... and then remove final row.
            return fr2.iloc[:-1]
        else:
            # For summable frames: first make averagable.
            avgable = fr.div(fr.duration, axis=0)
            avgable2 = _changefreq_general(avgable, freq, False)
            fr2 = avgable2.mul(avgable2.duration, axis=0)
            return fr2


def changefreq_sum(fr: NDFrame, freq: str = "MS") -> NDFrame:
    """
    Resample and aggregate DataFrame or Series with time-summable quantities.

    Parameters
    ----------
    fr : NDFrame
        Pandas Series or DataFrame.
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
    return _changefreq_general(fr, freq, True)


def changefreq_avg(fr: NDFrame, freq: str = "MS") -> NDFrame:
    """
    Resample and aggregate DataFrame or Series with time-averagable quantities.

    Parameters
    ----------
    fr : NDFrame
        Pandas Series or DataFrame.
    freq : str, optional
        The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
        for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
        quarterhour. The default is 'MS'.

    Returns
    -------
    DataFrame or Series

    Notes
    -----
    A 'time-averagable' quantity is one that can be averaged to an aggregate value, 
    like power [MW]. When downsampling, the values are weighted with their duration.    
    """
    return _changefreq_general(fr, freq, False)


def freq_up_or_down(freq_source, freq_target) -> int:
    """
    Compare source frequency with target frequency to see if it needs up- or downsampling.

    Parameters
    ----------
    freq_source, freq_target : frequencies to compare.

    Returns
    -------
    1 (-1, 0) if source frequency must be upsampled (downsampled, no change) to obtain
        target frequency.

    Notes
    -----
    Arbitrarily using a time point as anchor to calculate the length of the time period
    from. May have influence on the ratio (duration of a month, quarter, year etc are
    influenced by this), but, for most common frequencies, not on which is larger.
    """
    common_ts = pd.Timestamp("2020-01-01")
    ts1 = common_ts + pd.tseries.frequencies.to_offset(freq_source)
    ts2 = common_ts + pd.tseries.frequencies.to_offset(freq_target)
    if ts1 > ts2:
        return 1
    elif ts1 < ts2:
        return -1
    return 0


def _freq_longestshortest(shortest: bool, *freqs):
    """
    Determine which frequency denotes the shortest or longest time period.

    Parameters
    ----------
    shortest : bool
        True to find shortest, False to find longest frequency.
    *freqs : frequencies to compare (as string or other object).

    Returns
    -------
    Frequency.
    """
    common_ts = pd.Timestamp("2020-01-01")
    ts = [common_ts + pd.tseries.frequencies.to_offset(fr) for fr in freqs]
    i = (np.argmin if shortest else np.argmax)(ts)
    return freqs[i]


def freq_shortest(*freqs):
    """
    Returns shortest frequency in list.
    """
    return _freq_longestshortest(True, *freqs)


def freq_longest(*freqs):
    """
    Returns longest frequency in list.
    """
    return _freq_longestshortest(False, *freqs)