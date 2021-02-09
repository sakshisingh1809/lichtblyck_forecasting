"""Functions to work with pandas dataframes."""

from .pfseries_pfframe import PfFrame, FREQUENCIES
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


def _aggpf(pf: PfFrame) -> pd.Series:
    """
    Aggregation function for PfFrames.

    Parameters
    ----------
    pf : PfFrame
        Dataframe with (at least) 2 of the following columns: (w or q), p, r.

    Returns
    -------
    pd.Series
        The aggregated series with the aggregated values for q, r, w and p.
    """
    if not isinstance(pf, PfFrame):
        pf = PfFrame(pf)
    duration = pf.duration.sum()
    q = pf.q.sum(skipna=False)
    r = pf.r.sum(skipna=False)
    return pd.Series({"q": q, "r": r, "w": q / duration, "p": r / q})


def changefreq(pf: PfFrame, freq: str = "MS") -> PfFrame:
    """
    Resample and aggregate the DataFrame at a new frequency.

    Parameters
    ----------
    pf : PfFrame
        Portfolioframe to resample and aggregate.
    freq : str, optional
        The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
        for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
        quarterhour; None to aggregate over the entire time period. The default is 'MS'.

    Returns
    -------
    PfFrame
        Same data at different timescale.
    """
    # Some resampling labels are right-bound by default. Change to make left-bound.
    if freq == "M" or freq == "A" or freq == "Q":
        freq += "S"
    if freq is not None and freq not in FREQUENCIES:
        raise ValueError(
            f"Parameter `freq` must be None or one of {','.join(FREQUENCIES)}."
        )
    if type(pf) is not PfFrame:
        pf = PfFrame()

    # Don't resample, just aggregate.
    if freq is None:
        duration = pf.duration.sum()
        q = pf.q.sum(skipna=False)
        r = pf.r.sum(skipna=False)
        # Must return all data, because time information is lost.
        return pd.Series({"w": q / duration, "q": q, "p": r / q, "r": r})
        i = pd.date_range(
            start=pf.index[0], end=pf.ts_right[-1], periods=1, tz=pf.index.tz
        )
        return PfFrame([[q / duration, q, r / q, r]], i, columns=list("wqpr"))

    # Empty frame.
    if len(pf) == 0:
        return PfFrame(pf.resample(freq).mean())

    down_or_up = FREQUENCIES.index(freq) - FREQUENCIES.index(pf.index.freq)

    # Nothing more needed; dataframe already in desired frequency.
    if down_or_up == 0:
        return pf

    # Must downsample.
    elif down_or_up < 0:
        newpf = PfFrame(pf.resample(freq).apply(_aggpf))
        # Discard rows in new dataframe that are only partially present in original dataframe.
        newpf = newpf[
            (newpf.index >= pf.index[0]) & (newpf.ts_right <= pf.ts_right[-1])
        ]
        return PfFrame(newpf)

    # Must upsample.
    else:
        # Keep only w and p because these are averages that can be copied over to each child row.
        newpf = PfFrame({"w": pf.w, "p": pf.p}, pf.index)
        # Workaround to avoid missing final values:
        newpf.loc[newpf.ts_right[-1]] = [None, None]  # first, add additional row...
        newpf = newpf.resample(freq).ffill()  # ... then do upsampling ...
        newpf = newpf.iloc[:-1]  # ... and then remove final row.
        return PfFrame(newpf)

    # TODO: change/customize the columns in the returned dataframe.
