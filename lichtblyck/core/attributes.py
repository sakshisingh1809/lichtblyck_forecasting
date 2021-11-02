# -*- coding: utf-8 -*-
"""
Attributes for custom dataframes, series, and indices.
"""

import functools
import pandas as pd
import warnings


@functools.lru_cache
def _hours(td):
    return td.total_seconds() / 3600


def _ts_right_frame(fr: pd.core.generic.NDFrame) -> pd.Series:
    """
    Return right-bound timestamp of each timestamp in index of DataFrame or Series.
    """
    warnings.warn(
        "This method will be removed in future releases. Use pd.DatetimeIndex.ts_right instead."
    )
    return _ts_right(fr.index)


def _duration_frame(fr: pd.core.generic.NDFrame) -> pd.Series:
    """
    Return duration [h] of each timestamp in index of DataFrame or Series.
    """
    warnings.warn(
        "This method will be removed in future releases. Use pd.DatetimeIndex.duration instead."
    )
    return _duration(fr.index)


def _ts_right(i: pd.DatetimeIndex) -> pd.Series:
    """
    Return right-bound timestamp of each timestamp in index.
    """
    if i.tz is None:
        raise ValueError("Index is missing timezone information.")

    # Get right timestamp for each index value, based on the frequency.
    # . This one breaks for 'MS':
    # (i + pd.DateOffset(nanoseconds=i.freq.nanos))
    # . This drops a value at some DST transitions:
    # (i.shift(1))
    # . This one gives wrong value at DST transitions:
    # i + i.freq
    if i.freq == "15T":
        ts_right = i + pd.Timedelta(hours=0.25)
    elif i.freq == "H":
        ts_right = i + pd.Timedelta(hours=1)
    else:
        if i.freq == "D":
            kwargs = {"days": 1}
        elif i.freq == "MS":
            kwargs = {"months": 1}
        elif i.freq == "QS":
            kwargs = {"months": 3}
        elif i.freq == "AS":
            kwargs = {"years": 1}
        else:
            raise ValueError(f"Invalid frequency: {i.freq}.")
        ts_right = i + pd.DateOffset(**kwargs)
    return pd.Series(ts_right, i, name="ts_right")


def _duration(i: pd.DatetimeIndex) -> pd.Series:
    """
    Return duration [h] of each timestamp in index.
    """
    # Speed-up things for fixed-duration frequencies.
    if i.freq == "15T":
        return pd.Series(0.25, i).rename('duration')
    elif i.freq == "H":
        return pd.Series(1, i).rename('duration')
    # Old-fashioned individual calculations for non-fixed-duration frequencies.
    return (_ts_right(i) - i).apply(_hours).rename('duration')
