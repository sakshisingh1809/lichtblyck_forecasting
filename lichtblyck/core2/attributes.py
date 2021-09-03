# -*- coding: utf-8 -*-
"""
Attributes for custom dataframes, series, and indices.
"""

import pandas as pd

def _duration(fr: pd.core.generic.NDFrame) -> pd.Series:
    """
    Return duration [h] of each timestamp in index of DataFrame or Series.
    """
    hours = (_ts_right(fr) - fr.index).apply(lambda td: td.total_seconds() / 3600)
    return hours.rename("duration")


def _ts_right(fr: pd.core.generic.NDFrame) -> pd.Series:
    """
    Return right-bound timestamp of each timestamp in index of DataFrame or Series.
    """
    i = fr.index
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
