"""
Module for doing basic timestamp and frequency operations.
"""

from typing import Iterable, Union
import pandas as pd
import numpy as np

# Allowed frequencies.
# Perfect containment; a short-frequency time period always entirely falls within a single high-frequency time period.
# AS -> 4 QS; QS -> 3 MS; MS -> 28-31 D; D -> 23-25 H; H -> 4 15T
FREQUENCIES = ["AS", "QS", "MS", "D", "H", "15T"]


def floor_ts(
    ts: Union[pd.Timestamp, pd.DatetimeIndex], future: int = 0, freq=None
) -> Union[pd.Timestamp, pd.DatetimeIndex]:
    """Calculate (timestamp at beginning of) month, quarter, or year that a timestamp
    is in.

    Parameters
    ----------
    ts : Timestamp or DatetimeIndex.
        Timestamp(s) to floor.
    future : int, optional
        0 (default) to get start of period that ts in contained in. 1 (-1) to get start 
        of period after (before) that. 2 (-2) .. etc.
    freq : frequency, optional
        What to floor it to. One of {'D', 'MS', 'QS', 'AS'} for start of the day, month, 
        quarter, or year it's contained in. If none specified, use .freq attribute of 
        timestamp.

    Returns
    -------
    Timestamp(s)
        At begin of period.
    """
    if freq is None:
        freq = ts.freq
    ts = ts.floor("D")  # make sure we return a midnight value
    if freq == "D":
        return ts + pd.Timedelta(days=future)
    elif freq == "MS":
        return ts + pd.offsets.MonthBegin(1) + pd.offsets.MonthBegin(future - 1)
    elif freq == "QS":
        return (
            ts
            + pd.offsets.QuarterBegin(1, startingMonth=1)
            + pd.offsets.QuarterBegin(future - 1, startingMonth=1)
        )
    elif freq == "AS":
        return ts + pd.offsets.YearBegin(1) + pd.offsets.YearBegin(future - 1)
    else:
        raise ValueError("Argument `freq` must be one of {'D', 'MS', 'QS', 'AS'}.")


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