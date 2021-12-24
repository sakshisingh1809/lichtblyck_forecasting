"""
Module for doing basic timestamp and frequency operations.
"""

from typing import Iterable, Union, Tuple
import datetime as dt
import pandas as pd
import numpy as np
from .nits import Q_


# Allowed frequencies.
# Perfect containment; a short-frequency time period always entirely falls within a single high-frequency time period.
# AS -> 4 QS; QS -> 3 MS; MS -> 28-31 D; D -> 23-25 H; H -> 4 15T
FREQUENCIES = ["AS", "QS", "MS", "D", "H", "15T"]


def ts_right(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex]
) -> Union[pd.Timestamp, pd.Series]:
    """
    Return right-bound timestamp of a timestamp, or of each timestamp in index.
    """
    if ts_left.tz is None:
        raise ValueError("Index is missing timezone information.")

    # Get right timestamp for each index value, based on the frequency.
    # . This one breaks for 'MS':
    # (i + pd.DateOffset(nanoseconds=i.freq.nanos))
    # . This drops a value at some DST transitions:
    # (i.shift(1))
    # . This one gives wrong value at DST transitions:
    # i + i.freq
    if ts_left.freq == "15T":
        ts_right = ts_left + pd.Timedelta(hours=0.25)
    elif ts_left.freq == "H":
        ts_right = ts_left + pd.Timedelta(hours=1)
    else:
        if ts_left.freq == "D":
            kwargs = {"days": 1}
        elif ts_left.freq == "MS":
            kwargs = {"months": 1}
        elif ts_left.freq == "QS":
            kwargs = {"months": 3}
        elif ts_left.freq == "AS":
            kwargs = {"years": 1}
        else:
            raise ValueError(f"Invalid frequency: {ts_left.freq}.")
        ts_right = ts_left + pd.DateOffset(**kwargs)
    # Return in correct format.
    if isinstance(ts_left, pd.Timestamp):
        return ts_right
    else:
        return pd.Series(ts_right, ts_left, name="ts_right")


def duration(ts_left: Union[pd.Timestamp, pd.DatetimeIndex]) -> Union[Q_, pd.Series]:
    """
    Return duration of a timestamp, or of each timestamp in index.
    """
    if isinstance(ts_left, pd.Timestamp):
        if ts_left.freq in ["15T", "H"]:
            # Speed-up things for fixed-duration frequencies.
            return Q_(1 if ts_left.freq == "H" else 0.25, "h")
        else:
            # Individual calculations for non-fixed-duration frequencies.
            return Q_((ts_right(ts_left) - ts_left).total_seconds() / 3600, "h")
    else:
        if ts_left.freq in ["15T", "H"]:
            dur = 1 if ts_left.freq == "H" else 0.25
            return pd.Series(dur, ts_left, dtype="pint[h]").rename("duration")
        else:
            dur = [td.total_seconds() / 3600 for td in ts_right(ts_left) - ts_left]
            return pd.Series(dur, ts_left, dtype="pint[h]").rename("duration")


def floor_ts(
    ts: Union[pd.Timestamp, pd.DatetimeIndex], future: int = 0, freq=None
) -> Union[pd.Timestamp, pd.DatetimeIndex]:
    """Floor timestamp to period boundary.

    i.e., find (latest) period start that is on or before the timestamp.

    Parameters
    ----------
    ts : Timestamp or DatetimeIndex.
        Timestamp(s) to floor.
    future : int, optional
        0 (default) to get latest period start that is `ts` or earlier. 1 (-1) to get
        start of period after (before) that. 2 (-2) .. etc.
    freq : {'D' (day), 'MS' (month), 'QS' (quarter), 'AS' (year)}, optional
        What to floor it to, e.g. 'QS' to get start of quarter it's contained in. If
        none specified, use .freq attribute of timestamp.

    Returns
    -------
    Timestamp(s)
        At begin of period.

    Notes
    -----
    If `ts` is exactly at the start of the period, ceil_ts(ts, 0) == floor_ts(ts, 0) == ts.
    """
    if freq is None:
        freq = ts.freq

    if freq == "15T":
        return ts.floor("15T") + pd.Timedelta(minutes=future * 15)
    elif freq == "H":
        return ts.floor("H") + pd.Timedelta(hours=future)

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
        raise ValueError(f"Argument `freq` must be one of {','.join(FREQUENCIES)}.")


def ceil_ts(
    ts: Union[pd.Timestamp, pd.DatetimeIndex], future: int = 0, freq=None
) -> Union[pd.Timestamp, pd.DatetimeIndex]:
    """Ceil timestamp to period boundary.

    i.e., find (earliest) period start that is on or after the timestamp.

    Parameters
    ----------
    ts : Timestamp or DatetimeIndex.
        Timestamp(s) to ceil.
    future : int, optional
        0 (default) to get earliest period start that is `ts` or later. 1 (-1) to get
        start of period after (before) that. 2 (-2) .. etc.
    freq : {'D' (day), 'MS' (month), 'QS' (quarter), 'AS' (year)}, optional
        What to ceil it to, e.g. 'QS' to get start of quarter it's contained in. If
        none specified, use .freq attribute of timestamp.

    Returns
    -------
    Timestamp(s)
        At begin of period.

    Notes
    -----
    If `ts` is exactly at the start of the period, ceil_ts(ts, 0) == floor_ts(ts, 0) == ts.
    """
    offset = (
        1 if ts != floor_ts(ts, 0, freq) else 0
    )  # if ts at start of period, ceil==floor
    return floor_ts(ts, future + offset, freq)


def trim_index(i: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Trim index to only keep full periods.

    Parameters
    ----------
    i : pd.DatetimeIndex
        The (untrimmed) DatetimeIndex
    freq : str
        Frequency to trim to. E.g. 'MS' to only keep full months.

    Returns
    -------
    pd.DatetimeIndex
        Subset of `i`, with same frequency.
    """
    start = ceil_ts(i[0], 0, freq)
    end = floor_ts(ts_right(i[-1]), 0, freq)
    return i[(i >= start) & (i < end)]


def ts_leftright(left=None, right=None) -> Tuple:
    """Makes 2 timestamps coherent to one another.

    Parameters
    ----------
    left : timestamp, optional
    right : timestamp, optional
        If no value for ts_left is given, the beginning of the year of ts_right is given.
        If no value for ts_right is given, the end of the year of ts_left is given.
        If no values is given for either, the entire next year is given.
        If a value is given for each, they are swapped if their order is incorrect.
        If no time zone is provided for either timestamp, the Europe/Berlin timezone is
        assumed.

    Returns
    -------
    (localized timestamp, localized timestamp)
    """

    left, right = pd.Timestamp(left), pd.Timestamp(right)

    if right is pd.NaT:
        if left is pd.NaT:
            return ts_leftright(floor_ts(pd.Timestamp.now(), 1, "AS"))
        if left.tz is None:
            return ts_leftright(left.tz_localize("Europe/Berlin"))
        return ts_leftright(left, floor_ts(left, 1, "AS"))

    # if we land here, we at least know ts_right.
    if left is pd.NaT:
        back = -1 if right == floor_ts(right, 0, "AS") else 0
        return ts_leftright(floor_ts(right, back, "AS"), right)

    # if we land here, we know ts_left and ts_right.
    if right.tz is None:
        if left.tz is None:
            return ts_leftright(left.tz_localize("Europe/Berlin"), right)
        return ts_leftright(left, right.tz_localize(left.tz))

    # if we land here, we know ts_left and localized ts_right.
    if left.tz is None:
        return ts_leftright(left.tz_localize(right.tz), right)

    # if we land here, we know localized ts_left and localized ts_right
    if left > right:
        left, right = right, left

    # return values possibly with distinct timezones. We cannot easily avoid this,
    # because summer- and wintertime are also distinct timezones.
    return left, right


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
        Upsampling meaning, the number of values increases - one value in the source
        corresponds to multiple values in the target.

    Notes
    -----
    Arbitrarily using a time point as anchor to calculate the length of the time period
    from. May have influence on the ratio (duration of a month, quarter, year etc are
    influenced by this), but, for most common frequencies, not on which is larger.
    """
    common_ts = pd.Timestamp("2020-01-01 0:00")
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
