"""
Module for doing basic timestamp and frequency operations.
"""

from typing import Iterable, Optional, Union, Tuple
import datetime as dt
import warnings
import pandas as pd
import numpy as np
from .nits import Q_


# Allowed frequencies.
# Perfect containment; a short-frequency time period always entirely falls within a single high-frequency time period.
# AS -> 4 QS; QS -> 3 MS; MS -> 28-31 D; D -> 23-25 H; H -> 4 15T
FREQUENCIES = ["AS", "QS", "MS", "D", "H", "15T"]


def timedelta(freq: str) -> Union[pd.Timedelta, pd.DateOffset]:
    """Returns object that can be added to a left-bound Timestamp or DatetimeIndex to
    find the corresponding right-bound Timestamp or DatetimeIndex."""

    # Get right timestamp for each index value, based on the frequency.
    # . This one breaks for 'MS':
    # (i + pd.DateOffset(nanoseconds=i.freq.nanos))
    # . This drops a value at some DST transitions:
    # (i.shift(1))
    # . This one gives wrong value at DST transitions:
    # i + i.freq

    if freq == "15T":
        return pd.Timedelta(hours=0.25)
    elif freq == "H":
        return pd.Timedelta(hours=1)
    elif freq == "D":
        return pd.DateOffset(days=1)
    elif freq == "MS":
        return pd.DateOffset(months=1)
    elif freq == "QS":
        return pd.DateOffset(months=3)
    elif freq == "AS":
        return pd.DateOffset(years=1)
    else:
        raise ValueError(
            f"Parameter ``freq`` must be one of {', '.join(FREQUENCIES)}; got {freq}."
        )


def ts_right(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex], freq: str = None
) -> Union[pd.Timestamp, pd.Series]:
    """Right-bound timestamp belonging to left-bound timestamp.

    Parameters
    ----------
    ts_left : Union[pd.Timestamp, pd.DatetimeIndex]
        Timestamp(s) for which to calculate the right-bound timestamp.
    freq : {'15T' (quarter-hour), 'H' (hour), 'D' (day), 'MS' (month), 'QS' (quarter),
        'AS' (year)}, optional
        Frequency to use in determining the right-bound timestamp.
        If none specified, use ``.freq`` attribute of ``ts_left``.

    Returns
    -------
    Timestamp (if ts_left is Timestamp) or Series (if ts_left is DatetimeIndex).
    """

    if ts_left.tz is None:
        raise ValueError("Parameter ``ts_left`` is missing timezone information.")

    if isinstance(ts_left, pd.Timestamp) and freq is None:
        warnings.warn(
            "Using a Timestamp without passing the ``freq`` parameter will be deprecated in a future release."
        )

    if freq is None:
        freq = ts_left.freq

    ts_right = ts_left + timedelta(freq)

    if isinstance(ts_left, pd.Timestamp):
        return ts_right
    elif isinstance(ts_left, pd.DatetimeIndex):
        return pd.Series(ts_right, ts_left, name="ts_right")

    raise TypeError(
        f"Parameter ``ts_left`` must be ``pandas.Timestamp`` or ``pandas.DatetimeIndex`` instance; got {type(ts_left)}."
    )


def duration(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex], freq: str = None
) -> Union[Q_, pd.Series]:
    """Duration of a timestamp.

    Parameters
    ----------
    ts_left : Union[pd.Timestamp, pd.DatetimeIndex]
        Timestamp(s) for which to calculate the duration.
    freq : {'15T' (quarter-hour), 'H' (hour), 'D' (day), 'MS' (month), 'QS' (quarter),
        'AS' (year)}, optional
        Frequency to use in determining the duration.
        If none specified, use ``.freq`` attribute of ``ts_left``.

    Returns
    -------
    pint Quantity (if ts_left is Timestamp) or Series (if ts_left is DatetimeIndex).
    """

    if ts_left.tz is None:
        raise ValueError("Parameter ``ts_left`` is missing timezone information.")

    if isinstance(ts_left, pd.Timestamp) and freq is None:
        warnings.warn(
            "Using a Timestamp without passing the ``freq`` parameter will be deprecated in a future release."
        )

    if freq is None:
        freq = ts_left.freq

    if isinstance(ts_left, pd.Timestamp):
        if freq in ["15T", "H"]:
            return Q_(1 if freq == "H" else 0.25, "h")
        else:
            return Q_((ts_right(ts_left, freq) - ts_left).total_seconds() / 3600, "h")

    elif isinstance(ts_left, pd.DatetimeIndex):
        if freq in ["15T", "H"]:
            # Speed-up things for fixed-duration frequencies.
            hour = 1 if freq == "H" else 0.25
            return pd.Series(hour, ts_left, dtype="pint[h]").rename("duration")
        else:
            # Individual calculations for non-fixed-duration frequencies.
            hours = [
                td.total_seconds() / 3600 for td in ts_right(ts_left, freq) - ts_left
            ]
            return pd.Series(hours, ts_left, dtype="pint[h]").rename("duration")

    raise TypeError(
        f"Parameter ``ts_left`` must be ``pandas.Timestamp`` or ``pandas.DatetimeIndex`` instance; got {type(ts_left)}."
    )


def floor_ts(
    ts: Union[pd.Timestamp, pd.DatetimeIndex], freq=None, future: int = 0
) -> Union[pd.Timestamp, pd.DatetimeIndex]:
    """Floor timestamp to period boundary.

    i.e., find (latest) period start that is on or before the timestamp.

    Parameters
    ----------
    ts : Timestamp or DatetimeIndex.
        Timestamp(s) to floor.
    freq : {'15T' (quarter-hour), 'H' (hour), 'D' (day), 'MS' (month), 'QS' (quarter),
        'AS' (year)}, optional
        What to floor it to, e.g. 'QS' to get start of quarter it's contained in. If
        none specified, use .freq attribute of timestamp.
    future : int, optional (default: 0)
        0 to get latest period start that is `ts` or earlier. 1 (-1) to get
        start of period after (before) that. 2 (-2) .. etc.

    Returns
    -------
    Timestamp(s)
        At begin of period.

    Notes
    -----
    If `ts` is exactly at the start of the period, ceil_ts(ts, 0) == floor_ts(ts, 0) == ts.
    """

    if isinstance(ts, pd.Timestamp) and freq is None:
        warnings.warn(
            "Using a Timestamp without passing the ``freq`` parameter will be deprecated in a future release."
        )

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
        raise ValueError(
            f"Parameter ``freq`` must be one of {', '.join(FREQUENCIES)}; got {freq}."
        )


def ceil_ts(
    ts: Union[pd.Timestamp, pd.DatetimeIndex], freq=None, future: int = 0
) -> Union[pd.Timestamp, pd.DatetimeIndex]:
    """Ceil timestamp to period boundary.

    i.e., find (earliest) period start that is on or after the timestamp.

    Parameters
    ----------
    ts : Timestamp or DatetimeIndex.
        Timestamp(s) to ceil.
    freq : {'15T' (quarter-hour), 'H' (hour), 'D' (day), 'MS' (month), 'QS' (quarter),
        'AS' (year)}, optional
        What to ceil it to, e.g. 'QS' to get start of quarter it's contained in. If
        none specified, use .freq attribute of timestamp.
    future : int, optional
        0 (default) to get earliest period start that is `ts` or later. 1 (-1) to get
        start of period after (before) that. 2 (-2) .. etc.

    Returns
    -------
    Timestamp(s)
        At begin of period.

    Notes
    -----
    If `ts` is exactly at the start of the period, ceil_ts(ts) == floor_ts(ts) == ts.
    """
    # if ts at start of period, ceil==floor
    offset = 1 if ts != floor_ts(ts, freq, 0) else 0
    return floor_ts(ts, freq, future + offset)


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
    start = ceil_ts(i[0], freq, 0)
    end = floor_ts(ts_right(i[-1]), freq, 0)
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
            return ts_leftright(floor_ts(pd.Timestamp.now(), "AS", 1))
        if left.tz is None:
            return ts_leftright(left.tz_localize("Europe/Berlin"))
        return ts_leftright(left, floor_ts(left, "AS", 1))

    # if we land here, we at least know ts_right.
    if left is pd.NaT:
        back = -1 if right == floor_ts(right, "AS", 0) else 0
        return ts_leftright(floor_ts(right, "AS", back), right)

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
