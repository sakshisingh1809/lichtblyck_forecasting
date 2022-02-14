"""Utilities for calculating / manipulating price data."""

import warnings
from ..tools import stamps
from ..tools.types import Stamp
from ..tools.nits import Q_
from typing import Tuple, Iterable, Type, Union, Optional
import pandas as pd


def is_peak_hour(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex]
) -> Union[bool, pd.Series]:
    """
    Boolean value indicating if a timestamp is in a peak period or not.

    Parameters
    ----------
    ts_left : Union[pd.Timestamp, pd.DatetimeIndex]
        Timestamp(s) for which to calculate if it falls in a peak period.

    More precisely: if timestamp lies in one of the (left-closed) time intervals that
    define the peak hour periods.

    Returns
    -------
    bool (if ts_left is Timestamp) or Series (if ts_left is DatetimeIndex).
    """
    if isinstance(ts_left, pd.Timestamp):
        return ts_left.hour >= 8 and ts_left.hour < 20 and ts_left.isoweekday() < 6
    elif isinstance(ts_left, pd.DatetimeIndex):
        ispeak = [is_peak_hour(ts) for ts in ts_left]
        return pd.Series(ispeak, ts_left).rename("is_peak_hour")

    raise TypeError(
        f"Parameter ``ts_left`` must be ``Timestamp`` or ``DatetimeIndex`` instance; got {type(ts_left)}."
    )


def duration_peak(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex], freq: str = None
) -> Union[Q_, pd.Series]:
    """
    Total duration of peak periods in a timestamp.

    See also
    --------
    .tools.stamps.duration
    """

    if isinstance(ts_left, pd.Timestamp) and freq is None:
        warnings.warn(
            "Using a Timestamp without passing the ``freq`` parameter will be deprecated in a future release."
        )

    if freq is None:
        freq = ts_left.freq

    if freq not in stamps.FREQUENCIES:
        raise ValueError(
            f"Frequency must be one of {', '.join(stamps.FREQUENCIES)}; got {freq}."
        )

    if isinstance(ts_left, pd.Timestamp):
        if freq in ["15T", "H"]:
            return stamps.duration(ts_left, freq) * is_peak_hour(ts_left)
        elif freq == "D":
            return Q_(0 if ts_left.isoweekday() >= 6 else 12, "h")
        else:
            ts_right = stamps.ts_right(ts_left, freq)
            days = pd.date_range(ts_left, ts_right, freq="D", closed="left")
            return Q_(sum(days.map(lambda day: day.isoweekday() <= 6) * 12), "h")

    elif isinstance(ts_left, pd.DatetimeIndex):
        if freq in ["15T", "H"]:
            return stamps.duration(ts_left, freq) * is_peak_hour(ts_left)
        elif freq == "D":
            hours = ts_left.map(lambda ts: ts.isoweekday() < 6) * 12  # no unit
            return pd.Series(hours, ts_left, dtype="pint[h]")  # works even during dst
        else:
            # dur = ts_left.map(duration_peak)  # crashes due to behaviour of .map method
            hours = (duration_peak(ts, freq) for ts in ts_left)  # has unit
            return pd.Series(hours, ts_left, dtype="pint[h]")

    raise TypeError(
        f"Parameter ``ts_left`` must be ``Timestamp`` or ``DatetimeIndex`` instance; got {type(ts_left)}."
    )


def duration_offpeak(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex], freq: str = None
) -> Union[Q_, pd.Series]:
    """
    Total duration of offpeak periods in a timestamp.

    See also
    --------
    .tools.stamps.duration
    """
    return stamps.duration(ts_left, freq) - duration_peak(ts_left, freq)


def duration_bpo(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex], freq: str = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Duration of base, peak and offpeak periods in a timestamp.

    Parameters
    ----------
    ts_left : Union[pd.Timestamp, pd.DatetimeIndex]
        Timestamp(s) for which to calculate the base, peak and offpeak durations.
    freq : {'15T' (quarter-hour), 'H' (hour), 'D' (day), 'MS' (month), 'QS' (quarter),
        'AS' (year)}, optional
        Frequency to use in determining the durations.
        If none specified, use ``.freq`` attribute of ``ts_left``.

    Returns
    -------
    Series (if ts_left is Timestamp) or DataFrame (if ts_left is DatetimeIndex).
    """
    b = stamps.duration(ts_left, freq)  # quantity or pint-series
    p = duration_peak(ts_left, freq)  # quantity or pint-series

    if isinstance(ts_left, pd.Timestamp):
        return pd.Series({"base": b, "peak": p, "offpeak": b - p}, dtype="pint[h]")
    elif isinstance(ts_left, pd.DatetimeIndex):
        return pd.DataFrame({"base": b, "peak": p, "offpeak": b - p}, dtype="pint[h]")

    raise TypeError(
        f"Parameter ``ts_left`` must be ``pandas.Timestamp`` or ``pandas.DatetimeIndex`` instance; got {type(ts_left)}."
    )


def ts_leftright(
    ts_trade: Stamp, period_type: str = "m", period_start: int = 1
) -> Tuple[Stamp]:
    """
    Find start and end of delivery period.

    Parameters
    ----------
    ts_trade : datetime
        Trading timestamp
    period_type : {'d' (day), 'm' (month, default), 'q' (quarter), 's' (season), 'a' (year)}
    period_start : int
        1 = next/coming (full) period, 2 = period after that, etc.

    Returns
    -------
    (datetime, datetime)
        left and right timestamp of delivery period.
    """
    ts_left_trade = ts_trade.floor("d")  # start of day

    if period_type in ["m", "q", "a"]:
        freq = period_type.upper() + "S"
        ts_left = stamps.floor_ts(ts_left_trade, freq, period_start)
        ts_right = stamps.floor_ts(ts_left, freq, 1)
    elif period_type == "d":
        ts_left = ts_trade.floor("d") + pd.Timedelta(days=period_start)
        ts_right = ts_left + pd.Timedelta(days=1)
    elif period_type == "s":
        ts_left, ts_right = ts_leftright(ts_trade, "q", period_start * 2 - 1)
        nextq = pd.offsets.QuarterBegin(1, startingMonth=1)
        ts_right = ts_right + nextq  # make 6 months long
        if ts_left.month % 2 == 1:  # season must start on even month
            ts_left, ts_right = ts_left + nextq, ts_right + nextq
    else:
        raise ValueError("Invalid value for parameter `period_type`.")
    return ts_left, ts_right
