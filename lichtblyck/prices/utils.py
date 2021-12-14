"""
Utilities for calculating / manipulating price data.
"""

from ..tools.stamps import FREQUENCIES, floor_ts, duration, ts_right
from ..tools.types import Stamp
from ..tools.nits import Q_
from typing import Tuple, Iterable, Union
import pandas as pd


def is_peak_hour(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex]
) -> Union[bool, pd.Series]:
    """Return True/False if timestamp, or each timestamp in index, is a peak hour. More
    precisely: if `ts_left` lies in one of the (left-closed) time intervals that define 
    the peak hour periods."""
    if isinstance(ts_left, pd.Timestamp):
        return ts_left.hour >= 8 and ts_left.hour < 20 and ts_left.isoweekday() < 6
    elif isinstance(ts_left, pd.DatetimeIndex):
        values = [is_peak_hour(ts) for ts in ts_left]
        return pd.Series(values, ts_left).rename("is_peak_hour")
    else:
        raise TypeError(
            f"`ts_left` must be Timestamp or DatetimeIndex; got {type(ts_left)}."
        )


def duration_peak(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex]
) -> Union[Q_, pd.Series]:
    """
    Return total duration of peak periods in a timestamp, or of each timestamp in index.

    See also
    --------
    .tools.stamps.duration
    """
    if ts_left.freq not in FREQUENCIES:
        raise ValueError(f"`.freq` must be one of {', '.join(FREQUENCIES)}.")

    if isinstance(ts_left, pd.Timestamp):
        if ts_left.freq in ["15T", "H"]:
            return duration(ts_left) * is_peak_hour(ts_left)
        elif ts_left.freq == "D":
            return Q_(0 if ts_left.isoweekday() >= 6 else 12, "h")
        else:
            days = pd.date_range(ts_left, ts_right(ts_left), freq="D", closed="left")
            return Q_(sum(days.map(lambda day: day.isoweekday() <= 6) * 12), "h")
    elif isinstance(ts_left, pd.DatetimeIndex):
        if ts_left.freq in ["15T", "H"]:
            return duration(ts_left) * is_peak_hour(ts_left)
        elif ts_left.freq == "D":
            dur = ts_left.map(lambda ts: ts.isoweekday() < 6) * 12
            return pd.Series(dur, ts_left, dtype="pint[h]")  # works even during dst
        else:
            # dur = ts_left.map(duration_peak)  # crashes due to behaviour of .map method
            dur = (duration_peak(ts) for ts in ts_left)  # has unit
            return pd.Series(dur, ts_left, dtype="pint[h]")
    else:
        raise TypeError(
            f"`ts_left` must be Timestamp or DatetimeIndex; got {type(ts_left)}."
        )


def duration_offpeak(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex]
) -> Union[Q_, pd.Series]:
    """
    Return total duration of offpeak periods in a timestamp, or in each timestamp in index.

    See also
    --------
    .tools.stamps.duration
    """
    return duration(ts_left) - duration_peak(ts_left)


def duration_bpo(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex]
) -> Union[pd.Series, pd.DataFrame]:
    """
    Return number of base, peak and offpeak hours in a timestamp, or in each timestamp in index.
    """
    b = duration(ts_left)  # quantity or pint-series
    p = duration_peak(ts_left)  # quantity or pint-series
    if isinstance(ts_left, pd.Timestamp):
        return pd.Series({"base": b, "peak": p, "offpeak": b - p}, dtype="pint[h]")
    else:
        return pd.DataFrame({"base": b, "peak": p, "offpeak": b - p}, dtype="pint[h]")


def ts_leftright(
    ts_trade: Stamp, period_type: str = "m", period_start: int = 1
) -> Tuple[Stamp]:
    """
    Find start and end of delivery period.

    Parameters
    ----------
    ts_trade : datetime
        Trading timestamp
    period_type : str
        One of {'d' (day), 'm' (month, default), 'q' (quarter), 's' (season), 'a' (year)}
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
        ts_left = floor_ts(ts_left_trade, period_start, freq)
        ts_right = floor_ts(ts_left, 1, freq)
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

