"""
Utilities for calculating / manipulating price data.
"""

from typing import Tuple, Iterable
import pandas as pd
import numpy as np


def is_peak_hour(ts) -> bool:
    """Return True if timestamp 'ts' is a peak hour. More precisely: if 'ts'
    lies in one of the (left-closed) time intervals that define the peak hour
    periods."""
    if isinstance(ts, Iterable):
        return np.vectorize(is_peak_hour)(ts)
    return ts.hour >= 8 and ts.hour < 20 and ts.isoweekday() < 6


def duration_bpo(ts_left, ts_right) -> Tuple[float]:
    """
    Return number of base, peak and offpeak hours in interval [ts_left, ts_right).
    Timestamps must coincide with quarterhour start.
    """
    # if isinstance(ts_left, Iterable):
    #     return np.vectorize(duration_bpo)(ts_left, ts_right)
    for ts in [ts_left, ts_right]:
        if ts.second != 0 or ts.minute % 15 != 0:
            raise ValueError("Timestamps must cooincide with quarterhour start.")
    # Duration of base.
    duration_base = (ts_right - ts_left).total_seconds() / 3600
    # Duration of peak.
    duration_peak = 0
    # to speed things up, do quarterhours only for non-full days.
    ts_left_day, ts_right_day = ts_left.ceil("d"), ts_right.floor("d")  # full days
    for ts in pd.date_range(ts_left, ts_left_day, freq="15T", closed="left"):
        if is_peak_hour(ts):
            duration_peak += 0.25
    for ts in pd.date_range(ts_left_day, ts_right_day, freq="D", closed="left"):
        if ts.isoweekday() < 6:
            duration_peak += 12
    for ts in pd.date_range(ts_right_day, ts_right, freq="15T", closed="left"):
        if is_peak_hour(ts):
            duration_peak += 0.25
    return duration_base, duration_peak, duration_base - duration_peak


def ts_deliv(ts_trade, period_type: str = "m", period_start: int = 1):
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
    if period_start <= 0:  # to get current period or before: first go to next period
        plus, minus = 1, period_start - 1
    else:
        plus, minus = period_start, 0
    if period_type == "d":
        ts_left = ts_left_trade + pd.Timedelta(days=plus) + pd.Timedelta(days=minus)
        ts_right = ts_left + pd.Timedelta(days=1)
    elif period_type == "m":
        ts_left = (
            ts_left_trade + pd.offsets.MonthBegin(plus) + pd.offsets.MonthBegin(minus)
        )
        ts_right = ts_left + pd.offsets.MonthBegin(1)
    elif period_type == "q":
        ts_left = (
            ts_left_trade
            + pd.offsets.QuarterBegin(plus, startingMonth=1)
            + pd.offsets.QuarterBegin(minus, startingMonth=1)
        )
        ts_right = ts_left + pd.offsets.QuarterBegin(1, startingMonth=1)
    elif period_type == "s":
        ts_left, ts_right = ts_deliv(ts_trade, "q", period_start * 2 - 1)
        nextq = pd.offsets.QuarterBegin(1, startingMonth=1)
        ts_right = ts_right + nextq  # make 6 months long
        if ts_left.month % 2 == 1:  # season must start on even month
            ts_left, ts_right = ts_left + nextq, ts_right + nextq
    elif period_type == "a":
        ts_left = (
            ts_left_trade + pd.offsets.YearBegin(plus) + pd.offsets.YearBegin(minus)
        )
        ts_right = ts_left + pd.offsets.YearBegin(1)
    else:
        raise ValueError("Invalid value for parameter `period_type`.")
    return ts_left, ts_right


def p_offpeak(p_base, p_peak, duration_base, duration_peak) -> float:
    """Return offpeak price from base and peak price and number of base and peak hours."""
    # if isinstance(p_base, Iterable):
    #     return np.vectorize(p_offpeak)(p_base, p_peak, duration_base, duration_peak)
    return (p_base * duration_base - p_peak * duration_peak) / (
        duration_base - duration_peak
    )
