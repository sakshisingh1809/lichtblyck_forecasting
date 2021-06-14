# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:16:24 2021

@author: ruud.wijtvliet
"""
import datetime as dt
import numpy as np
import lichtblyck as lb
import pandas as pd
import functools
import pytest

_tz = "Europe/Berlin"


def assert_prices_close(p1, p2):
    assert np.abs(p1 - p2) < 0.02

@functools.lru_cache
def power_futures(period_type):
    return lb.prices.power_futures(period_type)

@pytest.mark.parametrize(
    ("trading_day", "period_type", "ts_left_deliv", "base_and_peakprice"),
    [
        (pd.Timestamp("2010-09-08", tz=_tz), "m", pd.Timestamp("2010-10-01", tz=_tz), (49.83, 62.42)),
        (pd.Timestamp("2020-11-11", tz=_tz), "m", pd.Timestamp("2020-12-01", tz=_tz), (32.58, 43.2)),
        (pd.Timestamp("2010-09-08", tz=_tz), "q", pd.Timestamp("2010-10-01", tz=_tz), (50.34, 63.54)),
        (pd.Timestamp("2020-11-11", tz=_tz), "q", pd.Timestamp("2021-01-01", tz=_tz), (38.51, 47.68)),
        (pd.Timestamp("2010-09-08", tz=_tz), "a", pd.Timestamp("2011-01-01", tz=_tz), (50.47, 64.13)),
        (pd.Timestamp("2020-11-11", tz=_tz), "a", pd.Timestamp("2021-01-01", tz=_tz), (39.79, 47.95)),
    ],
)
def test_power_futures(trading_day, period_type, ts_left_deliv, base_and_peakprice):
    p_base, p_peak = base_and_peakprice
    prices = power_futures(period_type)
    assert_prices_close(prices.loc[(ts_left_deliv, trading_day), "p_base"], p_base)
    assert_prices_close(prices.loc[(ts_left_deliv, trading_day), "p_peak"], p_peak)
    if period_type == "m":
        start = trading_day + pd.offsets.MonthBegin(1)
        end = start + pd.offsets.MonthBegin(1)
    elif period_type == "q":
        start = trading_day + pd.offsets.QuarterBegin(1, startingMonth=1)
        end = start + pd.offsets.QuarterBegin(1, startingMonth=1)
    elif period_type == "a":
        start = trading_day + pd.offsets.YearBegin(1)
        end = start + pd.offsets.YearBegin(1)
    else:
        raise ValueError
    b, p, o = lb.prices.duration_bpo(start, end)
    p_offpeak = (p_base * b - p_peak * p) / o
    assert_prices_close(prices.loc[(ts_left_deliv, trading_day), "p_offpeak"], p_offpeak)

# TODO: power spot
# TODO: gas