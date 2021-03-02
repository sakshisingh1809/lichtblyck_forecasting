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


@pytest.mark.parametrize(
    ("p_base", "p_peak", "h_base", "h_peak", "expected_p_offpeak"),
    [(100, 100, 10, 5, 100), (100, 200, 20, 4, 75)],
)
def test_p_offpeak(p_base, p_peak, h_base, h_peak, expected_p_offpeak):
    assert_prices_close(
        lb.prices.p_offpeak(p_base, p_peak, h_base, h_peak), expected_p_offpeak
    )


@functools.lru_cache
def power_futures(period_type, period_start):
    return lb.prices.power_futures(period_type, period_start)


@pytest.mark.parametrize(
    ("trading_day", "period_type", "period_start", "base_and_peakprice"),
    [
        (pd.Timestamp("2010-09-08", tz=_tz), "m", 1, (49.83, 62.42)),
        (pd.Timestamp("2020-11-11", tz=_tz), "m", 1, (32.58, 43.2)),
        (pd.Timestamp("2010-09-08", tz=_tz), "q", 1, (50.34, 63.54)),
        (pd.Timestamp("2020-11-11", tz=_tz), "q", 1, (38.51, 47.68)),
        (pd.Timestamp("2010-09-08", tz=_tz), "a", 1, (50.47, 64.13)),
        (pd.Timestamp("2020-11-11", tz=_tz), "a", 1, (39.79, 47.95)),
    ],
)
def test_power(trading_day, period_type, period_start, base_and_peakprice):
    p_base, p_peak = base_and_peakprice
    prices = power_futures(period_type, period_start)
    assert_prices_close(prices.loc[trading_day, "p_base"], p_base)
    assert_prices_close(prices.loc[trading_day, "p_peak"], p_peak)
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
    b, p, o = lb.prices.hours_bpo(start, end)
    p_offpeak = (p_base * b - p_peak * p) / o
    assert_prices_close(prices.loc[trading_day, "p_offpeak"], p_offpeak)
