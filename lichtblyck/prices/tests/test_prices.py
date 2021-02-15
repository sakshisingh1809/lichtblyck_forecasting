# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:16:24 2021

@author: ruud.wijtvliet
"""
import datetime as dt
import lichtblyck as lb
import pandas as pd
import numpy as np
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


@pytest.mark.parametrize(
    ("start", "end", "bp"),
    [
        (
            pd.Timestamp("2021-01-01", tz=_tz),
            pd.Timestamp("2021-01-03", tz=_tz),
            (48, 12),
        ),
        (
            pd.Timestamp("2021-01-02", tz=_tz),
            pd.Timestamp("2021-01-04", tz=_tz),
            (48, 0),
        ),
        (
            pd.Timestamp("2021-01-01", tz=_tz),
            pd.Timestamp("2021-02-01", tz=_tz),
            (744, 252),
        ),
        (
            pd.Timestamp("2021-02-01", tz=_tz),
            pd.Timestamp("2021-03-01", tz=_tz),
            (672, 240),
        ),
        (
            pd.Timestamp("2021-03-01", tz=_tz),
            pd.Timestamp("2021-04-01", tz=_tz),
            (743, 276),
        ),
    ],
)
def test_hours_bpo(start, end, bp):
    bpo = lb.prices.hours_bpo(start, end)
    assert bpo == (*bp, bp[0] - bp[1])


@pytest.mark.parametrize(
    ("year", "text", "expected"),
    [
        (2020, "Q3", dt.datetime(2020, 7, 1)),
        (2020, "HY2", dt.datetime(2020, 7, 1)),
        (2020, "Cal", dt.datetime(2020, 1, 1)),
        (2020, "Wi", dt.datetime(2020, 10, 1)),
        (2020, "Su", dt.datetime(2020, 4, 1)),
        (2020, "M3", dt.datetime(2020, 3, 1)),
    ],
)
def test_ts_left1(year, text, expected):
    for s in (
        f"{year}{text}",
        f"{year}-{text}",
        f"{year} {text}",
        f"{text}-{year}",
        f"{text} {year}",
    ):
        assert lb.prices.ts_left(s) == expected


@functools.lru_cache
def power_front(period_type):
    return lb.prices._power_front(period_type)


@pytest.mark.parametrize(
    ("period_type", "trading_day", "base_and_peakprice"),
    [
        ("m", pd.Timestamp("2010-09-08", tz=_tz), (49.83, 62.42)),
        ("m", pd.Timestamp("2020-11-11", tz=_tz), (32.58, 43.2)),
        ("q", pd.Timestamp("2010-09-08", tz=_tz), (50.34, 63.54)),
        ("q", pd.Timestamp("2020-11-11", tz=_tz), (38.51, 47.68)),
        ("a", pd.Timestamp("2010-09-08", tz=_tz), (50.47, 64.13)),
        ("a", pd.Timestamp("2020-11-11", tz=_tz), (39.79, 47.95)),
    ],
)
def test_power_front(period_type, trading_day, base_and_peakprice):
    p_base, p_peak = base_and_peakprice
    prices = power_front(period_type)
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
