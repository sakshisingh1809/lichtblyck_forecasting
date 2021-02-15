from datetime import datetime, timedelta
from lichtblyck.core.dev import get_index
import lichtblyck as lb
import pandas as pd
import numpy as np
import pytest


def random_ts(localized=False):
    ts = pd.Timestamp("2000-1-1") + timedelta(
        days=np.random.randint(15000),
        hours=np.random.randint(24),
        minutes=np.random.randint(60),
        seconds=np.random.randint(60),
    )
    if localized:
        ts = ts.tz_localize("Europe/Berlin")
    return ts


def random_ts2():
    return random_ts(np.random.choice([True, False]))


def prev_qh(ts):
    return ts - timedelta(
        minutes=ts.minute % 15, seconds=ts.second, microseconds=ts.microsecond
    )


def assert_w_vs_t_series(s, threshold=0.9):
    """Check if colder means more consumption. s is Series of consumption values 
    [MW] as function of single index (temperature [degC]). threshold = lowest 
    allowed fraction of values that defy the rule."""
    diff = s.sort_index().diff().dropna()
    total = len(diff)
    ok = len(diff <= 0)
    assert ok / total >= threshold


def assert_w_vs_w_series(w_more, w_less, threshold=0.9):
    """Check if consumption in series w_more is more than in w_less. threshold = 
    lowest allowed fraction of values that defy the rule."""
    diff = w_more - w_less
    total = len(diff)
    ok = len(diff <= 0)
    assert ok / total >= threshold


# def assert_w_and_t(w, t, influences=['time'], threshold=0.9):
#     """Check if colder means more consumption. w is Series of consumption values
#     [MW], and t is Series of temperature values; both with same (timestamp) index.
#     'influences': ['time', 'weekd', 'month'], combination of things that determine
#     consumption, apart from temperature.
#     threshold = lowest allowed fraction of values that defy the rule."""
#     df = pd.DataFrame({'w': w, 't': t, 'i': t.index})
#     split = pd.DataFrame(index=df.index)
#     if 'time' in influences:
#         split['time'] = split.index.dt.time
#     if 'weekd' in influences:
#         split['weekd'] = split.index.dt.weekday()
#     if 'month' in influences:
#         split['month'] = split.index.dt.month()


#     s = df.set_index('t').sort_index()

#     assert_w_vs_t(s, threshold)


@pytest.mark.parametrize("source", np.random.choice(lb.tlp.power.SOURCES, 8, False))
def test_power_series_fromsource(source):
    tlp_s = lb.tlp.power._series_fromsource(source["name"], spec=1)
    # Number of distinct times and temperatures.
    assert len(tlp_s.index.unique("time_left_local")) == 96
    assert len(tlp_s.index.unique("t")) > 20
    # Monotonously increasing function? (strict monotony too strong criterium)
    for time in tlp_s.index.unique("time_left_local"):
        assert_w_vs_t_series(tlp_s.xs(time, level="time_left_local"), 0.8)


@pytest.mark.parametrize("source", np.random.choice(lb.tlp.power.SOURCES, 8, False))
def test_power_fromsource(source):
    i = get_index(freq="D")  # some random time index in days
    t = lb.PfSeries(np.random.uniform(-30, 30, len(i)), i)
    tlp1 = lb.tlp.power.fromsource(source["name"], spec=1)
    w1 = tlp1(t)
    w1b = tlp1(lb.PfSeries(t - 5))
    tlp2 = lb.tlp.power.fromsource(
        source["name"], spec=2
    )  # should have double the consumption
    w2 = tlp2(t)
    assert np.allclose(w2, w1 * 2)
    assert_w_vs_w_series(w1b, w1)


def test_gas_D14():
    i = get_index(freq="D")  # some random time index in days
    t = lb.PfSeries(np.random.uniform(-30, 30, len(i)), i)
    tlp1 = lb.tlp.gas.D14(kw=1)
    w1 = tlp1(t)
    w1b = tlp1(lb.PfSeries(t - 5))
    tlp2 = lb.tlp.gas.D14(kw=2)
    w2 = tlp2(t)  # should have double the consumption
    assert np.allclose(w2, 2 * w1)
    assert_w_vs_w_series(w1b, w1, 1)


# def test_convert():
#     time = pd.date_range("2020-01-01", "2020-01-02", freq="15T", closed="left").time
#     t = np.arange(-20, 20)
#     tlp_s = pd.Series(
#         np.random.rand(len(t) * len(time)),
#         pd.MultiIndex.from_product([t, time], names=["t", "time_left_local"]),
#     )
#     tlp = lb.tlp.convert.series2function(tlp_s)
#     # Check exact matches.
#     for _ in range(40):
#         ts = datetime.combine(
#             pd.Timestamp("2000-1-1")
#             + timedelta(days=np.random.randint(15000)),
#             np.random.choice(time),
#         )
#         tmpr = np.random.choice(t)
#         assert np.isclose(tlp(tmpr, ts), tlp_s[(tmpr, ts.time())])
#     # Check random temperature.
#     for _ in range(40):
#         ts = datetime.combine(
#             pd.Timestamp("2000-1-1")
#             + timedelta(days=np.random.randint(15000)),
#             np.random.choice(time),
#         )
#         tmpr = -25 + 50 * np.random.rand()
#         near_tmpr = min(max(np.rint(tmpr), -20), 19)
#         assert np.isclose(tlp(tmpr, ts), tlp_s[(near_tmpr, ts.time())])
#     # Check random timestamp.
#     for _ in range(40):
#         ts = random_ts2()
#         near_ts = prev_qh(ts)
#         tmpr = np.random.choice(t)
#         assert np.isclose(tlp(tmpr, ts), tlp_s[(tmpr, near_ts.time())])
#     # Check random timestamp and temperature.
#     for _ in range(40):
#         ts = random_ts2()
#         near_ts = prev_qh(ts)
#         tmpr = -25 + 50 * np.random.rand()
#         near_tmpr = min(max(np.rint(tmpr), -20), 19)
#         assert np.isclose(tlp(tmpr, ts), tlp_s[(near_tmpr, near_ts.time())])


# def test_toload_smooth():
#     for _ in range(40):
#         ts = pd.date_range(
#             "2020-01-01", freq="D", periods=400, closed="left", tz="Europe/Berlin"
#         )
#         t = pd.Series((3 * (np.random.rand(len(ts)) - 0.5)).cumsum() + 10, ts)
#         t_smooth = lb.tlp.toload._smooth_temperature(t, [10, 8, 5, 2.5, 1])
#         correct = total = 0
#         for w1, w2 in zip(t.rolling(20), t_smooth.rolling(20)):
#             total += 1
#             if w1.std() >= w2.std():
#                 correct += 1
#         assert 0.65 < correct / total <= 1
