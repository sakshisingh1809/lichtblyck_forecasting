import datetime
import lichtblyck as lb
import pandas as pd
import numpy as np


def random_ts(localized=False):
    ts = pd.Timestamp("2000-1-1") + datetime.timedelta(
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
    return ts - datetime.timedelta(
        minutes=ts.minute % 15, seconds=ts.second, microseconds=ts.microsecond
    )


def test_power_series_fromsource():
    for source in lb.tlp.power.SOURCES:
        print(source)
        tlp_s = lb.tlp.power.series_fromsource(source["name"], spec=1)
        # Number of distinct times and temperatures.
        assert len(tlp_s.index.get_level_values("time_left_local").unique()) == 96
        assert len(tlp_s.index.get_level_values("t").unique()) > 20
        # Monotonously increasing function? (strict monotony too strong criterium)
        diff = tlp_s.sort_index().diff().dropna()
        assert sum(diff > 1e-3) / len(diff) < 0.2


def test_power_fromsource():
    tlp1 = lb.tlp.power.fromsource(2, spec=1)
    tlp2 = lb.tlp.power.fromsource(2, spec=2)
    for _ in range(50):
        ts = random_ts2()
        for t in np.linspace(-30, 30, 50):
            assert tlp1(t, ts) <= tlp1(t - 10, ts)
            assert tlp1(t, ts) <= tlp2(t, ts)


def test_gas_D14():
    tlp1 = lb.tlp.gas.D14(kw=1)
    tlp2 = lb.tlp.gas.D14(kw=2)
    for t in np.linspace(-30, 30, 50):
        assert tlp1(t) <= tlp1(t - 1)
        assert tlp1(t) <= tlp2(t)


def test_convert():
    time = pd.date_range("2020-01-01", "2020-01-02", freq="15T", closed="left").time
    t = np.arange(-20, 20)
    tlp_s = pd.Series(
        np.random.rand(len(t) * len(time)),
        pd.MultiIndex.from_product([t, time], names=["t", "time_left_local"]),
    )
    tlp = lb.tlp.convert.series2function(tlp_s)
    # Check exact matches.
    for _ in range(40):
        ts = datetime.datetime.combine(
            pd.Timestamp("2000-1-1")
            + datetime.timedelta(days=np.random.randint(15000)),
            np.random.choice(time),
        )
        tmpr = np.random.choice(t)
        assert np.isclose(tlp(tmpr, ts), tlp_s[(tmpr, ts.time())])
    # Check random temperature.
    for _ in range(40):
        ts = datetime.datetime.combine(
            pd.Timestamp("2000-1-1")
            + datetime.timedelta(days=np.random.randint(15000)),
            np.random.choice(time),
        )
        tmpr = -25 + 50 * np.random.rand()
        near_tmpr = min(max(np.rint(tmpr), -20), 19)
        assert np.isclose(tlp(tmpr, ts), tlp_s[(near_tmpr, ts.time())])
    # Check random timestamp.
    for _ in range(40):
        ts = random_ts2()
        near_ts = prev_qh(ts)
        tmpr = np.random.choice(t)
        assert np.isclose(tlp(tmpr, ts), tlp_s[(tmpr, near_ts.time())])
    # Check random timestamp and temperature.
    for _ in range(40):
        ts = random_ts2()
        near_ts = prev_qh(ts)
        tmpr = -25 + 50 * np.random.rand()
        near_tmpr = min(max(np.rint(tmpr), -20), 19)
        assert np.isclose(tlp(tmpr, ts), tlp_s[(near_tmpr, near_ts.time())])


def test_toload_smooth():
    for _ in range(40):
        ts = pd.date_range(
            "2020-01-01", freq="D", periods=400, closed="left", tz="Europe/Berlin"
        )
        t = pd.Series((3 * (np.random.rand(len(ts)) - 0.5)).cumsum() + 10, ts)
        t_smooth = lb.tlp.toload._smooth_temperature(t, [10, 8, 5, 2.5, 1])
        correct = total = 0
        for w1, w2 in zip(t.rolling(20), t_smooth.rolling(20)):
            total += 1
            if w1.std() >= w2.std():
                correct += 1
        assert 0.65 < correct / total <= 1
