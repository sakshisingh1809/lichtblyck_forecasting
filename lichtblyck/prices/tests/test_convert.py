from lichtblyck.prices import convert, utils
from lichtblyck.core import dev, functions
from lichtblyck.tools import tools
import numpy as np
import pandas as pd
import pytest
import functools


@pytest.mark.parametrize(
    ("p_b", "p_p", "p_o", "ts_left", "ts_right"),
    [
        (1, 1, 1, "2020-01-06", "2020-01-11"),  # working days; 50% peak
        (1, 2, 0, "2020-01-06", "2020-01-11"),
        (1, 3, -1, "2020-01-06", "2020-01-11"),
        (100, 100, 100, "2020-01-06", "2020-01-11"),
        (1.5, 4.5, 0.5, "2020-01-12", "2020-01-14"),  # sunday and monday; 25% peak
        (100, 200, 80, "2020-01-11", "2020-01-14"),  # weekend and monday; 16.667% peak
        (100, np.nan, 100, "2020-01-11", "2020-01-13"),  # weekend only; 0% peak
    ],
)
def test_pbaseppeakpoffpeak_explicit(p_b, p_p, p_o, ts_left, ts_right):
    assert np.isclose(convert.p_peak(p_b, p_o, ts_left, ts_right), p_p, equal_nan=True)
    if p_p is np.nan:
        p_p = 0
    assert np.isclose(convert.p_offpeak(p_b, p_p, ts_left, ts_right), p_o)
    assert np.isclose(convert.p_base(p_p, p_o, ts_left, ts_right), p_b)


@pytest.mark.parametrize(
    "bpoframe",
    [
        pd.DataFrame(
            {
                "p_peak": [100, 100, 100, 100],
                "p_base": [80, 80, 80, 80],
                "p_offpeak": [68.2051282, 69.4736842, 68.9770355, 68.421053],
            },
            pd.date_range("2020", periods=4, freq="MS", tz="Europe/Berlin"),
        ), 
        pd.DataFrame(
            {
                "p_peak": [100, 100, 100, 100],
                "p_base": [80, 80, 80, 80],
                "p_offpeak": [68.8510638, 68.8699360, 68.9361702, 68.9361702],
            },
            pd.date_range("2020", periods=4, freq="AS", tz="Europe/Berlin"),
        )
    ],
)
def test_completebpoframe_explicit(bpoframe):
    p_b, p_p, p_o = bpoframe["p_base"], bpoframe["p_peak"], bpoframe["p_offpeak"]
    p_b2 = convert.complete_bpoframe(pd.DataFrame({"p_peak": p_p, "p_offpeak": p_o}))[
        "p_base"
    ]

    p_o2 = convert.complete_bpoframe(pd.DataFrame({"p_peak": p_p, "p_base": p_b}))[
        "p_offpeak"
    ]

    p_p2 = convert.complete_bpoframe(pd.DataFrame({"p_offpeak": p_o, "p_base": p_b}))[
        "p_peak"
    ]
    pd.testing.assert_series_equal(p_b, p_b2, atol=0.01, check_dtype=False)
    pd.testing.assert_series_equal(p_o, p_o2, atol=0.01, check_dtype=False)
    pd.testing.assert_series_equal(p_p, p_p2, atol=0.01, check_dtype=False)




@pytest.fixture(params=["var", "MS", "QS", "AS"])
def short_freq(request):
    return request.param


@pytest.fixture(params=["MS", "QS", "AS"])
def long_freq(request):
    return request.param


@pytest.fixture(params=["H", "15T"])
def tseries_freq(request):
    return request.param


@pytest.fixture(scope="session")
def series_and_frames():

    i_15T = pd.date_range("2020", "2022", freq="15T", tz="Europe/Berlin", closed="left")

    def exp(mean_p, ampl_p, mean_op, ampl_o):
        tz = "Europe/Berlin"
        start, end = (pd.Timestamp("2020", tz=tz), pd.Timestamp("2021", tz=tz))
        angle = lambda ts: 2 * np.pi * (ts - start) / (end - start)

        def expectation(ts):
            m, a = (mean_p, ampl_p) if utils.is_peak_hour(ts) else (mean_op, ampl_o)
            alpha = angle(ts) + np.pi / 12
            return m + a * np.cos(alpha) + a / 10 * np.cos(alpha * 15)

        return np.vectorize(expectation)  # make sure it accepts arrays

    keys = ["peak", "offpeak", "base"]
    pricefunc = exp(50, 40, 30, 25)
    pricevals = np.random.normal(pricefunc(i_15T), 5)  # added noise
    source = pd.DataFrame({"p": pricevals, "ts": i_15T})  # with integer index

    # (Quarter)hourly timeseries with variable prices.
    tseries = {
        "var": {
            "15T": tools.set_ts_index(source, "ts")["p"],
            "H": tools.set_ts_index(
                source.groupby(source.index // 4).agg({"p": "mean", "ts": "first"}),
                "ts",
            )["p"],
        }
    }

    # Dataframes with base, peak, offpeak prices.

    def isstart(period):
        return lambda ts: ts.floor("D") == ts and getattr(ts, f"is_{period}_start")

    bpoframes_sourcedata = {
        freq: {**{key: [] for key in keys}, "index": [], "new": isstart(period)}
        for freq, period in {"MS": "month", "QS": "quarter", "AS": "year",}.items()
    }

    for ts, p in tseries["var"]["H"].items():
        ispeak = utils.is_peak_hour(ts)
        for freq, dic in bpoframes_sourcedata.items():
            if dic["new"](ts):
                dic["index"].append(ts)
                for key in keys:
                    dic[key].append([])

            dic["base"][-1].append(p)
            dic["peak" if ispeak else "offpeak"][-1].append(p)

    bpoframes = {}
    for freq, dic in bpoframes_sourcedata.items():
        for key in keys:
            dic[key] = [np.mean(l) if len(l) else np.nan for l in dic[key]]
        df = pd.DataFrame({f"p_{key}": dic[key] for key in keys}, dic["index"])
        bpoframe = tools.set_ts_index(df.resample(freq).asfreq())
        bpoframes[freq] = bpoframe

    # (Quarter)hourly timeseries with uniform peak and offpeak prices.
    for period, bpoframe in bpoframes.items():
        i0 = tseries["var"]["H"].index
        offset_f = {
            "MS": pd.offsets.MonthBegin,
            "QS": lambda ts: pd.offsets.QuarterBegin(ts, startingMonth=1),
            "AS": pd.offsets.YearBegin,
        }[period]
        i1 = i0.map(lambda ts: ts.floor("d") + offset_f(1) + offset_f(-1))
        ispeak = utils.is_peak_hour(i0)
        df = bpoframe.loc[i1, :]
        s = pd.Series(np.where(ispeak, df["p_peak"], df["p_offpeak"]), i0).rename("p")
        tseries[period] = {}
        tseries[period]["H"] = s
        tseries[period]["15T"] = functions.changefreq_avg(s, "15T")

    return tseries, bpoframes

_keys = ["p_base", "p_peak", "p_offpeak"]

def test_pbaseppeakpoffpeak(series_and_frames, long_freq):
    # long_freq: uniform frequency to start with {'ms', 'qs', 'as'}
    # after conversion, values must be same as one provided by fixture
    tseries, bpoframes = series_and_frames
    idx = np.random.randint(len(bpoframes[long_freq].index))
    ts_left = bpoframes[long_freq].index[idx]
    ts_right = bpoframes[long_freq].ts_right[idx]
    values_ref = bpoframes[long_freq].loc[ts_left, :]

    for key, f in [
        ("p_peak", convert.p_peak),
        ("p_base", convert.p_base),
        ("p_offpeak", convert.p_offpeak),
    ]:
        othervalues = [
            values_ref[k] for k in _keys if k != key
        ]
        value_test = f(*othervalues, ts_left, ts_right)
        assert np.isclose(value_test, values_ref[key])

def test_completebpoframe(series_and_frames, long_freq):
    # long_freq: uniform frequency to start with {'ms', 'qs', 'as'}
    # after conversion, values must be same as one provided by fixture
    tseries, bpoframes = series_and_frames
    df_ref = bpoframes[long_freq]

    for key in _keys:
        partialframe = df_ref[[okey for okey in _keys if okey != key]]
        df_test = convert.complete_bpoframe(partialframe)
        for testkey in _keys:
            pd.testing.assert_series_equal(df_test[testkey], df_ref[testkey])


def test_tseries2singlebpo(series_and_frames, long_freq, tseries_freq):
    # tseries_freq: hour or quarterhour {'h', '15t'}
    # long_freq: uniform frequency to start with {'ms', 'qs', 'as'}
    # after conversion, values must be same as one provided by fixture
    tseries, bpoframes = series_and_frames
    ts = np.random.choice(bpoframes[long_freq].index)
    values_ref = bpoframes[long_freq].loc[ts, :]

    s = tseries[long_freq][tseries_freq]
    s_source = s[(s.index >= ts) & (s.index < ts + ts.freq)]
    values_test = convert.tseries2singlebpo(s_source)

    for key in ["p_peak", "p_base", "p_offpeak"]:
        assert np.isclose(values_test[key], values_ref[key])


def test_bpoframe2tseries(series_and_frames, long_freq, tseries_freq):
    # tseries_freq: hour or quarterhour {'h', '15t'}
    # long_freq: uniform frequency to start with {'ms', 'qs', 'as'}
    # after conversion, timeseries must be same as one provided by fixture.
    tseries, bpoframes = series_and_frames
    df_source = bpoframes[long_freq]
    s_test = convert.bpoframe2tseries(df_source, tseries_freq)
    s_ref = tseries[long_freq][tseries_freq]
    pd.testing.assert_series_equal(s_test, s_ref)


def test_bpoframe2bpoframe(series_and_frames, short_freq, long_freq):
    # short_freq: uniform frequency to start with {'ms', 'qs', 'as'}
    # long_freq: uniform frequency to convert to {'ms', 'qs', 'as'}
    # after conversion, bpoframe must be same as one provided by fixture.
    tseries, bpoframes = series_and_frames
    if short_freq == "var" or functions.freq_up_or_down(short_freq, long_freq) > 0:
        return  # if upsampling, the resulting series will not equal the reference
    df_source = bpoframes[short_freq]
    df_test = convert.bpoframe2bpoframe(df_source, long_freq)
    df_ref = bpoframes[long_freq]
    for col in df_ref.columns:
        if col in df_test.columns:
            pd.testing.assert_series_equal(df_test[col], df_ref[col])


def test_tseries2bpoframe(series_and_frames, short_freq, long_freq, tseries_freq):
    # tseries_freq: hour or quarterhour {'h', '15t'}
    # short_freq: uniform frequency to start with {'var', 'ms', 'qs', 'as'}
    # long_freq: uniform frequency to convert to {'ms', 'qs', 'as'}
    # after conversion, bpoframe must be same as one provided by fixture.
    tseries, bpoframes = series_and_frames
    if short_freq != "var" and functions.freq_up_or_down(short_freq, long_freq) > 0:
        return  # if upsampling, the resulting series will not equal the reference
    s_source = tseries[short_freq][tseries_freq]
    df_test = convert.tseries2bpoframe(s_source, long_freq)
    df_ref = bpoframes[long_freq]
    for col in df_ref.columns:
        if col in df_test.columns:
            pd.testing.assert_series_equal(df_test[col], df_ref[col])


def test_tseries2tseries(series_and_frames, short_freq, long_freq, tseries_freq):
    # tseries_freq: hour or quarterhour {'h', '15t'}
    # short_freq: uniform frequency to start with {'var', 'ms', 'qs', 'as'}
    # long_freq: uniform frequency to convert to {'ms', 'qs', 'as'}
    # after conversion, timeseries must be same as one provided by fixture.
    tseries, bpoframes = series_and_frames
    if short_freq != "var" and functions.freq_up_or_down(short_freq, long_freq) > 0:
        return  # if upsampling, the resulting series will not equal the reference
    s_source = tseries[short_freq][tseries_freq]
    s_test = convert.tseries2tseries(s_source, long_freq)
    s_ref = tseries[long_freq][tseries_freq]
    pd.testing.assert_series_equal(s_test, s_ref)


# TODO: tests where the start and/or end of the timeseries do not fall on a natural period end
