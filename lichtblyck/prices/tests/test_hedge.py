from lichtblyck.prices import hedge
from lichtblyck.prices import utils
from lichtblyck.core import dev
from lichtblyck.core import functions
import numpy as np
import pandas as pd
import pytest
import functools


@pytest.fixture(params=["2020", "2020-04-15 9:00", "2020-03-29", "2020-10-25"])
def start(request):
    return request.param


@pytest.fixture(params=["H", "15T", "D", "MS"])
def freq_bpoFalse(request):
    return request.param


@pytest.fixture(params=["H", "15T"])
def freq_bpoTrue(request):
    return request.param


@pytest.fixture(params=[None, "Europe/Berlin"])
def tz(request):
    return request.param


@pytest.fixture(params=[2, 24, 100, 1000])
def count(request):
    return request.param


@pytest.fixture(params=["MS", "QS", "AS"])
def aggfreq(request):
    return request.param


@pytest.fixture(params=["vol", "val"])
def how(request):
    return request.param


def get_prices(tz, p_peak, p_offpeak):
    i = dev.get_index(tz, "H")
    mu = {True: p_peak, False: p_offpeak}
    values = [np.random.normal(mu[utils.is_peak_hour(ts)], 1) for ts in i]
    return pd.Series(values, i)

@pytest.mark.parametrize(
    ("values", "start", "notbpo", "bpo"),
    [
        ([1, 2, 3], "2020-01-01", 2, (np.nan, 2)),
        (range(12), "2020-01-01", 5.5, (9.5, 3.5)),
        ([*[10] * (23 + 8), *[15] * 5], "2020-03-29", 10.69444, (15, 10)),
        ([*[10] * (25 + 8), *[15] * 5], "2020-10-25", 10.65789, (15, 10)),
    ],
)
def test_basic_volhedge(values, start, notbpo, bpo):
    w = pd.Series(
        values, pd.date_range(start, freq="H", periods=len(values), tz="Europe/Berlin")
    )
    assert np.isclose(notbpo, hedge._w_hedge(w, how="vol", bpo=False))
    pd.testing.assert_series_equal(
        hedge._w_hedge(w, how="vol", bpo=True).sort_index(),
        pd.Series({"w_peak": bpo[0], "w_offpeak": bpo[1]}).dropna().sort_index(),
        check_dtype=False,
    )


@functools.lru_cache(maxsize=1024)
def get_hedgeresults(start, freq, length, tz, bpo, aggfreq):
    if functions.freq_up_or_down(freq, "H") <= 0:
        return get_hedgeresults_short(start, freq, length, tz, bpo, aggfreq)
    else:
        return get_hedgeresults_long(start, freq, length, aggfreq)


def get_hedgeresults_short(start, freq, length, tz, bpo, aggfreq):
    """For freq == 'H' or shorter.
    returns {(None or startts): {('val' or 'vol'): (peak, offpeak)}}
    """
    i = pd.date_range(start, freq=freq, periods=length, tz=tz)
    w_values = 100 + 100 * np.random.rand(len(i))
    p_values = 50 + 20 * np.random.rand(len(i))

    if aggfreq is None:
        resultkey = lambda ts: None
    else:  # key is start of delivery period
        resultkey = lambda ts: utils.ts_leftright(ts, aggfreq.lower()[0], 0)[0]

    result = {}
    duration = 0.25 if freq == "15T" else 1
    for ts, w, p in zip(i, w_values, p_values):

        key = resultkey(ts)
        if key not in result:
            result[key] = {
                "w.d": np.array([0.0, 0.0]),
                "d": np.array([0.0, 0.0]),
                "w.pd": np.array([0.0, 0.0]),
                "pd": np.array([0.0, 0.0]),
            }

        if bpo and not utils.is_peak_hour(ts):
            result[key]["w.d"] += [0, w * duration]
            result[key]["d"] += [0, duration]
            result[key]["w.pd"] += [0, p * w * duration]
            result[key]["pd"] += [0, p * duration]
        else:
            result[key]["w.d"] += [w * duration, 0]
            result[key]["d"] += [duration, 0]
            result[key]["w.pd"] += [p * w * duration, 0]
            result[key]["pd"] += [p * duration, 0]

    return (
        pd.Series(w_values, i),
        pd.Series(p_values, i),
        {
            key: {
                "val": values["w.pd"] / values["pd"],
                "vol": values["w.d"] / values["d"],
            }
            for key, values in result.items()
        },
    )


def get_hedgeresults_long(start, freq, length, aggfreq):
    """For freq == 'D' or longer.
    No split in peak and offpeak values possible.
    returns {(None or startts): {('val' or 'vol'): (value, 0)}}
    """
    i = pd.date_range(start, freq=freq, periods=length, tz="Europe/Berlin")
    w_values = 100 + 100 * np.random.rand(len(i))
    p_values = 50 + 20 * np.random.rand(len(i))
    duration_values = pd.Series(w_values, i).duration.values

    if aggfreq is None:
        resultkey = lambda ts: None
    else:  # key is start of delivery period
        resultkey = lambda ts: utils.ts_leftright(ts, aggfreq.lower()[0], 0)[0]

    result = {}
    for ts, w, p, duration in zip(i, w_values, p_values, duration_values):

        key = resultkey(ts)
        if key not in result:
            result[key] = {"w.d": 0, "d": 0, "w.pd": 0, "pd": 0}

        result[key]["w.d"] += w * duration
        result[key]["d"] += duration
        result[key]["w.pd"] += p * w * duration
        result[key]["pd"] += p * duration

    return (
        pd.Series(w_values, i),
        pd.Series(p_values, i),
        {
            key: {"val": (v["w.pd"] / v["pd"], 0), "vol": (v["w.d"] / v["d"], 0)}
            for key, v in result.items()
        },
    )


def test__w_hedge_bpoFalse(start, freq_bpoFalse, count, tz):
    freq = freq_bpoFalse
    w, p, ref_results = get_hedgeresults(start, freq, count, tz, False, None)
    for how in ["vol", "val"]:
        test_result = hedge._w_hedge(w, p, how, False)
        ref_result = ref_results[None][how][0]
        assert np.isclose(test_result, ref_result)


def test__w_hedge_bpoTrue(start, freq_bpoTrue, count, tz):
    freq = freq_bpoTrue
    w, p, ref_results = get_hedgeresults(start, freq, count, tz, True, None)
    for how in ["vol", "val"]:
        test_result = hedge._w_hedge(w, p, how, True).sort_index()
        ref_bpo = ref_results[None][how]
        records = {"w_peak": ref_bpo[0], "w_offpeak": ref_bpo[1]}
        ref_result = pd.Series(records).dropna().sort_index()
        pd.testing.assert_series_equal(test_result, ref_result)


def test_wide_bpoFalse(start, freq_bpoFalse, count, tz, aggfreq, how):
    freq = freq_bpoFalse
    w, p, ref_results = get_hedgeresults(start, freq, count, tz, False, aggfreq)
    test_result = hedge.hedge(w, p, aggfreq, how, False, False).sort_index()
    records = {ts: values[how][0] for ts, values in ref_results.items()}
    ref_result = pd.Series(records).sort_index().dropna()
    pd.testing.assert_series_equal(test_result, ref_result, check_freq=False)


def test_wide_bpoTrue(start, freq_bpoTrue, count, tz, aggfreq, how):
    freq = freq_bpoTrue
    w, p, ref_results = get_hedgeresults(start, freq, count, tz, True, aggfreq)
    test_result = hedge.hedge(w, p, aggfreq, how, True, False).sort_index()
    records = {
        ts: {"w_peak": values[how][0], "w_offpeak": values[how][1]}
        for ts, values in ref_results.items()
    }
    ref_result = pd.DataFrame.from_records(records).sort_index().T.dropna(1)
    pd.testing.assert_frame_equal(test_result, ref_result, check_freq=False)


def test_narrow_bpoFalse(start, freq_bpoFalse, count, tz, aggfreq, how):
    freq = freq_bpoFalse
    w, p, ref_results = get_hedgeresults(start, freq, count, tz, False, aggfreq)
    test_result = hedge.hedge(w, p, aggfreq, how, False, True)
    availkeys = np.array([key for key in ref_results.keys()])
    keys = availkeys[np.searchsorted(availkeys, w.index, side="right") - 1]
    ref_result = pd.Series([ref_results[key][how][0] for key in keys], w.index)
    pd.testing.assert_series_equal(test_result, ref_result, check_names=False)


def test_narrow_bpoTrue(start, freq_bpoTrue, count, tz, aggfreq, how):
    freq = freq_bpoTrue
    w, p, ref_results = get_hedgeresults(start, freq, count, tz, True, aggfreq)
    test_result = hedge.hedge(w, p, aggfreq, how, True, True)
    availkeys = np.array([key for key in ref_results.keys()])
    keys = availkeys[np.searchsorted(availkeys, w.index, side="right") - 1]
    inds = [0 if utils.is_peak_hour(ts) else 1 for ts in w.index]
    ref_values = [ref_results[key][how][ind] for key, ind in zip(keys, inds)]
    ref_result = pd.Series(ref_values, w.index)
    pd.testing.assert_series_equal(test_result, ref_result, check_names=False)

