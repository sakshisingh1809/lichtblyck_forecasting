"""Testing PfSeries and PfFrame."""

from lichtblyck.core.dev import get_index, get_dataframe, get_series, OK_FREQ
import pandas as pd
import numpy as np
import pytest


def assert_all_same_time(ps):
    assert len(ps.index.map(lambda ts: ts.time()).unique()) == 1
    assert len(ps.ts_right.map(lambda ts: ts.time()).unique()) == 1


@pytest.mark.parametrize("freq", [*OK_FREQ, "A", "Q", "M", "5T", "T"])
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("get_obj_func", [get_series, get_dataframe])
def test_index(freq, tz, get_obj_func):
    i = get_index(tz=tz, freq=freq)
    pf = get_obj_func(i)
    if freq in OK_FREQ and tz is not None:
        assert len(pf.duration) == len(pf)
        assert np.isclose(
            pf.duration[:-1].sum(), (pf.index[-1] - pf.index[0]).total_seconds() / 3600,
        )
        assert np.allclose(
            pf.duration, (pf.ts_right - pf.index).dt.total_seconds() / 3600
        )
        assert pf.index.freq is not None
    else:
        with pytest.raises(ValueError):
            pf.duration
        with pytest.raises(ValueError):
            pf.ts_right


@pytest.mark.parametrize("get_obj_func", [get_series, get_dataframe])
def test_index_wt_st(get_obj_func):
    i = pd.date_range("2020-03-28", freq="D", periods=3, tz="Europe/Berlin")
    ps = get_obj_func(i)
    assert_all_same_time(ps)
    assert ps.duration[-2] == 23
    ps = get_obj_func(i[:-1])
    assert_all_same_time(ps)
    assert ps.duration[-1] == 23
    i = pd.date_range("2020-03-28", freq="H", periods=71, tz="Europe/Berlin")
    ps = get_obj_func(i)
    assert (ps.duration == 1).all()
    assert ps.ts_right[-1].time() == ps.index[0].time()


@pytest.mark.parametrize("get_obj_func", [get_series, get_dataframe])
def test_index_st_wt(get_obj_func):
    i = pd.date_range("2020-10-24", freq="D", periods=3, tz="Europe/Berlin")
    ps = get_obj_func(i)
    assert_all_same_time(ps)
    assert ps.duration[-2] == 25
    ps = get_obj_func(i[:-1])
    assert_all_same_time(ps)
    assert ps.duration[-1] == 25
    i = pd.date_range("2020-10-24", freq="H", periods=73, tz="Europe/Berlin")
    ps = get_obj_func(i)
    assert (ps.duration == 1).all()
    assert ps.ts_right[-1].time() == ps.index[0].time()


def test_sameobject():
    i = get_index()

    # Check that creating object from existing object makes a copy.

    vals = np.random.rand(len(i), 2)
    pf1 = pd.DataFrame(vals, i, list("wp"))
    pf2 = pd.DataFrame(pf1)
    pd.testing.assert_frame_equal(pf1, pf2)

    vals = np.random.rand(len(i), 1)
    pf1 = pd.DataFrame(vals, i, list("w"))
    pf2 = pd.DataFrame(pf1)
    pd.testing.assert_frame_equal(pf1, pf2)

    vals = np.random.rand(len(i))
    ps1 = pd.Series(vals, i, name="w")
    ps2 = pd.Series(ps1)
    pd.testing.assert_series_equal(ps1, ps2)