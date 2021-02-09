"""Testing PfSeries and PfFrame."""

from lichtblyck import PfSeries, PfFrame
from lichtblyck.core.dev import get_index, get_pfframe, get_pfseries, OK_FREQ
import pandas as pd
import numpy as np
import pytest


def assert_all_same_time(ps):
    assert len(ps.index.map(lambda ts: ts.time()).unique()) == 1
    assert len(ps.ts_right.map(lambda ts: ts.time()).unique()) == 1


@pytest.mark.parametrize("freq", [*OK_FREQ, "A", "Q", "M", "5T", "T"])
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("get_obj_func", [get_pfseries, get_pfframe])
def test_index(freq, tz, get_obj_func):
    i = pd.date_range("2020-03-28", freq=freq, periods=3, tz=tz)
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


@pytest.mark.parametrize("get_obj_func", [get_pfseries, get_pfframe])
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


@pytest.mark.parametrize("get_obj_func", [get_pfseries, get_pfframe])
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

    # Compare (DataFrame and Series) with (PfFrame and PfSeries).
    # Former don't have .duration or .ts_right attribute, but should have equal values.

    vals = np.random.rand(len(i), 2)
    df1 = pd.DataFrame(vals, i, list("wp"))
    pf1 = PfFrame(vals, i, list("wp"))
    assert ((df1.index - pf1.index) < pd.Timedelta(minutes=1)).all()
    assert ((df1.index - pf1.index) > pd.Timedelta(minutes=-1)).all()
    np.testing.assert_array_almost_equal(df1.w.values, pf1.w.values)
    np.testing.assert_array_almost_equal(df1.p.values, pf1.p.values)

    vals = np.random.rand(len(i))
    s1 = pd.Series(vals, i, name="w")
    ps1 = PfSeries(vals, i, name="w")
    assert ((s1.index - ps1.index) < pd.Timedelta(minutes=1)).all()
    assert ((s1.index - ps1.index) > pd.Timedelta(minutes=-1)).all()
    np.testing.assert_array_almost_equal(s1.values, ps1.values)

    # Check that creating object from existing object makes a copy.

    vals = np.random.rand(len(i), 2)
    pf1 = PfFrame(vals, i, list("wp"))
    pf2 = PfFrame(pf1)
    pd.testing.assert_frame_equal(pf1, pf2)

    vals = np.random.rand(len(i), 1)
    pf1 = PfFrame(vals, i, list("w"))
    pf2 = PfFrame(pf1)
    pd.testing.assert_frame_equal(pf1, pf2)

    vals = np.random.rand(len(i))
    ps1 = PfSeries(vals, i, name="w")
    ps2 = PfSeries(ps1)
    pd.testing.assert_series_equal(ps1, ps2)


def test_conversion():
    pf = get_pfframe()
    df = pd.DataFrame(pf)
    assert type(df) is pd.DataFrame
    pf = PfFrame(df)
    assert type(pf) is PfFrame

    ps = get_pfseries()
    s = pd.Series(ps)
    assert type(s) is pd.Series
    ps = PfSeries(s)
    assert type(ps) is PfSeries
