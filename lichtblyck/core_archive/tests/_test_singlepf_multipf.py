"""Testing SinglePf and MultiPf."""

from lichtblyck import SinglePf, MultiPf
from lichtblyck.core.dev import (
    get_index,
    get_dataframe,
    get_singlepf,
    get_multipf_standardcase,
    get_multipf_allcases,
    OK_FREQ,
    OK_COL_COMBOS,
)
from typing import Union
import pandas as pd
import numpy as np
import pytest


@pytest.fixture(params=OK_COL_COMBOS)
def columns(request):
    return request.param


columns1 = columns2 = columns


@pytest.fixture(params=[1, 2, 3])
def levels(request):
    return request.param


levels1 = levels2 = levels


def assert_raises_attributeerror(df: pd.DataFrame, yes=None, no=None):
    if yes is not None:
        for a in yes:
            with pytest.raises(AttributeError):
                getattr(df, a)
    if no is not None:
        for a in no:
            getattr(df, a)


def assert_w_q_compatible(pf: Union[SinglePf, MultiPf]):
    if pf.index.freq == "15T":
        np.testing.assert_allclose(pf.q, pf.w * 0.25)
    elif pf.index.freq == "H":
        np.testing.assert_allclose(pf.q, pf.w)
    elif pf.index.freq == "D":
        assert (pf.q > pf.w * 22.99).all()
        assert (pf.q < pf.w * 25.01).all()
    elif pf.index.freq == "MS":
        assert (pf.q > pf.w * 27 * 24).all()
        assert (pf.q < pf.w * 32 * 24).all()
    elif pf.index.freq == "QS":
        assert (pf.q > pf.w * 89 * 24).all()
        assert (pf.q < pf.w * 93 * 24).all()
    elif pf.index.freq == "AS":
        assert (pf.q > pf.w * 8759.9).all()
        assert (pf.q < pf.w * 8784.1).all()
    else:
        raise ValueError("Uncaught value for .index.freq: {p.index.freq}.")


def assert_p_q_r_compatible(df: pd.DataFrame):
    np.testing.assert_allclose(df.r, df.q * df.p)


def get_by_attr_or_key(obj, a):
    try:
        return getattr(obj, a)
    except:
        return obj[a]


def assert_pfolio_attributes_equal(p1, p2, attributes="wqpr"):
    for a in attributes:
        np.testing.assert_allclose(get_by_attr_or_key(p1, a), get_by_attr_or_key(p2, a))


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("freq", OK_FREQ)
def test_singlepf_correctvalues(tz, freq):
    # Without timezone there should be an error.

    if tz is None:
        for columns in ["q", "w", "qr"]:
            with pytest.raises(ValueError):
                df = get_dataframe(get_index(freq, tz), columns)
                SinglePf(df, name="test").duration
        return

    # Specify one. That's good, if it's (w or q).

    for column in ["q", "w"]:
        df = get_dataframe(get_index(freq, tz), column)
        sp = SinglePf(df, name="test")
        assert_w_q_compatible(sp)
        assert_raises_attributeerror(sp, no="wpqr")
        assert sp.r.isna().all()
        assert sp.p.isna().all()
        assert sp.index.freq == freq
        assert sp.index.tz is not None

    for column in ["p", "r"]:
        with pytest.raises(ValueError):  # information about power is missing.
            df = get_dataframe(get_index(freq, tz), column)
            SinglePf(df, name="test").duration

    # Specify two. That's good, if it's not (w and q).

    for columns in ["pr", "qr", "pq", "wp", "wr"]:
        df = get_dataframe(get_index(freq, tz), columns)
        sp = SinglePf(df, name="test")
        assert_w_q_compatible(sp)
        assert_p_q_r_compatible(sp)
        assert_raises_attributeerror(sp, no="wpqr")
        assert sp.index.freq == freq
        assert sp.index.tz is not None

    with pytest.raises(ValueError):
        df = get_dataframe(get_index(freq, tz), "wq")
        SinglePf(df, name="test").duration

    # Specify three or four. Always incompatible.

    for columns in ["pqr", "wpr", "qwp", "qwr", "pqrw"]:
        with pytest.raises(ValueError):
            df = get_dataframe(get_index(freq, tz), columns)
            SinglePf(df, name="test").duration


def test_init_1(columns):  # init singlepf with various objects
    i = get_index()

    ref = get_dataframe(i, columns)
    sp = SinglePf(ref, name="test")  # data passed as pfframe
    assert_pfolio_attributes_equal(sp, ref, columns)
    ref = pd.DataFrame(ref)
    sp = SinglePf(ref, name="test")  # data passed as dataframe
    assert_pfolio_attributes_equal(sp, ref, columns)
    ref = {key: ref[key] for key in ref.columns}
    sp = SinglePf(ref, name="test")  # data passed as dictionary
    assert_pfolio_attributes_equal(sp, ref, columns)

    ref = get_singlepf(i, columns)
    sp = SinglePf(ref)  # data passed as other singlepf
    assert_pfolio_attributes_equal(sp, ref, columns)
    pd.testing.assert_frame_equal(ref.df(), sp.df(), check_names=False)


def test_init_2(levels):  # init singlepf with multipf
    ref = get_multipf_standardcase(levels=levels)
    sp = SinglePf(ref)  # data passed as multipf with 1 level
    assert_pfolio_attributes_equal(sp, ref)


def test_init_3(levels):  # init multipf with multipf
    ref = get_multipf_standardcase(levels=levels)
    mp = MultiPf(ref)
    pd.testing.assert_frame_equal(ref.df(), mp.df())
    assert ref.name == mp.name


@pytest.mark.parametrize("freq", np.random.choice(OK_FREQ, 3, False))
@pytest.mark.parametrize("newfreq", np.random.choice(OK_FREQ, 3, False))
@pytest.mark.parametrize("columns", np.random.choice(OK_COL_COMBOS, 3, False))
def test_change_freq(freq, newfreq, columns):
    df = get_dataframe(get_index(freq, "Europe/Berlin"), columns)
    spf1 = SinglePf(df, name="test")
    spf2 = spf1.changefreq(newfreq)
    # Compare the dataframes.
    # To compare, only keep time intervals that are in both objects.
    df1, df2 = spf1.df("qr"), spf2.df("qr")
    if df2.empty:
        return
    df1 = df1[(df1.index >= df2.index[0]) & (df1.ts_right <= df2.ts_right[-1])]
    if df1.empty:
        return
    assert np.isclose(df1.r.sum(), df2.r.sum())
    assert np.isclose(df1.q.sum(), df2.q.sum())


def test_addition_1(columns1, columns2):  # singlepf +- singlepf
    i = get_index()
    sp1 = get_singlepf(i, columns1)
    sp2 = get_singlepf(i, columns2)

    sp_sum = sp1 + sp2
    pd.testing.assert_series_equal(sp_sum.q, sp1.q + sp2.q, check_names=False)
    pd.testing.assert_series_equal(sp_sum.w, sp1.w + sp2.w, check_names=False)
    if ("r" in columns1 or "p" in columns1) and ("r" in columns2 or "p" in columns2):
        pd.testing.assert_series_equal(sp_sum.r, sp1.r + sp2.r, check_names=False)
    else:
        assert sp_sum.r.isna().all()
        assert sp_sum.p.isna().all()
    
    sp_diff = sp1 - sp2
    pd.testing.assert_series_equal(sp_diff.q, sp1.q - sp2.q, check_names=False)
    pd.testing.assert_series_equal(sp_diff.w, sp1.w - sp2.w, check_names=False)
    if ("r" in columns1 or "p" in columns1) and ("r" in columns2 or "p" in columns2):
        pd.testing.assert_series_equal(sp_diff.r, sp1.r - sp2.r, check_names=False)
    else:
        assert sp_diff.r.isna().all()
        assert sp_diff.p.isna().all()

def test_addition_2(levels1, levels2):  # multipf +- multipf
    i = get_index()
    mp1 = get_multipf_standardcase(i, levels1)
    mp2 = get_multipf_standardcase(i, levels2)
    
    sp_sum = mp1 + mp2
    pd.testing.assert_series_equal(sp_sum.q, mp1.q + mp2.q, check_names=False)
    pd.testing.assert_series_equal(sp_sum.w, mp1.w + mp2.w, check_names=False)
    pd.testing.assert_series_equal(sp_sum.r, mp1.r + mp2.r, check_names=False)
    
    sp_diff = mp1 - mp2
    pd.testing.assert_series_equal(sp_diff.q, mp1.q - mp2.q, check_names=False)
    pd.testing.assert_series_equal(sp_diff.w, mp1.w - mp2.w, check_names=False)
    pd.testing.assert_series_equal(sp_diff.r, mp1.r - mp2.r, check_names=False)


def test_addition_3(columns, levels):  # singlepf +- multipf
    i = get_index()
    sp = get_singlepf(i, columns)
    mp = get_multipf_standardcase(i, levels)

    for sp_sum in [sp + mp, mp + sp]:
        pd.testing.assert_series_equal(sp_sum.q, sp.q + mp.q, check_names=False)
        pd.testing.assert_series_equal(sp_sum.w, sp.w + mp.w, check_names=False)
        if "r" in columns or "p" in columns:
            pd.testing.assert_series_equal(sp_sum.r, sp.r + mp.r, check_names=False)
        else:
            assert sp_sum.r.isna().all()
            assert sp_sum.p.isna().all()
    
    sp_diff = sp - mp
    pd.testing.assert_series_equal(sp_diff.q, sp.q - mp.q, check_names=False)
    pd.testing.assert_series_equal(sp_diff.w, sp.w - mp.w, check_names=False)
    if "r" in columns or "p" in columns:
        pd.testing.assert_series_equal(sp_diff.r, sp.r - mp.r, check_names=False)
    else:
        assert sp_diff.r.isna().all()
        assert sp_diff.p.isna().all()

    sp_diff = mp - sp
    pd.testing.assert_series_equal(sp_diff.q, mp.q - sp.q, check_names=False)
    pd.testing.assert_series_equal(sp_diff.w, mp.w - sp.w, check_names=False)
    if "r" in columns or "p" in columns:
        pd.testing.assert_series_equal(sp_diff.r, mp.r - sp.r, check_names=False)
    else:
        assert sp_diff.r.isna().all()
        assert sp_diff.p.isna().all()


def test_addition_distinctindices():
    pass

def test_multiplication():
    pass

def test_value():
    pass