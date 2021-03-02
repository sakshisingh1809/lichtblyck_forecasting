"""Testing Portfolio."""

from lichtblyck import SinglePf, MultiPf
from lichtblyck.core.dev import (
    get_index,
    get_pfframe,
    get_singlepf,
    get_multipf,
    OK_FREQ,
    OK_COL_COMBOS,
)
from typing import Union
import pandas as pd
import numpy as np
import pytest


def assert_raises_attributeerror(df: pd.DataFrame, yes=None, no=None):
    if yes is not None:
        for a in yes:
            with pytest.raises(AttributeError):
                getattr(df, a)
    if no is not None:
        for a in no:
            getattr(df, a)


def assert_w_q_compatible(p: Union[SinglePf, MultiPf]):
    if p.index.freq == "15T":
        np.testing.assert_allclose(p.q, p.w * 0.25)
    elif p.index.freq == "H":
        np.testing.assert_allclose(p.q, p.w)
    elif p.index.freq == "D":
        assert (p.q > p.w * 22.99).all()
        assert (p.q < p.w * 25.01).all()
    elif p.index.freq == "MS":
        assert (p.q > p.w * 27 * 24).all()
        assert (p.q < p.w * 32 * 24).all()
    elif p.index.freq == "QS":
        assert (p.q > p.w * 89 * 24).all()
        assert (p.q < p.w * 93 * 24).all()
    elif p.index.freq == "AS":
        assert (p.q > p.w * 8759.9).all()
        assert (p.q < p.w * 8784.1).all()
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
    if tz is None:
        for columns in ["q", "w", "qr"]:
            with pytest.raises(ValueError):
                pf = get_pfframe(get_index(tz, freq), columns)
                SinglePf(pf, "test").duration
        return

    # Specify one. That's good, if it's (w or q).

    for column in ["q", "w"]:
        pf = get_pfframe(get_index(tz, freq), column)
        sp = SinglePf(pf, "test")
        assert_w_q_compatible(sp)
        assert_raises_attributeerror(sp, no="wpqr")
        assert sp.r.isna().all()
        assert sp.p.isna().all()
        assert sp.index.freq == freq
        assert sp.index.tz is not None

    for column in ["p", "r"]:
        with pytest.raises(ValueError):  # information about power is missing.
            pf = get_pfframe(get_index(tz, freq), column)
            SinglePf(pf, "test").duration

    # Specify two. That's good, if it's not (w and q).

    for columns in ["pr", "qr", "pq", "wp", "wr"]:
        pf = get_pfframe(get_index(tz, freq), columns)
        sp = SinglePf(pf, "test")
        assert_w_q_compatible(sp)
        assert_p_q_r_compatible(sp)
        assert_raises_attributeerror(sp, no="wpqr")
        assert sp.index.freq == freq
        assert sp.index.tz is not None

    with pytest.raises(ValueError):
        pf = get_pfframe(get_index(tz, freq), "wq")
        SinglePf(pf, "test").duration

    # Specify three or four. Always incompatible.

    for columns in ["pqr", "wpr", "qwp", "qwr", "pqrw"]:
        with pytest.raises(ValueError):
            pf = get_pfframe(get_index(tz, freq), columns)
            SinglePf(pf, "test").duration


@pytest.mark.parametrize("columns", np.random.choice(OK_COL_COMBOS, 5, False))
def test_singlepf_init(columns):
    i = get_index()

    ref = get_pfframe(i, columns)
    sp = SinglePf(ref, "test")  # data passed as pfframe
    assert_pfolio_attributes_equal(sp, ref, columns)
    ref = pd.DataFrame(ref)
    sp = SinglePf(ref, "test")  # data passed as dataframe
    assert_pfolio_attributes_equal(sp, ref, columns)
    ref = {key: ref[key] for key in ref.columns}
    sp = SinglePf(ref, "test")  # data passed as dictionary
    assert_pfolio_attributes_equal(sp, ref, columns)

    ref = get_singlepf(i, columns)
    sp = SinglePf(ref, "test")  # data passed as other singlepf
    assert_pfolio_attributes_equal(sp, ref)

    ref = get_multipf(i, columns)
    sp = SinglePf(ref, "test")  # data passed as multipf
    assert_pfolio_attributes_equal(sp, ref)


@pytest.mark.parametrize("freq", np.random.choice(OK_FREQ, 3, False))
@pytest.mark.parametrize("newfreq", np.random.choice(OK_FREQ, 3, False))
@pytest.mark.parametrize("columns", np.random.choice(OK_COL_COMBOS, 3, False))
def test_change_freq(freq, newfreq, columns):
    df = get_pfframe(get_index("Europe/Berlin", freq), columns)
    spf1 = SinglePf(df, "test")
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


def test_sameobject():

    sp1 = get_singlepf()
    sp2 = SinglePf(sp1)
    pd.testing.assert_frame_equal(sp1.df(), sp2.df())
    assert sp1.name == sp2.name

    mp1 = get_multipf()
    mp2 = MultiPf(mp1)
    pd.testing.assert_frame_equal(mp1.df(), mp2.df())
    assert mp1.name == mp2.name
