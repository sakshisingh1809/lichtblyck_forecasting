"""Testing PfSeries and PfFrame."""

from lichtblyck import PfSeries, PfFrame
import pandas as pd
import numpy as np
import pytest


def get_index(tz='Europe/Berlin', freq='D'):
    count = {"QS": 4, "MS": 10, "D": 100, "H": 1000, "15T": 1000}[freq]
    periods = np.random.randint(count, count * 10)
    a, m, d = np.array([2016, 1, 1]) + np.random.randint(0, 12, 3)
    return pd.date_range(f"{a}-{m}-{d}", freq=freq, periods=periods, tz=tz)


def get_pfframe(i=None, columns='wp'):
    if i is None:
        i = get_index()
    return PfFrame(np.random.rand(len(i), len(columns)), i, list(columns))


def assert_raises_attributeerror(pf: PfFrame, yes=None, no=None):
    if yes is not None:
        for a in yes:
            with pytest.raises(AttributeError):
                getattr(pf, a)
    if no is not None:
        for a in no:
            getattr(pf, a)


def assert_w_q_compatible(pf: PfFrame, freq: str):
    if freq == "15T":
        np.testing.assert_allclose(pf.q, pf.w * 0.25)
    elif freq == "H":
        np.testing.assert_allclose(pf.q, pf.w)
    elif freq == "D":
        assert (pf.q > pf.w * 22.99).all()
        assert (pf.q < pf.w * 25.01).all()
    elif freq == "MS":
        assert (pf.q > pf.w * 27 * 24).all()
        assert (pf.q < pf.w * 32 * 24).all()
    elif freq == "QS":
        assert (pf.q > pf.w * 89 * 24).all()
        assert (pf.q < pf.w * 93 * 24).all()


def assert_p_q_r_compatible(pf: PfFrame):
    np.testing.assert_allclose(pf.r, pf.q * pf.p)


def test_index():
    for freq in ["QS", "MS", "D", "H", "15T"]:
        pf = get_pfframe(get_index("Europe/Berlin", freq), "qr")
        assert len(pf.duration) == len(pf)
        assert np.isclose(
            pf.duration[:-1].sum(),
            (pf.index[-1] - pf.index[0]).total_seconds() / 3600,
        )
        assert pf.index.freq is not None


def test_index_wt_st():
    i = pd.date_range("2020-03-28", freq="D", periods=3, tz="Europe/Berlin")
    pf = PfSeries(np.random.rand(len(i)), i)
    assert pf.duration[-2] == 23
    pf = PfSeries(pf[:-1])
    assert pf.duration[-1] == 23
    i = pd.date_range("2020-03-28", freq="H", periods=72, tz="Europe/Berlin")
    pf = PfSeries(np.random.rand(len(i)), i)
    assert (pf.duration == 1).all()


def test_index_st_wt():
    i = pd.date_range("2020-10-24", freq="D", periods=3, tz="Europe/Berlin")
    pf = PfSeries(np.random.rand(len(i)), i)
    assert pf.duration[-2] == 25
    pf = PfSeries(pf[:-1])
    assert pf.duration[-1] == 25
    i = pd.date_range("2020-10-24", freq="H", periods=72, tz="Europe/Berlin")
    pf = PfSeries(np.random.rand(len(i)), i)
    assert (pf.duration == 1).all()


def test_series():
    for tz in [None, "Europe/Berlin"]:
        for freq in ["QS", "MS", "D", "H", "15T"]:
            i = get_index(tz, freq)
            s = PfSeries(np.random.rand(len(i)), i)
            if tz is None:
                with pytest.raises(AttributeError):
                    s.duration
            else:
                assert len(s.duration) == len(i)


def test_frame():
    for tz in [None, "Europe/Berlin"]:
        for freq in ["QS", "MS", "D", "H", "15T"]:
            # Specify one. That's never enough.

            pf = get_pfframe(get_index(tz, freq), "q")
            assert_raises_attributeerror(pf, "rp", "q")
            if tz is None:
                assert_raises_attributeerror(pf, "w")
            else:
                assert_w_q_compatible(pf, freq)

            pf = get_pfframe(get_index(tz, freq), "q")
            assert_raises_attributeerror(pf, "rp", "q")
            if tz is None:
                assert_raises_attributeerror(pf, "w")
            else:
                assert_w_q_compatible(pf, freq)

            pf = get_pfframe(get_index(tz, freq), "r")
            assert_raises_attributeerror(pf, "pqw", "r")

            pf = get_pfframe(get_index(tz, freq), "p")
            assert_raises_attributeerror(pf, "rqw", "p")

            # Specify two. That's good, as long as w xor q is specified.

            # . w not specified.

            pf = get_pfframe(get_index(tz, freq), "pr")
            if tz is None:
                assert_raises_attributeerror(pf, "w", "pqr")
            else:
                assert_raises_attributeerror(pf, no="wpqr")
                assert_w_q_compatible(pf, freq)
            assert_p_q_r_compatible(pf)

            pf = get_pfframe(get_index(tz, freq), "qr")
            if tz is None:
                assert_raises_attributeerror(pf, "w", "pqr")
            else:
                assert_raises_attributeerror(pf, no="wpqr")
                assert_w_q_compatible(pf, freq)
            assert_p_q_r_compatible(pf)

            pf = get_pfframe(get_index(tz, freq), "pq")
            if tz is None:
                assert_raises_attributeerror(pf, "w", "pqr")
            else:
                assert_raises_attributeerror(pf, no="wpqr")
                assert_w_q_compatible(pf, freq)
            assert_p_q_r_compatible(pf)

            # . w specified, q not specified.

            pf = get_pfframe(get_index(tz, freq), "wr")
            if tz is None:
                assert_raises_attributeerror(pf, "qp", "wr")
            else:
                assert_raises_attributeerror(pf, no="wpqr")
                assert_w_q_compatible(pf, freq)
                assert_p_q_r_compatible(pf)

            pf = get_pfframe(get_index(tz, freq), "wp")
            if tz is None:
                assert_raises_attributeerror(pf, "qr", "wp")
            else:
                assert_raises_attributeerror(pf, no="wpqr")
                assert_w_q_compatible(pf, freq)
                assert_p_q_r_compatible(pf)

            # . w specified and q specified.

            pf = get_pfframe(get_index(tz, freq), "wq")
            assert_raises_attributeerror(pf, "pr", "wq")
            with pytest.raises(AssertionError):
                assert_w_q_compatible(pf, freq)  # both specified: won't be compatible

            # Specify three. Always incompatible.

            pf = get_pfframe(get_index(tz, freq), "pqr")
            if tz is None:
                assert_raises_attributeerror(pf, "w", "pqr")
            else:
                assert_raises_attributeerror(pf, no="wpqr")
                assert_w_q_compatible(pf, freq)
            with pytest.raises(AssertionError):
                assert_p_q_r_compatible(pf)  # all specified: won't be compatible

            pf = get_pfframe(get_index(tz, freq), "wpr")
            assert_raises_attributeerror(pf, no="wqpr")
            if tz is not None:
                with pytest.raises(AssertionError):
                    assert_p_q_r_compatible(pf)

            pf = get_pfframe(get_index(tz, freq), "wpq")
            assert_raises_attributeerror(pf, no="wpqr")
            assert_p_q_r_compatible(pf)
            with pytest.raises(AssertionError):
                assert_w_q_compatible(pf, freq)  # both specified: won't be compatible

            pf = get_pfframe(get_index(tz, freq), "wqr")
            assert_raises_attributeerror(pf, no="wpqr")
            assert_p_q_r_compatible(pf)
            with pytest.raises(AssertionError):
                assert_w_q_compatible(pf, freq)  # both specified: won't be compatible

            # Specify four. Always incompatible.

            pf = get_pfframe(get_index(tz, freq), "wpqr")
            assert_raises_attributeerror(pf, no="wpqr")
            with pytest.raises(AssertionError):
                assert_p_q_r_compatible(pf)  # all specified: won't be compatible
            with pytest.raises(AssertionError):
                assert_w_q_compatible(pf, freq)  # both specified: won't be compatible


def test_changefreq():
    for freq in ["QS", "MS", "D", "H", "15T"]:
        for columns in ["qr", "wr", "qp", "wp", "pr"]:
            for newfreq in [None, "AS", "QS", "MS", "D", "H", "15T"]:
                pf = get_pfframe(get_index("Europe/Berlin", freq), columns)
                newpf = pf.changefreq(newfreq)
                if newfreq is None:
                    assert np.isclose(pf.r.sum(), newpf.r.sum())
                    assert np.isclose(pf.q.sum(), newpf.q.sum())
                else:
                    if len(newpf) == 0:
                        continue
                    oldpf = pf[
                        (pf.index >= newpf.index[0])
                        & (pf.ts_right <= newpf.ts_right[-1])
                    ]
                    if len(oldpf) == 0:
                        continue
                    oldpf = PfFrame(oldpf)
                    assert np.isclose(oldpf.r.sum(), newpf.r.sum())
                    assert np.isclose(oldpf.q.sum(), newpf.q.sum())
