import lichtblyck
import numpy as np
import pandas as pd
import pytest


def test_duration():
    # Hourly.
    i = pd.date_range("2020-01-01", "2020-12-31", freq="H", tz="Europe/Berlin")
    s = pd.Series(np.random.rand(len(i)), i)
    df = pd.DataFrame({"a": s})
    #   Stand-alone
    assert i.duration.max() == 1
    assert i.duration.median() == 1
    assert i.duration.min() == 1
    #   Series
    assert s.index.duration.max() == 1
    assert s.index.duration.median() == 1
    assert s.index.duration.min() == 1
    #   DataFrame
    assert df.index.duration.max() == 1
    assert df.index.duration.median() == 1
    assert df.index.duration.min() == 1

    # Quarter-hourly.
    i = pd.date_range("2020-01-01", "2020-12-31", freq="15T", tz="Europe/Berlin")
    s = pd.Series(np.random.rand(len(i)), i)
    df = pd.DataFrame({"a": s})
    #   Stand-alone
    assert i.duration.max() == 0.25
    assert i.duration.median() == 0.25
    assert i.duration.min() == 0.25
    #   Series
    assert s.index.duration.max() == 0.25
    assert s.index.duration.median() == 0.25
    assert s.index.duration.min() == 0.25
    #   DataFrame
    assert df.index.duration.max() == 0.25
    assert df.index.duration.median() == 0.25
    assert df.index.duration.min() == 0.25

    # Daily.
    i = pd.date_range("2020-01-01", "2020-12-31", freq="D", tz="Europe/Berlin")
    s = pd.Series(np.random.rand(len(i)), i)
    df = pd.DataFrame({"a": s})
    #   Stand-alone
    assert i.duration.max() == 25
    assert i.duration.median() == 24
    assert i.duration.min() == 23
    #   Series
    assert s.index.duration.max() == 25
    assert s.index.duration.median() == 24
    assert s.index.duration.min() == 23
    #   DataFrame
    assert df.index.duration.max() == 25
    assert df.index.duration.median() == 24
    assert df.index.duration.min() == 23

    # Missing tz.
    i = pd.date_range("2020-01-01", "2020-12-31", freq="H")
    s = pd.Series(np.random.rand(len(i)), i)
    df = pd.DataFrame({"a": s})
    with pytest.raises(AttributeError):
        i.duration
    with pytest.raises(AttributeError):
        s.index.duration
    with pytest.raises(AttributeError):
        df.index.duration


def test_q():
    # Hourly.
    i = pd.date_range("2020-01-01", "2020-12-31", freq="H", tz="Europe/Berlin")
    values1 = np.random.rand(len(i))
    values2 = np.random.rand(len(i))
    #   Series
    s = pd.Series(values1, i)
    pd.testing.assert_series_equal(s.q, s * 1, check_names=False)
    #   Dataframe missing q
    df = pd.DataFrame({"w": values1, "x": values2}, i)
    pd.testing.assert_series_equal(df.q, df.w * 1, check_names=False)
    #   Dataframe already having q
    df = pd.DataFrame({"w": values1, "q": values2}, i)
    pd.testing.assert_series_equal(df.q, df["q"], check_names=False)
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(df.q, df.w * 1, check_names=False)

    # Quarter-hourly.
    i = pd.date_range("2020-01-01", "2020-12-31", freq="15T", tz="Europe/Berlin")
    values1 = np.random.rand(len(i))
    values2 = np.random.rand(len(i))
    #   Series
    s = pd.Series(values1, i)
    pd.testing.assert_series_equal(s.q, s * 0.25, check_names=False)
    #   Dataframe missing q
    df = pd.DataFrame({"w": values1, "x": values2}, i)
    pd.testing.assert_series_equal(df.q, df.w * 0.25, check_names=False)
    #   Dataframe already having q
    df = pd.DataFrame({"w": values1, "q": values2}, i)
    pd.testing.assert_series_equal(df.q, df["q"], check_names=False)
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(df.q, df.w * 0.25, check_names=False)

    # Daily.
    i = pd.date_range("2020-01-01", "2020-12-31", freq="D", tz="Europe/Berlin")
    dura = i.duration
    values1 = np.random.rand(len(i))
    values2 = np.random.rand(len(i))
    #   Series
    s = pd.Series(values1, i)
    assert sum(s.q == s * 24) == len(s) - 2
    assert sum(s.q == s * 23) == 1
    assert sum(s.q == s * 25) == 1
    #   Dataframe missing q
    df = pd.DataFrame({"w": values1, "x": values2}, i)
    pd.testing.assert_series_equal(df.q, df.w * dura, check_names=False)
    #   Dataframe already having q
    df = pd.DataFrame({"w": values1, "q": values2}, i)
    pd.testing.assert_series_equal(df.q, df["q"], check_names=False)
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(df.q, df.w * dura, check_names=False)


def test_r():
    for freq in ["H", "15T", "D"]:
        i = pd.date_range("2020-01-01", "2020-12-31", freq=freq, tz="Europe/Berlin")
        dura = i.duration
        values1 = np.random.rand(len(i))
        values2 = np.random.rand(len(i))
        #   Dataframe missing r
        df = pd.DataFrame({"q": values1, "p": values2}, i)
        pd.testing.assert_series_equal(df.r, df.q * df.p, check_names=False)
        #   Dataframe missing r and missing q
        df = pd.DataFrame({"w": values1, "p": values2}, i)
        pd.testing.assert_series_equal(df.r, df.q * df.p, check_names=False)
        pd.testing.assert_series_equal(df.r, df.w * dura * df.p, check_names=False)
        if freq == "H":
            pd.testing.assert_series_equal(df.r, df.w * 1 * df.p, check_names=False)
        elif freq == "15T":
            pd.testing.assert_series_equal(df.r, df.w * 0.25 * df.p, check_names=False)
        #   Dataframe already having r
        df = pd.DataFrame({"q": values1, "p": values2, "r": values2}, i)
        pd.testing.assert_series_equal(df.r, df["r"], check_names=False)
        with pytest.raises(AssertionError):
            pd.testing.assert_series_equal(df.r, df.q * df.p, check_names=False)
