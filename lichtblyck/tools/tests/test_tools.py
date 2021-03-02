from lichtblyck import tools
from numpy import nan
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    ("values", "maxgap", "gapvalues"),
    [
        ([1, 2, 3, 4, 25, 7, 8], 1, []),
        ([1, 2, 3, 4, nan, 7, 8], 1, [5.5]),
        ([1, 2, 3, 4, nan, 7, 8], 2, [5.5]),
        ([1, 2, 3, 4, nan, 7, 8], 3, [5.5]),
        ([3, 2, 1, nan, nan, 7, 8], 1, [nan, nan]),
        ([3, 2, 1, nan, nan, 7, 8], 2, [3, 5]),
        ([3, 2, 1, nan, nan, 7, 8], 3, [3, 5]),
        ([nan, 2, 1, nan, nan, 7, nan], 1, [nan, nan, nan, nan]),
        ([nan, 2, 1, nan, nan, 7, nan], 2, [nan, 3, 5, nan]),
    ],
)
@pytest.mark.parametrize(
    ("index", "tol"),
    [
        (range(7), 0),
        (range(-3, 4), 0),
        (pd.date_range("2020", periods=7, freq="D"), 0),
        (pd.date_range("2020", periods=7, freq="M", tz="Europe/Berlin"), 0.04),
    ],
)
def test_fill_gaps(values, index, maxgap, gapvalues, tol):
    s = pd.Series(values, index)
    s_new = tools.fill_gaps(s, maxgap)
    s[s.isna()] = gapvalues
    pd.testing.assert_series_equal(s_new, s, rtol=tol)


def test_wavg():
    values1 = [1, 2, 3, -1.5]
    values2 = [1, -2, 3, -1.5]
    weights = [1, 1, 1, 2]

    # Series
    assert np.isclose(tools.wavg(pd.Series(values1), weights), 0.6)  # align by position
    assert np.isclose(
        tools.wavg(pd.Series(values1), pd.Series(weights, [3, 2, 1, 0])), 1.1
    )  # align by index
    # DataFrame
    pd.testing.assert_series_equal(
        tools.wavg(pd.DataFrame({"a": values1, "b": values1}), weights),
        pd.Series({"a": 0.6, "b": 0.6}),
    )
    pd.testing.assert_series_equal(
        tools.wavg(pd.DataFrame({"a": values1, "b": values2}), weights),
        pd.Series({"a": 0.6, "b": -0.2}),
    )
    pd.testing.assert_series_equal(
        tools.wavg(
            pd.DataFrame({"a": values1, "b": values2}), pd.Series(weights, [3, 2, 1, 0])
        ),
        pd.Series({"a": 1.1, "b": 0.3}),
    )  # align by index
    pd.testing.assert_series_equal(
        tools.wavg(
            pd.DataFrame({"a": values1, "b": values1, "c": values1, "d": values2}),
            weights,
            axis=0,
        ),
        pd.Series({"a": 0.6, "b": 0.6, "c": 0.6, "d": -0.2}),
    )
    pd.testing.assert_series_equal(
        tools.wavg(
            pd.DataFrame({"a": values1, "b": values1, "c": values1, "d": values2}),
            weights,
            axis=1,
        ),
        pd.Series([1, 0.4, 3, -1.5]),
    )  # row-averages
    pd.testing.assert_series_equal(
        tools.wavg(
            pd.DataFrame({"a": values1, "b": values1, "c": values1, "d": values2}),
            pd.Series(weights, ["d", "c", "b", "a"]),
            axis=1,
        ),
        pd.Series([1, 1.2, 3, -1.5]),
    )  # align by index and row-averages


def test_tsindex_normaldays():
    tsindextest(
        np.random.random(48),
        pd.date_range(
            "2020-03-01", "2020-03-03", freq="H", closed="left", tz="Europe/Berlin"
        ),
    )


def test_tsindex_wtst():
    # Check for days incl WT->ST.
    tsindextest(
        np.random.random(47),
        pd.date_range(
            "2020-03-28", "2020-03-30", freq="H", closed="left", tz="Europe/Berlin"
        ),
    )


def test_tsindex_stwt():
    # Check for days incl ST->WT.
    tsindextest(
        np.random.random(49),
        pd.date_range(
            "2020-10-24", "2020-10-26", freq="H", closed="left", tz="Europe/Berlin"
        ),
    )


def tsindextest(values, ts):
    expected = pd.DataFrame({"a": values}, index=ts.rename("ts_left"))

    # Using expected frame.
    pd.testing.assert_frame_equal(tools.set_ts_index(expected), expected)

    for bound in ["left", "right"]:

        for tz in [0, 1, 2, 3]:

            ts1 = ts.copy()
            timz = "Europe/Berlin"

            if tz == 1:
                ts1 = ts1.tz_localize(None)
            elif tz == 2:
                ts1 = ts1.tz_convert("Asia/Kolkata")
            elif tz == 3:
                ts1 = ts1.tz_convert("Asia/Kolkata").tz_localize(None)
                timz = "Asia/Kolkata"

            if bound == "right":
                ts1 = ts1 + pd.Timedelta("1H")

            pd.testing.assert_frame_equal(
                tools.set_ts_index(
                    pd.DataFrame({"a": values}, ts1), bound=bound, tz=timz
                ),
                expected,
            )  # use index
            pd.testing.assert_frame_equal(
                tools.set_ts_index(
                    pd.DataFrame({"a": values, "t": ts1}), "t", bound, timz
                ),
                expected,
            )  # use column as index
