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
    # Test as Series.
    s = pd.Series(values, index)
    s_new = tools.fill_gaps(s, maxgap)
    s[s.isna()] = gapvalues
    pd.testing.assert_series_equal(s_new, s, rtol=tol)

    # Test as DataFrame.
    df = pd.DataFrame({"a": values}, index)
    df_new = tools.fill_gaps(df, maxgap)
    df[df.isna()] = gapvalues
    pd.testing.assert_frame_equal(df_new, df, rtol=tol)


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


def values_and_index():
    return [
        (
            np.random.random(48),
            pd.date_range(
                "2020-03-01", "2020-03-03", freq="H", closed="left", tz="Europe/Berlin"
            ),
        ),
        (  # Check for days incl ST->WT.
            np.random.random(49),
            pd.date_range(
                "2020-10-24", "2020-10-26", freq="H", closed="left", tz="Europe/Berlin"
            ),
        ),
        (  # Check for days incl WT->ST.
            np.random.random(47),
            pd.date_range(
                "2020-03-28", "2020-03-30", freq="H", closed="left", tz="Europe/Berlin"
            ),
        ),
    ]


@pytest.fixture(params=["Europe/Berlin", "Asia/Kolkata"])
def timezone(request):
    return request.param


@pytest.fixture(params=[True, False])
def do_localize(request):
    return request.param


@pytest.fixture(params=[True, False])
def shiftright(request):
    return request.param


@pytest.mark.parametrize(("values", "i"), values_and_index())
def test_set_ts_index(values, i, shiftright, do_localize, timezone):
    expected_df = pd.DataFrame({"a": values}, index=i.rename("ts_left"))
    expected_s = pd.Series(values, i.rename("ts_left"))

    # Using expected frame: should stay the same.
    pd.testing.assert_frame_equal(tools.set_ts_index(expected_df), expected_df)
    pd.testing.assert_series_equal(tools.set_ts_index(expected_s), expected_s)

    i = i.tz_convert(timezone)
    if not do_localize:
        i = i.tz_localize(None)
    else:
        # If supplied timezone is not 'Europe/Berlin', this value is contradictory to
        # what is in the index. Test if timezone in Index is given preference.
        timezone = "Europe/Berlin"

    if shiftright:
        bound = "right"
        i = i + pd.Timedelta("1H")
    else:
        bound = "left"

    # Dataframe with index.
    pd.testing.assert_frame_equal(
        tools.set_ts_index(pd.DataFrame({"a": values}, i), bound=bound, tz=timezone),
        expected_df,
    )

    # Dataframe with column that must become index.
    pd.testing.assert_frame_equal(
        tools.set_ts_index(pd.DataFrame({"a": values, "ts": i}), "ts", bound, timezone),
        expected_df,
    )

    # Series.
    pd.testing.assert_series_equal(
        tools.set_ts_index(pd.Series(values, i), bound=bound, tz=timezone), expected_s
    )

# TODO: add test that uses rightA timestamps, and one that uses rightB