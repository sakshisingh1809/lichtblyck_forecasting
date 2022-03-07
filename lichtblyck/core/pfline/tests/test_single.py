from lichtblyck import testing
from lichtblyck.core.pfline.single import SinglePfLine
from lichtblyck.core.pfline.multi import MultiPfLine
from lichtblyck.core.develop import dev
from lichtblyck.tools.frames import set_ts_index
from lichtblyck.tools.stamps import FREQUENCIES
import pandas as pd
import numpy as np
import pytest


@pytest.mark.parametrize("freq", FREQUENCIES[::2])
@pytest.mark.parametrize("columns", ["w", "q", "p", "pr", "qr", "pq", "wp", "wr"])
@pytest.mark.parametrize("inputtype", ["df", "dict", "singlepfline", "multipfline"])
def test_singlepfline_init(freq, columns, inputtype):
    """Test if object can be initialized correctly, and attributes return correct values."""

    i = dev.get_index(freq, "Europe/Berlin")
    df = dev.get_dataframe(i, columns)
    if inputtype == "df":
        data_in = df
    elif inputtype == "dict":
        data_in = {name: s for name, s in df.items()}
    elif inputtype == "singlepfline":
        data_in = SinglePfLine(df)
    else:  # inputtype multipfline
        if columns in ["w", "q", "p", "qr", "wr"]:
            df1 = 0.4 * df
            df2 = 0.6 * df
        else:  # has price column
            othercol = columns.replace("p", "")
            df1 = df.mul({"p": 1, othercol: 0.4})
            df2 = df.mul({"p": 1, othercol: 0.6})
        data_in = MultiPfLine({"a": SinglePfLine(df1), "b": SinglePfLine(df2)})

    result = SinglePfLine(data_in)
    result_df = result.df(columns)
    expected_df = set_ts_index(df)
    if columns in ["w", "q"]:  # kind == 'q'
        expectedkind = "q"
        expectedavailable = "wq"
        expectedsummable = "q"
    elif columns in ["p"]:  # kind == 'p'
        expectedkind = "p"
        expectedavailable = "p"
        expectedsummable = "p"
    else:  # kind == 'all'
        expectedkind = "all"
        expectedavailable = "wqpr"
        expectedsummable = "qr"

    assert type(result) is SinglePfLine
    testing.assert_frame_equal(result_df, expected_df)
    assert result.kind == expectedkind
    assert set(list(result.available)) == set(list(expectedavailable))
    assert set(list(result.summable)) == set(list(expectedsummable))
    assert result.children == {}


@pytest.mark.parametrize("columns", ["w", "q", "p", "pr", "qr", "pq", "wp", "wr"])
def test_singlepfline_access(columns):
    """Test if core data can be accessed by item and attribute."""

    df = dev.get_dataframe(columns=columns)
    df["na"] = np.nan  # add nancolumn
    result = SinglePfLine(df)

    testing.assert_index_equal(result.index, df.index)
    testing.assert_index_equal(result["index"], df.index)

    def test_series_equal(col, expected):
        testing.assert_series_equal(getattr(result, col), expected)
        testing.assert_series_equal(result[col], expected)
        testing.assert_series_equal(getattr(result.df(col), col), expected)
        testing.assert_series_equal(result.df(col)[col], expected)

    for col in list("wqpr"):
        if col in columns:
            test_series_equal(col, df[col])
        elif col not in result.available:
            test_series_equal(col, df.na.rename(col))


# . check correct working of attributes .asfreq().
@pytest.mark.parametrize("freq", ["H", "D", "MS", "QS", "AS"])  # not do all (many!)
@pytest.mark.parametrize("newfreq", ["H", "D", "MS", "QS", "AS"])  # not do all (many!)
@pytest.mark.parametrize("columns", ["pr", "qr", "pq", "wp", "wr"])
def test_singlepfline_asfreqpossible(freq, newfreq, columns):
    """Test if changing frequency is done correctly (when it's possible)"""

    # Includes at 2 full years
    a, m, d = np.array([2016, 1, 1]) + np.random.randint(0, 12, 3)  # each + 0..11
    start = f"{a}-{m}-{d}"
    a, (m, d) = a + 3, np.array([1, 1]) + np.random.randint(0, 12, 2)  # each + 0..11
    end = f"{a}-{m}-{d}"

    i = pd.date_range(start, end, freq=freq, tz="Europe/Berlin")
    df = dev.get_dataframe(i, columns)
    pfl1 = SinglePfLine(df)
    pfl2 = pfl1.asfreq(newfreq)

    # Compare the dataframes, only keep time intervals that are in both objects.
    summable = pfl1.summable
    df1, df2 = pfl1.df(summable), pfl2.df(summable)
    mask1 = (df1.index >= df2.index[0]) & (df1.index.ts_right <= df2.index.ts_right[-1])
    mask2 = (df2.index >= df1.index[0]) & (df2.index.ts_right <= df1.index.ts_right[-1])
    df1, df2 = df1[mask1], df2[mask2]

    if df2.empty or df1.empty:
        return

    testing.assert_series_equal(df1.apply(sum), df2.apply(sum))


@pytest.mark.parametrize("freq", ["15T", "H", "D"])
@pytest.mark.parametrize("newfreq", ["MS", "QS", "AS"])
@pytest.mark.parametrize("kind", ["p", "q", "all"])
def test_singlepfline_asfreqimpossible(freq, newfreq, kind):
    """Test if changing frequency raises error if it's impossible."""

    periods = {"H": 200, "15T": 2000, "D": 20}[freq]
    i = pd.date_range("2020-04-06", freq=freq, periods=periods, tz="Europe/Berlin")
    pfl = dev.get_singlepfline(i, kind)
    with pytest.raises(ValueError):
        _ = pfl.asfreq(newfreq)


@pytest.mark.parametrize("kind", ["p", "q", "all"])
@pytest.mark.parametrize("col", ["w", "q", "p", "r"])
def test_singlepfline_setseries(kind, col):
    """Test if series can be set on existing pfline."""

    pfl_in = dev.get_singlepfline(kind=kind)
    s = dev.get_series(pfl_in.index, col)

    if kind == "all" and col == "r":  # Expecting error
        with pytest.raises(NotImplementedError):
            _ = pfl_in.set_r(s)
        return

    result = getattr(pfl_in, f"set_{col}")(s)
    testing.assert_series_equal(result[col], s)
    assert col in result.available
    if kind == "q" and col in ["w", "q"]:
        expectedkind = "q"
    elif kind == "p" and col == "p":
        expectedkind = "p"
    else:
        expectedkind = "all"
    assert result.kind == expectedkind
