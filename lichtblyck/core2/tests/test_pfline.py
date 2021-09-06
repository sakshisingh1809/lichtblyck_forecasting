# Assert correct working of _make_df:
# . can be called with dictionary, with dataframe, with pfline, with named tuple.
# . check with various combinations of keys: p, q, w, p and q, q and w, etc.
# . check that inconsistent data raises error.
# . check with keys having unequal indexes: unequal freq, timeperiod.
# . check if missing values have expected result.

# Assert correct working of pfline:
# . initialisation with dictionary, with dataframe, with named tuple.
# . initialisation with pfline returns identical pfline.
# . .kind property always correctly set.
# . timeseries can be accessed with .q, .p, .r, .w, ['q'], ['p'], etc. Check for various kinds.
# . check correct working of attributes .df() and .changefreq().
# . check correct working of dunder methods. E.g. assert correct addition:
# . . pflines having same or different kind
# . . pflines having same or different frequency
# . . pflines covering same or different time periods

from lichtblyck.core2.pfline import _make_df
from lichtblyck.core2 import dev
from lichtblyck.tools.frames import set_ts_index
import pandas as pd
import numpy as np
import pytest

# from typing import Union

# . can be called with dictionary, with dataframe, with pfline, with named tuple.
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("freq", ["MS", "D"])
def test_makedf1(freq, tz):
    i = dev.get_index(tz, freq)
    q = dev.get_series(i, "q")
    testresult1 = _make_df({"q": q})

    expected = pd.DataFrame({"q": q})
    if tz is None:
        expected = expected.tz_localize("Europe/Berlin")
    expected.index.freq = freq

    pd.testing.assert_frame_equal(testresult1, expected, check_names=False)

    if tz:
        w = q / q.duration
        testresult2 = _make_df({"w": w})
        pd.testing.assert_frame_equal(testresult2, expected, check_names=False)


def assert_w_q_compatible(freq, w, q):
    if freq == "15T":
        pd.testing.assert_series_equal(q, w * 0.25, check_names=False)
    elif freq == "H":
        pd.testing.assert_series_equal(q, w, check_names=False)
    elif freq == "D":
        assert (q > w * 22.99).all()
        assert (q < w * 25.01).all()
    elif freq == "MS":
        assert (q > w * 27 * 24).all()
        assert (q < w * 32 * 24).all()
    elif freq == "QS":
        assert (q > w * 89 * 24).all()
        assert (q < w * 93 * 24).all()
    elif freq == "AS":
        assert (q > w * 8759.9).all()
        assert (q < w * 8784.1).all()
    else:
        raise ValueError("Uncaught value for freq: {freq}.")


def assert_p_q_r_compatible(r, p, q):
    pd.testing.assert_series_equal(r, q * p, check_names=False)


def combination_of_two(choice, df):
    if choice == "wp":
        df["q"] = df.w * df.w.duration
        df["r"] = df.p * df.q

    elif choice == "rp":
        df["q"] = df.r / df.p
        df["w"] = df.q / df.duration

    elif choice == "pq":
        df["r"] = df.p * df.q
        df["w"] = df.q / df.duration

    elif choice == "wr":
        df["q"] = df.w * df.w.duration
        df["p"] = df.r / df.q

    else:
        df["p"] = df.r / df.q
        df["w"] = df.q / df.duration

    return df


# . check with various combinations of keys: p, q, w, p and q, q and w, etc.
# . check that inconsistent data raises error.
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("freq", ["MS", "D"])
@pytest.mark.parametrize(
    "choice",
    [
        "r",
        "p",
        "q",
        "w",
        "qw",
        "rp",
        "wp",
        "pq",
        "qr",
        "wr",
        "qwp",
        "qpr",
        "qrw",
        "pwr",
        "qwpr",
    ],
)
def test_combinations(tz, freq, choice):
    i = dev.get_index(tz, freq)
    df = dev.get_dataframe(i, choice)
    # dic = {key: df[key] for key in choice}

    if choice in ["r", "qw", "qwp", "qrw", "pwr", "qpr", "qwpr"]:  # error cases
        with pytest.raises(ValueError):
            result = _make_df(df)
        # with pytest.raises(ValueError):
        #     result = _make_df(dic)
        return

    elif choice == "p":  # kind == "p"
        result = _make_df(df)
        expectedresult = set_ts_index(pd.DataFrame({"p": df.p}))

    elif choice == "q":  # kind == "q"
        result = _make_df(df)
        expectedresult = set_ts_index(pd.DataFrame({"q": df.q}))

    elif choice == "w":  # kind == "q"
        result = _make_df(df)
        df["q"] = df.w * df.w.duration
        assert_w_q_compatible(freq, df.w, result["q"])
        expectedresult = set_ts_index(pd.DataFrame({"q": df.q}))

    elif choice in ["rp", "pq", "wp", "qr", "wr"]:  # kind == "all"
        # calculate r, w and q first
        # df = combination_of_two(choice, df)

        if choice == "wp":
            df["q"] = df.w * df.w.duration
            df["r"] = df.p * df.q

        elif choice == "rp":
            df["q"] = df.r / df.p
            df["w"] = df.q / df.duration

        elif choice == "pq":
            df["r"] = df.p * df.q
            df["w"] = df.q / df.duration

        elif choice == "wr":
            df["q"] = df.w * df.w.duration
            df["p"] = df.r / df.q

        else:
            df["p"] = df.r / df.q
            df["w"] = df.q / df.duration

        result = _make_df(df)
        assert_p_q_r_compatible(result.r, df.p, result.q)
        assert_w_q_compatible(freq, df.w, result.q)

        expectedresult = set_ts_index(pd.DataFrame({"q": df.q, "r": df.r})).dropna()

    pd.testing.assert_frame_equal(result, expectedresult)


freqs_small_to_large = ["T", "5T", "15T", "30T", "H", "2H", "D", "MS", "QS", "AS"]


@pytest.fixture(params=freqs_small_to_large)
def freq(request):
    return request.param


freq1 = freq2 = freq


# . check with keys having unequal indexes: unequal frequency
@pytest.mark.parametrize("freq1", ["15T", "H", "2H", "D", "MS", "QS", "AS"])
@pytest.mark.parametrize("freq2", ["15T", "H", "2H", "D", "MS", "QS", "AS"])
@pytest.mark.parametrize("choice", ["qw", "rp", "wp", "pq", "qr", "wr"])
def test_pfline_equalfrequencies(freq1, freq2, choice):

    i1 = pd.date_range(start="2020", end="2021", freq=freq1, closed="left")
    i2 = pd.date_range(start="2020", end="2021", freq=freq2, closed="left")

    df1 = dev.get_dataframe(i1, choice)
    df2 = dev.get_dataframe(i2, choice)

    if choice == "qw":  # error case
        with pytest.raises(ValueError):
            result = _make_df(df1)
        return

    df1 = combination_of_two(choice, df1)
    df2 = combination_of_two(choice, df2)

    result = _make_df({"q": df1.q, "r": df1.r})
    assert_p_q_r_compatible(result.r, df2.p, result.q)
    assert_w_q_compatible(freq, df2.w, result.q)
    expectedresult = set_ts_index(pd.DataFrame({"q": df2.q, "r": df2.r})).dropna()

    # CASE 1 : UNEQUAL FREQUENCY
    if freq1 != freq2:
        with pytest.raises(ValueError):
            result = _make_df(df1)
        return

    pd.testing.assert_frame_equal(result, expectedresult)


# . check with keys having unequal indexes: unequal timeperiod.
@pytest.mark.parametrize("freq", ["MS", "D"])
@pytest.mark.parametrize("choice", ["qw", "rp", "wp", "pq", "qr", "wr"])
def test_pfline_equaltimeperiods(freq, choice):

    i1 = pd.date_range(start="01-01-2020", end="01-06-2020", freq=freq, closed="left")
    i2 = pd.date_range(start="01-03-2020", end="01-09-2020", freq=freq, closed="left")

    df1 = dev.get_dataframe(i1, choice)
    df2 = dev.get_dataframe(i2, choice)

    if choice == "qw":  # error case
        with pytest.raises(ValueError):
            result = _make_df(df1)
        return

    df1 = combination_of_two(choice, df1)
    df2 = combination_of_two(choice, df2)

    result = _make_df({"q": df1.q, "r": df1.r})
    assert_p_q_r_compatible(result.r, df2.p, result.q)
    assert_w_q_compatible(freq, df2.w, result.q)
    expectedresult = set_ts_index(pd.DataFrame({"q": df2.q, "r": df2.r})).dropna()

    # CASE 2 : UNEQUAL TIMESERIES
    intersection = result.index.intersection(expectedresult)

    if intersection is None:
        # raise ValueError("The two timeseries do not have anything in common.")
        with pytest.raises(ValueError):
            result = _make_df(df1)
        return

    else:
        # check result have same intersection as that of expected result
        pd.testing.assert_frame_equal(result, expectedresult)


# Assert correct working of pfline:
# . initialisation with dictionary, with dataframe, with named tuple.
# . initialisation with pfline returns identical pfline.
# . .kind property always correctly set.
# . timeseries can be accessed with .q, .p, .r, .w, ['q'], ['p'], etc.
# . check correct working of attributes .df() and .changefreq().
# . check correct working of dunder methods. E.g. assert correct addition:
# . . pflines having same or different kind
# . . pflines having same or different frequency
# . . pflines covering same or different time periods
