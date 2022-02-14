from lichtblyck import testing
from lichtblyck.core.pfline import single_helper
from lichtblyck.core.develop import dev
from lichtblyck.tools.frames import set_ts_index
from lichtblyck.tools.nits import Q_
from lichtblyck.tools.stamps import FREQUENCIES
import pandas as pd
import pytest


def assert_w_q_compatible(freq, w, q):
    if freq == "15T":
        testing.assert_series_equal(q, w * Q_(0.25, "h"), check_names=False)
    elif freq == "H":
        testing.assert_series_equal(q, w * Q_(1, "h"), check_names=False)
    elif freq == "D":
        assert (q > w * Q_(22.99, "h")).all()
        assert (q < w * Q_(25.01, "h")).all()
    elif freq == "MS":
        assert (q > w * 27 * Q_(24, "h")).all()
        assert (q < w * 32 * Q_(24, "h")).all()
    elif freq == "QS":
        assert (q > w * 89 * Q_(24, "h")).all()
        assert (q < w * 93 * Q_(24, "h")).all()
    elif freq == "AS":
        assert (q > w * Q_(8759.9, "h")).all()
        assert (q < w * Q_(8784.1, "h")).all()
    else:
        raise ValueError("Uncaught value for freq: {freq}.")


def assert_p_q_r_compatible(r, p, q):
    testing.assert_series_equal(r, q * p, check_names=False)


@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("freq", FREQUENCIES)
def test_makedataframe_freqtz(freq, tz):
    """Test if dataframe can made from data with various timezones and frequencies."""

    i = dev.get_index(freq, tz)
    q = dev.get_series(i, "q")
    result1 = single_helper.make_dataframe({"q": q})

    expected = pd.DataFrame({"q": q})
    if tz is None:
        expected = expected.tz_localize("Europe/Berlin")
    expected.index.freq = freq

    testing.assert_frame_equal(result1, expected, check_names=False)

    if tz:
        w = q / q.index.duration
        result2 = single_helper.make_dataframe({"w": w})
        testing.assert_frame_equal(
            result2, expected, check_names=False, check_dtype=False
        )


@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("freq", ["MS", "D"])
@pytest.mark.parametrize(
    "columns",
    [
        "r",
        "p",
        "q",
        "w",
        "wq",
        "pr",
        "wp",
        "qp",
        "qr",
        "wr",
        "wqp",
        "qpr",
        "wqr",
        "wpr",
        "wqpr",
    ],
)
def test_makedataframe_consistency(tz, freq, columns):
    """Test if conversions are done correctly and inconsistent data raises error."""

    i = dev.get_index(freq, tz)
    df = dev.get_dataframe(i, columns)
    # dic = {key: df[key] for key in choice}

    if columns in ["r", "wq", "wqp", "wqr", "wpr", "qpr", "wqpr"]:  # error cases
        with pytest.raises(ValueError):
            result = single_helper.make_dataframe(df)
        # with pytest.raises(ValueError):
        #     result = single_helper.make_dataframe(dic)
        return

    result = single_helper.make_dataframe(df)
    df = set_ts_index(df)

    if columns == "p":  # kind == "p"
        expected = df[["p"]]

    elif columns in ["q", "w"]:  # kind == "q"
        if columns == "w":
            df["q"] = df.w * df.w.index.duration
        expected = df[["q"]]

    elif columns in ["pr", "qp", "wp", "qr", "wr"]:  # kind == "all"
        # fill dataframe first.
        if columns == "wp":
            df["q"] = df.w * df.w.index.duration
            df["r"] = df.p * df.q

        elif columns == "pr":
            df["q"] = df.r / df.p
            df["w"] = df.q / df.index.duration

        elif columns == "qp":
            df["r"] = df.p * df.q
            df["w"] = df.q / df.index.duration

        elif columns == "wr":
            df["q"] = df.w * df.w.index.duration
            df["p"] = df.r / df.q

        else:
            df["p"] = df.r / df.q
            df["w"] = df.q / df.index.duration

        assert_p_q_r_compatible(result.r, df.p, result.q)
        assert_w_q_compatible(freq, df.w, result.q)
        expected = df[["q", "r"]].dropna()

    testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("freq1", ["15T", "D", "MS", "QS"])  # don't do all - many!
@pytest.mark.parametrize("freq2", ["15T", "H", "D", "MS", "QS"])
@pytest.mark.parametrize("columns", ["rp", "wp", "pq", "qr", "wr"])
def test_makedataframe_unequalfrequencies(freq1, freq2, columns):
    """Test if error is raised when creating a dataframe from series with unequal frequencies."""

    kwargs = {"start": "2020", "end": "2021", "closed": "left", "tz": "Europe/Berlin"}
    i1 = pd.date_range(**kwargs, freq=freq1)
    i2 = pd.date_range(**kwargs, freq=freq2)

    s1 = dev.get_series(i1, columns[0])
    s2 = dev.get_series(i2, columns[1])

    dic = {columns[0]: s1, columns[1]: s2}

    if freq1 != freq2:
        with pytest.raises(ValueError):
            _ = single_helper.make_dataframe(dic)


@pytest.mark.parametrize("freq", ["15T", "H", "D", "MS"])
@pytest.mark.parametrize("overlap", [True, False])
def test_pfline_unequaltimeperiods(freq, overlap):
    """Test if only intersection is kept for overlapping series, and error is raised
    for non-overlapping series."""

    i1 = pd.date_range(
        start="2020-01-01",
        end="2020-06-01",
        freq=freq,
        closed="left",
        tz="Europe/Berlin",
    )
    start = "2020-03-01" if overlap else "2020-07-01"
    i2 = pd.date_range(
        start=start,
        end="2020-09-01",
        freq=freq,
        closed="left",
        tz="Europe/Berlin",
    )
    s1 = dev.get_series(i1, "q")
    s2 = dev.get_series(i2, "r")

    intersection = s1.index.intersection(s2.index)

    if not overlap:
        # raise ValueError("The two timeseries do not have anything in common.")
        with pytest.raises(ValueError):
            result = single_helper.make_dataframe({"q": s1, "r": s2})
        return

    result = single_helper.make_dataframe({"q": s1, "r": s2})
    testing.assert_series_equal(result.q, s1.loc[intersection])
    testing.assert_series_equal(result.r, s2.loc[intersection])
    testing.assert_index_equal(result.index, intersection)
