from pandas._libs.tslibs.timestamps import Timestamp
from lichtblyck.tools import stamps
import pandas as pd
import numpy as np
import pytest

freqs_small_to_large = ["T", "5T", "15T", "30T", "H", "2H", "D", "MS", "QS", "AS"]


@pytest.mark.parametrize("iter", [False, True])  # make iterable or not
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("ts", "fut", "freq", "expected"),
    [
        (pd.Timestamp("2020"), 0, "D", pd.Timestamp("2020")),
        (pd.Timestamp("2020"), 0, "MS", pd.Timestamp("2020")),
        (pd.Timestamp("2020"), 0, "QS", pd.Timestamp("2020")),
        (pd.Timestamp("2020"), 0, "AS", pd.Timestamp("2020")),
        (pd.Timestamp("2020"), 1, "D", pd.Timestamp("2020-01-02")),
        (pd.Timestamp("2020"), 1, "MS", pd.Timestamp("2020-02")),
        (pd.Timestamp("2020"), 1, "QS", pd.Timestamp("2020-04")),
        (pd.Timestamp("2020"), 1, "AS", pd.Timestamp("2021")),
        (pd.Timestamp("2020"), -1, "D", pd.Timestamp("2019-12-31")),
        (pd.Timestamp("2020"), -1, "MS", pd.Timestamp("2019-12")),
        (pd.Timestamp("2020"), -1, "QS", pd.Timestamp("2019-10")),
        (pd.Timestamp("2020"), -1, "AS", pd.Timestamp("2019")),
        (pd.Timestamp("2020-01-01 23:55"), 0, "D", pd.Timestamp("2020")),
        (pd.Timestamp("2020-01-24 1:32"), 0, "MS", pd.Timestamp("2020")),
        (pd.Timestamp("2020-03-03 3:33"), 0, "QS", pd.Timestamp("2020")),
        (pd.Timestamp("2020-10-11 12:34"), 0, "AS", pd.Timestamp("2020")),
        (pd.Timestamp("2020-01-01 23:55"), 1, "D", pd.Timestamp("2020-01-02")),
        (pd.Timestamp("2020-01-24 1:32"), 1, "MS", pd.Timestamp("2020-02")),
        (pd.Timestamp("2020-03-03 3:33"), 1, "QS", pd.Timestamp("2020-04")),
        (pd.Timestamp("2020-10-11 12:34"), 1, "AS", pd.Timestamp("2021")),
        (pd.Timestamp("2020-01-01 23:55"), -1, "D", pd.Timestamp("2019-12-31")),
        (pd.Timestamp("2020-01-24 1:32"), -1, "MS", pd.Timestamp("2019-12")),
        (pd.Timestamp("2020-03-03 3:33"), -1, "QS", pd.Timestamp("2019-10")),
        (pd.Timestamp("2020-10-11 12:34"), -1, "AS", pd.Timestamp("2019")),
    ],
)
def test_floorts(ts, fut, freq, expected, tz, iter):
    if tz:
        ts = ts.tz_localize(tz)
        expected = expected.tz_localize(tz)

    if not iter:
        # Test single value.
        assert stamps.floor_ts(ts, freq, fut) == expected
    else:
        # Test index.
        periods = np.random.randint(4, 40)

        index = pd.date_range(ts, periods=periods, freq=freq)  # ts no longer at index 0
        index += ts - index[0]  # index starts again with ts and has non-floored values

        result = stamps.floor_ts(index, freq, fut)
        result.freq = None  # disregard checking frequencies here
        expected = pd.date_range(expected, periods=periods, freq=freq)
        expected.freq = None  # disregard checking frequencies here

        pd.testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("tz_left", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("tz_right", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("tss", "expected"),
    [
        (("2020-01-01", "2020-02-02"), ("2020-01-01", "2020-02-02")),
        (("2020-02-02", "2020-01-01"), ("2020-01-01", "2020-02-02")),
        (("2020-01-01", None), ("2020-01-01", "2021-01-01")),
        ((None, "2020-02-02"), ("2020-01-01", "2020-02-02")),
        (("2020-01-01", "2020-01-01"), ("2020-01-01", "2020-01-01")),
        (("2020-03-03 3:33", "2021-10-09"), ("2020-03-03 3:33", "2021-10-09")),
        (("2021-10-09", "2020-03-03 3:33"), ("2020-03-03 3:33", "2021-10-09")),
        (("2020-03-03 3:33", "2021-10-09"), ("2020-03-03 3:33", "2021-10-09")),
        (("2020-03-03 3:33", None), ("2020-03-03 3:33", "2021-01-01")),
        ((None, "2021-10-09"), ("2021-01-01 0:00", "2021-10-09")),
        (
            (None, None),
            (
                pd.Timestamp(pd.Timestamp.today().year + 1, 1, 1),
                pd.Timestamp(pd.Timestamp.today().year + 2, 1, 1),
            ),
        ),
    ],
)
def test_tsleftright(tss, expected, tz_left, tz_right):
    tzs = [tz_left, tz_right]
    tss = [pd.Timestamp(ts) for ts in tss]  # turn into timestamps for comparison
    if tss[0] == tss[1] and tz_left != tz_right:
        return  # too complicated to test; would have to recreate function here.
    swap = tss[0] > tss[1]
    tss = [ts.tz_localize(tz) for ts, tz in zip(tss, tzs)]  # add timezone info

    result = stamps.ts_leftright(*tss)

    exp_tzs = [tz for ts, tz in zip(tss, tzs) if tz is not None and ts is not pd.NaT]
    if swap:
        exp_tzs.reverse()
    if not len(exp_tzs):
        exp_tzs = ["Europe/Berlin"]
    if len(exp_tzs) == 1:
        exp_tzs = exp_tzs * 2
    exp_result = [pd.Timestamp(ts).tz_localize(tz) for ts, tz in zip(expected, exp_tzs)]

    for a, b in zip(result, exp_result):
        assert a == b


@pytest.mark.parametrize("freq1", freqs_small_to_large)
@pytest.mark.parametrize("freq2", freqs_small_to_large)
def test_frequpordown(freq1, freq2):
    i1 = freqs_small_to_large.index(freq1)
    i2 = freqs_small_to_large.index(freq2)
    outcome = np.sign(i1 - i2)
    assert stamps.freq_up_or_down(freq1, freq2) == outcome


@pytest.mark.parametrize("count", range(1, 30))
def test_longestshortestfreq(count):
    indices = np.random.randint(0, len(freqs_small_to_large), count)
    freqs = np.array(freqs_small_to_large)[indices]
    assert stamps.freq_longest(*freqs) == freqs_small_to_large[max(indices)]
    assert stamps.freq_shortest(*freqs) == freqs_small_to_large[min(indices)]
