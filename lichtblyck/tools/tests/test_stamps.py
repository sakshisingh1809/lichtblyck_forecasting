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
        assert stamps.floor_ts(ts, fut, freq) == expected
    else:
        # Test index.
        periods = np.random.randint(4, 40)

        index = pd.date_range(ts, periods=periods, freq=freq) #ts no longer at index 0
        index += ts - index[0] #index starts again with ts and has non-floored values

        result = stamps.floor_ts(index, fut, freq)
        result.freq = None  # disregard checking frequencies here
        expected = pd.date_range(expected, periods=periods, freq=freq)
        expected.freq = None  # disregard checking frequencies here

        pd.testing.assert_index_equal(result, expected)




@pytest.mark.parametrize("freq1", freqs_small_to_large)
@pytest.mark.parametrize("freq2", freqs_small_to_large)
def test_frequpordown(freq1, freq2):
    i1 = freqs_small_to_large.index(freq1)
    i2 = freqs_small_to_large.index(freq2)
    outcome = np.sign(i1 - i2)
    assert stamps.freq_up_or_down(freq1, freq2) == outcome


@pytest.mark.parametrize('count', range(1, 30))
def test_longestshortestfreq(count):
    indices = np.random.randint(0, len(freqs_small_to_large), count)
    freqs = np.array(freqs_small_to_large)[indices]
    assert stamps.freq_longest(*freqs) == freqs_small_to_large[max(indices)]
    assert stamps.freq_shortest(*freqs) == freqs_small_to_large[min(indices)]
