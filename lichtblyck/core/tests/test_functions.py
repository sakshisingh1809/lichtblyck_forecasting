from lichtblyck.core import functions
import numpy as np
import pandas as pd
import pytest


freqs_small_to_large = ["T", "5T", "15T", "30T", "H", "2H", "D", "MS", "QS", "AS"]


@pytest.fixture(params=freqs_small_to_large)
def freq1(request):
    return request.param


@pytest.fixture(params=freqs_small_to_large)
def freq2(request):
    return request.param


def test_freq_up_or_down(freq1, freq2):
    i1 = freqs_small_to_large.index(freq1)
    i2 = freqs_small_to_large.index(freq2)
    outcome = np.sign(i1 - i2)
    assert functions.freq_up_or_down(freq1, freq2) == outcome


def get_summable_combinations_nodst_nojagged():
    combinations = []

    def get_index(freq):
        return pd.date_range(
            "2020-01-01", "2020-03-01", freq=freq, tz="Europe/Berlin", closed="left"
        )

    series_uniformshort = [
        pd.Series([5.0] * 24 * (31 + 29), get_index("H"), name="a"),
        pd.Series([5.0 * 24] * (31 + 29), get_index("D"), name="a"),
        pd.Series([5.0 * 24 * 31, 5.0 * 24 * 29], get_index("MS"), name="a"),
    ]
    series_uniformlong = [
        pd.Series(
            [*[5.0 / 31 / 24] * 31 * 24, *[5.0 / 29 / 24] * 29 * 24],
            get_index("H"),
            name="a",
        ),
        pd.Series([*[5.0 / 31] * 31, *[5.0 / 29] * 29], get_index("D"), name="a"),
        pd.Series([5.0, 5.0], get_index("MS"), name="a"),
    ]
    combis = [
        (s1, s2, f"{s1.index.freq}-{s2.index.freq}")
        for series in [series_uniformshort, series_uniformlong]
        for s1 in series
        for s2 in series
    ]

    # As series.
    combinations.extend([(s1, s2, f"series-{descr}") for s1, s2, descr in combis])

    # As dataframes.
    combinations.extend(
        [
            (pd.DataFrame(s1), pd.DataFrame(s2), f"dataframe-{descr}")
            for s1, s2, descr in combis
        ]
    )
    return combinations


def get_averagable_combinations_nodst_nojagged():
    combinations = []

    def get_index(freq):
        return pd.date_range(
            "2020-01-01", "2020-03-01", freq=freq, tz="Europe/Berlin", closed="left"
        )

    series_uniform = [
        pd.Series([5.0] * 24 * (31 + 29), get_index("H"), name="a"),
        pd.Series([5.0] * (31 + 29), get_index("D"), name="a"),
        pd.Series([5.0] * 2, get_index("MS"), name="a"),
    ]
    combis = [
        (s1, s2, f"{s1.index.freq}-{s2.index.freq}")
        for s1 in series_uniform
        for s2 in series_uniform
    ]

    # As series.
    combinations.extend([(s1, s2, f"series-{descr}") for s1, s2, descr in combis])

    # As dataframes.
    combinations.extend(
        [
            (pd.DataFrame(s1), pd.DataFrame(s2), f"dataframe-{descr}")
            for s1, s2, descr in combis
        ]
    )
    return combinations


def changfreq_combinations():
    return [
        *[
            (*params, functions.changefreq_sum)
            for params in get_summable_combinations_nodst_nojagged()
        ],
        *[
            (*params, functions.changefreq_avg)
            for params in get_averagable_combinations_nodst_nojagged()
        ],
    ]


@pytest.mark.parametrize(
    "fr1,fr2,description,changefreq_function", changfreq_combinations()
)
def test_changefreq(fr1, fr2, description, changefreq_function):
    testfr = changefreq_function(fr1, fr2.index.freq)
    if isinstance(fr1, pd.Series):
        pd.testing.assert_series_equal(testfr, fr2)
    else:
        pd.testing.assert_frame_equal(testfr, fr2)


# TODO: test where timeseries is downsampled that starts/ends in e.g. middle of month.

