import datetime as dt
import numpy as np
from lichtblyck.prices import utils
import pandas as pd
import pytest

@pytest.fixture(params=[None, "Europe/Berlin"])
def tz(request):
    return request.param

@pytest.mark.parametrize(
    ("value", "ispeak"),
    [
        ("2020-01-01 01:00", False),
        ("2020-01-01 07:00", False),
        ("2020-01-01 08:00", True),
        ("2020-01-01 19:00", True),
        ("2020-01-01 20:00", False),
        ("2020-01-03 07:59", False),
        ("2020-01-03 08:00", True),
        ("2020-01-03 19:59", True),
        ("2020-01-03 20:00", False),
        ("2020-01-04 07:59", False),
        ("2020-01-04 08:00", False),
        ("2020-01-04 19:59", False),
        ("2020-01-04 20:00", False),
        ("2020-01-05 07:59", False),
        ("2020-01-05 08:00", False),
        ("2020-01-05 19:59", False),
        ("2020-01-05 20:00", False),
        ("2020-03-29 01:00", False),
        ("2020-03-29 03:00", False),
        ("2020-10-25 01:00", False),
        ("2020-10-25 03:00", False),
    ],
)
def test_is_peak_hour(value, tz, ispeak):
    ts = pd.Timestamp(value, tz=tz)
    assert utils.is_peak_hour(ts) == ispeak


@pytest.mark.parametrize(
    ("left", "add_left"),
    [
        ("2020-01-01", (1440, 516)),
        ("2020", (1440, 516)),
        ("2020-01-01 12:00", (1428, 512)),
        ("2020-01-05 12:00", (1332, 480)),
    ],
)
@pytest.mark.parametrize(
    ("tz", "add_middle"), [("Europe/Berlin", (743, 264)), (None, (744, 264))]
)
@pytest.mark.parametrize(
    ("right", "add_right"),
    [
        ("2020-04-01", (0, 0)),
        ("2020-04", (0, 0)),
        ("2020-04-01 12:00", (12, 4)),
        ("2020-04-05 12:00", (108, 36)),
    ],
)
def test_duration_bpo(left, right, tz, add_left, add_middle, add_right):
    # add_left is duration (base, peak) until 01 march 2020 0:00;
    # add_middle is duration (base, peak) in march 2020;
    # add_right is duration (base, peak) after 01 april 2020 0:00
    duration_base = add_left[0] + add_middle[0] + add_right[0]
    duration_peak = add_left[1] + add_middle[1] + add_right[1]
    ts_left, ts_right = pd.Timestamp(left, tz=tz), pd.Timestamp(right, tz=tz)
    durations = utils.duration_bpo(ts_left, ts_right)
    assert durations == (duration_base, duration_peak, duration_base - duration_peak)


@pytest.mark.parametrize(
    ("ts_trade", "period_type", "period_start", "expected_left"),
    [
        ("2020-1-1", "d", 0, "2020-1-1"),
        ("2020-1-1", "m", 0, "2020-1-1"),
        ("2020-1-1", "q", 0, "2020-1-1"),
        ("2020-1-1", "s", 0, "2019-10-1"),
        ("2020-1-1", "a", 0, "2020-1-1"),
        ("2020-1-1", "d", 1, "2020-1-2"),
        ("2020-1-1", "m", 1, "2020-2-1"),
        ("2020-1-1", "q", 1, "2020-4-1"),
        ("2020-1-1", "s", 1, "2020-4-1"),
        ("2020-1-1", "a", 1, "2021-1-1"),
        ("2020-1-1", "d", 3, "2020-1-4"),
        ("2020-1-1", "m", 3, "2020-4-1"),
        ("2020-1-1", "q", 3, "2020-10-1"),
        ("2020-1-1", "s", 3, "2021-4-1"),
        ("2020-1-1", "a", 3, "2023-1-1"),
        ("2020-1-31", "d", 0, "2020-1-31"),
        ("2020-1-31", "m", 0, "2020-1-1"),
        ("2020-1-31", "q", 0, "2020-1-1"),
        ("2020-1-31", "s", 0, "2019-10-1"),
        ("2020-1-31", "a", 0, "2020-1-1"),
        ("2020-1-31", "d", 1, "2020-2-1"),
        ("2020-1-31", "m", 1, "2020-2-1"),
        ("2020-1-31", "q", 1, "2020-4-1"),
        ("2020-1-31", "s", 1, "2020-4-1"),
        ("2020-1-31", "a", 1, "2021-1-1"),
        ("2020-1-31", "d", 3, "2020-2-3"),
        ("2020-1-31", "m", 3, "2020-4-1"),
        ("2020-1-31", "q", 3, "2020-10-1"),
        ("2020-1-31", "s", 3, "2021-4-1"),
        ("2020-1-31", "a", 3, "2023-1-1"),
        ("2020-2-14", "s", 0, "2019-10-1"),
        ("2020-3-14", "s", 0, "2019-10-1"),
        ("2020-4-14", "s", 0, "2020-4-1"),
        ("2020-5-14", "s", 0, "2020-4-1"),
        ("2020-6-14", "s", 0, "2020-4-1"),
        ("2020-7-14", "s", 0, "2020-4-1"),
        ("2020-8-14", "s", 0, "2020-4-1"),
        ("2020-9-14", "s", 0, "2020-4-1"),
        ("2020-10-14", "s", 0, "2020-10-1"),
        ("2020-11-14", "s", 0, "2020-10-1"),
        ("2020-12-14", "s", 0, "2020-10-1"),
        ("2020-2-14", "s", 1, "2020-4-1"),
        ("2020-3-14", "s", 1, "2020-4-1"),
        ("2020-4-14", "s", 1, "2020-10-1"),
        ("2020-5-14", "s", 1, "2020-10-1"),
        ("2020-6-14", "s", 1, "2020-10-1"),
        ("2020-7-14", "s", 1, "2020-10-1"),
        ("2020-8-14", "s", 1, "2020-10-1"),
        ("2020-9-14", "s", 1, "2020-10-1"),
        ("2020-10-14", "s", 1, "2021-4-1"),
        ("2020-11-14", "s", 1, "2021-4-1"),
        ("2020-12-14", "s", 1, "2021-4-1"),
        ("2020-2-14", "s", 3, "2021-4-1"),
        ("2020-3-14", "s", 3, "2021-4-1"),
        ("2020-4-14", "s", 3, "2021-10-1"),
        ("2020-5-14", "s", 3, "2021-10-1"),
        ("2020-6-14", "s", 3, "2021-10-1"),
        ("2020-7-14", "s", 3, "2021-10-1"),
        ("2020-8-14", "s", 3, "2021-10-1"),
        ("2020-9-14", "s", 3, "2021-10-1"),
        ("2020-10-14", "s", 3, "2022-4-1"),
        ("2020-11-14", "s", 3, "2022-4-1"),
        ("2020-12-14", "s", 3, "2022-4-1"),
    ],
)
def test_ts_deliv(ts_trade, period_type, period_start, tz, expected_left):
    ts_trade = pd.Timestamp(ts_trade, tz=tz)
    expected_left = pd.Timestamp(expected_left, tz=tz)
    ts_deliv = utils.ts_deliv(ts_trade, period_type, period_start)
    assert ts_deliv[0] == expected_left
    try:
        add = {"m": 1, "q": 3, "s": 6, "a": 12}[period_type]
        assert ts_deliv[1] == expected_left + pd.offsets.MonthBegin(add)
    except KeyError:
        assert ts_deliv[1] == expected_left + dt.timedelta(1)


@pytest.mark.parametrize(
    ("p_b", "p_p", "duration_b", "duration_p", "p_op_expected"),
    [
        (1, 1, 100, 50, 1),
        (1, 2, 100, 50, 0),
        (1, 3, 100, 50, -1),
        (1.5, 4.5, 200, 50, 0.5),
        (100, 100, 10, 5, 100),
        (100, 200, 20, 4, 75),
    ],
)
def test_p_peak(p_b, p_p, duration_b, duration_p, p_op_expected):
    assert np.isclose(utils.p_offpeak(p_b, p_p, duration_b, duration_p), p_op_expected)
