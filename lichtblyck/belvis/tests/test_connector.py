from typing import Iterable

import py
from lichtblyck.belvis import connector
import pandas as pd
import pytest

_didauth = False


@pytest.fixture(autouse=True)
def run_before_tests():
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want
    global _didauth

    if not _didauth:  # check if authentication (either by session or token) is done.
        connector.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")
        _didauth = True

    yield  # this is where the testing happens


@pytest.mark.parametrize(
    ("id", "partial_result"),
    [
        (
            43554048,
            {
                "dataExchangeNumber": None,
                "instanceName": "Ludwig",
                "instanceToken": "LUD",
            },
        )
    ],
)
def test_info(id, partial_result):
    """Test if correct timeseries info can be retrieved."""
    result = connector.info("power", id)
    for key, value in partial_result.items():
        assert result[key] == value


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize(
    ("commodity", "name", "list_if_not_strict"),
    [
        (
            "power",
            "udwig",
            {
                "DA_Ludwig": "DayAhead_Ludwig",
                "Deutschland": "EDG_(Ludwig)",
                "ID_Ludwig": "Intraday_Ludwig",
                "LUD": "Ludwig",
                "LUD_HKN": "Ludwig_HKN",
            },
        ),
        (
            "power",
            "Lud",
            {
                "LUD_HKN": "Ludwig_HKN",
                "Ludwig_Rechnungspruefung": "Ludwig_Rechnungspruefung",
                "ID_Ludwig": "Intraday_Ludwig",
                "DA_Ludwig": "DayAhead_Ludwig",
                "T_Ludwig": "Termin_Ludwig",
            },
        ),
        (
            "power",
            "Privatkunden_Neu",
            {
                "PK_Neu": "Privatkunden_Neu",
                "PK_Neu_FLX": "Privatkunden_Neu_Flex",
                "PK_Neu_NSP": "Privatkunden_Neu_NSP",
                "PK_Neu_WP": "Privatkunden_Neu_WP",
                "PK_Neu_FLX_SiM": "Privatkunden_Neu_Flex_Sichere_Menge",
                "PK_Neu_NSP_SiM": "Privatkunden_Neu_NSP_Sichere_Menge",
                "PK_Neu_WP_SiM": "Privatkunden_Neu_WP_Sichere_Menge",
            },
        ),
        ("power", "Nonexistingname", {}),
    ],
)
def test_find_pfids(commodity, name, list_if_not_strict, strict):
    """Test if portfolio id can be found from their name."""
    if strict:
        expected = {
            key: value for key, value in list_if_not_strict.items() if value == name
        }
    else:
        expected = list_if_not_strict

    if not expected:
        with pytest.raises(ValueError):
            _ = connector.find_pfids(commodity, name, strict=strict)
    else:
        result = connector.find_pfids(commodity, name, strict=strict)
        for key, value in expected.items():
            assert key in result
            assert result[key] == value


tsidtestcases = [
    (
        "power",
        "LUD",
        "#LB FRM Procurement/Forward - MW - excl subpf",
        [44133207],
        44133207,
    ),  # 1 result which is also exact
    (
        "power",
        "PKG",
        "#LB Saldo aller Spotgeschäfte +UB",
        [44133207, 44133212, 44133274, 44133279],
        None,
    ),  # >1 results, 0 exact result
    (
        "power",
        "PKG",
        "#LB CPM Wert HI-Geschaefte ohne Spot",
        [38721055, 38721057],
        38721055,
    ),  # >1 result, 1 exact
    ("power", "PKG", "Noneexistingtimeseries", [], None),  # 0 results
]  # "commodity", "pfid", "name", "list_if_not_strict", "value_if_strict"


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize(
    ("commodity", "pfid", "name", "list_if_not_strict", "value_if_strict"),
    tsidtestcases,
)
def test_find_tsids(
    commodity, pfid, name, list_if_not_strict, value_if_strict, strict
):
    """Test if timeseries ids can be found from their name."""
    if strict:
        expectedkeys = [value_if_strict] if value_if_strict is not None else []
    else:
        expectedkeys = list_if_not_strict

    result = connector.find_tsids(commodity, pfid, name, strict=strict)
    for key in expectedkeys:
        assert key in result


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize(
    ("commodity", "pfid", "name", "list_if_not_strict", "value_if_strict"),
    tsidtestcases,
)
def test_find_tsid(
    commodity, pfid, name, list_if_not_strict, value_if_strict, strict
):
    """Test if timeseries id can be found from its name."""
    if strict:
        expected = [value_if_strict] if value_if_strict is not None else []
    else:
        expected = list_if_not_strict

    if len(expected) == 1:
        result = connector.find_tsids(commodity, pfid, name, strict=strict)
        assert result == expected[0]
    else:
        with pytest.raises(ValueError):
            _ = connector.find_tsids(commodity, pfid, name, strict=strict)


@pytest.mark.parametrize("method", ["records", "series"])
@pytest.mark.parametrize(
    ("commodity", "tsid", "ts_left", "ts_right", "has_result"),
    [
        ("power", 42818406, "2020-01-01 00:00:00", "2020-01-31 23:59:59", True),
        ("power", 99999999, "2020-01-01 00:00:00", "2020-01-31 23:59:59", False),
    ],
)
def test_records(method, commodity, tsid, ts_left, ts_right, has_result):
    """Test if data can be retrieved for existing timeseries."""
    ts_left, ts_right = pd.Timestamp(ts_left), pd.Timestamp(ts_right)

    # Test the correct method.
    if method == "records":
        getter = connector.records  # returns list
    else:
        getter = connector.series  # returns series

    if has_result:
        result = getter(commodity, tsid, ts_left, ts_right)
        assert isinstance(result, Iterable)
        assert len(result) > 0
    else:
        with pytest.raises((RuntimeError, ValueError)):
            _ = getter(commodity, tsid, ts_left, ts_right)
