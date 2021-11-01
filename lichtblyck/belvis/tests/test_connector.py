from lichtblyck.belvis import connector
import json
import requests
import pandas as pd
import pytest


def test_authenication():
    # Without authenication, we should get an error.
    with pytest.raises():
        connector.connection_alive()

    # Authentication with incorrect credentials should give error.
    with pytest.raises():
        connector.auth_with_password("nonexstinguser", "")

    # Authentication with correct credentials should work.
    connector.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")

    # Authentication with token should work.connector.auth_with_token()

    # See if we can get information on existing timeseries.
    info = connector.info(123)

    # Changing the commodity should cause renewed authentication.
    connector.set_commodity("gas")
    connector.set_commodity("power")
    with pytest.raises():
        info = connector.info(123)


_didauth = False


@pytest.fixture(autouse=True)
def run_before_tests():
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want
    global _didauth

    if not _didauth:  # check if authentication (either by session or token) is done.
        connector.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")
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
    result = connector.info(id)
    for key, value in partial_result.items():
        assert result[key] == value


def pf_and_allids():
    return [
        ("LUD", [44133187, 44133192, 44133197, 44133202]),
        ("PKG", [16238, 16240, 16242, 16244, 16246]),
        ("B2B", []),
    ]


# @pytest.mark.parametrize(("pf", "partial_result"), pf_and_allids())
@pytest.mark.parametrize(
    ("pf", "partial_result"),
    [
        ("LUD", [44133187, 44133192, 44133197, 44133202]),
        ("PKG", [16238, 16240, 16242, 16244, 16246]),
        ("B2B", []),
    ],
)
def test_all_ids_in_pf(pf, partial_result):
    if partial_result:
        result = connector.all_ids_in_pf(pf)
        for key, value in partial_result.items():
            assert result[key] == value

    else:
        with pytest.raises(ValueError):
            connector.all_ids_in_pf(pf)


@pytest.mark.parametrize(
    ("partial_or_exact_pf_name", "partial_result"),
    [
        (
            "udwig",
            {
                "DA_Ludwig": "DayAhead_Ludwig",
                "Deutschland": "EDG_(Ludwig)",
                "ID_Ludwig": "Intraday_Ludwig",
                "LUD": "Ludwig",
                "LUD_HKN": "Ludwig_HKN",
            },
        )(
            "LUD",
            {
                "01.08.2020 - 01.09.2020": "20200729_S_HI_Verkauf_Base_LUD_NSp_SiM_7_MW",
                "01.09.2020 - 01.10.2020": "20200825_S_HI_Verkauf_Peak_LUD_NSp_SiM_2_MW",
                "01.09.2021 - 01.10.2021": "20210824_S_HI_Verkauf_Peak_LUD_WP_0_MWh",
                "LBK_HBK_DMY_50H_prod": "S_LPRG_SiM_Gegengeschaeft_LUD_WP_MWh",
                "LBK_HBK_TEN_cons": "S_PRG_LUD_WP_MWh",
            },
        ),
        (
            "PKG",
            {
                "01.03.2016 - 01.04.2016": "20160223_S_HI_Verkauf_Base_PKG_8_MW",
                "01.03.2017 - 01.04.2017": "20170227_S_HI_Verkauf_Base_PKG_0_MW",
                "01.03.2018 - 01.04.2018": "20180219_S_HI_Verkauf_Base_PKG_3_MW",
                "01.03.2019 - 01.04.2019": "20190226_S_HI_Kauf_Base_PKG_1_MW",
            },
        ),
    ],
)
def test_find_pfs(partial_or_exact_pf_name, partial_result):
    if partial_result:
        result = connector.find_pfs(partial_or_exact_pf_name)
        for key, value in partial_result.items():
            assert result[key] == value

    else:
        with pytest.raises(ValueError):
            connector.find_pfs(partial_or_exact_pf_name)


@pytest.mark.parametrize(
    ("pf", "name", "partial_result"),
    [
        ("LUD", "#LB FRM Procurement/Forward - MW - excl subpf", 44133207),
        ("PKG", "#LB Saldo aller Spotgeschäfte +UB", 42818406),
    ],
)
def test_find_id(pf, name, partial_result):
    if partial_result:
        result = connector.find_id(pf, name)
        if result == partial_result:
            assert result

    else:
        with pytest.raises(ValueError):
            connector.find_id(pf, name)


@pytest.mark.parametrize(
    ("id", "ts_left", "ts_right", "partial_result"),
    [(42818406, "2020-01-01 00:00:00", "2020-12-31 23:59:59", {})],
)
def test_records(id, ts_left, ts_right, partial_result):
    result = connector.records(id, ts_left, ts_right)
    for key, value in partial_result.items():
        assert result[key] == value


@pytest.mark.parametrize(
    ("id", "ts_left", "ts_right", "partial_result"),
    [(42818406, "2020-01-01 00:00:00", "2020-12-31 23:59:59", {})],
)
def test_series(id, ts_left, ts_right, partial_result):
    result = connector.series(id, ts_left, ts_right)
    for key, value in partial_result.items():
        assert result[key] == value
