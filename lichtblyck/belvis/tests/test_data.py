"""
Test cases to get data from the Belvis API.
"""

from typing import Tuple
from lichtblyck.belvis import connector, data
from lichtblyck.testing import testing
import datetime as dt
import numpy as np
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


def start_and_end(past: bool) -> Tuple[str]:
    thisyear = dt.date.today().year
    year = thisyear + (-1 if past else 1)
    return f"{year}", f"{year+1}"


def expectedfreq(commodity: str) -> str:
    return "15T" if commodity == "power" else "D"


@pytest.mark.parametrize(("commodity", "pfid"), [("power", "WP")])
@pytest.mark.parametrize("past", (True, False))
def test_offtakevolume(commodity, pfid, past):
    """Test if offtake volume can be fetched, for present and past years."""
    # main test: do we get any data at all
    result = data.offtakevolume(commodity, pfid, *start_and_end(past))
    assert result.kind == "q"
    assert result.index.freq == expectedfreq(commodity)
    assert (result.w <= 0).all()


@pytest.mark.parametrize(("commodity", "pfid"), [("power", "WP")])
@pytest.mark.parametrize("past", (True, False))
def test_forward(commodity, pfid, past):
    """Test if forward/futures sourced volume and price can be fetched, for present and past years."""
    # main test: do we get any data at all
    result = data.forward(commodity, pfid, *start_and_end(past))
    assert result.kind == "all"
    assert result.index.freq == expectedfreq(commodity)
    assert (result.w >= 0).all()


@pytest.mark.parametrize(("commodity", "pfid"), [("power", "WP")])
@pytest.mark.parametrize("past", (True, False))
def test_spot(commodity, pfid, past):
    """Test if spot sourced volume and price can be fetched, for present and past years."""
    # main test: do we get any data at all
    result = data.spot(commodity, pfid, *start_and_end(past))
    assert result.kind == "all"
    assert result.index.freq == expectedfreq(commodity)
    # for future, all spot volumes must be 0
    if not past:
        assert np.allclose(result.w.pint.m, 0)


@pytest.mark.parametrize(("commodity", "pfid"), [("power", "WP")])
@pytest.mark.parametrize("past", (True, False))
def test_sourced(commodity, pfid, past):
    """Test if sourced volume and price can be fetched, for present and past years."""
    # main test: do we get any data at all
    result = data.sourced(commodity, pfid, *start_and_end(past))
    assert result.kind == "all"
    assert result.index.freq == expectedfreq(commodity)
    assert (result.w >= 0).all()
    assert "forward" in result
    assert "spot" in result
    if not past:
        assert np.allclose(result.spot.w.pint.m, 0)


@pytest.mark.parametrize("commodity", ("power", "gas"))
@pytest.mark.parametrize("past", (True, False))
def test_unsourcedprice(commodity, past):
    """Test if unsourced prices can be fetched."""
    # main test: do we get any data at all
    result = data.unsourcedprice(commodity, *start_and_end(past))
    assert result.kind == "p"
    assert result.index.freq == expectedfreq(commodity)
    # for future, all prices must be positive
    if not past:
        assert (result.p >= 0).all()
