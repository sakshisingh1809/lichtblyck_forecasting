from lichtblyck.belvis import connector
import pytest


@pytest.mark.parametrize("case", ["beforeauth", "wrongcred", "correctcred", "token"])
def test_authenication(case: str):
    """Test if authentication works as expected."""

    if case == "beforeauth":
        # Without authenication, we should get an error.
        with pytest.raises(ConnectionError):
            connector.connection_alive()
        return

    elif case == "wrongcred":
        # Authentication with incorrect credentials should give error.
        with pytest.raises(ConnectionError):
            connector.auth_with_password("nonexstinguser", "")
        return

    elif case == "correctcred":
        # Authentication with correct credentials should work.
        # connector.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")
        connector.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")

    elif case == "token":
        # Authentication with token should work.connector.auth_with_token()
        connector.auth_with_token()

    # See if we can get information on existing timeseries.
    info = connector.info("power", 123)

    with pytest.raises(ValueError):
        info = connector.info("gas", 123)
