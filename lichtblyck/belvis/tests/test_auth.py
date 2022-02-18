from lichtblyck.belvis import connector
from pathlib import Path
import pytest


@pytest.mark.parametrize("how", ["usrpwdstr", "usrpwdfile", "token"])
@pytest.mark.parametrize("case", ["before", "wrong", "correct"])
def test_authenication(how: str, case: str):
    """Test if authentication works as expected."""

    if case == "before":
        if how == "usrpwdstr":
            # Without authenication, we should get an error.
            with pytest.raises(ConnectionError):
                connector.connection_alive()
        return

    if how == "usrpwdstr":

        if case == "wrong":
            # Authentication with incorrect credentials should give error.
            with pytest.raises(ConnectionError):
                connector.auth_with_password("nonexstinguser", "")
            return

        elif case == "correct":
            # Authentication with correct credentials should work.
            connector.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")

    elif how == "usrpwdfile":

        if case == "wrong":
            # Authentication with incorrect credentials should give error.
            with pytest.raises(ConnectionError):
                path = Path(__file__).parent / "wrongcreds.txt"
                connector.auth_with_passwordfile(path)
            return

        elif case == "correct":
            # Authentication with correct credentials should work.
            path = Path(__file__).parent / "correctcreds.txt"
            connector.auth_with_passwordfile(path)

    elif how == "token":
        # Authentication with token should work.connector.auth_with_token()
        connector.auth_with_token()

    # See if we can get information on existing timeseries.
    info = connector.info("power", 123)

    with pytest.raises(ValueError):
        info = connector.info("gas", 123)
