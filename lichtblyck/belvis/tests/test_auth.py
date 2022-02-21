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
            with pytest.raises(PermissionError):
                connector.connection_alive("power")
        return

    if how == "usrpwdstr":

        if case == "wrong":
            # Authentication with incorrect credentials should give error.
            with pytest.raises(PermissionError):
                connector.auth_with_password("nonexstinguser", "")
            return

        elif case == "correct":
            # Authentication with correct credentials should work.
            connector.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")

    elif how == "usrpwdfile":

        if case == "wrong":
            # Authentication with incorrect credentials should give error.
            path = Path(__file__).parent / "wrongcreds.txt"
            with pytest.raises(PermissionError):
                connector.auth_with_passwordfile(path)
            return

        elif case == "correct":
            # Authentication with correct credentials should work.
            path = Path(__file__).parent / "correctcreds.txt"
            connector.auth_with_passwordfile(path)

    elif how == "token":
        # Authentication with token should work.
        connector.auth_with_token("API-User-FRM")

    # See if we can get information on existing timeseries.
    info = connector.info("power", 38721055)

    with pytest.raises(RuntimeError):
        info = connector.info("power", 9999999)
