from lichtblyck.belvis import raw, connect
from pathlib import Path
import pytest


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("correct", [True, False])
@pytest.mark.parametrize("tenant", ["PFMGAS", "PFMSTROM"])
def test_auth_usrpwd(tenant: str, direct: bool, correct: bool):
    """Test if authentication with username and password works as expected."""

    if direct:
        klass = connect.Connection_From_UsrPwd
    else:
        klass = connect.BelvisConnection.from_usrpwd

    if correct:
        # Authentication with correct credentials should work.
        _ = klass(tenant, "API-User-FRM", "boring!Apfelmexiko85hirsch")
    else:
        # Authentication with incorrect credentials should give error.
        with pytest.raises(PermissionError):
            _ = klass(tenant, "nonexstinguser", "incorrectpwd")
        return


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("correct", [True, False])
@pytest.mark.parametrize("tenant", ["PFMGAS", "PFMSTROM"])
def test_auth_token(tenant: str, direct: bool, correct: bool):
    """Test if authentication with token works as expected."""

    if direct:
        klass = connect.Connection_From_Token
    else:
        klass = connect.BelvisConnection.from_token

    if correct:
        # Authentication with correct credentials should work.
        _ = klass(tenant, "API-User-FRM")
    else:
        # Authentication with incorrect credentials should give error.
        with pytest.raises(PermissionError):
            _ = klass(tenant, "nonexstinguser")
        return


@pytest.mark.parametrize("how", ["usrpwdstr", "usrpwdfile", "token"])
@pytest.mark.parametrize("case", ["before", "wrong", "correct"])
def test_authentication(how: str, case: str):
    """Test if authentication works as expected at 'raw' module."""

    if case == "before":
        if how == "usrpwdstr":
            # Without authenication, we should get an error.
            with pytest.raises(PermissionError):
                raw.connection_alive("power")
        return

    if how == "usrpwdstr":

        if case == "wrong":
            # Authentication with incorrect credentials should give error.
            with pytest.raises(PermissionError):
                raw.auth_with_password("nonexstinguser", "")
            return

        elif case == "correct":
            # Authentication with correct credentials should work.
            raw.auth_with_password("API-User-FRM", "boring!Apfelmexiko85hirsch")

    elif how == "usrpwdfile":

        if case == "wrong":
            # Authentication with incorrect credentials should give error.
            path = Path(__file__).parent / "wrongcreds.txt"
            with pytest.raises(PermissionError):
                raw.auth_with_passwordfile(path)
            return

        elif case == "correct":
            # Authentication with correct credentials should work.
            path = Path(__file__).parent / "correctcreds.txt"
            raw.auth_with_passwordfile(path)

    elif how == "token":
        # Authentication with token should work.
        raw.auth_with_token("API-User-FRM")

    # See if we can get information on existing timeseries.
    info = raw.info("power", 38721055)

    with pytest.raises(RuntimeError):
        info = raw.info("power", 9999999)
