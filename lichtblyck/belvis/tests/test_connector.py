from lichtblyck.belvis import connector
import pytest

def test_authenication():
    # Without authenication, we should get an error.
    with pytest.raises():
        connector.connection_alive()
    # Authentication with incorrect credentials should give error.
    with pytest.raises():
        connector.auth_with_password('nonexstinguser', '')
    # Authentication with correct credentials should work.
    connector.auth_with_password('Ruud.Wijtvliet', 'Ammm1mmm2mmm3mmm')
    # Authentication with token should work.
    connector.auth_with_token()

    # See if we can get information on existing timeseries.
    info = connector.info(123)

    # Changing the commodity should cause renewed authentication.
    connector.set_commodity('gas')
    connector.set_commodity('power')
    with pytest.raises():
        info = connector.info(123)


_didauth = False


def test_info():
    global _didauth
    if not _didauth:
        connector.auth_with_token()
        _didauth = True
    
    pass

def test_getallpfs():
    global _didauth
    if not _didauth:
        connector.auth_with_token()
        _didauth = True
        
    
    pass


    


# 