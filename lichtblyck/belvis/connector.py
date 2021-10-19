"""
Retrieve data from Belvis using Rest-API.
"""

# Developer notes, Belvis API document:
# 1. general info
# 2.2 finding timeseries
# 2.3 metadata timeseries
# 2.5 reading timeseries values
# 7 how to connect

from requests.exceptions import ConnectionError
from typing import Tuple, Dict, List, Union, Iterable
from urllib import parse
from pathlib import Path
import pandas as pd
import datetime as dt
import jwt
import time
import json
import requests

# import numpy as np
# import datetime as dt

# import subprocess
# import pathlib
# from OpenSSL import crypto
# from socket import gethostname

_PFDATAFILEPATH = Path(__file__).parent / "memoized" / "metadata.txt"

_server = "http://lbbelvis01:8040"
_auth = None
_commodity = "power"


def _tenant() -> str:
    """Convenience function to find Belvis tenant belonging to selected commodity."""
    return {"power": "PFMSTROM", "gas": "PFMGAS"}[_commodity]


def _getreq(path, *queryparts) -> requests.request:
    string = f"{_server}{path}"
    if queryparts:
        queryparts = [parse.quote(qp, safe=":=") for qp in queryparts]
        string += "?" + "&".join(queryparts)
    # print(string)
    try:
        if "session" in _auth:
            return _auth["session"].get(string)
        elif "token" in _auth:
            return requests.get(
                string, headers={"Authorization": "Bearer " + _auth["token"]}
            )
        else:
            raise ValueError(
                "First authenicate with `auth_with_password` or `auth_with_token`."
            )
    except ConnectionError as e:
        raise RuntimeError("Check if VPN connection to Lichtblick exists.") from e


def auth_with_password(usr: str = None, pwd: str = None) -> None:
    """Authenticaten with username and password; open a session.

    Parameters
    ----------
    usr : str, optional
        Belvis username. If none provided, use previously specified username.
    pwd : str, optional
        Belvis password for the given user. If none provided, use previously specified
        password.
    """
    global _auth

    # Handle case when usr and pwd not specified.
    if usr is None and "usr" in _auth:
        usr = _auth["usr"]
    if pwd is None and "pwd" in _auth:
        pwd = _auth["pwd"]
    if usr is None or pwd is None:
        raise ValueError("Username and/or password missing.")

    # Store for later use and log in.
    _auth = {"usr": usr, "pwd": pwd, "session": requests.Session()}
    _getreq("/rest/session", f"usr={usr}", f"pwd={pwd}", f"tenant={_tenant()}")

    # Check if successful.
    if not connection_alive():
        _auth = None
        raise ConnectionError("No connection exists. Username and password incorrect?")


def auth_with_token() -> None:
    """This method is used by current REST clients or Libraries supported.
    A trustworthy body generates a key pair from which it can be used for
    authorized persons Clients generate strings, so-called Bearer tokens.
    """
    global _auth
    # Open private key to sign token with.
    with open("privatekey.txt", "r") as f:
        private_key = f.read()

    # Create token that is valid for a given amount of time.
    claims = {
        "preferred_username": "Ruud.Wijtvliet",
        "clientId": _tenant(),
        "exp": int(time.time()) + 10 * 60,
    }

    # "RSA 512 bit" in the PKCS standard for your client.
    token = jwt.encode(claims, private_key, algorithm="RS512")
    _auth = {"token": token}

    # Check if successful.
    if not connection_alive():
        _auth = None
        raise ConnectionError("No connection exists. Token incorrect?")


def set_commodity(commodity: str) -> None:
    """Set commodity to get information for. One of {'power', 'gas'}."""
    global _commodity
    if commodity == _commodity:
        return  # commodity already set correctly, nothing to do
    if commodity not in ["power", "gas"]:
        raise ValueError("`commodity` must be 'power' or 'gas'.")
    _commodity = commodity
    if "session" in _auth:
        auth_with_password()  # redo authentication
    elif "token" in _auth:
        auth_with_token()  # redo authentication


def _object(response) -> Union[Dict, List]:
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        raise RuntimeError(response)


def connection_alive():
    """Return True if connection is still up."""
    return _getreq("/rest/belvis/internal/heartbeat/ping").status_code == 200


def info(id: int) -> Dict:
    """Get dictionary with information about timeseries with id `id`."""
    response = _getreq(f"/rest/energy/belvis/{_tenant()}/timeSeries/{id}")
    return _object(response)


def all_ids_in_pf(pf: str) -> List[int]:
    """Gets ids of all timeseries in offtake portfolio `pf`.

    Parameters
    ----------
    pf : str
        Portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM')

    Returns
    -------
    List[int]
        Found timeseries ids.
    """
    response = _getreq(
        f"/rest/energy/belvis/{_tenant()}/timeseries", f"instancetoken={pf}"
    )
    restlist = _object(response)
    ids = [int(entry.split("/")[-1]) for entry in restlist]
    if not ids:
        raise ValueError("No timeseries found. Check the portfolio abbreviation.")
    return ids


def fetch_pfinfo() -> None:
    """This is an expensive function, which is called only once from main().
    This function searches the whole belvis database, creates a list of all ids, and 
    stores the information in json format in a textfile for later use.
    """
    # Get all pfs.
    response = _getreq(f"/rest/energy/belvis/{_tenant()}/timeseries")
    restlist = _object(response)
    ids = [int(entry.split("/")[-1]) for entry in restlist]

    # Create json file with all pf names ()
    metadata = []
    for id in ids:
        record = [None] * 2
        record[0] = info(id)["instanceToken"]
        record[1] = info(id)["instanceName"]

        metadata = sorted(metadata)
        if record not in metadata:
            metadata.append(record)

    with open(_PFDATAFILEPATH, "w") as outfile:
        json.dump(metadata, outfile)


def find_pfs(partial_or_exact_pf_name: str, refresh: bool = False) -> str:
    """Find the exact portfolio abbreviation given any 'pf' names (full or partial).

    Parameters
    ----------
    partial_or_exact_pf_name : str
        Exact or partial name of portfolio.
    refresh : bool, optional
        If true, refreshes (and stores) all portfolio information. NB: Expensive function
        (takes a long time), only run if any changes to the portfolio or the timeseries
        have been made.

    Returns
    -------
    str
        Portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM').

    Raises
    ------
    ValueError
        If no, or more than 1, matching portfolio is found.
    """
    if refresh:
        fetch_pfinfo()

    # Get info of each id.
    with open(_PFDATAFILEPATH) as json_file:
        data = json.load(json_file)

    # Convert data(list of list) into list of dictionaries
    hits = {d[0]: d[1] for d in data if partial_or_exact_pf_name in d[1]}

    # Raise error if 0 or > 1 found.
    if len(hits) == 0:
        raise ValueError("No portfolios found. Check parameters; use .find_id")

    return hits


def find_id(pf: str, name: str) -> int:
    """In offtake portfolio `pf`, find id of timeseries with name `name`.

    Parameters
    ----------
    pf : str
        Portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM').
    name : str
        Name of timeseries (e.g. '#LB FRM Procurement/Forward - MW - excl subpf').
        Partial names also work, as long as they are unique to the timeseries.

    Returns
    -------
    int
        id of found timeseries.
    """
    # Get all ids belonging to pf.
    all_ids = all_ids_in_pf(pf)

    # Get info of each id.
    metadata = []
    for ids in all_ids:
        record = info(ids)
        metadata.append({key: record[key] for key in ["id", "timeSeriesName"]})

    # Keep ids where name includes partialname
    hits = [record for record in metadata if name in record["timeSeriesName"]]

    # Raise error if 0 or > 1 found.
    if len(hits) == 0:
        raise ValueError(
            "No timeseries found. Check parameters; use .find_pf"
            + "to check if pf abbreviation is correct."
        )
    elif len(hits) > 1:
        raise ValueError(
            'Found more than 1 timeseries, i.e. with ids {",".join(hits)}.'
        )

    return hits[0]["id"]


def records(
    id: int,
    ts_left: Union[pd.Timestamp, dt.datetime],
    ts_right: Union[pd.Timestamp, dt.datetime],
) -> Iterable[Dict]:
    """Return values from timeseries with id `id` in given delivery time interval.

    See also
    --------
    .series
    """
    response = _getreq(
        f"/rest/energy/belvis/{_tenant()}/timeSeries/{id}/values",
        f"timeRange={ts_left.isoformat()}--{ts_right.isoformat()}",
        "timeRangeType=inclusive-exclusive",
    )
    return _object(response)


def series(
    id: int,
    ts_left: Union[pd.Timestamp, dt.datetime],
    ts_right: Union[pd.Timestamp, dt.datetime],
) -> pd.Series:
    """Return series from timeseries with id `id` in given delivery time interval.

    Parameters
    ----------
    id : int
        Timeseries id.
    ts_left : Union[pd.Timestamp, dt.datetime]
    ts_right : Union[pd.Timestamp, dt.datetime]

    Returns
    -------
    pd.Series
        with resulting information.
    """
    vals = records(id, ts_left, ts_right)
    df = pd.DataFrame.from_records(vals)
    mask = df["pf"] == "missing"
    s = pd.Series(
        df["v"].to_list(), pd.DatetimeIndex(df["ts"]).tz_convert("Europe/Berlin")
    )
    return s
