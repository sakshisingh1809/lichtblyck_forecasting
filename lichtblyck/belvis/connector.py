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
import pandas as pd
import numpy as np
import datetime as dt
import subprocess
import pathlib
import requests
import json
from OpenSSL import crypto
from socket import gethostname
import jwt
import time


__usr = "Ruud.Wijtvliet"
__pwd = "Ammm1mmm2mmm3mmm"
__tenant = "PFMSTROM"
__server = "http://lbbelvis01:8040"
_session = None


def _getreq(path, *queryparts) -> requests.request:
    if _session is None:
        _startsession_and_authenticate()
    string = f"{__server}{path}"
    if queryparts:
        queryparts = [parse.quote(qp, safe=":=") for qp in queryparts]
        string += "?" + "&".join(queryparts)
    print(string)
    try:
        return _session.get(string)
    except ConnectionError as e:
        raise RuntimeError("Check if VPN connection to Lichtblick exists.") from e


def _startsession_and_authenticate() -> None:
    """Start session and get token."""
    global _session
    _session = requests.Session()
    _getreq("/rest/session", f"usr={__usr}", f"pwd={__pwd}", f"tenant={__tenant}")


def _generate_token() -> requests:
    """This method is used by current REST clients or Libraries supported.
    A trustworthy body generates a key pair from which it can be used for
    authorized persons Clients generate strings, so-called Bearer tokens.

    Returns:
        requests: requests.response
    """
    # Open private key to sign token with
    with open("privatekey.txt", "r") as f:
        private_key = f.read()

    # Create token that is valid for a given amount of time.
    claims = {
        "preferred_username": __usr,
        "clientId": __tenant,
        "exp": int(time.time()) + 10 * 60,
    }

    # "RSA 512 bit" in the PKCS standard for your client
    token = jwt.encode(claims, private_key, algorithm="RS512")
    headers = {"Authorization": "Bearer " + token}

    response = requests.get(__server, headers=headers)

    return response
    # return _object(response)


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
    response = _getreq(f"/rest/energy/belvis/{__tenant}/timeSeries/{id}")
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
        f"/rest/energy/belvis/{__tenant}/timeseries", f"instancetoken={pf}"
    )
    restlist = _object(response)
    ids = [int(entry.split("/")[-1]) for entry in restlist]
    if not ids:
        raise ValueError("No timeseries found. Check the portfolio abbreviation.")
    return ids


def fetch_pfinfo():
    """This is an expensive function, which is called only once from main().
    This function searches the whole belvis database and creates a list of all ids.
    The list is passed to another function store_all_pfinfo(), which stores the
    pf names and their abbreviations in a json file.
    """
    # Get all pfs.
    response = _getreq(f"/rest/energy/belvis/{__tenant}/timeseries")
    restlist = _object(response)
    ids = [int(entry.split("/")[-1]) for entry in restlist]

    # Create json file with all pf names ()
    store_all_pfinfo(ids)


def store_all_pfinfo(ids: List):
    """This function takes the list of all ids in the belvis database and stores the
    pf names & their abbreviations in a json object in belvis/memoized location.

    Args:
        ids_list: List
            list of all portfolio ids
    """
    metadata = []
    for id in ids:
        record = [None] * 2
        record[0] = info(id)["instanceToken"]
        record[1] = info(id)["instanceName"]

        metadata = sorted(metadata)
        if record not in metadata:
            metadata.append(record)

    with open("metadata.txt", "w") as outfile:
        json.dump(metadata, outfile)


def find_pf(partial_or_exact_pf_name: str) -> str:
    """Find the exact portfolio abbrevaition given any 'pf' names (full or partial).

    Parameters:
    ----------
        partial_or_exact_pf_name : str
            Exact or partial name of portfolio

    Raises:
    -------
        ValueError : If no matching timeseries is found or 
            If more than 1 matching timeseries are found

    Returns:
    -------
        instanceToken : str
            Portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM')
    """
    # Get info of each id.
    with open("metadata.txt") as json_file:
        data = json.load(json_file)

    # Convert data(list of list) into list of dictionaries
    metadata = {}
    for d in data:
        metadata[d[0]] = d[1]

    # Keep ids where name includes partialname
    hits = [
        record for record in metadata if partial_or_exact_pf_name in metadata[record]
    ]

    # Raise error if 0 or > 1 found.
    if len(hits) == 0:
        raise ValueError("No timeseries found. Check parameters; use .find_id")
    elif len(hits) > 1:
        raise ValueError("Found more than 1 timeseries, i.e. with ids: {", hits, "}")

    return hits[0]


def find_id(pf: str, name: str) -> int:
    """In offtake portfolio `pf`, find id of timeseries with name `name`.

    Parameters
    ----------
    pf : str
        Portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM')
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


def records(id: int, ts_left=None, ts_right=None) -> Iterable[Dict]:
    """Return values from timeseries with id `id` in given delivery time interval.

    See also
    --------
    .series
    """

    response = _getreq(
        f"/rest/energy/belvis/{__tenant}/timeSeries/{id}/values",
        f"timeRange={ts_left.isoformat()}--{ts_right.isoformat()}",
        "timeRangeType=inclusive-exclusive",
    )
    return _object(response)


def series(id: int, ts_left=None, ts_right=None) -> pd.Series:
    """Return series from timeseries with id `id` in given delivery time interval.

    Parameters
    ----------
    id : int
        Timeseries id.
    ts_left : timestamp, optional
    ts_right : timestamp, optional

    Returns
    -------
    pd.Series
        with resulting information.

    See also
    --------
    .ts_leftright
    """
    vals = records(id, ts_left, ts_right)
    df = pd.DataFrame.from_records(vals)
    mask = df["pf"] == "missing"
    s = pd.Series(
        df["v"].to_list(), pd.DatetimeIndex(df["ts"]).tz_convert("Europe/Berlin")
    )
    return s


if __name__ == "__main__":
    # fetch_pfinfo()
    # pf = find_pf("udwig")  # returs 'LUD' or ValueError if 0 of > 1
    id = find_id(
        "LUD", "#LB FRM Procurement/Forward - MW - excl subpf"
    )  # returns id or ValueError if 0 or > 1

    i = info(id)
    r = records(id)
    s = series(id, "2020-02")


# %%

# TODO:
# . use token instead of password
# . function to get QHPFC


# . function to find PF-abbrev (e.g. 'LUD') from part of pf-name (e.g. 'udwig')

# I don't know pf abbreviation:
# . find_pf: partial_or_exact_pf_name -> pf

# I know exact name or partial name of the ts:
# . find_id: pf, name -> id
# . series: id -> data

# I know the id of the ts:
# . series: id -> data
