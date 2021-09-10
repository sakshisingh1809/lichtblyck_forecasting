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


def _object(response) -> Union[Dict, List]:
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        raise RuntimeError(response)


def connection_alive():
    """Return True if connection is still up."""
    return _getreq("/rest/belvis/internal/heartbeat/ping").status_code == 200


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

    # Get info of each id.

    # Keep ids where name includes partialname

    # response = _getreq(
    #     f"/rest/energy/belvis/{__tenant}/timeseries",
    #     f"instancetoken={pf}",
    #     f"timeseriesname={name}",
    # )
    # restlist = _object(response)
    # ids = [int(entry.split("/")[-1]) for entry in restlist]
    
    # Raise error if 0 or >1 found.
    if not ids:
        raise ValueError(
            "No timeseries found. Check parameters; use .find_ids "
            + "to check if pf abbreviation is correct."
        )
    if len(ids) > 1:
        # TODO also return the names of the timeseries in error message.
        raise ValueError('Found more than 1 timeseries, i.e. with ids {",".join(ids)}.')
    # Return id.
    return ids[-1]


def info(id: int) -> Dict:
    """Get dictionary with information about timeseries with id `id`."""
    response = _getreq(f"/rest/energy/belvis/{__tenant}/timeSeries/{id}")
    return _object(response)

def records(id: int, ts_left, ts_right) -> Iterable[Dict]:
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


def series(id: int, ts_left, ts_right) -> pd.Series:
    """Return series from timeseries with id `id` in given delivery time interval.

    Parameters
    ----------
    id : int
        Timeseries id.
    ts_left : timestamp
    ts_right : timestamp

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
    id = find_id("LUD", "#LB FRM Procurement/Forward - MW - excl subpf")
    i = info(id)
    r = records(id)
    s = series(id, "2020-02")
    print(s)


# %%

# TODO:
# . use token instead of password
# . function to get QHPFC
# . function to find PF-abbrev (e.g. 'LUD') from part of pf-name (e.g. 'udwig')



# I don't know pf abbreviation:
# . find_pf: partial_or_exact_pf_name -> j

# I know exact name or partial name of the ts:
# . find_id: pf, name -> id
# . series: id -> data

# I know the id of the ts:
# . series: id -> data
