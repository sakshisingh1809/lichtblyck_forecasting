"""
Retrieve data from Belvis using Rest-API.
"""

# from ..core.singlepf_multipf import SinglePf, MultiPf
# from .books import find_node
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

TSNAMES = {
    "LUD": {
        "qo": "#LB Saldo aller Prognosegeschäfte +UB",
        "qs": "#LB Saldo aller Termingeschäfte +UB",
    }
    #etc.
}


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


def find_ids(pf: str) -> List[int]:
    """Find ids of all timeseries in offtake portfolio `pf`.

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

    Returns
    -------
    int
        id of found timeseries.
    """
    response = _getreq(
        f"/rest/energy/belvis/{__tenant}/timeseries",
        f"instancetoken={pf}",
        f"timeseriesname={name}",
    )
    restlist = _object(response)
    ids = [int(entry.split("/")[-1]) for entry in restlist]
    if not ids:
        raise ValueError(
            "No timeseries found. Check parameters; use .find_ids "
            + "to check if pf abbreviation is correct."
        )
    if len(ids) > 1:
        raise ValueError('Found more than 1 timeseries, i.e. with ids {",".join(ids)}.')
    return ids[-1]


def info(id: int) -> Dict:
    """Get dictionary with information about timeseries with id `id`."""
    response = _getreq(f"/rest/energy/belvis/{__tenant}/timeSeries/{id}")
    return _object(response)


def ts_leftright(ts_left=None, ts_right=None) -> Tuple:
    """Makes 2 timestamps coherent to one another.

    Parameters
    ----------
    ts_left : timestamp, optional
    ts_right : timestamp, optional

    If no value for ts_left is given, the beginning of the year of ts_right is given.
    If no value for ts_right is given, the end of the year of ts_left is given.
    If no values is given for either, the entire current year is given. If no time
    zone is provided for either timestamp, the Europe/Berlin timezone is assumed.

    Returns
    -------
    (localized timestamp, localized timestamp)
    """

    def start(middle):
        return middle + pd.offsets.YearBegin(0) + pd.offsets.YearBegin(-1)

    def end(middle):
        return middle + pd.offsets.YearBegin(1)

    ts_left, ts_right = pd.Timestamp(ts_left), pd.Timestamp(ts_right)

    if ts_right is pd.NaT:
        if ts_left is pd.NaT:
            return ts_leftright(start(dt.date.today()))
        if ts_left.tz is None:
            return ts_leftright(ts_left.tz_localize("Europe/Berlin"))
        return ts_leftright(ts_left, end(ts_left))

    # if we land here, we at least know ts_right.
    if ts_left is pd.NaT:
        return ts_leftright(start(ts_right), ts_right)

    # if we land here, we know ts_left and ts_right.
    if ts_right.tz is None:
        if ts_left.tz is None:
            return ts_leftright(ts_left.tz_localize("Europe/Berlin"), ts_right)
        return ts_leftright(ts_left, ts_right.tz_localize(ts_left.tz))

    # if we land here, we know ts_left and localized ts_right.
    if ts_left.tz is None:
        return ts_leftright(ts_left.tz_localize(ts_right.tz), ts_right)

    # if we land here, we know localized ts_left and localized ts_right.
    if ts_left.tz != ts_right.tz:
        raise ValueError("Timestamps have non-matching timezones.")

    return ts_left, ts_right


def records(id: int, ts_left=None, ts_right=None) -> Iterable[Dict]:
    """Return values from timeseries with id `id` in given delivery time interval.

    See also
    --------
    .series
    """
    ts_left, ts_right = ts_leftright(ts_left, ts_right)
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


def pf_own(pf: str, ts_left=None, ts_right=None):
    """Return MultiPf for a the 'own' timeseries of a Belvis portfolio - i.e., excluding 
    children.

    Returns
    -------
    MultiPf
        for Belvis portfolio but excluding children.

    See also
    --------
    .pf_complete
    """
    pass
    # # Find relevant timeseries.
    # tseries = []
    # for id in find_ids(pf):
    #     name = info(id)["timeSeriesName"]
    #     if "#LB FRM" in name:
    #         tseries.append({"name": name, "id": id, "s": series(id, ts_left, ts_right)})

    # def find_data(contains, excl_subpf=True):
    #     subpf = ("excl" if excl_subpf else "incl") + " subpf"
    #     relevant = [s for s in tseries if contains in s["name"] and subpf in s["name"]]
    #     data = {}
    #     for s in relevant:
    #         if "- MW -" in s["name"]:
    #             if "w" in data:
    #                 raise ValueError(f"Found > 1 MW timeseries with filter {contains}.")
    #             data["w"] = s["s"]
    #         if "- EUR (contract" in s["name"]:
    #             if "r" in data:
    #                 raise ValueError(
    #                     f"Found > 1 EUR timeseries with filter {contains}."
    #                 )
    #             data["r"] = s["s"]
    #     return data

    # offtake = SinglePf(find_data("Offtake"), "Offtake")
    # fwd = SinglePf(find_data("Forward"), "Forward")
    # spotda = SinglePf(find_data("Spot/DA"), "Da")
    # spotid = SinglePf(find_data("Spot/ID"), "Id")

    # procurement = MultiPf([fwd, MultiPf([spotda, spotid], "Spot")], "Procurement")

    # return MultiPf([procurement, offtake], "Own")


def pf_complete(pf: str, ts_left=None, ts_right=None):
    """Return MultiPf for a complete Belvis portfolio, including children.

    Parameters
    ----------
    pf : str
        Portfolio abbreviation, i.e., 'LUD'
    ts_left : timestamp, optional
    ts_right : timestamp, optional
        Delivery time interval.

    Returns
    -------
    MultiPf
        for complete Belvis portfolio, including children.

    See also
    --------
    .ts_leftright
    """
    pass
    # node = find_node(pf)

    # def pf_from_node(n):
    #     own = pf_own(n.name, ts_left, ts_right)
    #     children = [pf_from_node(childnode) for childnode in n.children]
    #     return MultiPf([own, *children], pf)

    # return pf_from_node(node)


if __name__ == "__main__":
    id = find_id("LUD", "#LB FRM Procurement/Forward - MW - excl subpf")
    i = info(id)
    r = records(id)
    s = series(id, "2020-02")


# %%

# Belvis API document: 
# 1. general info
# 2.2 finding timeseries
# 2.3 metadata timeseries
# 2.5 reading timeseries values
# 7 how to connect