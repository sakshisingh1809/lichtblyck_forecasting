"""Retrieve raw data from Belvis using Rest-API."""

# Developer notes, Belvis API document:
# 1. general info
# 2.2 finding timeseries
# 2.3 metadata timeseries
# 2.5 reading timeseries values
# 7 how to connect

from .connect import BelvisConnection
from typing import Tuple, Dict, Union
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime as dt
import json


_COMMOTEN = {
    "power": "TESTPFMSTROM",
    "gas": "TESTPFMGAS",
}  # commodity: tenant dictionary


class _Cache:
    """Dict-like class to access cached Belvis information (timeseries metadata)."""

    _FOLDER = Path(__file__).parent / "cache"

    def __init__(self, filename: str):
        self._filepath = self._FOLDER / filename
        self.reload()

    __getitem__ = lambda self, *args, **kwargs: self._dic.__getitem__(*args, **kwargs)
    __contains__ = lambda self, *args, **kwargs: self._dic.__contains__(*args, **kwargs)
    __getattr__ = lambda self, *args, **kwargs: getattr(self._dic, *args, **kwargs)

    def update(self, dic) -> None:
        lis = [(key, val) for key, val in dic.items()]  # avoid int key in json
        json.dump(lis, open(self._filepath, "w"))
        self.reload()

    def reload(self) -> None:
        lis = json.load(open(self._filepath, "r"))
        self._dic = {key: val for key, val in lis}


class _Source:
    """Class to get data from Belvis data source, including authentication and caching."""

    _CACHED_TS_KEYS = ("instanceToken", "measurementUnit", "timeSeriesName")

    def __init__(self, connection: BelvisConnection):
        self._connection = connection
        self._tsscache = _Cache(f"{connection.tenant}_timeseries.json")
        self._pfscache = _Cache(f"{connection.tenant}_portfolios.json")

    connection = property(lambda self: self._connection)
    tsscache = property(lambda self: self._tsscache)
    pfscache = property(lambda self: self._pfscache)

    def update_cache_files(self):
        """Update cache files. Expensive function that can take >1h."""

        def couldbepf(pf: str) -> bool:
            """Return True if `pf` might be a portfolio."""
            for char in ["-", ".", " ", ":", *list("0123456789")]:
                pf = pf.replace(char, "")
            return bool(pf)

        # Get all timeseries ids.
        paths = self._connection.query_timeseries("")
        tsids = [int(path.split("/")[-1]) for path in paths]

        # Create dictionaries with metadata.
        tss, pfs = {}, {}
        for tsid in tqdm(tsids):
            inf = self._connection.query_timeseries(f"/{tsid}")
            # Check if want to store.
            pfid = inf["instanceToken"]
            if not couldbepf(pfid):
                continue
            # Relevant timeseries information. key = timeseries id.
            tss[tsid] = {key: inf[key] for key in self._CACHED_TS_KEYS}
            # Relevant portfolio information. key = portfolio abbreviation.
            if pfid not in pfs:
                pfs[pfid] = {"name": inf["instanceName"], "tsids": []}
            pfs[pfid]["tsids"].append(tsid)

        # Update (save and reload).
        self._tsscache.update(tss)
        self._pfscache.update(pfs)


_sources = {}


def _source(commodity):
    if commodity not in _COMMOTEN:
        raise ValueError(f"`commodity` must be one of {_COMMOTEN.keys()}.")
    if commodity not in _sources:
        raise PermissionError(
            "First authenticate using `auth_with_password()`, `auth_with_passwordfile()` or `auth_with_token()`"
        )
    return _sources[commodity]


def auth_with_password(usr: str, pwd: str):
    """Authentication with username ``usr`` and password ``pwd``."""
    for commodity, tenant in _COMMOTEN.items():
        connection = BelvisConnection.from_usrpwd(tenant, usr, pwd)
        _sources[commodity] = _Source(connection)


def auth_with_passwordfile(path: Path):
    """Authentication with username and password stored on first 2 lines of textfile.
    (NB: Whitespace is stripped.)"""
    with open(path, "r") as f:
        usr, pwd, *rest = f.readlines()
    auth_with_password(usr.strip(), pwd.strip())


def auth_with_token(usr: str):
    """Authentication with private-public key pair."""
    for commodity, tenant in _COMMOTEN.items():
        connection = BelvisConnection.from_token(tenant, usr)
        _sources[commodity] = _Source(connection)


def update_cache_files(commodity: str = None):
    """Update the cache files for all timeseries of a commodity (or all commodities, if
    none specified). Takes a long time, only run when portfolio structure changed or
    when relevant timeseries added/changed."""
    if not commodity:
        for commodity in _sources:
            update_cache_files(commodity)
    print(f"Updating files of commodity '{commodity}'.")
    _source(commodity).update_cache_files()
    print("Done.")


def connection_alive(commodity: str):
    """Return True if connection to Belvis tenant for `commodity` is (still) alive."""
    connection = _source(commodity).connection
    url = connection.url("/rest/belvis/internal/heartbeat/ping")
    return connection.request(url).status_code == 200


def info(commodity: str, id: int) -> Dict:
    """Get information about timeseries.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    id : int
        Timeseries id.

    Returns
    -------
    dict
        Metadata about the timeseries.
    """
    return _source(commodity).connection.query_timeseries(f"/{id}")


def find_pfids(commodity: str, name: str, strict: bool = False) -> Dict[str, str]:
    """Find portfolios by name or id.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    name : str
        Name of portfolio.
    strict : bool, optional (default: False)
        If True, only returns portfolios if the name or pfid exactly matches. Otherwise,
        also return if name partially matches. Always case insensitive.

    Returns
    -------
    Dict[str, str]
        Dictionary of matching portfolios. Key = portfolio abbreviation (e.g. 'LUD' or
        'LUD_SIM', value = portfolio name (e.g. 'Ludwig Sichere Menge').

    Notes
    -----
    Always uses cached information. If portfolio structure in Belvis is changed, run
    the `.update_cache_files()` function to manually update the cache.
    """
    name = name.lower()

    def matchingname(pfname: str) -> bool:
        return name == pfname.lower() if strict else name in pfname.lower()

    # Keep the portfolio abbreviations with matching names.
    hits = {
        pfid: inf["name"]
        for pfid, inf in _source(commodity).pfscache.items()
        if matchingname(inf["name"]) or matchingname(pfid)
    }

    # Raise error if 0 found.
    if len(hits) == 0:
        raise ValueError(
            f"No portfolio with name '{name}' found in commodity {commodity}."
        )

    return hits


def find_tsids(
    commodity: str,
    pfid: str = None,
    name: str = "",
    strict: bool = False,
    use_cache: bool = True,
) -> Dict[int, Tuple[str]]:
    """Gets ids of all timeseries in a portfolio.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    pfid : str, optional (default: search in all portfolios. Only possible if use_cache)
        Portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM')
    name : str, optional (default: return all timeseries).
        Name of timeseries (e.g. '#LB FRM Procurement/Forward - MW - excl subpf').
    strict : bool, optional (default: False)
        If True, only returns timeseries if the name exactly matches. Otherwise, also
        return if name partially matches. Always case insensitive.
    use_cache : bool, optional (default: True)
        If True, use the cached information to find the ids. Otherwise, use the API.

    Returns
    -------
    Dict[int, Tuple[str]]
        Dictionary with found timeseries. Keys are the timeseries ids, the values are
        (portfolio abbreviation, timeseries name)-tuples.
    """
    name = name.lower()

    def matchingname(tsname: str) -> bool:
        return name == tsname.lower() if strict else name in tsname.lower()

    if pfid is None and not use_cache:
        raise ValueError(
            "Must specify `pfid` when using 'live' (i.e., not cached) information."
        )

    hits = {}
    if use_cache:

        # Filter on pf.
        pfs = _source(commodity).pfscache

        if pfid is not None and pfid not in pfs:
            raise ValueError(
                f"No Portfolio with abbreviation '{pfid}' found in commodity '{commodity}'."
            )
        elif pfid is None:
            # Get all timeseries ids of all portfolios.
            tsids = [tsid for pf in pfs.values() for tsid in pf["tsids"]]
        else:
            # Get all timeseries ids in the portfolio.
            tsids = pfs[pfid]["tsids"]

        # Filter on timeseries name.
        for tsid, inf in _source(commodity).tsscache.items():
            if tsid in tsids and matchingname(inf["timeSeriesName"]):
                hits[tsid] = (inf["instanceToken"], inf["timeSeriesName"])
    else:

        # Filter on pf.
        # Get all timeseries ids in the portfolio.
        paths = _source(commodity).connection.query_timeseries(
            "", f"instancetoken={pfid}"
        )
        tsids = [int(path.split("/")[-1]) for path in paths]

        # Filter on timeseries name.
        for tsid in tsids:
            inf = info(commodity, tsid)
            if matchingname(inf["timeSeriesName"]):
                hits[int(inf["id"])] = (inf["instanceToken"], inf["timeSeriesName"])

    return hits


def find_tsid(
    commodity: str, pfid: str, name: str, strict: bool = False, use_cache: bool = True
) -> int:
    """Find id of unique timeseries.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    pfid : str
        Portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM').
    name : str
        Name of timeseries (e.g. '#LB FRM Procurement/Forward - MW - excl subpf').
    strict : bool, optional (default: False)
        If True, only returns timeseries if the name exactly matches. Otherwise, also
        return if name partially matches. Always case insensitive.
    use_cache : bool, optional (default: True)
        If True, use the cached information to find the id. Otherwise, use the API.

    Returns
    -------
    int
        id of found timeseries.
    """
    # Find all hits
    hits = find_tsids(commodity, pfid, name, strict, use_cache)

    # custom quick-fix
    if set(hits.keys()) == set([44578448, 44580972]):
        hits = {44578448: hits[44578448]}

    # Raise error if 0 or > 1 found.
    if len(hits) == 0:
        raise ValueError(
            f"No timeseries with name '{name}' found in commodity '{commodity}' and portfolio '{pfid}'. Use `.find_pfids` to check if `pfid` is correct."
        )
    elif len(hits) > 1:
        raise ValueError(f"Found more than 1 timeseries: {hits}.")

    return next(iter(hits.keys()))  # return only the tsid of the (only) hit.


def series(
    commodity: str,
    tsid: int,
    ts_left: Union[pd.Timestamp, dt.datetime],
    ts_right: Union[pd.Timestamp, dt.datetime],
    *,
    leftrange: str = "exclusive",
    rightrange: str = "inclusive",
    missing2zero: bool = True,
    blocking: bool = True,
) -> pd.Series:
    """Return series from timeseries with id `id` in given delivery time interval.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    id : int
        Timeseries id.
    ts_left : Union[pd.Timestamp, dt.datetime]
    ts_right : Union[pd.Timestamp, dt.datetime]
    leftrange : str, optional (default: 'exclusive')
        'inclusive' ('exclusive') to get values with timestamp that is >= (>) ts_left.
        Default: 'exclusive' because timestamps in Belvis are *usually* right-bound.
    rightrange : str, optional (default: 'inclusive')
        'inclusive' ('exclusive') to get values with timestamp that is <= (<) ts_right.
        Default: 'inclusive' because timestamps in Belvis are *usually* right-bound.
    missing2zero : bool, optional (default: True)
        What to do with values that are flagged as 'missing'. True to replace with 0,
        False to replace with nan.
    blocking : bool, optional (default: True)
        If True, recalculates data that is not up-to-date before returning; might take
        long time or result in internal-server-error. If False, return most up-to-date
        data that is available without recalculating.

    Returns
    -------
    pd.Series
        with resulting information.

    Notes
    -----
    Returns series with data as found in Belvis; no correction (e.g. for right-bounded
    timestamps) done.
    """
    # Default: left=exclusive, right=inclusive, because timestamps in belvis are
    # *usually* right-bound, for some reason.
    records = _source(commodity).connection.query_timeseries(
        f"/{tsid}/values",
        f"timeRange={ts_left.isoformat()}--{ts_right.isoformat()}",
        f"timeRangeType={leftrange}-{rightrange}",
        f"blocking={str(blocking).lower()}",
    )
    df = pd.DataFrame.from_records(records)
    mask = df["pf"] == "missing"
    df.loc[mask, "v"] = 0 if missing2zero else np.na
    s = pd.Series(
        df["v"].to_list(), pd.DatetimeIndex(df["ts"]).tz_convert("Europe/Berlin")
    )
    return s
