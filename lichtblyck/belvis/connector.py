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
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime as dt
import jwt
import time
import datetime
import json
import requests

_server = "http://lbbelvis01:8040"

_COMMOTEN = {"power": "PFMSTROM", "gas": "PFMGAS"}  # commodity: tenant dictionary


class _Connection:
    """Class to connect to a Belvis tenant, including authentication and querying."""

    _AUTHFOLDER = Path(__file__).parent / "auth"

    def __init__(self, tenant: str):
        self._details = {}
        self._tenant = tenant
        self._lastquery = None

    tenant = property(lambda self: self._tenant)

    def auth_with_password(self, usr: str, pwd: str) -> None:
        """Authentication with username `usr` and password `pwd`; open a session."""
        self._details = {"usr": usr, "pwd": pwd, "session": requests.Session()}
        self.query_general(
            "/rest/session", f"usr={usr}", f"pwd={pwd}", f"tenant={self._tenant}"
        )
        self._lastquery = None  # reset to keep track of this auth method's validity
        # Check if successful.
        if not self.auth_successful():
            raise ConnectionError("No connection exists. Username/password incorrect?")

    def encode_with_token(self, user_id, private_key):
        # Create token that is valid for a given amount of time.
        try:
            claims = {
                "name": self._tenant,
                "sub": user_id,
                "exp": datetime.datetime.utcnow()
                + datetime.timedelta(days=0, seconds=120),
                "iat": datetime.datetime.utcnow(),
            }

            # "RSA 512 bit" in the PKCS standard for your client.
            return jwt.encode(payload=claims, key=private_key, algorithm="RS512")

        except Exception as e:
            return e

    def decode_with_token(self, token, private_key):
        try:
            payload = jwt.decode(
                token, key=private_key, options={"verify_signature": True}
            )
            return payload["sub"]
        except jwt.ExpiredSignatureError:
            return "Signature expired. Please log in again."
        except jwt.InvalidTokenError:
            return "Invalid token. Please log in again."

    def auth_with_token(self, usrID) -> None:
        """Authentication with public-private key pair."""

        # Open private key to sign token with.
        with open(self._AUTHFOLDER / "privatekey.txt", "r") as f:
            private_key = f.read()

        token = self.encode_with_token(usrID)
        self.assertTrue(isinstance(token, bytes))

        self._details = {"headers": {"Authorization": f"Bearer {token}"}}
        self._lastquery = None  # reset to keep track of this auth method's validity

        # Check if successful.
        if not self.auth_successful():  # check for connection failures
            raise ConnectionError("No connection exists. Token incorrect?")

        self.assertTrue(
            self.decode_with_token(token) == 1
        )  # decode the token and assert

    def auth_successful(self) -> bool:
        return connection_alive(self._tenant)

    def redo_auth(self) -> None:
        """Redo authentication. Necessary after timeout of log-in."""
        if "session" in self._details:
            self.auth_with_password(self._details["usr"], self._details["pwd"])
        elif "headers" in self._details:
            self.auth_with_token()
        else:
            raise PermissionError(
                "First authenicate with `auth_with_password` or `auth_with_token`."
            )

    def request(self, path: str, *queryparts: str) -> requests.request:
        string = f"{_server}{path}"
        if queryparts:
            queryparts = [parse.quote(qp, safe=":=") for qp in queryparts]
            string += "?" + "&".join(queryparts)
        try:
            if "session" in self._details:
                req = self._details["session"].get(string)
            elif "headers" in self._details:
                req = requests.get(string, headers=self._details["headers"])
            else:
                raise PermissionError(
                    "First authenicate with `auth_with_password` or `auth_with_token`."
                )
        except ConnectionError as e:
            raise ConnectionError(
                "Check if VPN connection to Lichtblick exists."
            ) from e
        self._lastquery = dt.datetime.now()
        return req

    def query_general(self, path: str, *queryparts: str) -> Union[Dict, List]:
        """Query connection for general information."""
        response = self.request(path, *queryparts)
        if response.status_code == 200:
            return json.loads(response.text)
        elif self._lastquery is not None:  # authentication might be expired.
            self.redo_auth()
            return self.query_general(path, *queryparts)  # retry.
        else:
            raise RuntimeError(response)

    def query_timeseries(
        self, remainingpath: str, *queryparts: str
    ) -> Union[Dict, List]:
        """Query connection for timeseries information."""
        path = f"/rest/energy/belvis/{self.tenant}/timeSeries{remainingpath}"
        return self.query_general(path, *queryparts)


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

    def __init__(self, tenant: str):
        self._connection = _Connection(tenant)
        self._tsscache = _Cache(f"{tenant}_timeseries.json")
        self._pfscache = _Cache(f"{tenant}_portfolios.json")

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


_sources = {commodity: _Source(tenant) for commodity, tenant in _COMMOTEN.items()}


def _source(commodity):
    try:
        return _sources[commodity]
    except KeyError:
        raise ValueError(f"`commodity` must be one of {_COMMOTEN.keys()}.")


def auth_with_password(usr: str, pwd: str):
    """Authentication with username ``usr`` and password ``pwd``."""
    for source in _sources.values():
        source.connection.auth_with_password(usr, pwd)


def auth_with_passwordfile(path: Path):
    """Authentication with username and password stored on first 2 lines of textfile.
    (NB: Whitespace is stripped.)"""
    with open(path, "r") as f:
        usr, pwd, *rest = f.readlines()
    auth_with_password(usr.strip(), pwd.strip())


def auth_with_token(usrID: int):
    """Authentication with private-public key pair."""
    for source in _sources.values():
        source.connection.auth_with_token(usrID)


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
    return (
        _source(commodity)
        .connection.rawrequest("/rest/belvis/internal/heartbeat/ping")
        .status_code
        == 200
    )


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
    """Find portfolios by name.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    name : str
        Name of portfolio.
    strict : bool, optional (default: False)
        If True, only returns portfolios if the name exactly matches. Otherwise, also
        return if name partially matches. Always case insensitive.

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
        if matchingname(inf["name"])
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
            inf = info(tsid)
            if matchingname(inf["timeseriesName"]):
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


def records(
    commodity: str,
    tsid: int,
    ts_left: Union[pd.Timestamp, dt.datetime],
    ts_right: Union[pd.Timestamp, dt.datetime],
) -> Iterable[Dict]:
    """Return values from timeseries with id `id` in given delivery time interval.

    See also
    --------
    .series
    """
    return _source(commodity).connection.query_timeseries(
        f"/{tsid}/values",
        f"timeRange={ts_left.isoformat()}--{ts_right.isoformat()}",
        "timeRangeType=exclusive-inclusive",
    )  # exclusive-inclusive because timestamps in belvis are right-bound for some reason


def series(
    commodity: str,
    tsid: int,
    ts_left: Union[pd.Timestamp, dt.datetime],
    ts_right: Union[pd.Timestamp, dt.datetime],
    missing2zero: bool = True,
) -> pd.Series:
    """Return series from timeseries with id `id` in given delivery time interval.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    id : int
        Timeseries id.
    ts_left : Union[pd.Timestamp, dt.datetime]
    ts_right : Union[pd.Timestamp, dt.datetime]
    missing2zero : bool, optional (default: True)
        What to do with values that are flagged as 'missing'. True to replace with 0,
        False to replace with nan.

    Returns
    -------
    pd.Series
        with resulting information.

    Notes
    -----
    Returns series with data as found in Belvis; no correction (e.g. for right-bounded
    timestamps) done.
    """
    vals = records(commodity, tsid, ts_left, ts_right)
    df = pd.DataFrame.from_records(vals)
    mask = df["pf"] == "missing"
    df.loc[mask, "v"] = 0 if missing2zero else np.na
    s = pd.Series(
        df["v"].to_list(), pd.DatetimeIndex(df["ts"]).tz_convert("Europe/Berlin")
    )
    return s
