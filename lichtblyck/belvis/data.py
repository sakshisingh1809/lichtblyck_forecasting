"""Module that uses connection to Belvis to generate PfLine and PfState objects."""

from ..tools import stamps
from ..core2 import factory
from ..core2.pfline import PfLine
from ..core2.pfstate import PfState
from . import connector
from typing import Union
import functools
import datetime as dt
import pandas as pd

# Names of the portfolios and of the timeseries in them.

DEFAULTTSNAMES = {
    "wo": "#LB Saldo aller Prognosegeschäfte +UB",
    "ws": "#LB Saldo aller Termingeschäfte +UB",
    "rs": "#LB FRM Procurement/Forward - EUR (contract) - incl subpf",
}
TSNAMES = {  # All allowed portfolios must have a key here.
    "LUD": {},  # any
    "PKG": {"wo": "#FRM offtake - MW"}
    # etc.
}


def _get_tsname(pf, ts):
    """Convenience function to find name of timeseries in Belvis portfolio."""
    try:
        tsnames = TSNAMES[pf]
    except KeyError:
        raise ValueError(f"`pf` must be one of {', '.join(TSNAMES.keys())}.")
    try:
        defaulttsname = DEFAULTTSNAMES[ts]
    except KeyError:
        raise ValueError(f"`ts` must be one of {', '.join(DEFAULTTSNAMES.keys())}.")
    return tsnames.get(ts, defaulttsname)


@functools.lru_cache()
def _ts_leftright(ts_left, ts_right):
    return stamps.ts_leftright(ts_left, ts_right)


def offtakevolume(
    pf: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> PfLine:
    """Get offtake volume for a certain portfolio from Belvis.

    Parameters
    ----------
    pf : str
        Portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM').
    ts_left : Union[str, dt.datetime, pd.Timestamp], optional
        Start of delivery period.
    ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        End of delivery period.

    Returns
    -------
    PfLine
    """
    # Fix timestamps (if necessary).
    ts_left, ts_right = _ts_leftright(ts_left, ts_right)
    # Get timeseries.
    tsname = _get_tsname(pf, "wo")
    tsid = connector.find_id(pf, tsname)
    s = connector.series(tsid, ts_left, ts_right)
    return PfLine({"w": s})


def sourced(
    pf: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> PfLine:
    """Get sourced volume and price for a certain portfolio from Belvis.

    Parameters
    ----------
    pf : str
        Portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM').
    ts_left : Union[str, dt.datetime, pd.Timestamp], optional
        Start of delivery period.
    ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        End of delivery period.

    Returns
    -------
    PfLine
    """
    # Fix timestamps (if necessary).
    ts_left, ts_right = _ts_leftright(ts_left, ts_right)
    # Get timeseries.
    data = {}
    for n, ts in {"w": "ws", "r": "rs"}.items():
        tsname = _get_tsname(pf, ts)
        tsid = connector.find_id(pf, tsname)
        data[n] = connector.series(tsid, ts_left, ts_right)
    return PfLine(data)


@functools.lru_cache()  # memoization
def unsourcedprice(
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> PfLine:
    """Get market prices from Belvis.

    Parameters
    ----------
    ts_left : Union[str, dt.datetime, pd.Timestamp], optional
        Start of delivery period.
    ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        End of delivery period.

    Returns
    -------
    PfLine
    """
    # Fix timestamps (if necessary).
    ts_left, ts_right = _ts_leftright(ts_left, ts_right)
    # Get timeseries.
    tsid = connector.find_id_not_in_pf("name of qhpfc")  # TODO
    s = connector.series(tsid, ts_left, ts_right)
    return PfLine({"p": s})


def pfstate(
    pf: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> PfState:
    """Get state of portfolio from Belvis

    Parameters
    ----------
    pf : str
        Portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM').
    ts_left : Union[str, dt.datetime, pd.Timestamp], optional
        Start of the delivery period. If none provided, use start of coming year.
    ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        End of the delivery period. If none provided, use end of year of `ts_left`.

    Returns
    -------
    PfState
    """
    # Get portfolio lines.
    pfl_offtakevolume = offtakevolume(pf, ts_left, ts_right)
    pfl_sourced = sourced(pf, ts_left, ts_right)
    pfl_unsourcedprice = unsourcedprice(ts_left, ts_right)
    # Create portfolio state.
    pfstate = PfState(pfl_offtakevolume, pfl_unsourcedprice, pfl_sourced)
    return pfstate


# future goal
# def givemepfstate(
#     pf="LUD", viewon="2021-04-01", deliveryperiod=["2022-01-01", "2022-02-01"]
# ) -> PfState:
#     pass

# Add constructors to the objects directly.
factory.register_pfstate_source("from_belvis", pfstate)
factory.register_pfline_source("from_belvis_offtakevolume", offtakevolume)
factory.register_pfline_source("from_belvis_sourced", sourced)
factory.register_pfline_source("from_belvis_forwardpricecurve", unsourcedprice)
