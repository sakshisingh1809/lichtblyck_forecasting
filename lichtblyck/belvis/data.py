"""Module that uses connection to Belvis to generate PfLine and PfState objects."""

from ..tools import stamps, frames
from ..core2.pfline import PfLine
from ..core2.pfstate import PfState
from . import connector
from typing import Union, Tuple
import functools
import datetime as dt
import pandas as pd

# For offtake volume, sourced volume, and sourced revenue: timeseries names.
DEFAULTTSNAMES_PER_COMMODITY = {  # commodity, ts, name.
    "power": {
        "wo": "#LB Saldo aller Prognosegeschäfte +UB",
        "ws": "#LB Saldo aller Termingeschäfte +UB",
        "rs": "#LB CPM Wert HI-Geschaefte ohne Spot",
    },
    "gas": {
        "wo": "",
        "ws": "",
        "rs": "",
    },
}
TSNAMES_PER_COMMODITY_AND_PF = {  # commodity, pfid, ts, name. All allowed portfolios must have a key here.
    "power": {
        "LUD": {},  # any
        "PKG": {},
        "PK_Neu": {},
        "WP": {},
        "NSp": {},
        "GK": {},
        "SBSG": {},
    },
    "gas": {"SKK9_G": {"wo": "#LB PFMG Absatz SBK9 Gesamt"}, "SK": {}},
}
# For unsourced prices: portfolio id and timeseries name.
PFID_AND_TSNAME_FOR_PU = {
    "power": ("Deutschland", "Spot (Mix 15min) / QHPFC (aktuell)"),
    "gas": (),
}


def _tsname(commodity, pfid, ts):
    """Convenience function to find name of timeseries in Belvis portfolio."""
    try:
        tsnames_per_pf = TSNAMES_PER_COMMODITY_AND_PF[commodity]
        defaulttsnames = DEFAULTTSNAMES_PER_COMMODITY[commodity]
    except KeyError:
        raise ValueError("`commodity` must be one of {'power', 'gas'}.")
    try:
        tsnames = tsnames_per_pf[pfid]
    except KeyError:
        raise ValueError(f"`pfid` must be one of {', '.join(tsnames_per_pf.keys())}.")
    try:
        defaulttsname = defaulttsnames[ts]
    except KeyError:
        raise ValueError(f"`ts` must be one of {', '.join(defaulttsnames.keys())}.")
    return tsnames.get(ts, defaulttsname)


def _pfid_and_tsname_for_pu(commodity: str) -> Tuple:
    try:
        return PFID_AND_TSNAME_FOR_PU[commodity]
    except KeyError:
        raise ValueError("`commodity` must be one of {'power', 'gas'}.")


@functools.lru_cache()
def _ts_leftright(ts_left, ts_right):
    return stamps.ts_leftright(ts_left, ts_right)


def offtakevolume(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> PfLine:
    """Get offtake volume for a certain portfolio from Belvis.

    Parameters
    ----------
    commodity : str
        Commodity. One of {'power', 'gas'}.
    pfid : str
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
    tsname = _tsname(commodity, pfid, "wo")
    tsid = connector.find_tsid(commodity, pfid, tsname, strict=True)
    s = connector.series(commodity, tsid, ts_left, ts_right)
    s = frames.set_ts_index(s, bound='right')
    return PfLine({"w": s})


def sourced(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> PfLine:
    """Get sourced volume and price for a certain portfolio from Belvis.

    Parameters
    ----------
    commodity : str
        Commodity. One of {'power', 'gas'}.
    pfid : str
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
        tsname = _tsname(commodity, pfid, ts)
        tsid = connector.find_tsid(commodity, pfid, tsname, strict=True)
        s = connector.series(commodity, tsid, ts_left, ts_right)
        data[n] = frames.set_ts_index(s, bound='right')
    return PfLine(data)


@functools.lru_cache()  # memoization
def unsourcedprice(
    commodity: str = "power",
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> PfLine:
    """Get market prices from Belvis.

    Parameters
    ----------
    commodity : str
        Commodity for which to get the market prices. One of {'power', 'gas'}.
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
    pfid, tsname = _pfid_and_tsname_for_pu(commodity)
    tsid = connector.find_tsid(commodity, pfid, tsname, strict=True)
    s = connector.series(commodity, tsid, ts_left, ts_right)
    s = frames.set_ts_index(s, bound='right')
    return PfLine({"p": s})


def pfstate(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> PfState:
    """Get state of portfolio from Belvis.

    Parameters
    ----------
    commodity : str, 
        Commodity. One of {'power', 'gas'}
    pfid : str
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
    pfl_offtakevolume = offtakevolume(commodity, pfid, ts_left, ts_right)
    pfl_sourced = sourced(commodity, pfid, ts_left, ts_right)
    pfl_unsourcedprice = unsourcedprice(commodity, ts_left, ts_right)
    # Create portfolio state.
    pfstate = PfState(pfl_offtakevolume, pfl_unsourcedprice, pfl_sourced)
    return pfstate


# future goal
# def givemepfstate(
#     pf="LUD", viewon="2021-04-01", deliveryperiod=["2022-01-01", "2022-02-01"]
# ) -> PfState:
#     pass

# Add constructors to the objects directly. --> Done directly now, below.
# factory.register_pfline_source("from_belvis_offtakevolume", offtakevolume)
# factory.register_pfline_source("from_belvis_sourced", sourced)
# factory.register_pfline_source("from_belvis_forwardpricecurve", unsourcedprice)
# factory.register_pfstate_source("from_belvis", pfstate)
PfState.from_belvis = pfstate
PfLine.from_belvis_offtakevolume = offtakevolume
PfLine.from_belvis_sourced =  sourced
PfLine.from_belvis_forwardpricecurve =  unsourcedprice