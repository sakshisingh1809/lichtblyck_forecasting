"""Module that uses connection to Belvis to generate PfLine and PfState objects."""

from . import connector
from ..tools import stamps, frames
from ..core.pfline import PfLine, SinglePfLine
from ..core.pfstate import PfState
from typing import Union, Tuple
import functools
import datetime as dt
import pandas as pd

# For offtake volume, sourced volume, and sourced revenue: timeseries names.
DEFAULTTSNAMES_PER_COMMODITY = {  # commodity, ts, name or list of names.
    "power": {
        # currently used:
        "wo": "#LB FRM Offtake - MW - incl subpf",
        "ws": (
            "#LB FRM Procurement/Forward - MW - incl subpf",
            "#LB FRM Procurement/SPOT/DA - MW - incl subpf",
            "#LB FRM Procurement/SPOT/ID - MW - incl subpf",
        ),
        "rs": (
            "#LB FRM Procurement/Forward - EUR (contract) - incl subpf",
            "#LB FRM Procurement/SPOT/DA - EUR (contract) - incl subpf",
            "#LB FRM Procurement/SPOT/ID - EUR (contract) - incl subpf",
        ),
    },
    "gas": {
        "wo": "",
        "ws": "",
        "rs": "",
    },
}
TSNAMES_PER_COMMODITY_AND_PF = {  # commodity, pfid, ts, name or list of names. All allowed portfolios must have a key here.
    "power": {
        "PKG": {"wo": "#LB FRM Offtake - MW - excl subpf"},
        "WP": {},
        "NSp": {},
        "LUD": {},
        "LUD_NSp": {"wo": "#LB FRM Offtake - MW - excl subpf"},
        "LUD_NSp_SiM": {},
        "LUD_Stg": {"wo": "#LB FRM Offtake - MW - excl subpf"},
        "LUD_Stg_SiM": {},
        "LUD_WP": {"wo": "#LB FRM Offtake - MW - excl subpf"},
        "LUD_WP_SiM": {},
        "PK_SiM": {},
        "PK_Neu_FLX": {"wo": "#LB FRM Offtake - MW - excl subpf"},
        "PK_Neu_FLX_SiM": {},
        "PK_Neu_NSP": {"wo": "#LB FRM Offtake - MW - excl subpf"},
        "PK_Neu_NSP_SiM": {},
        "PK_Neu_WP": {"wo": "#LB FRM Offtake - MW - excl subpf"},
        "PK_Neu_WP_SiM": {},
        "GK": {},
        "SBSG": {},
    },
    "gas": {
        "SBK1_G": {"wo": "#LB PFMG Absatz SBK1 Gesamt"},
        "SBK6_G": {"wo": "#LB PFMG Absatz SBK6 Gesamt"},
        "SBK9_G": {"wo": "#LB PFMG Absatz SBK9 Gesamt"},
    },
}
# For unsourced prices: portfolio id and timeseries name.
PFID_AND_TSNAME_FOR_PU = {
    "power": ("Deutschland", "Spot (Mix 15min) / QHPFC (aktuell)"),
    "gas": ("PEGAS_THE_H", "Aktuellste DFC THE LB_d"),
}


def _print_status(msg: str):
    print(f"{dt.datetime.now().isoformat()} {msg}")


def _tsname(commodity: str, pfid: str, ts: str):
    """Convenience function to find name of timeseries in Belvis portfolio. Case insensitive."""
    tsnames_per_pf = TSNAMES_PER_COMMODITY_AND_PF.get(commodity)
    defaulttsnames = DEFAULTTSNAMES_PER_COMMODITY.get(commodity)
    if tsnames_per_pf is None or defaulttsnames is None:
        raise ValueError("`commodity` must be one of {'power', 'gas'}.")
    tsnames = tsnames_per_pf.get(pfid)
    if tsnames is None:
        raise ValueError(
            f"`pfid` '{pfid}' not found. Must be one of {', '.join(tsnames_per_pf.keys())}."
        )
    defaulttsname = defaulttsnames.get(ts)
    if defaulttsname is None:
        raise ValueError(
            f"``ts`` '{ts}' not found. Must be one of {', '.join(defaulttsnames.keys())}."
        )
    return tsnames.get(ts, defaulttsname)


def _pfid_and_tsname_for_pu(commodity: str) -> Tuple:
    try:
        return PFID_AND_TSNAME_FOR_PU[commodity]
    except KeyError:
        raise ValueError("``commodity`` must be one of {'power', 'gas'}.")


@functools.lru_cache()
def _ts_leftright(ts_left, ts_right):
    return stamps.ts_leftright(ts_left, ts_right)


def _series(commodity, pfid, ts, ts_left, ts_right):
    _print_status(
        f"For commodity '{commodity}' and portfolio '{pfid}', getting the '{ts}'-data, for delivery from (incl) {ts_left} to (excl) {ts_right}."
    )
    tsnames = _tsname(commodity, pfid, ts)
    if isinstance(tsnames, str):
        tsnames = (tsnames,)  # turn into (1-element-) iterable
    series = []
    for tsname in tsnames:
        _print_status(f". {tsname}")
        tsid = connector.find_tsid(commodity, pfid, tsname, strict=True)
        series.append(connector.series(commodity, tsid, ts_left, ts_right))
    return frames.set_ts_index(sum(series), bound="right")


def offtakevolume(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> SinglePfLine:
    """Get offtake volume for a certain portfolio from Belvis.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    pfid : str
        Belvis portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM').
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
    s = _series(commodity, pfid, "wo", ts_left, ts_right)
    return SinglePfLine({"w": s})


def sourced(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> PfLine:
    """Get sourced volume and price for a certain portfolio from Belvis.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    pfid : str
        Belvis portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM').
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
    data = {
        n: _series(commodity, pfid, ts, ts_left, ts_right)
        for n, ts in {"w": "ws", "r": "rs"}.items()
    }
    return SinglePfLine(data)


@functools.lru_cache()  # memoization
def unsourcedprice(
    commodity: str = "power",
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> PfLine:
    """Get market prices from Belvis.

    Parameters
    ----------
    commodity : {'power', 'gas'}
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
    s = frames.set_ts_index(s, bound="right")
    return SinglePfLine({"p": s})


def pfstate(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> PfState:
    """Get state of portfolio from Belvis.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    pfid : str
        Belvis portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM').
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
PfLine.from_belvis_sourced = sourced
PfLine.from_belvis_forwardpricecurve = unsourcedprice
