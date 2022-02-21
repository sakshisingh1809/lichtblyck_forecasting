"""Module that uses connection to Belvis to generate PfLine and PfState objects."""

from numpy import isin
from . import connector
from ..tools import stamps, frames
from ..core.pfline import PfLine, SinglePfLine, MultiPfLine
from ..core.pfstate import PfState
from typing import Dict, Iterable, Union, Tuple
import functools
import datetime as dt
import pandas as pd

# commodity [power/gas] : part [offtake/forward/spot] : col [w/q/r/p] : tsname or tsnames
DEFAULT = {
    "power": {
        "offtake": {"w": "#LB FRM Offtake - MW - incl subpf"},
        "forward": {
            "w": "#LB FRM Procurement/Forward - MW - incl subpf",
            "r": "#LB FRM Procurement/Forward - EUR (contract) - incl subpf",
        },
        "spot": {
            "w": (
                "#LB FRM Procurement/SPOT/DA - MW - incl subpf",
                "#LB FRM Procurement/SPOT/ID - MW - incl subpf",
            ),
            "r": (
                "#LB FRM Procurement/SPOT/DA - EUR (contract) - incl subpf",
                "#LB FRM Procurement/SPOT/ID - EUR (contract) - incl subpf",
            ),
        },
    },
    "gas": {
        "offtake": {"w": ""},
        "forward": {"w": "", "r": ""},
        "spot": {"w": "", "r": ""},
    },
}

# All allowed portfolios must have a key here.
# commodity [power/gas] : pfid : part [offtake/forward/spot]: tsname or tsnames
SPECIFICS = {
    "power": {
        "PKG": {"offtake": {"w": "#LB FRM Offtake - MW - excl subpf"}},
        "WP": {},
        "NSp": {},
        "LUD": {},
        "LUD_NSp": {"offtake": {"w": "#LB FRM Offtake - MW - excl subpf"}},
        "LUD_NSp_SiM": {},
        "LUD_Stg": {"offtake": {"w": "#LB FRM Offtake - MW - excl subpf"}},
        "LUD_Stg_SiM": {},
        "LUD_WP": {"offtake": {"w": "#LB FRM Offtake - MW - excl subpf"}},
        "LUD_WP_SiM": {},
        "PK_SiM": {},
        "PK_Neu_FLX": {"offtake": {"w": "#LB FRM Offtake - MW - excl subpf"}},
        "PK_Neu_FLX_SiM": {},
        "PK_Neu_NSP": {"offtake": {"w": "#LB FRM Offtake - MW - excl subpf"}},
        "PK_Neu_NSP_SiM": {},
        "PK_Neu_WP": {"offtake": {"w": "#LB FRM Offtake - MW - excl subpf"}},
        "PK_Neu_WP_SiM": {},
        "GK": {},
        "SBSG": {},
    },
    "gas": {
        "SBK1_G": {"offtake": {"w": "#LB PFMG Absatz SBK1 Gesamt"}},
        "SBK6_G": {"offtake": {"w": "#LB PFMG Absatz SBK6 Gesamt"}},
        "SBK9_G": {"offtake": {"w": "#LB PFMG Absatz SBK9 Gesamt"}},
    },
}
# For unsourced prices: portfolio id and timeseries name.
UNSOURCEDPRICE = {
    "power": ("Deutschland", "Spot (Mix 15min) / QHPFC (aktuell)"),
    "gas": ("PEGAS_THE_H", "Aktuellste DFC THE LB_d"),
}


def _print_status(msg: str):
    print(f"{dt.datetime.now().isoformat()} {msg}")


def _tsnamedict(commodity: str, pfid: str, part: str) -> Dict:
    """Lookup function to create dictionary that can be used as input to create a SinglePfLine instance."""

    # Get default part dictionary.
    partdict1 = DEFAULT.get(commodity)
    if partdict1 is None:
        raise ValueError("Parameter ``commodity`` must be one of {'power', 'gas'}.")

    # Get part dictionary that is specific for this portfolio.
    partdict2 = SPECIFICS.get(commodity).get(pfid)
    if partdict2 is None:
        raise ValueError(
            f"Parameter ``pfid`` must be one of {', '.join(SPECIFICS.get(commodity).keys())}; got '{pfid}'."
        )

    # Merge the two and get the correct part.
    tsnamedict = {**partdict1, **partdict2}.get(part)
    if tsnamedict is None:
        raise ValueError(
            f"Parameter ``part`` must be one of {','.join(DEFAULT[commodity].keys())}; got '{part}'."
        )

    return tsnamedict


def _pftsid_unsourced(commodity: str) -> Tuple[str]:
    if commodity not in UNSOURCEDPRICE:
        raise ValueError("Parameter ``commodity`` must be one of {'power', 'gas'}.")
    return UNSOURCEDPRICE[commodity]


def _series(commodity, pfid, tsnames, ts_left, ts_right):
    if isinstance(tsnames, str):
        tsnames = (tsnames,)  # turn into (1-element-) iterable
    series = []
    for tsname in tsnames:
        _print_status(f". {tsname}")
        tsid = connector.find_tsid(commodity, pfid, tsname, strict=True)
        series.append(connector.series(commodity, tsid, ts_left, ts_right))
    return frames.set_ts_index(sum(series), bound="right")


def _singlepfline(commodity, pfid, part, ts_left, ts_right):
    # Fix timestamps (if necessary).
    ts_left, ts_right = stamps.ts_leftright(ts_left, ts_right)
    if not (ts_left < ts_right):
        raise ValueError("Left timestamp must be strictly before right timestamp.")
    # Get timeseries names.
    tsnamedict = _tsnamedict(commodity, pfid, part)
    # Collect data.
    data = {}
    for col, tsnames in tsnamedict.items():
        _print_status(f"{commodity} | {pfid} | {ts_left} (incl) - {ts_right} (excl)")
        data[col] = _series(commodity, pfid, tsnames, ts_left, ts_right)
    # Create SinglePfLine.
    return SinglePfLine(data)


def offtakevolume(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> SinglePfLine:
    """Get offtake (volume) for a certain portfolio from Belvis.

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
    return _singlepfline(commodity, pfid, "offtake", ts_left, ts_right)


def forward(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> SinglePfLine:
    """Get sourced forward/futures (volume and price) for a certain portfolio from Belvis.

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
    return _singlepfline(commodity, pfid, "forward", ts_left, ts_right)


def spot(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> SinglePfLine:
    """Get sourced spot (volume and price) for a certain portfolio from Belvis.

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
    return _singlepfline(commodity, pfid, "spot", ts_left, ts_right)


def sourced(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> MultiPfLine:
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
    data = {
        "forward": forward(commodity, pfid, ts_left, ts_right),
        "spot": spot(commodity, pfid, ts_left, ts_right),
    }
    return MultiPfLine(data)


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
    ts_left, ts_right = stamps.ts_leftright(ts_left, ts_right)
    # Where to find this data.
    pfid, tsname = _pftsid_unsourced(commodity)
    # Get the data.
    data = {"p": _series(commodity, pfid, tsname, ts_left, ts_right)}
    return SinglePfLine(data)


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
# TODO: ADD AS CLASSMETHOD.
PfState.from_belvis = pfstate
PfLine.from_belvis_offtakevolume = offtakevolume
PfLine.from_belvis_sourced = sourced
PfLine.from_belvis_sourcedspot = spot
PfLine.from_belvis_sourcedforward = forward
PfLine.from_belvis_forwardpricecurve = unsourcedprice
