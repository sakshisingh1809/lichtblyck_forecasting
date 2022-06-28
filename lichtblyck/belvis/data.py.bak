"""Module that uses connection to Belvis to generate PfLine and PfState objects."""

from . import raw
from ..tools import stamps, frames
from ..core.pfline import PfLine, SinglePfLine, MultiPfLine
from typing import Dict, Union, Tuple
import functools
import datetime as dt
import pandas as pd

# commodity [power/gas] : part [offtake/sourced] : ... : col [w/q/r/p] : tsname or tsnames
DEFAULT = {
    "power": {
        "offtake": {"w": "#LB FRM Offtake - MW - incl subpf"},
        "sourced": {
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
    },
    "gas": {
        "offtake": {"w": ""},
        "sourced": {
            "forward": {
                "w": "#LB RM Saldo Menge Anteil Termin",
                "r": "#LB RM Kostensumme Anteil Termin",
            },
            "spot": {
                "w": "#LB RM Saldo Menge Anteil Spot",
                "r": "#LB RM Kostensumme Anteil Spot",
            },
            "conversion": {
                "w": "#LB RM Saldo Menge Anteil Konvertierung",
                "r": "#LB RM Kostensumme Anteil Konvertierung",
            },
            "top": {
                "w": "#LB RM Saldo Menge Anteil ToP",
                "r": "#LB RM Kostensumme Anteil ToP",
            },
        },
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
        # B2C
        "SBK1_G": {
            "offtake": {"w": "#LB PFMG Absatz SBK1 Gesamt"},
            "sourced": {
                "transfer": {
                    "w": "#LB RM Saldo Menge Anteil SBK6",
                    "r": "#LB RM Kostensumme Anteil SBK6",
                },
                "storage": {
                    "w": "#LB RM Saldo Menge Anteil Speicher",
                    "r": "#LB RM Kostensumme Anteil Speicher",
                },
                "tempr": {
                    "w": "#LB RM Saldo Menge Anteil Temperatur",
                    "r": "#LB RM Kostensumme Anteil Temperatur",
                },
            },
        },
        "SBK6_G": {
            "offtake": {"w": "#LB PFMG Absatz SBK6 Gesamt"},
            "sourced": {
                "transfer": {
                    "w": "#LB RM Saldo Menge Anteil SBK6",
                    "r": "#LB RM Kostensumme Anteil SBK6",
                },
                "storage": {
                    "w": "#LB RM Saldo Menge Anteil Speicher",
                    "r": "#LB RM Kostensumme Anteil Speicher",
                },
                "biogas": {
                    "w": "#LB RM Saldo Menge Anteil Biogas",
                    "r": "#LB RM Kostensumme Anteil Biogas",
                },
                "tempr": {
                    "w": "#LB RM Saldo Menge Anteil Temperatur",
                    "r": "#LB RM Kostensumme Anteil Temperatur",
                },
            },
        },
        # B2C New tariffs
        "SBK9_G": {"offtake": {"w": "#LB PFMG Absatz SBK9 Gesamt"}},
        # B2B BtB
        "SBK5_G": {
            "offtake": {"w": "#LB PFMG Absatz SBK5 Gesamt"},
            "sourced": {
                "transfer": {
                    "w": "#LB RM Saldo Menge Anteil SBK4",
                    "r": "#LB RM Kostensumme Anteil SBK4",
                },
                "tempr": {
                    "w": "#LB RM Saldo Menge Anteil Temperatur",
                    "r": "#LB RM Kostensumme Anteil Temperatur",
                },
            },
        },
        "SBK8_G": {
            "offtake": {"w": "#LB PFMG Absatz SBK8 Gesamt"},
            "sourced": {
                "transfer": {
                    "w": (
                        "#LB RM Saldo Menge Anteil SBK6",
                        "#LB RM Saldo Menge Anteil SBK4",
                    ),
                    "r": (
                        "#LB RM Kostensumme Anteil SBK6",
                        "#LB RM Kostensumme Anteil SBK4",
                    ),
                },
                "tempr": {
                    "w": "#LB RM Saldo Menge Anteil Temperatur",
                    "r": "#LB RM Kostensumme Anteil Temperatur",
                },
            },
        },
        # B2B Contingent
        "SBK4_G": {
            "offtake": {"w": "#LB PFMG Absatz SBK4 Gesamt"},
            "sourced": {
                "transfer": {
                    "w": "#LB RM Saldo Menge Anteil SBK4",
                    "r": "#LB RM Kostensumme Anteil SBK4",
                },
                "tempr": {
                    "w": "#LB RM Saldo Menge Anteil Temperatur",
                    "r": "#LB RM Kostensumme Anteil Temperatur",
                },
            },
        },
        # B2B RLM
        "SBK7_G": {
            "offtake": {"w": "#LB PFMG Absatz SBK7 Gesamt"},
            "sourced": {
                "transfer": {
                    "w": "#LB RM Saldo Menge Anteil SBK4",
                    "r": "#LB RM Kostensumme Anteil SBK4",
                },
            },
        },
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
    """Lookup function to create dictionary that can be used as input to create a SinglePfLine instance.
    ``part``: {'offtake' or 'sourced'}."""

    if commodity not in ["power", "gas"]:
        raise ValueError(
            f"Parameter ``commodity`` must be one of 'power', 'gas'; got '{commodity}'."
        )
    if part not in ["offtake", "sourced"]:
        raise ValueError(
            f"Parameter ``part`` must be one of 'offtake', 'sourced'; got '{part}'."
        )
    if pfid not in SPECIFICS[commodity]:
        raise ValueError(
            f"Parameter ``pfid`` must be one of {', '.join(SPECIFICS[commodity].keys())}; got '{pfid}'."
        )

    # Get default dictionary for this part.
    dic1 = DEFAULT[commodity][part]

    # Get dictionary for this part, that is specific for this portfolio (if any specified).
    dic2 = SPECIFICS[commodity][pfid].get(part, {})

    # Merge the two.
    # Result: dictionary with 'w' (and possibly 'r') as keys, or nested dict with (eventually) 'w' (and possibly 'r') as keys.
    return {**dic1, **dic2}


def _pfidtsname_unsourced(commodity: str) -> Tuple[str]:
    if commodity not in UNSOURCEDPRICE:
        raise ValueError("Parameter ``commodity`` must be one of {'power', 'gas'}.")
    return UNSOURCEDPRICE[commodity]


def _series(commodity, pfid, tsnames, ts_left, ts_right, *, recalc, **kwargs):
    if isinstance(tsnames, str):
        tsnames = (tsnames,)  # turn into (1-element-) iterable
    series = []
    for tsname in tsnames:
        tsid = raw.find_tsid(commodity, pfid, tsname, strict=True)
        _print_status(f". {tsid}: {tsname}")
        series.append(
            raw.series(commodity, tsid, ts_left, ts_right, blocking=recalc, **kwargs)
        )
    return sum(series)


def _pfline(commodity, pfid, part, ts_left, ts_right, *, recalc) -> PfLine:
    """Get portfolio line for certain commodity (power, gas), pfid (LUD, WP) and part (offtake, sourced)."""
    # Fix timestamps (if necessary).
    ts_left, ts_right = stamps.ts_leftright(ts_left, ts_right)
    if not (ts_left < ts_right):
        raise ValueError("Left timestamp must be strictly before right timestamp.")
    # Status.
    _print_status(f"{commodity} | {pfid} | {ts_left} (incl) - {ts_right} (excl)")

    def tsnamedict2pfline(tsnamedict) -> PfLine:
        data = {}
        if "w" in tsnamedict:  # Bottom level: create SinglePfLine.
            for col, tsnames in tsnamedict.items():
                s = _series(commodity, pfid, tsnames, ts_left, ts_right, recalc=recalc)
                # Correction for bad Belvis implementation: turn right-bound into left-bound timestamps.
                data[col] = frames.set_ts_index(s, bound="right")
            return SinglePfLine(data)
        else:  # Higher level: create MultiPfLine.
            for name, subdict in tsnamedict.items():
                if pfl := tsnamedict2pfline(subdict):
                    data[name] = pfl  # only add if relevant information.
            return MultiPfLine(data) if data else 0.0

    # Get timeseries names and data, and construct pfline.
    return tsnamedict2pfline(_tsnamedict(commodity, pfid, part))


","


def offtakevolume(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
    *,
    recalc: bool = True,
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
    recalc : bool, optional (default: True)
        If True, recalculate data that is not up-to-date. If False, return data without
        recalculating.

    Returns
    -------
    PfLine
    """
    return _pfline(commodity, pfid, "offtake", ts_left, ts_right, recalc=recalc)


def sourced(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
    *,
    recalc: bool = True,
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
    recalc : bool, optional (default: True)
        If True, recalculate data that is not up-to-date. If False, return data without
        recalculating.

    Returns
    -------
    PfLine
    """
    return _pfline(commodity, pfid, "sourced", ts_left, ts_right, recalc=recalc)


@functools.lru_cache()  # memoization
def unsourcedprice(
    commodity: str,
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
    pfid, tsname = _pfidtsname_unsourced(commodity)
    # Get the data.
    if commodity == "power":
        # Correction for bad Belvis implementation: turn right-bound into left-bound timestamps.
        s = _series(commodity, pfid, tsname, ts_left, ts_right, recalc=True)
        data = {"p": frames.set_ts_index(s, bound="right")}
    else:
        # Correction for bad Belvis implementation: gas DFC is not DST-adjusted.
        s = _series(
            commodity,
            pfid,
            tsname,
            ts_left,
            ts_right,
            leftrange="inclusive",
            rightrange="exclusive",
            recalc=True,
        )
        s = s.tz_convert("+01:00").tz_localize(None).tz_localize("Europe/Berlin")
        data = {"p": frames.set_ts_index(s, bound="left")}
    return SinglePfLine(data)


# Add constructors to the objects directly. --> Done directly now, below.
# factory.register_pfline_source("from_belvis_offtakevolume", offtakevolume)
# factory.register_pfline_source("from_belvis_sourced", sourced)
# factory.register_pfline_source("from_belvis_forwardpricecurve", unsourcedprice)
# factory.register_pfstate_source("from_belvis", pfstate)
# TODO: ADD AS CLASSMETHOD.
PfLine.from_belvis_offtakevolume = offtakevolume
PfLine.from_belvis_sourced = sourced
PfLine.from_belvis_forwardpricecurve = unsourcedprice
