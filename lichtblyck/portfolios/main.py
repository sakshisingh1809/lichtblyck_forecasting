"""Module to set up the portfolios as they are used and reported."""


from ..core.pfstate import PfState
from ..core.pfline.single import SinglePfLine
from .. import belvis  # to add functionality to pfline and pfstate
from typing import Union
import datetime as dt
import pandas as pd

# Mapping of belvis portfolios onto 'useful' portfolios with 'arbitrary' names.
#
# Some corrections/calculations are needed, and because power and gas are not implemented
# equally, the mapping dictionaries for power and gas have distinct structures.

# . For original power portfolios (as found in Belvis):
#   POWER_ORIGINAL:
#       pf-name: {'offtake': {'100%': Iterable, 'certain', Iterable} 'sourced': Iterable}
#   The iterable elements are portfolio abbreviations in Belvis and must be present in lichtblyck.belvis.data
POWER_ORIGINAL = {
    "PKG": {"offtake": {"100%": ("PKG",), "certain": ("PK_SiM",)}, "sourced": ("PKG",)},
    "NSP": {"offtake": {"100%": ("NSp",), "certain": ("NSp",)}, "sourced": ("NSp",)},
    "WP": {"offtake": {"100%": ("WP",), "certain": ("WP",)}, "sourced": ("WP",)},
    "LUD_STG": {
        "offtake": {"100%": ("LUD_Stg",), "certain": ("LUD_Stg_SiM",)},
        "sourced": ("LUD_Stg",),
    },
    "LUD_NSP": {
        "offtake": {"100%": ("LUD_NSp",), "certain": ("LUD_NSp_SiM",)},
        "sourced": ("LUD_NSp",),
    },
    "LUD_WP": {
        "offtake": {"100%": ("LUD_WP",), "certain": ("LUD_WP_SiM",)},
        "sourced": ("LUD_WP",),
    },
    "B2C_HH_NEW": {
        "offtake": {"100%": ("PK_Neu_FLX",), "certain": ("PK_Neu_FLX_SiM",)},
        "sourced": ("PK_Neu_FLX",),
    },
    "B2C_RH_NEW": {
        "offtake": {"100%": ("PK_Neu_NSP",), "certain": ("PK_Neu_NSP_SiM",)},
        "sourced": ("PK_Neu_NSP",),
    },
    "B2C_HP_NEW": {
        "offtake": {"100%": ("PK_Neu_WP",), "certain": ("PK_Neu_WP_SiM",)},
        "sourced": ("PK_Neu_WP",),
    },
}

# . For synthetic power portfolios (not found in Belvis):
#   POWER_SYNTHETIC:
#       pf-name: Iterable
#   The iterable elements are pf-names in POWER_ORIGINAL
POWER_SYNTHETIC = {  # pf-name: (pf-names in POWER_ORIGINAL)
    "B2C_P2H_LEGACY": ("NSP", "WP", "LUD_NSP", "LUD_WP"),
    "B2C_HH_LEGACY": ("PKG", "LUD_STG"),
    "B2C_P2H_NEW": ("B2C_HP_NEW", "B2C_RH_NEW"),
}

# . For original gas portfolios (as found in Belvis):
#   GAS_ORIGINAL:
#       pf-name: belvispfid
GAS_ORIGINAL = {
    "SBK1": "SBK1_G",
    "SBK6": "SBK6_G",
    "SBK9": "SBK9_G",
    "SBK5": "SBK5_G",
    "SBK8": "SBK8_G",
    "SBK4": "SBK4_G",
    "SBK7": "SBK7_G",
}

# . For synthetic gas portfolios (not found in Belvis):
#   GAS_SYNTHETIC:
#       pf-name: Iterable
#   The iterable elements are pf-names in GAS_ORIGINAL
GAS_SYNTHETIC = {
    "B2C_LEGACY": ("SBK1", "SBK6"),
    "B2C_NEW": ("SBK9",),
    "B2B_BTB": ("SBK5", "SBK8"),
    "B2B_CONTI": ("SBK4",),
    "B2B_RLM": ("SBK7",),
    "B2B": ("SBK5", "SBK8", "SBK4", "SBK7"),
}

PFNAMES = {
    "power": [*POWER_ORIGINAL.keys(), *POWER_SYNTHETIC.keys()],
    "gas": [*GAS_ORIGINAL.keys(), *GAS_SYNTHETIC.keys()],
}


def pfstate(
    commodity: str,
    pfname: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
    *,
    recalc: bool = True,
) -> PfState:
    """Get sourced volume and price for a certain portfolio.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    pfname : str
        Portfolio name. See .PFNAMES for allowed values.
    ts_left, ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        Start and end of delivery period (left-closed).
    recalc : bool, optional (default: True)
        If True, recalculate data that is not up-to-date. If False, return data without
        recalculating.

    Returns
    -------
    PfState
    """

    if commodity not in ("power", "gas"):
        raise ValueError(
            f"Parameter ``commodity`` must be one of 'power', 'gas'; got {commodity}."
        )
    if pfname not in PFNAMES[commodity]:
        raise ValueError(
            f"Parameter ``pfname`` must be one of {', '.join(PFNAMES[commodity])}; got {pfname}."
        )

    if commodity == "power":
        return _pfstate_power(pfname, ts_left, ts_right, recalc=recalc)
    else:
        return _pfstate_gas(pfname, ts_left, ts_right, recalc=recalc)


def _pfstate_power(pfname: str, ts_left, ts_right, *, recalc) -> PfState:
    # Portfolio is sum of several portfolios.
    if pfnames := POWER_SYNTHETIC.get(pfname):
        return sum(
            _pfstate_power(pfn, ts_left, ts_right, recalc=recalc) for pfn in pfnames
        )

    # Portfolio is original portfolio.
    pf_dic = POWER_ORIGINAL[pfname]

    # Offtake: combine two curves.
    offtakevolume_100 = sum(
        belvis.data.offtakevolume("power", pfid, ts_left, ts_right, recalc=recalc)
        for pfid in pf_dic["offtake"]["100%"]
    )
    offtakevolume_certain = sum(
        belvis.data.offtakevolume("power", pfid, ts_left, ts_right, recalc=recalc)
        for pfid in pf_dic["offtake"]["certain"]
    )
    now = pd.Timestamp.now().tz_localize("Europe/Berlin").floor("D")
    cutoff = now + pd.Timedelta(days=40)
    df1 = offtakevolume_100.df()[offtakevolume_100.index < cutoff]
    df2 = offtakevolume_certain.df()[offtakevolume_certain.index >= cutoff]
    offtakevolume = SinglePfLine(pd.concat([df1, df2]).w)

    # Sourced: add forward and spot from individual components.
    sourced = sum(
        belvis.data.sourced("power", pfid, ts_left, ts_right, recalc=recalc)
        for pfid in pf_dic["sourced"]
    )

    # Unsourcedprice.
    unsourcedprice = belvis.data.unsourcedprice("power", ts_left, ts_right)

    # Combine.
    return PfState(offtakevolume, unsourcedprice, sourced)


def _pfstate_gas(pfname: str, ts_left, ts_right, *, recalc) -> PfState:
    # Portfolio is sum of several portfolios.
    if pfnames := GAS_SYNTHETIC.get(pfname):
        return sum(
            _pfstate_gas(pfn, ts_left, ts_right, recalc=recalc) for pfn in pfnames
        )

    # Portfolio is original portfolio.
    pfid = GAS_ORIGINAL[pfname]

    # No changes necessary - offtake etc. correct in Belvis.
    offtakevolume = belvis.data.offtakevolume(
        "gas", pfid, ts_left, ts_right, recalc=recalc
    )
    sourced = belvis.data.sourced("gas", pfid, ts_left, ts_right, recalc=recalc)
    unsourcedprice = belvis.data.unsourcedprice("gas", ts_left, ts_right)

    # Market prices are in daily frequency; make sure other data is also in daily frequency.
    offtakevolume = offtakevolume.asfreq("D")
    sourced = sourced.asfreq("D")
    return PfState(offtakevolume, unsourcedprice, sourced)
