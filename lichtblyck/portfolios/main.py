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
#       list of belvispfid
#   The iterable elements are portfolio abbreviations in Belvis and must be present in lichtblyck.belvis.data
POWER_ORIGINAL = [
    "P_SAL",
    "P_B2C",
    "P_B2C_HH",
    "P_B2C_HH_FIX",
    "P_B2C_HH_FLEX",
    "P_B2C_HH_SPOT",
    "P_B2C_P2H",
    "P_B2C_P2H_HP",
    "P_B2C_P2H_HP_FLEX",
    "P_B2C_P2H_HP_FIX",
    "P_B2C_P2H_NSH",
    "P_B2C_P2H_NSH_FLEX",
    "P_B2C_P2H_NSH_FIX",
    "P_B2B",
    "P_B2B_BTB",
    "P_B2B_EXT",
    "P_B2B_SPOT",
    "P_B2B_STR",
]

# . For synthetic power portfolios (not found in Belvis):
#   POWER_SYNTHETIC:
#       pf-name: Iterable
#   The iterable elements are pf-names in POWER_ORIGINAL
POWER_SYNTHETIC = {
    "P_B2C_P2H_FLEX": ["P_B2C_P2H_HP_FLEX", "P_B2C_P2H_NSH_FLEX"],
    "P_B2C_P2H_FIX": ["P_B2C_P2H_HP_FIX", "P_B2C_P2H_NSH_FIX"],
}  # pf-name: (pf-names in POWER_ORIGINAL)

# . For original gas portfolios (as found in Belvis):
#   GAS_ORIGINAL:
#       list of belvispfid
GAS_ORIGINAL = [
    "SBK1_G",
    "SBK6_G",
    "SBK9_G",
    "SBK5_G",
    "SBK8_G",
    "SBK4_G",
    "SBK7_G",
]

# . For synthetic gas portfolios (not found in Belvis):
#   GAS_SYNTHETIC:
#       pf-name: Iterable
#   The iterable elements are pf-names in GAS_ORIGINAL
GAS_SYNTHETIC = {
    "B2C_LEGACY": ("SBK1_G", "SBK6_G"),
    "B2C_NEW": ("SBK9_G",),
    "B2B_BTB": ("SBK5_G", "SBK8_G"),
    "B2B_CONTI": ("SBK4_G",),
    "B2B_RLM": ("SBK7_G",),
    "B2B": ("SBK5_G", "SBK8_G", "SBK4_G", "SBK7_G"),
}

PFNAMES = {
    "power": [*POWER_ORIGINAL, *POWER_SYNTHETIC.keys()],
    "gas": [*GAS_ORIGINAL, *GAS_SYNTHETIC.keys()],
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
    pfid = pfname

    # No changes necessary - offtake etc. correct in Belvis.
    offtakevolume = belvis.data.offtakevolume(
        "power", pfid, ts_left, ts_right, recalc=recalc
    )
    sourced = belvis.data.sourced("power", pfid, ts_left, ts_right, recalc=recalc)
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
    pfid = pfname

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
