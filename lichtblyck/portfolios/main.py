"""Module to set up the portfolios as they are used and reported."""

from ..core.pfstate import PfState
from ..core.pfline.single import SinglePfLine
from .. import belvis  # to add functionality to pfline and pfstate
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
#       pf-name: {}
GAS_ORIGINAL = {"SBK1": "SBK1_G", "SBK6": "SBK6_G"}

# . For synthetic gas portfolios (not found in Belvis):
#   GAS_SYNTHETIC:
#       pf-name: Iterable
#   The iterable elements are pf-names in GAS_ORIGINAL
GAS_SYNTHETIC = {"B2C_Legacy": ("SBK1", "SBK6")}

PFNAMES = {
    "power": [*POWER_ORIGINAL.keys(), *POWER_SYNTHETIC.keys()],
    "gas": [*GAS_ORIGINAL.keys(), *GAS_SYNTHETIC.keys()],
}


def pfstate(commodity: str, pfname: str, ts_left=None, ts_right=None) -> PfState:
    """Get sourced volume and price for a certain portfolio.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    pfname : str
        Portfolio name. See .PFNAMES for allowed values.
    ts_left : Union[str, dt.datetime, pd.Timestamp], optional
        Start of delivery period.
    ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        End of delivery period.

    Returns
    -------
    PfState
    """

    if commodity not in ("power", "gas"):
        raise ValueError(
            f"Values of parameter ``commodity`` must be 'power' or 'gas'; got {commodity}."
        )
    if pfname not in PFNAMES[commodity]:
        raise ValueError(
            f"Parameter ``pfname`` must be one of {PFNAMES[commodity]}; got {pfname}."
        )

    if commodity == "power":
        return pfstate_power(pfname, ts_left, ts_right)
    else:
        return pfstate_gas(pfname, ts_left, ts_right)


def pfstate_power(pfname: str, ts_left, ts_right) -> PfState:
    commodity = "power"

    # Portfolio is sum of several portfolios.
    if pfnames := POWER_SYNTHETIC.get(pfname):
        return sum(pfstate_power(pfn, ts_left, ts_right) for pfn in pfnames)

    # Portfolio is original portfolio.
    pf_dic = POWER_ORIGINAL[pfname]

    # Offtake: combine two curves.
    offtakevolume_100 = sum(
        belvis.data.offtakevolume(commodity, pfid, ts_left, ts_right)
        for pfid in pf_dic["offtake"]["100%"]
    )
    offtakevolume_certain = sum(
        belvis.data.offtakevolume(commodity, pfid, ts_left, ts_right)
        for pfid in pf_dic["offtake"]["certain"]
    )
    now = pd.Timestamp.now().tz_localize("Europe/Berlin").floor("D")
    cutoff = now + pd.Timedelta(days=4)
    df1 = offtakevolume_100.df()[offtakevolume_100.index < cutoff]
    df2 = offtakevolume_certain.df()[offtakevolume_certain.index >= cutoff]
    offtakevolume = SinglePfLine(pd.concat([df1, df2]).w)

    # Sourced.
    sourced = sum(
        belvis.data.sourced(commodity, pfid, ts_left, ts_right)
        for pfid in pf_dic["sourced"]
    )

    # Unsourcedprice.
    unsourcedprice = belvis.data.unsourcedprice(commodity, ts_left, ts_right)

    # Combine.
    return PfState(offtakevolume, unsourcedprice, sourced)


def pfstate_gas(pfname: str, ts_left, ts_right) -> PfState:
    commodity = "gas"

    # Portfolio is sum of several portfolios.
    if pfnames := GAS_SYNTHETIC.get(pfname):
        return sum(pfstate_gas(pfn, ts_left, ts_right) for pfn in pfnames)
