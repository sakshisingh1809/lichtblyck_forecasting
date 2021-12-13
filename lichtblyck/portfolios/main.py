"""Module to set up the portfolios as they are used and reported."""

from ..core.pfstate import PfState
from ..core.pfline import PfLine
from .. import belvis  # to add functionality to pfline and pfstate
import pandas as pd
import datetime as dt


POWER = {  # pf-name: {'offtake': {'100%': (), 'certain', ()} 'sourced': ()}
    "LUD_STG": {
        "offtake": {"100%": ("LUD_Stg",), "certain": ("LUD_Stg_SiM",)},
        "sourced": ("LUD_Stg",),
    },
    "LUD_WP": {
        "offtake": {"100%": ("LUD_WP",), "certain": ("LUD_WP_SiM",)},
        "sourced": ("LUD_WP",),
    },
    "LUD_NSP": {
        "offtake": {"100%": ("LUD_NSp",), "certain": ("LUD_NSp_SiM",)},
        "sourced": ("LUD_NSp",),
    },
    "WP": {"offtake": {"100%": ("WP",), "certain": ("WP",)}, "sourced": ("WP",)},
    "NSP": {"offtake": {"100%": ("NSp",), "certain": ("NSp",)}, "sourced": ("NSp",)},
    "B2C_P2H": {
        "offtake": {
            "100%": ("LUD_NSp", "LUD_WP", "WP", "NSp"),
            "certain": ("LUD_NSp_SiM", "LUD_WP_SiM", "WP", "NSp"),
        },
        "sourced": ("LUD_NSp", "LUD_WP", "WP", "NSp"),
    },
    "B2C_HH": {
        "offtake": {"100%": ("PKG", "LUD_Stg"), "certain": ("PK_SiM", "LUD_Stg_SiM"),},
        "sourced": ("PKG", "LUD_Stg"),
    },
}

GAS = {}




def pfstate(commodity: str, pfname: str, ts_left=None, ts_right=None) -> PfState:
    commodity_dic = {"power": POWER, "gas": GAS}.get(commodity)
    if not commodity_dic:
        raise ValueError("`commodity` must be 'power' or 'gas'.")

    pf_dic = commodity_dic.get(pfname)
    if not pf_dic:
        raise ValueError(f"`pfname` must be one of {', '.join(commodity_dic.keys())}.")

    if commodity == "power":
        # Offtake: combine two curves.
        offtakevolume_100 = sum(
            belvis.data.offtakevolume(commodity, pfid, ts_left, ts_right)
            for pfid in pf_dic["offtake"]["100%"]
        )
        offtakevolume_certain = sum(
            belvis.data.offtakevolume(commodity, pfid, ts_left, ts_right)
            for pfid in pf_dic["offtake"]["certain"]
        )
        cutoff = pd.Timestamp.now().tz_localize("Europe/Berlin").floor("D") + pd.Timedelta(days=40)
        df1 = offtakevolume_100.df()[offtakevolume_100.index < cutoff]
        df2 = offtakevolume_certain.df()[offtakevolume_certain.index >= cutoff]
        offtakevolume = PfLine(pd.concat([df1, df2]).w)
        # Sourced.
        sourced = sum(
            belvis.data.sourced(commodity, pfid, ts_left, ts_right)
            for pfid in pf_dic["sourced"]
        )
        # Unsourcedprice.
        unsourcedprice = belvis.data.unsourcedprice(commodity, ts_left, ts_right)
        print("Unsourced prices")
        return PfState(offtakevolume, unsourcedprice, sourced)

