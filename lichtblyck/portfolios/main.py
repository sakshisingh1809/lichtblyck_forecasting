"""Module to set up the portfolios as they are used and reported."""

from ..core.pfstate import PfState
from ..core.pfline import PfLine
from .. import belvis  # to add functionality to pfline and pfstate
import pandas as pd
import datetime as dt


POWER_ORIGINAL = {  # pf-name: {'offtake': {'100%': (), 'certain', ()} 'sourced': ()}
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
}
POWER_COMBINED = {  # pf-name: (pf-names in POWER_ORIGINAL)
    "B2C_P2H": ("NSP", "WP", "LUD_NSP", "LUD_WP"),
    "B2C_HH": ("PKG", "LUD_STG"),
}


GAS = {}

PFNAMES = {
    "power": [*POWER_ORIGINAL.keys(), *POWER_COMBINED.keys()],
    "gas": [*GAS.keys()],
}


def pfstate(commodity: str, pfname: str, ts_left=None, ts_right=None) -> PfState:
    """Get sourced volume and price for a certain portfolio.

    Parameters
    ----------
    commodity : str
        Commodity. One of {'power', 'gas'}.
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
        raise ValueError("`commodity` must be 'power' or 'gas'.")
    if pfname not in PFNAMES[commodity]:
        raise ValueError(f"`pfname` must be one of {', '.join(PFNAMES[commodity])}.")

    if commodity == "power":

        # Portfolio is sum of several portfolios.
        if (pfnames := POWER_COMBINED.get(pfname)) :
            return sum(pfstate(commodity, pfn, ts_left, ts_right) for pfn in pfnames)

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
        offtakevolume = PfLine(pd.concat([df1, df2]).w)
        # Sourced.
        sourced = sum(
            belvis.data.sourced(commodity, pfid, ts_left, ts_right)
            for pfid in pf_dic["sourced"]
        )
        # Unsourcedprice.
        unsourcedprice = belvis.data.unsourcedprice(commodity, ts_left, ts_right)
        # Combine.
        return PfState(offtakevolume, unsourcedprice, sourced)
