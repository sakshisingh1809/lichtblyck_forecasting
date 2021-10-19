from ..tools import stamps
from ..core2.pfline import PfLine
from ..core2.pfstate import PfState
from . import connector
from typing import Optional, Union
import functools
import datetime as dt
import pandas as pd


# Intermediate goals:
# pf, deliveryperiod -> ws, rs, wo, pu
# ws, rs -> sourced = PfLine
# wo -> offtakevolume = PfLine
# pu -> unsourcedprice = PfLine
# sourced, offtakevolume, unsourcedprice -> PfState

PUNAME = {}

TSNAMES = {
    "DEFAULT": {
        "wo": "#LB Saldo aller Prognosegeschäfte +UB",
        "ws": "#LB Saldo aller Termingeschäfte +UB",
        "rs": "#LB Wert aller Termingeschäfte +UB",
    },
    "LUD": {},  # any
    "PKG": {"wo": "#FRM offtake - MW"}
    # etc.
}


def offtakevolume(
    pf,
    ts_left: Optional[Union[str, dt.datetime]] = None,
    ts_right: Optional[Union[str, dt.datetime]] = None,
) -> PfLine:
    """[summary]

    Args:
        pf ([type]): [description]
        ts_left (Optional[Union[str, dt.dt.datetime]], optional): [description]. Defaults to None.
        ts_right (Optional[Union[str, dt.dt.datetime]], optional): [description]. Defaults to None.

    Returns:
        PfLine: returns the PfLine for offtake o
    """
    ts_left, ts_right = stamps.ts_leftright(ts_left, ts_right)
    tsname = get_tsnames(pf)
    id_wo = connector.find_id(pf, tsname["wo"])  # "name of offtake timeseries"
    wo = connector.series(id_wo, ts_left, ts_left)

    return PfLine({"w": wo})


def sourced(
    pf,
    ts_left: Optional[Union[str, dt.dt.datetime]] = None,
    ts_right: Optional[Union[str, dt.dt.datetime]] = None,
) -> PfLine:
    """[summary]

    Args:
        pfname ([type]): [description]
        tsname_ws ([type]): [description]
        tsname_rs ([type]): [description]
        ts_left (Optional[Union[str, dt.dt.datetime]], optional): [description]. Defaults to None.
        ts_right (Optional[Union[str, dt.dt.datetime]], optional): [description]. Defaults to None.

    Returns:
        PfLine: returns the PfLine for sourced volume w & revenue r
    """
    ts_left, ts_right = stamps.ts_leftright(ts_left, ts_right)
    tsname = get_tsnames(pf)
    id_ws = connector.find_id(pf, tsname["ws"])  # "name of sourced volume timeseries"
    id_rs = connector.find_id(pf, tsname["rs"])  # "name of sourced revenue timeseries"
    ws = connector.series(id_ws, ts_left, ts_left)
    rs = connector.series(id_rs, ts_left, ts_left)

    return PfLine({"w": ws, "r": rs})


@functools.lru_cache()  # memoization
def unsourcedprice(
    ts_left: Optional[Union[str, dt.dt.datetime]] = None,
    ts_right: Optional[Union[str, dt.dt.datetime]] = None,
) -> PfLine:
    """[summary]

    Args:
        ts_left (Optional[Union[str, dt.dt.datetime]], optional): [description]. Defaults to None.
        ts_right (Optional[Union[str, dt.dt.datetime]], optional): [description]. Defaults to None.

    Returns:
        PfLine: returns the PfLine for unsourced price p
    """
    ts_left, ts_right = stamps.ts_leftright(ts_left, ts_right)
    id_pu = connector.find_id_not_in_pf("Spot (Mix 15min) / QHPFC (aktuell)")
    pu = connector.series(id_pu, ts_left, ts_left)

    return PfLine({"p": pu})


# current final goal
def pfstate(
    pf="LUD",
    ts_left: Optional[Union[str, dt.dt.datetime]] = None,
    ts_right: Optional[Union[str, dt.dt.datetime]] = None,
) -> PfState:
    """[summary]

    Parameters
    ----------
    pf : str, optional
        [description], by default "LUD"
    ts_left : str | dt.datetime, optional
        Start of the delivery period. If none provided, use start of coming year.
    ts_right : Optional[Union[str, dt.dt.datetime]], optional
        End of the delivery period. If none provided, use end of year of `ts_left`.

    Returns
    -------
    PfState
    """
    ts_left, ts_right = stamps.ts_leftright(ts_left, ts_right)

    pfl_offtakevolume = offtakevolume(pf, ts_left, ts_right)
    pfl_sourced = sourced(pf, ts_left, ts_right)
    pfl_unsourcedprice = unsourcedprice(ts_left, ts_right)

    pfstate = PfState(pfl_offtakevolume, pfl_unsourcedprice, pfl_sourced)

    return pfstate


# future goal
def givemepfstate(
    pf="LUD", viewon="2021-04-01", deliveryperiod=["2022-01-01", "2022-02-01"]
) -> PfState:
    pass


def get_tsnames(pf):

    dic = TSNAMES[pf]
    for key in ["wo", "ws", "rs", "pu"]:
        if key not in dic:
            dic[key] = TSNAMES["DEFAULT"][key]

    return dic


if __name__ == "__main__":

    pf = "udwig"
    # tsname = "#LB FRM Procurement/Forward - MW - incl subpf"

    pfs = pfstate(pf)
    print(pfs)
