from ..tools.stamps import ts_leftright
from ..core2.pfline import PfLine
from ..core2.pfstate import PfState
from . import connector
from typing import Optional, Union
import datetime as dt


# Intermediate goals:
# pf, deliveryperiod -> ws, rs, wo, pu
# ws, rs -> sourced = PfLine
# wo -> offtakevolume = PfLine
# pu -> unsourcedprice = PfLine
# sourced, offtakevolume, unsourcedprice -> PfState

PUNAME = {}

TSNAMES = {
    "LUD": {
        "wo": "#LB Saldo aller Prognosegeschäfte +UB",
        "ws": "#LB Saldo aller Termingeschäfte +UB",
        "rs": "#LB Wert aller Termingeschäfte +UB",
    },
    "PKG": {
        "wo": "#LB Saldo aller Prognosegeschäfte +UB",
        "ws": "#LB Saldo aller Termingeschäfte +UB",
        "rs": "#LB Wert aller Termingeschäfte +UB",
    }
    # etc.
}


def offtakevolume(
    pf="LUD",
    ts_left: Optional[Union[str, dt.dt.datetime]] = None,
    ts_right: Optional[Union[str, dt.dt.datetime]] = None,
) -> PfLine:
    id_wo = connector.find_id("LUD", "name of offtake volume timeseries")
    wo = connector.series(id_wo, ts_left, ts_left)
    return PfLine({"w": wo})


def sourced(
    pf="LUD",
    ts_left: Optional[Union[str, dt.dt.datetime]] = None,
    ts_right: Optional[Union[str, dt.dt.datetime]] = None,
) -> PfLine:
    id_ws = connector.find_id("LUD", "name of sourced volume timeseries")
    id_rs = connector.find_id("LUD", "name of sourced revenue timeseries")
    ws = connector.series(id_ws, ts_left, ts_left)
    rs = connector.series(id_rs, ts_left, ts_left)
    return PfLine({"w": ws, "r": rs})


def unsourcedprice(
    ts_left: Optional[Union[str, dt.dt.datetime]] = None,
    ts_right: Optional[Union[str, dt.dt.datetime]] = None,
) -> PfLine:
    id_pu = connector.find_id_not_in_pf("name of qhpfc")
    pu = connector.series(id_pu, ts_left, ts_left)
    return PfLine({"p": pu})


# current final goal
def givemecurrentpfstate(
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
        Start of the delivery period  If none provided, use start of coming year.
    ts_right : Optional[Union[str, dt.dt.datetime]], optional
        [description], by default None

    Returns
    -------
    PfState
        [description]
    """
    ts_left, ts_right = ts_leftright(ts_left, ts_right)

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
