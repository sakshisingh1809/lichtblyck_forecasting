"""Module to get data from Belvis."""

import datetime as dt
import os
import pathlib
from typing import Dict, Union

import belvys
import pandas as pd
import portfolyo as pf

tenants: Dict[str, belvys.Tenant] = {}

THISFOLDER = pathlib.Path(__file__).parent


def aftercare_power(s, tsid, pfid, tsname):
    s = belvys.adjustment.convert_to_berlin(s)
    s = belvys.adjustment.infer_frequency(s)
    s = belvys.adjustment.makeleft(s)
    return s


def aftercare_gas(s, tsid, pfid, tsname):
    if tsid == 23346575:
        s = belvys.adjustment.cet_to_berlin(s)
    else:
        s = belvys.adjustment.convert_to_berlin(s)
    s = belvys.adjustment.infer_frequency(s)
    s = belvys.adjustment.makeleft(s)
    return s


# def precare_factory(tsid: int, pfid: str, tsname: str) -> belvys.Precare:

#     if tsid == 23346575:

#         def precare(ts_left: pd.Timestamp, ts_right: pd.Timestamp):
#             return ts_left, ts_right, "inclusive", "exclusive"

#         return precare

#     return belvys.precare_store.precare_gas_hourly


# def aftercare_factory(tsid: int, pfid: str, tsname: str) -> belvys.Aftercare:

#     if tsid == 23346575:

#         def aftercare(s: pd.Series):
#             s = s.tz_convert("+01:00").tz_localize(None)
#             return pf.standardize(s, "aware", "left", "Europe/Berlin")

#         return aftercare

#     return belvys.aftercare_store.aftercare_gas_hourly


def create_tenants() -> None:
    """Create tenants"""
    tenant_info = {
        "power": ("lichtblick_api_power", "lichtblick_structure_power"),
        "gas": ("lichtblick_api_gas", "lichtblick_structure_gas"),
    }
    for id, (apifile, structfile) in tenant_info.items():
        s = belvys.Structure.from_file(THISFOLDER / f"{structfile}.yaml")
        a = belvys.Api.from_file(THISFOLDER / f"{apifile}.yaml")
        tenant = belvys.Tenant(s, a)
        if id == "gas":
            # tenant.prepend_aftercare(gas_aftercare)
            tenant.aftercare = aftercare_gas
        else:
            tenant.aftercare = aftercare_power
        tenants[id] = tenant


def auth_with_environ(usr: str = "BELVIS_USR", pwd: str = "BELVIS_PWD") -> None:
    """Authorize with belvis tenants using environment variables.

    Parameters
    ----------
    usr : str
        Name of environment variable where belvis username is stored.
    pwd : str
        Name of environment variable where belvis password is stored.
    """
    auth_with_password(os.environ[usr], os.environ[pwd])


def auth_with_password(usr: str, pwd: str) -> None:
    """Authorize with belvis tenants using username and password.

    Parameters
    ----------
    usr : str
        Name of belvis username.
    pwd : str
        Name of belvis password.
    """
    for tenant in tenants.values():
        tenant.api.access_from_usr_pwd(usr, pwd)


def update_cache_files():
    """Update all cache files. NB: might take long time!"""
    for tenant in tenants.values():
        tenant.update_cache()


def offtakevolume(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> pf.PfLine:
    """Get offtake (volume) for a certain portfolio from Belvis.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    pfid : str
        Belvis portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM').
    ts_left : Union[str, dt.datetime, pd.Timestamp], optional
    ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        Start and end of delivery period. Left-closed, right-open.

    Returns
    -------
    PfLine
    """
    return tenants[commodity].portfolio_pfl(pfid, "offtake", ts_left, ts_right)


def sourced(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> pf.PfLine:
    """Get sourced volume and price for a certain portfolio from Belvis.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    pfid : str
        Belvis portfolio abbreviation (e.g. 'LUD' or 'LUD_SIM').
    ts_left : Union[str, dt.datetime, pd.Timestamp], optional
    ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        Start and end of delivery period. Left-closed, right-open.

    Returns
    -------
    PfLine
    """
    return tenants[commodity].portfolio_pfl(pfid, "sourced", ts_left, ts_right)


def unsourcedprice(
    commodity: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> pf.PfLine:
    """Get forward curve prices for a certain portfolio from Belvis.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    ts_left : Union[str, dt.datetime, pd.Timestamp], optional
    ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        Start and end of delivery period. Left-closed, right-open.

    Returns
    -------
    PfLine
    """
    priceid = {"power": "qhpfc", "gas": "dpfc"}[commodity]
    ts_left = pd.Timestamp(ts_left)  # + dt.timedelta(days=-1)  # workaround gas
    return tenants[commodity].price_pfl(priceid, ts_left, ts_right)


def pfstate(
    commodity: str,
    pfid: str,
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> pf.PfState:
    """Get sourced volume and price for a certain portfolio.

    Parameters
    ----------
    commodity : {'power', 'gas'}
    pfid : str
        Id (= short name) of portfolio.
    ts_left, ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        Start and end of delivery period. Left-closed, right-open.

    Returns
    -------
    PfState
    """
    o = offtakevolume(commodity, pfid, ts_left, ts_right)
    u = unsourcedprice(commodity, ts_left, ts_right)
    s = sourced(commodity, pfid, ts_left, ts_right)

    return pf.PfState(o, u, s)


create_tenants()

if __name__ == "__main__":
    auth_with_environ()
    left = dt.date.today()
    right = dt.date.today() + dt.timedelta(days=300)
    og = offtakevolume("gas", "SBK1_G", left, right)
    ug = unsourcedprice("gas", left, right)
    op = offtakevolume("power", "P_B2B", left, right)
    up = unsourcedprice("power", left, right)
