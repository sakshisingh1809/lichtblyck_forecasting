# -*- coding: utf-8 -*-
"""
Standardized temperature load profiles for gas consumption.
"""

from . import convert
from typing import Callable, Iterable, Union
from pathlib import Path
import datetime as dt
import pandas as pd

SOURCEPATH = Path(__file__).parent / "sourcedata" / "gas" / "AllSigmoid.xlsx"


def fromscratch(
    A: float,
    B: float,
    C: float,
    Dprime: float,
    theta0: float = 40,
    mh: float = 0,
    bh: float = 0,
    mw: float = 0,
    bw: float = 0,
    monsunfactors: Iterable[float] = None,
    *,
    kw: float,
) -> Callable[[pd.Series], pd.Series]:
    """
    Create a function to calculate gas consumption as function of temperature.

    Parameters
    ----------
    A, B, C, Dprime, theta0 : float
        Parameters to the sigmoid part of the standard function.
    mh, bh, mw, bw: float, optional
        Parameters to the linear part of the standard function.
    monsunfactors : Iterable of floats, optional
        Multiplication factor for each day of the week (Mon-Sun). If none: 1 for each.
    kw : float
        'Kundenwert'. The offtake [kWh/day] at 8 degC, averaged over all weekdays.

    Returns
    -------
    Callable[[pd.Series], pd.Series]
        Function that takes a temperature [degC] timeseries as input and
        returns the consumption [MW] timeseries as output.
    """

    if monsunfactors is None:
        monsunfactors = [1] * 7

    def fwt(ts: dt.datetime) -> float:
        return monsunfactors[ts.weekday()]

    def h(theta: float) -> float:
        sig = (A / (1 + (B / (theta - theta0)) ** C)) + Dprime  # sigmoid part
        lin = max([mh * theta + bh, mw * theta + bw])  # linear part
        return (sig + lin) * 0.001 / 24  # kWh/d --> MW

    def offtake(t: float, ts: dt.datetime) -> float:
        return kw * h(t) * fwt(ts)

    return convert.function2function(offtake)


def fromsource(code: Union[str, int], *, kw: float) -> Callable[[pd.Series], pd.Series]:
    """
    Standardized temperature-dependent load profile.

    Parameters
    ----------
    code : Union[str, int]
        Code (e.g. 'BD4'), or its index position (>= 0), of a TLP in the source file.
    kw : float
        'Kundenwert'. The offtake [kWh/day] at 8 degC, averaged over all weekdays.

    Returns
    -------
    Callable[[pd.Series], pd.Series]
        Function that takes a temperature [degC] timeseries as input and
        returns the consumption [MW] timeseries as output.
    """
    df = pd.read_excel(sheet_name="TLP", header=4, io=SOURCEPATH)
    df = df.set_index("code")
    if isinstance(code, str):
        try:
            v = df.loc[code]
        except KeyError:
            raise ValueError(
                f"`code` must be a TLP code or its index position in the following list: {df.index.to_list()}"
            )
    else:
        try:
            v = df.iloc[code]
        except IndexError:
            raise ValueError(
                f"`code` must be a TLP code or its index position in the following list: {df.index.to_list()}"
            )
    f = [v.Mo, v.Di, v.Mi, v.Do, v.Fr, v.Sa, v.So]
    return fromscratch(v.A, v.B, v.C, v.D, v.theta0, v.mH, v.bH, v.mW, v.bW, f, kw=kw)


def D14(*, kw: float) -> Callable:
    return fromsource("D14", kw=kw)


def HD4(*, kw: float) -> Callable:
    return fromsource("HD4", kw=kw)


def weights() -> pd.DataFrame:
    """
    Weights of the individual TLPs in the lichtblick portfolio.
    
    Parameters
    ----------
    None

    Returns
    -------
    pd.DataFrame
        with TLP codes as index, and customer segment as columns.
    """
    df = pd.read_excel(sheet_name="weights", header=0, io=SOURCEPATH)
    df = df.set_index("code")
    return df
