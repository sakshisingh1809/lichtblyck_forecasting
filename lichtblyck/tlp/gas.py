# -*- coding: utf-8 -*-
"""
Standardized temperature load profiles for gas consumption.
"""

from . import convert
from typing import Callable


def fromscratch(A: float, B: float, C: float, Dprime: float, *, kw: float) -> Callable:
    """
    Create a function to calculate gas consumption as function of temperature.

    Parameters
    ----------
    A, B, C, Dprime : float
        Parameters to the standard function.
    kw : float
        'Kundenwert', i.e., value that determines the yearly (temperature-
        corrected) consumption.

    Returns
    -------
    Callable[[pd.Series], pd.Series]
        Function that takes a temperature [degC] timeseries as input and
        returns the consumption [MW] timeseries as output.
    """

    def f(t: float) -> float:  # original formula gives kWh per day, so kWh/day --> MW
        return kw * ((A / (1 + (B / (t - 40)) ** C)) + Dprime) * 0.001 / 24

    return convert.function2function(f)


def D14(*, kw: float) -> Callable:
    """
    Create function to calculate gas consumption of D14-profile.

    Notes
    -----
    See `fromscratch` documentation for parameters and return value.
    """
    return fromscratch(3.1850191, -37.4124155, 6.1723179, 0.07610960000, kw=kw)
