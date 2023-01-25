"""Create timeseries with norm temperatures for past and future dates."""

from .sourcedata.climate_zones import norm_fourier_coefficients, forallzones
from ..tools import stamps

from typing import Union
import pandas as pd
import numpy as np
import datetime as dt


def _tmpr(climate_zone: int, ts_left, ts_right) -> pd.Series:
    """Return timeseries with norm daily climate data for specified climate zone,
    from ``ts_left`` (inclusive) to ``ts_right`` (exclusive). The timeseries has the
    same timezone as ``ts_left`` and ``ts_right``.

    Returns
    -------
    Series
        With daily temperature values. Index: timestamp (daily). Values: norm
        temperature at corresponding day in degC.
    """

    # Get coefficients...
    coeff = norm_fourier_coefficients(climate_zone)

    # ...index and t-values (=years since 2000)...
    i = pd.date_range(ts_left, ts_right, freq="D", closed="left")
    t = (i - pd.Timestamp("2000-01-01", tz=i.tz)) / pd.Timedelta(days=365.25)

    # ...and use to calcutae values...
    values = (
        coeff["a_0"]
        + coeff["a_1"] * t
        + coeff["a_2"] * np.cos(1 * 2 * np.pi * (t - coeff["a_3"]))
        + coeff["a_4"] * np.cos(2 * 2 * np.pi * (t - coeff["a_5"]))
        + coeff["a_6"] * np.cos(3 * 2 * np.pi * (t - coeff["a_7"]))
        + coeff["a_8"] * np.cos(4 * 2 * np.pi * (t - coeff["a_9"]))
    )

    return pd.Series(values, i, name="t")


def tmpr(
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Return the expected (i.e., norm) daily temperature for each climate zone.

    Parameters
    ----------
    ts_left, ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        Start and end of time period (left-closed).

    Returns
    -------
    Dataframe
        With daily temperature values. Index: timestamp (daily). Columns: climate zones
        (1..15). Values: norm temperature for corresponding day and climate zone in
        degC.
    """
    # Fix timestamps (if necessary).
    ts_left, ts_right = stamps.ts_leftright(ts_left, ts_right)
    return forallzones(lambda cz: _tmpr(cz, ts_left, ts_right))
