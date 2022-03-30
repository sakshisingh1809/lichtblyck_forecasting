"""Create timeseries with norm temperatures for future dates."""

import pandas as pd
import numpy as np
from .sourcedata.climate_zones import futurefourierdata, forallzones


def _tmpr(climate_zone: int, ts_left, ts_right) -> pd.Series:
    """Return timeseries with future daily climate data for specified climate zone,
    from ``ts_left`` (inclusive) to ``ts_right`` (exclusive). The timeseries has the
    same timezone as ``ts_left`` and ``ts_right``.

    Returns
    -------
    Series
        With daily temperature values. Index: timestamp (daily). Values: expected
        temperature at corresponding day in degC.
    """

    # Get coefficients...
    coeff = futurefourierdata(climate_zone)

    # ...index and t-values (=years since 2000)...
    i = pd.date_range(ts_left, ts_right, freq="D")
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


def tmpr(ts_left, ts_right) -> pd.DataFrame:
    """
    Return the expected future daily temperature for each climate zone.

    Returns
    -------
    Dataframe
        With daily temperature values. Index: timestamp (daily). Columns: climate zones
        (1..15). Values: expected temperature for corresponding day and climate zone in
        degC.
    """
    return forallzones(lambda cz: _tmpr(cz, ts_left, ts_right))
