# -*- coding: utf-8 -*-
"""
Module used for reading historic temperature data from disk.

Source: DWD
https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/

"""

from sourcedata.climate_zones import historicdata, forallzones
from ..tools.frames import set_ts_index
from ..tools import stamps, frames
from sklearn.linear_model import LinearRegression
from typing import Callable, Union
from datetime import datetime
from io import StringIO
from pathlib import Path
import pandas as pd
import numpy as np
import datetime as dt


def climate_data(climate_zone: Union[int, Path]) -> pd.DataFrame:
    """Return dataframe with historic daily climate data for specified climate zone (if
    int) or from specified file (if path). Values before 1917 are dropped. Index is
    gapless with missing values filled with np.nan. Column names not standardized but
    rather as found in source file. Index is timezone-agnostic, because UTC-offset of
    old values is not same as UTC-offset of modern values."""

    # Get file content and turn into dataframe...
    bytes_data = historicdata(climate_zone)
    data = StringIO(str(bytes_data, "utf-8"))
    df = pd.read_csv(data, sep=";")
    # ...then do some cleaning up...
    df.columns = df.columns.str.strip()
    df.drop("eor", axis=1, inplace=True)
    df.replace(-999, np.nan, inplace=True)  # -999 represents missing value
    df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d")
    df = df[df["MESS_DATUM"] >= "1917"]  # Problems with earlier data.
    # ...and set correct index and make gapless.
    df = set_ts_index(df, "MESS_DATUM", "left", continuous=False).tz_localize(None)
    df = df.resample("D").asfreq()  # add na-values for missing rows.
    return df


def _tmpr(climate_zone: int, ts_left, ts_right) -> pd.Series:
    """Return timeseries with historic daily climate data for specified climate zone,
    from ``ts_left`` (inclusive) to ``ts_right`` (exclusive). The timeseries has the
    same timezone as ``ts_left`` and ``ts_right``.

    Returns
    -------
    Series
        With daily temperature values. Index: timestamp (daily). Values: average
        temperature at corresponding day in degC.
    """
    df = climate_data(climate_zone)
    s = df["TMK"].rename("t")
    if s.index.tz != ts_left.tz:
        s = s.tz_localize(ts_left.tz)
    mask = (s.index >= ts_left) & (s.index < ts_right)
    return s[mask]


def tmpr(
    ts_left: Union[str, dt.datetime, pd.Timestamp] = None,
    ts_right: Union[str, dt.datetime, pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Return the daily temperatures for each climate zone.

    Parameters
    ----------
    ts_left, ts_right : Union[str, dt.datetime, pd.Timestamp], optional
        Start and end of time period (left-closed).

    Returns
    -------
    Dataframe
        With daily temperature values. Index: timestamp (daily). Columns: climate zones
        (1..15). Values: average temperature for corresponding day and climate zone in
        degC.
    """
    # Fix timestamps (if necessary).
    ts_left, ts_right = stamps.ts_leftright(ts_left, ts_right)
    return forallzones(lambda cz: _tmpr(cz, ts_left, ts_right))


def _tmpr_monthlyavg(climate_zone: int) -> pd.Series:
    """
    Return monthly average temperatures for the specified climate zone.

    Returns:
        Series with average temperature values. Index: timestamp (monthly).
            Values: average temperatures for corresponding month in degC.
    """
    s = _tmpr(climate_zone)
    # TODO: deal with with nan-values.
    return s.resample("MS").mean()


def tmpr_monthlyavg() -> pd.DataFrame:
    """
    Return monthly average temperatures for each climate zone.

    Returns:
        Dataframe with average temperature values. Index: timestamp (monthly).
            Columns: climate zones (1..15). Values: average temperatures for
            corresponding month and climate zone in degC.
    """
    return forallzones(_tmpr_monthlyavg)


def _tmpr_monthlymovingavg(climate_zone: int, window: int = 5) -> pd.Series:
    """
    Return monthly moving average temperatures for the specified climate zone.
    If e.g. 'window'==10, the value for May 2020 is the average May temperature
    in the time period from 2010 until and including 2019.

    Returns:
        Series with moving average temperature values. Index: timestamp (monthly).
            Values: average temperature in preceding 'window' years for that
            month-of-year in degC.
    """
    t = _tmpr_monthlyavg(climate_zone)
    mavg = t.groupby(t.index.map(lambda ts: ts.month)).rolling(window=window).mean()
    # Do corrections.
    mavg.index = mavg.index.droplevel(0)  # drop month-level that was added
    mavg = mavg.sort_index()  # put back into chronological order
    mavg = mavg.iloc[(window - 1) * 12 :]  # drop first few months
    mavg.index = mavg.index + pd.offsets.DateOffset(years=1)  # add one year
    mavg.index.freq = pd.infer_freq(mavg.index)
    return mavg


def tmpr_monthlymovingavg(window: int = 5) -> pd.DataFrame:
    """
    Return monthly moving average temperatures for each climate zone.
    If e.g. 'window'==10, the value for May 2020 is the average May temperature
    in 2010 until and including 2019.

    Returns:
        Dataframe with moving average temperature values. Index: timestamp
            (monthly). Columns: climate zones (1..15). Values: average
            temperature in preceding 'windows' years for that month-of-year
            in degC.
    """
    return forallzones(lambda cz: _tmpr_monthlymovingavg(cz, window))


def tmpr_struct(
    year_start: int = 2005, year_end: int = 2019, f: Callable = np.std
) -> pd.DataFrame:
    """
    Return a year with a typical temperature *structure* for each climate
    zone. The monthly average is zero, so, when using the result to create
    actual temperatures, the wanted monthly averages for each zone must be
    added.

    Arguments:
        year_start, year_end: interval (closed on both sides) used to calculate
          the standardized structure.
        f: function used to determine the representative year for each month-
          of-year. E.g.: mean or standard-deviation. For each month-of-year,
          that year is chosen, whose aggregate value is closest to the aggre-
          gate value of the entire interval.

    Returns:
        Dataframe with temperature structure. Index: (month-of-year,
            day-of-month). Columns: climate zones (1..15). Values: temperature
            deviation from month average in degC.
    """
    # Get historic daily temperatures for each climate zones...
    t = tmpr()
    # NB: may contain gaps e.g. due to broken weather station.

    # ...keep only wanted interval...
    t = t[
        (t.index >= pd.Timestamp(datetime(year_start, 1, 1), tz="Europe/Berlin"))
        & (t.index < pd.Timestamp(datetime(year_end + 1, 1, 1), tz="Europe/Berlin"))
    ]

    # ...and find the most representative year for each month.
    #    1: calculate monthly aggregate.
    yymm = t.groupby([t.index.month, t.index.year]).apply(f)
    yymm.index.rename(["MM", "YY"], inplace=True)
    mm = yymm.groupby("MM").mean()

    #    2: calculate geographic average; weigh with consumption / customer presence in each zone
    weights = pd.DataFrame(
        {
            "power": [
                60,
                717,
                1257,
                1548,
                1859,
                661,
                304,
                0,
                83,
                61,
                0,
                1324,
                1131,
                0,
                21,
            ],
            "gas": [
                729,
                3973,
                13116,
                28950,
                13243,
                3613,
                2898,
                0,
                1795,
                400,
                0,
                9390,
                3383,
                9,
                113,
            ],
        },
        index=range(1, 16),
    )  # MWh/a in each zone
    weights = (
        weights["power"] / weights["power"].sum()
        + weights["gas"] / weights["gas"].sum()
    )
    yymm["t_germany"] = frames.wavg(yymm, weights, axis=1)
    mm["t_germany"] = frames.wavg(mm, weights, axis=1)
    #    3: compare to, for each month, find year with lowest deviation from the long-term average
    yymm["t_delta"] = yymm.apply(
        lambda row: row["t_germany"] - mm["t_germany"][row.name[0]], axis=1
    )
    idx = yymm["t_delta"].groupby("MM").apply(lambda df: df.apply(abs).idxmin())
    bestfit = yymm.loc[idx, "t_germany":"t_delta"]

    # Then, create single representative year from these individual months...
    keep = t.index.map(lambda idx: (idx.month, idx.year) in bestfit.index)
    repryear = t[keep]
    repryear.index = repryear.index.map(lambda ts: (ts.month, ts.day)).set_names(
        ["MM", "DD"]
    )
    repryear.index.rename(["MM", "DD"], inplace=True)
    if (2, 29) not in repryear.index:  # add 29 feb if doesn't exist yet.
        toadd = pd.Series(
            repryear[repryear.index.map(lambda idx: idx[0] == 2)].mean(), name=(2, 29)
        )
        repryear = repryear.append(toadd)
    repryear = repryear.sort_index()

    # ... and return (zero-averaged) structure.
    struct = repryear - repryear.groupby("MM").mean()
    return struct


def fill_gaps(t: pd.DataFrame) -> pd.DataFrame:
    """Fills gaps in temperature dataframe by comparing one climate zone to all others.

    Parameters
    ----------
    t : pd.DataFrame
        Dataframe with temperature timeseries. Each column is different geographic location.

    Returns
    -------
    pd.DataFrame
        Temperature dataframe with (some) gaps filled.
    """
    # Keep only days with at most 1 missing climate zone.
    # remove days with >1 missing value. (.copy() only needed to stop 'A value is trying to be set on a copy of a slice' warning.)
    t = t[t.isna().sum(axis=1) < 2].copy()

    # For each missing value, get estimate. Using average difference to other stations' values.
    complete = t.dropna()  # all days without any missing value
    for col in t:
        isna = t[col].isna()
        if not isna.any():
            continue
        x_fit = complete.drop(col, axis=1).mean(axis=1).values.reshape(-1, 1)
        y_fit = complete[col].values.reshape(-1, 1)
        model = LinearRegression().fit(x_fit, y_fit)  # perform linear regression
        x = t.drop(col, axis=1).loc[isna].mean(axis=1).values.reshape(-1, 1)
        y_pred = model.predict(x)
        t.loc[isna, col] = y_pred.reshape(-1)
    return t
