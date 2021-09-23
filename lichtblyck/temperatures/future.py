# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:42:08 2020

@author: ruud.wijtvliet
"""

from typing import Iterable, Union
import pandas as pd
import numpy as np
from .sourcedata.climate_zones import futuredata, forallzones
from . import historic


def climate_data(climate_zone: int) -> pd.DataFrame:
    """Return dataframe with future daily climate data for specified climate
    zone. Column names not standardized, but rather as found in source data."""
    # Get file content and turn into dataframe...
    file = futuredata(climate_zone)
    with open(file) as f:
        for line in f:
            line = line.strip()
            if len(line):
                zerolen = False
            elif not zerolen:
                zerolen = True
            else:  # Two empty lines in a row: now the table starts.
                df = pd.read_csv(f, delim_whitespace=True, skiprows=[1])
                break
        else:
            raise ValueError(
                f"Couldn't find 2 empty lines (that mark table start) in {file.name}."
            )
    return df.set_index(["MM", "DD", "HH"])


# Series (res = 1 day, len = 1 year) with temperatures.
def _tmpr_1year(climate_zone: int) -> pd.Series:
    """
    Return the forecast future daily temperature for specified climate zone.

    Returns:
        Dataframe with daily temperature values. Index: (MM, DD).  Values:
            average temperature at corresponding day in degC.
    """
    df = climate_data(climate_zone)
    s = df["t"].groupby(["MM", "DD"]).mean().rename("t")
    if (2, 29) not in s.index:  # Add 29 feb.
        s.loc[(2, 29)] = s[s.index.map(lambda idx: idx[0] == 2)].mean()
    return s.sort_index()


def tmpr_1year() -> pd.DataFrame:
    """
    Return the forecast future daily temperature each climate zone.

    Returns:
        Dataframe with daily temperature values. Index: (MM, DD).  Values:
            average temperature at corresponding day in degC.
    """
    return forallzones(_tmpr_1year)


# Series (res = 1 day, len = several years)
def tmpr_concat(
    tmpr_1year: Union[pd.Series, pd.DataFrame], year_start: int, year_end: int
) -> Union[pd.Series, pd.DataFrame]:
    """
    Turn single temperature profile 'tmpr_1year' (res = d, len = 1 year)
    into a temperature timeseries (res = d, len = 'year_start'
    until (and including) 'year_end') by repetition.

    'tmpr_1year': Series with (month-of-year, day-of-month) index.
    """
    idxTs = pd.date_range(
        start=pd.Timestamp(year=year_start, month=1, day=1),
        end=pd.Timestamp(year=year_end + 1, month=1, day=1),
        closed="left",
        freq="D",
        tz="Europe/Berlin",
    )
    idxMD = idxTs.map(lambda ts: (ts.month, ts.day)).rename(["MM", "DD"])

    s = tmpr_1year.copy()
    s.index.rename(["MM", "DD"], inplace=True)
    if (2, 29) not in s.index:  # Add 29 feb.
        s.loc[(2, 29)] = s[s.index.map(lambda idx: idx[0] == 2)].mean()

    tmpr = s.loc[idxMD]
    tmpr.set_axis(idxTs, inplace=True)
    tmpr.index.rename("ts_left", inplace=True)
    return tmpr


def tmpr_standardized() -> pd.DataFrame:
    """
    Return standardized temperature year (res=1d, len=2020-2030) for 15 climate zones.
    """
    # Get historic daily temperatures for each climate zones...
    tmpr_hist = (
        historic.tmpr()
    )  # NB: may contain gaps e.g. due to broken weather station.
    # ...keep only 2005-2019...
    tmpr_hist = tmpr_hist[(tmpr_hist.index >= "2005") & (tmpr_hist.index < "2020")]
    idx = tmpr_hist.index.map(lambda ts: (ts.year, ts.month)).set_names(["YY", "MM"])
    tmpr_hist = tmpr_hist.set_index(idx)
    # ...and find the monthly averages.
    tmpr2012 = tmpr_hist.groupby("MM").mean()

    # Also find the future monthly averages.
    tmpr2045 = tmpr_1year()
    tmpr2045 = tmpr2045.groupby("MM").mean()  # Per month and climate zone.

    # As well as the structure added to each month.
    tmprstruct = historic.tmpr_struct(
        2005, 2019, np.std
    )  # or np.mean, to get the pfms/pfmg year

    # Finally, combine into a daily time series with 'standardized' temperatures.
    year_start = 2020
    year_end = 2030
    idxTs = pd.date_range(
        start=pd.Timestamp(year=year_start, month=1, day=1),
        end=pd.Timestamp(year=year_end + 1, month=1, day=1),
        closed="left",
        freq="D",
        tz="Europe/Berlin",
        name="ts_left",
    )
    idxY = idxTs.map(lambda ts: ts.year).rename("YY")
    idxM = idxTs.map(lambda ts: ts.month).rename("MM")
    idxMD = idxTs.map(lambda ts: (ts.month, ts.day)).rename(["MM", "DD"])
    factor2045 = pd.Series(
        (np.arange(2012, 2046) - 2012) / (2045 - 2012), index=range(2012, 2046)
    ).rename_axis("YY")
    f = factor2045.loc[idxY]
    f.index = idxTs
    stdz_tmpr = (
        tmprstruct.loc[idxMD].set_index(idxTs)
        + tmpr2012.loc[idxM].set_index(idxTs).multiply(1 - f, axis=0)
        + tmpr2045.loc[idxM].set_index(idxTs).multiply(f, axis=0)
    )

    return stdz_tmpr
