# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:42:08 2020

@author: ruud.wijtvliet
"""

from typing import Iterable
import pandas as pd
import numpy as np
from lichtblick.tools.tools import set_ts_index
from lichtblick.temperatures.sourcedata.climate_zones import futuredata 

def climate_data(climate_zone:int) -> pd.DataFrame:
    """Return dataframe with future daily climate data for specified climate zone."""
    # Get file content and turn into dataframe...
    file = futuredata(climate_zone)
    with open(file) as f:
        for line in f:
            line = line.strip()
            if len(line):
                zerolen = False
            elif not zerolen:
                zerolen = True
            else: #Two empty lines in a row: now the table starts.
                df = pd.read_csv(f, delim_whitespace=True, skiprows=[1])
                break
        else:
            raise ValueError(f"Couldn't find 2 empty lines (that mark table start) in {file.name}.")
    return df

# Series (res = 1 day, len = 1 year) with temperatures.
def tmpr(climate_zone:int) -> pd.Series:
    """    
    Return the daily temperatures for the specified climate zone.
    returns: pandas Series with average temperature values (in [degC]) and 
        (month-of-year, day-of-month) row index.
    """
    df = climate_data(climate_zone)
    s = df.groupby(['MM', 'DD']).mean()['t'].rename('tmpr')
    if (2, 29) not in s.index:    # Add 29 feb.
        s.loc[(2, 29)] = s[s.index.map(lambda idx: idx[0] == 2)].mean()
    return s.sort_index()


# Series (res = 1 day, len = several years)
def tmpr_concat(tmpr_1year:pd.Series, year_start:int, year_end:int) -> pd.Series:
    """
    Turn single temperature profile 'tmpr_1year' (res = d, len = 1 year)
    into a temperature timeseries (res = d, len = 'year_start'
    until (and including) 'year_end') by repetition.
                                                   
    'tmpr_1year': Series with (month-of-year, day-of-month) index.
    """
    idxTs = pd.date_range(start=pd.Timestamp(year=year_start, month=1, day=1),
                          end=pd.Timestamp(year=year_end+1, month=1, day=1),
                          closed='left', freq='D', tz='Europe/Berlin')
    idxMD = idxTs.map(lambda ts: (ts.month, ts.day)).rename(['MM', 'DD'])

    s = tmpr_1year.copy()
    s.index.rename(['MM', 'DD'], inplace=True)
    if (2, 29) not in s.index:    # Add 29 feb.
        s.loc[(2, 29)] = s[s.index.map(lambda idx: idx[0] == 2)].mean()
        
    tmpr = s.loc[idxMD]
    tmpr.set_axis(idxTs, inplace=True)
    tmpr.index.rename('ts_left', inplace=True)
    return tmpr
