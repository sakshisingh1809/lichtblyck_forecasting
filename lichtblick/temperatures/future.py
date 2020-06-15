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

# Series (res = 1 day, len = 1 day) with temperatures.
def tmpr(climate_zone:int) -> pd.Series:
    """    
    Return the daily temperatures for the specified climate zone.
    returns: pandas Series with average temperature values (in [degC]) and 
        datetime row index, in daily resolution.
    """
    df = climate_data(climate_zone)
    s = df.groupby(['MM', 'DD']).mean()['t'].rename('tmpr')
    if (2, 29) not in s.index:    # Add 29 feb.
        s.loc[(2, 29)] = s[s.index.map(lambda idx: idx[0] == 2)].mean()
    return s.sort_index()