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

def tmpr(climate_zone:int) -> pd.Series:
    """
    Return standardized temperature for the specified climate zone (1-15).
    returns: pandas Series with 'prognosed' temperature values (in [degC]) and 
        datetime row index, in daily resolution, for future years until 2030.
    """
    df = pd.read_excel("lichtblick/temperatures/sourcedata/standard_tmpr_year.xlsx",
                       sheet_name=1, header=1)
    df = set_ts_index(df, df.columns[0], 'left')
    return df[climate_zone].rename('tmpr')

def tmpr_concat(tmpr_1year:Iterable[float], year_start:int, year_end:int) -> pd.Series:
    """
    Turn single temperature profile 'tmpr_1year' (res = d, len = 365 or 366
    days) into a temperature timeseries (res = d, len = 'year_start'
    until (and including) 'year_end') by repetition.
                                                   
    'tmpr_1year': Iterable of floats with length 365 or 366. (If 365 values are
        supplied, last value is repeated for leap years. If 366 values are 
        supplied, last value is left out for non-leap years).
    """
    ts_idx = pd.date_range(start=pd.Timestamp(year=year_start, month=1, day=1),
                           end=pd.Timestamp(year=year_end+1, month=1, day=1),
                           closed='left', freq='D', tz='Europe/Berlin')
    row_idx = np.minimum(ts_idx.dayofyear - 1, len(tmpr_1year) - 1)
    return pd.Series(data=tmpr_1year[row_idx].values, index=ts_idx, name='tmpr')


def climate_data(climate_zone:int) -> pd.DataFrame:
    """Return dataframe with future daily climate data for specified climate zone."""
    # Get file content and turn into dataframe...
    file = futuredata(climate_zone)
    for line in open(file):
        line = line.strip()
        if len(line):
            zerolen = False
        elif not zerolen:
            zerolen = True
        else: #Two empty lines in a row: now the table starts.
            df = pd.read_csv(f, delim_whitespace=True, skiprows=[1])
            break

            
            
            
            
    
    data = StringIO(str(bytes_data, 'utf-8'))
    df = pd.read_csv(file, header=31, sep=' ')
    # ...then do some cleaning up...
    df.columns = df.columns.str.strip()
    df.drop('eor', axis=1, inplace=True)
    df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'], format='%Y%m%d')
    df = set_ts_index(df, 'MESS_DATUM', 'left')
    df.index.freq = pd.infer_freq(df.index) # TODO: this is currently not working; freq == None
    return df

#%%

same = 0
diff = 0
for r1, r2 in zip(df['HH'][1:], df['HH'][:-1]):
    if r1 == r2: 
        same += 1
    else:
        diff +=1