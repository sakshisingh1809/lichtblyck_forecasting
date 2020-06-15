# -*- coding: utf-8 -*-
"""
Module used for reading historic temperature data from disk.

Source: DWD
https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/

"""

import pandas as pd
from io import StringIO
from lichtblick.tools.tools import set_ts_index
from lichtblick.temperatures.sourcedata.climate_zones import historicdata

def climate_data(climate_zone:int) -> pd.DataFrame:
    """Return dataframe with historic daily climate data for specified climate zone."""
    # Get file content and turn into dataframe...
    bytes_data = historicdata(climate_zone)
    data = StringIO(str(bytes_data, 'utf-8'))
    df = pd.read_csv(data, sep=";")
    # ...then do some cleaning up...
    df.columns = df.columns.str.strip()
    df.drop('eor', axis=1, inplace=True)
    df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'], format='%Y%m%d')
    df = set_ts_index(df, 'MESS_DATUM', 'left')
    df.index.freq = pd.infer_freq(df.index) # TODO: this is currently not working; freq == None
    return df

# Series (res = 1 day, len = several years) with temperatures.
def tmpr(climate_zone:int) -> pd.Series:
    """    
    Return the daily temperatures for the specified climate zone.
    returns: pandas Series with average temperature values (in [degC]) and 
        datetime row index, in daily resolution.
    """
    df = climate_data(climate_zone)
    s = df['TMK'].rename('tmpr')
    # Keep correct values.
    s = s[s>-950]
    return s
