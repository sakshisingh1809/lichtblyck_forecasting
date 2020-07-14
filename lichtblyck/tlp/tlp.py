# -*- coding: utf-8 -*-
"""
Reading standardized temperature load profiles;
Converting them into loads with help of a temperature timeseries.
"""

import pandas as pd
import datetime
import numpy as np
from typing import Union

SOURCEPATH = 'lichtblyck/tlp/sourcedata/'
SOURCES = [
    {'name': 'avacon_hz0',     'io': SOURCEPATH + 'AVACON_HZ0.xlsx'},
    {'name': 'avacon_hzs',     'io': SOURCEPATH + 'AVACON_HZS.xlsx'},
    {'name': 'bayernwerk_nsp', 'io': SOURCEPATH + 'BAYERNWERK_NSP.xlsx'},
    {'name': 'bayernwerk_wp',  'io': SOURCEPATH + 'BAYERNWERK_WP.xlsx'},
    {'name': 'edis_n21',       'io': SOURCEPATH + 'EDIS_N21.xlsx'},
    {'name': 'edis_w21',       'io': SOURCEPATH + 'EDIS_W21.xlsx'},
    {'name': 'enmitte_enm',    'io': SOURCEPATH + 'ENMITTE_ENM.xlsx'},
    {'name': 'netzbw_ez2',     'io': SOURCEPATH + 'NETZBW_EZ2.xlsx'},
    {'name': 'shnetz_e1',      'io': SOURCEPATH + 'SHNETZ_E1.xlsx'},
    {'name': 'shnetz_e2',      'io': SOURCEPATH + 'SHNETZ_E2.xlsx'},
    {'name': 'shnetz_w1',      'io': SOURCEPATH + 'SHNETZ_W1.xlsx'},
    {'name': 'westnetz_wk2',   'io': SOURCEPATH + 'WESTNETZ_WK2.xlsx'},
    {'name': 'wwnetz_nsp',     'io': SOURCEPATH + 'WWNETZ_NSP.xlsx'}
]

# Series (res = qh, len = 1 day) with consumption for various temperatures.
def standardized_tmpr_loadprofile(source:Union[str, int]) -> pd.Series:
    """
    Return the standardized temperature load profile ('Normiertes Lastprofil') 
    as specified by 'source'.
    
    source: index position, or value of 'name'-key, for any of the dictionaries
        in the SOURCES list.
    
    returns: pandas Series with load values (in [K/h]), as function of 2-level
        row index (time of day, temperature), in qh resolution.
    """
    try:
        if isinstance(source, str):
            io = [s for s in SOURCES if s['name'] == source][0]['io']
        else:
            io = SOURCES[source]['io']
    except:
        raise ValueError("Value for argument 'source' must be one of {" + 
                         ", ".join([s['name'] for s in SOURCES]) +
                         "}, or the index position of one of them.")
    #Requirements/assumptions for any tlp excel file:
    #. Have column called 'Uhrzeit' with 15min resolution and right-bound timestamp.
    df = pd.read_excel(**{'header':0, 'sheet_name':0, 'io': io})
    df['ts_right_local'] = pd.to_datetime(df['Uhrzeit'], format='%H:%M:%S')
    df['time_left_local'] = (df['ts_right_local'] + datetime.timedelta(hours=-0.25)).dt.time
    df = df.drop(columns=['Nr.', 'Uhrzeit', 'ts_right_local'])
    # Put in correct output format (long table).
    df = pd.melt(df, id_vars=['time_left_local'], var_name='tmpr', value_name='std_tmpr_lp')
    df = df.set_index(['time_left_local', 'tmpr'])
    return df['std_tmpr_lp']

# Function to convert temperature into load using TLP.
def tmpr2load(std_tmpr_lp:pd.Series, tmpr:pd.Series, spec:float) -> pd.Series:
    """
    Turn temperature timeseries 'tmpr' (with res = 1d) into a load timeseries,
    with help of standardized temperature load profile series 'std_tmpr_lp'. The 
    resulting timeseries has same resolution as 'std_tmpr_lp', and same length 
    as 'tmpr'.
    
    Parameters:
    'std_tmpr_lp': Series with multilevel-index. level 0: time-of-day timestamp.
        Level 1: temperature in [degC]. Values: load (in [K/h]) at given time 
        and temperature.
    'tmpr': Series with temperature values (in [degC]). Index = date timestamp.
    'spec': Specific electrical load [kWh/K] with which to scale the profile.
        It describes the heating energy needed by the customer during a single
        day, per degC that the average outdoor temperature of that day is
        below a certain reference value.
    
    returns: 
        Timeseries with the electrical load in MW.
    """
    available_tmpr = std_tmpr_lp.index.get_level_values('tmpr')
    def nearest_available(tmpr):
        idx = (np.abs(available_tmpr - tmpr)).argmin()
        return available_tmpr[idx]
    
    new_timestamps = pd.date_range(
        tmpr.index[0], tmpr.index[-1] + datetime.timedelta(1), 
        freq='15T', closed='left', tz='Europe/Berlin') #TODO: change freq to be same as 'tlp'
    
    # Put into correct time resolution.
    df = pd.DataFrame({'tmpr': tmpr})
    df['tmpr_avail'] = df['tmpr'].apply(nearest_available)
    # df['date'] = df.index.map(lambda ts: ts.date)
    df = df.reindex(new_timestamps).ffill()
    df['time'] = df.index.map(lambda ts: ts.time)
    
    # Add corresponding standardized load.
    merged = df.merge(std_tmpr_lp, left_on=('time', 'tmpr_avail'), right_index=True)
    merged = merged[['tmpr', 'std_tmpr_lp']]
    merged.sort_index(inplace=True)
    
    # Convert into actual power.
    load = merged['std_tmpr_lp'] * spec * 0.001 #kW to MW
    load.index.freq = pd.infer_freq(load.index)
    return load.rename('w')