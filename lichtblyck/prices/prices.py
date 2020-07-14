# -*- coding: utf-8 -*-
"""
Module to read price data from disk.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lichtblyck import tools
from sqlalchemy import create_engine

SPOTPRICEFILE = 'lichtblyck/prices/sourcedata/spot.tsv'

def spot() -> pd.Series:
    """Return spot price timeseries."""
    data = pd.read_csv(SPOTPRICEFILE, header=None, sep='\t', 
                       names=['date', 'time', 'price', 'anmerkungen', 'empty'])
    data['ts_right'] = pd.to_datetime(data["date"] + " " + data["time"], format="%d.%m.%Y %H:%M:%S")
    #Replace missing values with NaN, convert others to float:
    data['p_spot'] = data['price'].apply(lambda x: np.NaN if x == '---' else float(x.replace(',', '.'))) 
    spot = tools.set_ts_index(data, 'ts_right', 'right')
    return spot['p_spot']


def futures(prod:str='y') -> pd.DataFrame:
    """
    Return futures prices timeseries.
    
    Arguments:
        prod: product to return the prices of. One of {'y', 'q', 'm'}.
        
    Returns: 
        Dataframe with EEX futures prices. Index: Multiindex (with levels 0: 
            date of trading, 1: product {'y', 'q', 'm'}, 2: time stamp of delivery
            start). Columns: p_base (base price), p_peak (peak price), 
            p_offpeak (offpeak price); Values in Eur/MWh. deliv_ts_right (time
            stamp of delivery end). basehours, peakhours, offpeakhours in delivery
            period; int values.
    """
    engine = create_engine('mssql+pymssql://sqlbiprod/__Staging')
    
    try:
        dur_months = {'m':1, 'q':3, 'y':12}[prod]
    except KeyError:
        raise ValueError("Argument 'prod' must be in {'y', 'q', 'm'}.")
    
    # Get values from server...
    df = pd.read_sql_query("SELECT * FROM dbo.Fakten_EW_EEX_Preise_fkunden " +
                           f"WHERE Typ='{prod.upper()}'", engine)
    # ...massage to get correct format...
    df = df.rename(columns={'Typ': 'deliv_prod', 'Peak_Preis': 'p_peak',
                            'Base_Preis': 'p_base', 'EEX_Handelstag': 'trade_ts_left',
                            'Lieferzeitraum': 'deliv_ts_left'})
    df = df.drop('ID', axis = 1)
    df['trade_ts_left'] = pd.to_datetime(df['trade_ts_left']).dt.tz_localize('Europe/Berlin')
    df['deliv_ts_left'] = pd.to_datetime(df['deliv_ts_left']).dt.tz_localize('Europe/Berlin')
    df['deliv_ts_right'] = df['deliv_ts_left'] + pd.offsets.DateOffset(months=dur_months)
    df = df.set_index(['deliv_prod', 'deliv_ts_left', 'trade_ts_left'])
    # ...get peak and base and offpeak hours...
    #TODO: group by deliv_ts_left and deliv_ts_right to reduce number of calculations.
    bpop = _basepeakoffpeak(df.index.get_level_values('deliv_ts_left'), df['deliv_ts_right'])
    df[['basehours', 'peakhours', 'offpeakhours']] = pd.DataFrame(np.transpose(bpop), index=df.index)
    # ...and use to calculate offpeak prices.
    df['p_offpeak'] = _offpeakprice(df['p_base'], df['p_peak'], df['basehours'], df['peakhours'])
    # Finally, return in correct row and column order.
    return df[['deliv_ts_right', 'p_base', 'p_peak', 'p_offpeak', 'basehours', 'peakhours', 'offpeakhours']].sort_index()

@np.vectorize
def _basepeakoffpeak(ts_left, ts_right) -> float:
    """Return number of base, peak and offpeak hours in interval [ts_left, ts_right).
    Timestamps must coincide with day start."""
    days = pd.date_range(ts_left, ts_right, freq='D', closed='left')
    base = (ts_right - ts_left).total_seconds()/3600 # to correctly deal with summer-/wintertime changeover
    fullweeks, individual = divmod(len(days), 7)
    peak = fullweeks * 60.0
    for day in days[:individual]:
        if day.isoweekday() < 6:
            peak += 12
    return base, peak, base - peak

@np.vectorize
def _offpeakprice(p_base, p_peak, basehours, peakhours):
    """Return offpeak price from base and peak price and number of base and peak hours in period."""
    return (p_base * basehours - p_peak * peakhours) / (basehours - peakhours)

