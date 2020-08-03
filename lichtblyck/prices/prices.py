# -*- coding: utf-8 -*-
"""
Module to read price data from disk.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lichtblyck import tools
from sqlalchemy import create_engine
from typing import Tuple, Union, Iterable

SPOTPRICEFILE = 'lichtblyck/prices/sourcedata/spot.tsv'

def spot() -> pd.Series:
    """Return spot price timeseries."""
    data = pd.read_csv(SPOTPRICEFILE, header=None, sep='\t', 
                       names=['date', 'time', 'price', 'anmerkungen', 'empty'])
    data['ts_right'] = pd.to_datetime(data["date"] + " " + data["time"], format="%d.%m.%Y %H:%M:%S")
    #Replace missing values with NaN, convert others to float:
    data['p_spot'] = data['price'].apply(lambda x: np.NaN if x == '---' else float(x.replace(',', '.'))) 
    spot = tools.set_ts_index(data, 'ts_right', 'right')
    spot = spot.resample('H').asfreq()
    return spot['p_spot']


def futures(prod:str='y', earliest_trading=None, earliest_delivery=None) -> pd.DataFrame:
    """
    Return futures prices timeseries.
    
    Arguments:
        prod: product to return the prices of. One of {'y', 'q', 'm'}.
        earliest_trading: only return prices after this date. (default: all)
        earliest_delivery: only return prices for products whose delivery is
            on or after this date. (default: all)
        
    Returns: 
        Dataframe with EEX futures prices. Index: Multiindex (with levels 0: 
            product {'y', 'q', 'm'}, 1: time stamp of delivery start, 2: date 
            of trading). Columns: p_base (base price), p_peak (peak price), 
            p_offpeak (offpeak price). ts_right_deliv (time stamp of delivery 
            end). basehours, peakhours, offpeakhours in delivery period.
    """
    engine = create_engine('mssql+pymssql://sqlbiprod/__Staging')
    
    try:
        dur_months = {'m':1, 'q':3, 'y':12}[prod]
    except KeyError:
        raise ValueError("Argument 'prod' must be in {'y', 'q', 'm'}.")
    
    # Get values from server...
    where = f"WHERE Typ='{prod.upper()}'"
    if earliest_trading:
        where += f" AND EEX_Handelstag > '{earliest_trading}'"
    if earliest_delivery:
        where += f" AND Lieferzeitraum > '{earliest_delivery}'"
    df = pd.read_sql_query(f"SELECT * FROM dbo.Fakten_EW_EEX_Preise_fkunden {where}", engine)
    # ...massage to get correct format...
    df = df.rename(columns={'Typ': 'deliv_prod', 'Peak_Preis': 'p_peak',
                            'Base_Preis': 'p_base', 'EEX_Handelstag': 'ts_left_trade',
                            'Lieferzeitraum': 'ts_left_deliv'})
    df = df.drop('ID', axis = 1)
    df['ts_left_trade'] = pd.to_datetime(df['ts_left_trade']).dt.tz_localize('Europe/Berlin')
    df['ts_left_deliv'] = pd.to_datetime(df['ts_left_deliv']).dt.tz_localize('Europe/Berlin')
    df['ts_right_deliv'] = df['ts_left_deliv'] + pd.offsets.DateOffset(months=dur_months)
    df['trade_before_deliv'] = df['ts_left_deliv'] - df['ts_left_trade']
    # ...get peak and base and offpeak hours...
    h = df[['ts_left_deliv', 'ts_right_deliv']].drop_duplicates()
    bpo = hours_bpo(h['ts_left_deliv'], h['ts_right_deliv'])
    h['basehours'], h['peakhours'], h['offpeakhours'] = bpo
    df = pd.merge(df, h, on=['ts_left_deliv', 'ts_right_deliv'])
    # ...and use to calculate offpeak prices.
    df['p_offpeak'] = p_offpeak(df['p_base'], df['p_peak'], df['basehours'], df['peakhours'])
    # Finally, return in correct row and column order.
    # TODO: add frequency to index, if possible
    df = df.set_index(['deliv_prod', 'ts_left_deliv', 'ts_left_trade']).sort_index()
    return df[['ts_right_deliv', 'trade_before_deliv', 'p_base', 'p_peak', 'p_offpeak', 
               'basehours', 'peakhours', 'offpeakhours']]


def hours_bpo(ts_left, ts_right) -> float:
    """Return number of base, peak and offpeak hours in interval [ts_left, ts_right).
    Timestamps must coincide with day start."""
    if isinstance(ts_left, Iterable):
        return np.vectorize(hours_bpo)(ts_left, ts_right)
    days = pd.date_range(ts_left, ts_right, freq='D', closed='left')
    base = (ts_right - ts_left).total_seconds()/3600 # to correctly deal with summer-/wintertime changeover
    fullweeks, individual = divmod(len(days), 7)
    peak = fullweeks * 60.0
    for day in days[:individual]:
        if day.isoweekday() < 6:
            peak += 12
    return base, peak, base - peak


def p_offpeak(p_base, p_peak, basehours, peakhours) -> float:
    """Return offpeak price from base and peak price and number of base and peak hours in period."""
    if isinstance(p_base, Iterable):
        return np.vectorize(p_offpeak)(p_base, p_peak, basehours, peakhours)
    return (p_base * basehours - p_peak * peakhours) / (basehours - peakhours)


def is_peak_hour(ts) -> bool:
    """Return True if timestamp 'ts' is a peak hour. More precisely: if 'ts' 
    lies in one of the (left-closed) time intervals that define the peak hour 
    periods."""
    if isinstance(ts, Iterable):
        return np.vectorize(is_peak_hour)(ts)
    return ts.hour >= 8 and ts.hour < 20 and ts.isoweekday() < 6    


def p_bpo(s:pd.Series) -> pd.Series:
    """
    Aggregate price series to base, peak and offpeak prices.
    
    Arguments:
        s: price timeseries with (left-bound) DatetimeIndex.
    
    Returns:
        Series with prices. Index: p_base, p_peak, p_offpeak.
    """
    is_peak = is_peak_hour(s.index)
    grouped = s.groupby(is_peak).mean()
    return pd.Series({'p_base': s.mean(), 'p_peak': grouped[True], 'p_offpeak': grouped[False]})


def p_bpo_grouped(s:pd.Series, freq:str='MS') -> pd.DataFrame:
    """
    Aggregate price series to base, peak and offpeak prices. Grouped by time 
    interval.
    
    Arguments:
        s: price timeseries with (left-bound) DatetimeIndex.
        freq: grouping frequency. One of {'MS', 'QS', 'YS'} for month, quarter,
            or year prices.
        
    Returns: 
        Dataframe with base, peak and offpeak prices. Index: time stamp of 
            delivery start. Columns: p_base (base price), p_peak (peak price), 
            p_offpeak (offpeak price).
    """
    if freq not in ('MS', 'QS', 'YS'):
        raise ValueError("Argument 'freq' must be in {'MS', 'QS', 'YS'}.")
    return s.groupby(pd.Grouper(freq=freq)).apply(p_bpo).unstack()

def vola(df: Union[pd.Series, pd.DataFrame], window:int=100
         ) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate volatility in [fraction/year] from price time series/dataframe.
    
    Arguments:
        df: Series or Dataframe with price values indexed by (trading day) timestamps.
        window: number of observations for volatility estimate.
    
    Returns:
        Series or Dataframe with volatility calculated with rolling window.
    """
    df = df.apply(np.log).diff()     #change as fraction
    volas = df.rolling(window).std() #volatility per average timedelta
    av_years = df.rolling(window).apply(lambda s: ((s.index[-1] - s.index[0])/window).total_seconds()/3600/24/365.24)
    volas /= av_years.apply(np.sqrt) #volatility per year
    return volas.dropna()