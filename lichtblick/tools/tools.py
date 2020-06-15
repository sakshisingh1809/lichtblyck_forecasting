# -*- coding: utf-8 -*-
"""
Module with tools to modify and standardize dataframes.
"""
from typing import Optional, Iterable
import pandas as pd
import datetime

# The files we want to import don't have a standard layout. Importantly, the 
# timestamp is not always left-bound. Therefore, we create a function to deal with this
# (and do some other standardization).
def set_ts_index(df:pd.DataFrame, column:str, bound="try", tz:str='Europe/Berlin') -> pd.DataFrame:
    """
    Use column 'column' of dataframe 'df' to create a left-bound timestamp index 
    in timezone 'tz'. If bound=='left', will assume timestamps are already 
    leftbound; if bound=='right', will subtract one 'timestep' first; if
    bound=='try', will first try the timestamps unchanged, and subtract one 
    'timestep' if that fails. NB: bound=='try' will give false results if the 
    data contains no summertime/wintertime changeover and the times are rightbound.
    
    Returns dataframe with 'column' removed, and index renamed to 'ts_left'.
    """    
    if bound=="left":
        df = df.set_index(column)
        df.index.name = "ts_left"
    elif bound=="right":
        minutes = (df.iloc[1][column] - df.iloc[0][column]).seconds / 60
        df = df.copy() #don't change passed-in df
        df["ts_left"] = df[column] + datetime.timedelta(minutes = -minutes)
        df = df.set_index("ts_left").drop(column, axis=1)
    else:
        try:
            return set_ts_index(df, column, tz, "left")
        except:
            return set_ts_index(df, column, tz, "right")
    try:
        return df.tz_localize(tz, ambiguous='infer')
    except:
        return df.tz_localize(tz, ambiguous='NaT')


def wavg(df:pd.DataFrame, weights:Optional[Iterable]=None, axis:int=0) -> pd.DataFrame:
    """Returns column-average over all rows (if axis==0, default) or row-average
    over all columns (if axis==1) in dataframe 'df'. If provided, weighted with 
    values in 'weights', which must be of equal length.
    source: http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns"""
    if axis == 1:
        df = df.T
    if weights is None:
        return df.mean()
    if len(df) != len(weights):
        raise ValueError('Dataframe and weights have unequal length.')
    return df.mul(weights, axis=0).sum() / weights.sum()