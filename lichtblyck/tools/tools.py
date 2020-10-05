# -*- coding: utf-8 -*-
"""
Module with tools to modify and standardize dataframes.
"""
from typing import Optional, Iterable, Callable
import pandas as pd
import numpy as np

# The files we want to import don't have a standard layout. Importantly, the 
# timestamp is not always left-bound. Therefore, we create a function to deal with this
# (and do some other standardization).
def set_ts_index(df:pd.DataFrame, column:str=None, bound="try", 
                 tz:str='Europe/Berlin') -> pd.DataFrame:
    """
    Use column 'column' (or the index, if column not specified) of dataframe 
    'df' to create a left-bound timestamp index in the Europe/Berlin timezone.
    
    Other arguments:
        tz: timezone of the input dataframe. 
        bound: if bound=='left', will assume timestamps are already leftbound; 
            if bound=='right', will subtract one 'timestep' first; if bound==
            'try', will first assume bound=='left', then try with bound=='right'
            if that fails. NB: bound=='try' will give false results if the data
            contains no summertime/wintertime changeover and the times are 
            rightbound.
    
    Returns dataframe with 'column' removed, and index renamed to 'ts_left'.
    """
    if column:
        df = df.set_index(column)
    else:
        df = df.copy() #don't change passed-in df
        
    df.index = pd.DatetimeIndex(df.index) #turn / try to turn into datetime
    
    if bound=="left":
        pass
    elif bound=="right":
        minutes = (df.index[1] - df.index[0]).seconds / 60
        df.index += pd.Timedelta(minutes = -minutes)
    else:
        try:
            return set_ts_index(df, None, "left", tz)
        except:
            return set_ts_index(df, None, "right", tz)
    df.index.name = "ts_left"
    
    if df.index.tz is None:
        try:
            df = df.tz_localize(tz, ambiguous='infer')
        except:
            df = df.tz_localize(tz, ambiguous='NaT')
    return df.tz_convert('Europe/Berlin')
    

@property
def duration(i:pd.core.indexes.datetimes.DatetimeIndex) -> pd.Series:
    """
    return duration of each timestamp (in hours).
    """
    duration = (i[1:] - i[:-1]).seconds/3600 #get duration in h for each except final datapoint 
    if i.freq is not None:
        final_duration = ((i[-1] + i.freq) - i[-1]).seconds/3600
    else:
        final_duration = np.median(duration)
    return pd.Series(np.append(duration, final_duration), i) #add duration of final datapoint (guessed)

@property
def quantity(fr:pd.core.generic.NDFrame) -> pd.Series:
    """
    return quantity [MWh] series, if power [MW] series is given.
    """
    
    if isinstance(fr, pd.Series):
        if fr.name != 'w' and not fr.name.startswith('w_'):
            raise ValueError("Series name is not 'w' and does not start with 'w_'.")
        w = fr # Series
    else:
        if not 'w' in fr.columns:
            raise ValueError("Dataframe does not have a column named 'w'.")
        w = fr['w'] # Series
    return w * fr.index.duration


def wavg(df:pd.DataFrame, weights:Optional[Iterable]=None, axis:int=0) -> pd.DataFrame:
    """Returns each column's average over all rows (if axis==0, default) or each
    row's average over all columns (if axis==1) in dataframe 'df'. If provided,
    weighted with values in 'weights', which must be of equal length.
    source: http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns"""
    if axis == 1:
        df = df.T
    if weights is None:
        return df.mean()
    if len(df) != len(weights):
        raise ValueError('Dataframe and weights have unequal length.')
    try:
        weights = weights.values
    except AttributeError:
        pass
    return df.mul(weights, axis=0).sum(skipna=False) / weights.sum()


def __is(letter:str) -> Callable[[str], bool]:
    """Returns function that tests if its argument is 'letter' or starts with 
    'letter_'."""
    @np.vectorize
    def check(name):
        return name == letter or name.startswith(letter + "_")
    return check
 
is_price = __is('p')
is_quantity = __is('q')
is_temperature = __is('t')
is_revenue = __is('r')
is_power = __is('w')
