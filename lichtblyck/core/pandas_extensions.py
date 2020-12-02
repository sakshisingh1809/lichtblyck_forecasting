"""
Custom extensions (attributes) to the pandas objects
"""


import pandas as pd
import numpy as np

@property
def _duration(i:pd.core.indexes.datetimes.DatetimeIndex) -> pd.Series:
    """
    Return duration of each timestamp (in hours).
    """
    duration = (i[1:] - i[:-1]).total_seconds()/3600 #get duration in h for each except final datapoint 
    if i.freq is not None:
        final_duration = ((i[-1] + i.freq) - i[-1]).total_seconds()/3600
    else:
        final_duration = np.median(duration)
    return pd.Series(np.append(duration, final_duration), i) #add duration of final datapoint (guessed)

@property
def _quantity(fr:pd.core.generic.NDFrame) -> pd.Series:
    """
    Return quantity [MWh] series, if power [MW] series is given.
    """
    try:
        return fr['q'] #Return quantity column, if it is present.
    except KeyError:
        pass
    if isinstance(fr, pd.Series):
        # if fr.name != 'w' and not fr.name.startswith('w_'):
        #     raise ValueError("Series name is not 'w' and does not start with 'w_'.")
        w = fr # Series
    else:
        if not 'w' in fr.columns:
            raise ValueError("Dataframe does not have a column named 'w'.")
        w = fr['w'] # Series
    return w * fr.index.duration

@property
def _revenue(fr:pd.core.frame.DataFrame) -> pd.Series:
    """
    Return revenue [Eur] series, if price and quantity series are present in DataFrame.
    """
    try:
        return fr['r'] #Return revenue column, if it is present.
    except KeyError:
        return fr.q * fr.p


#Extend attributes of DateTimeIndex
pd.core.indexes.datetimes.DatetimeIndex.duration = _duration 
#Extend attributes of Series and DataFrames
pd.core.generic.NDFrame.q = _quantity 
pd.DataFrame.r = _revenue 
    