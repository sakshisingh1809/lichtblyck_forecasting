"""
Custom extensions (attributes) to the pandas objects
"""


import pandas as pd
import numpy as np
from ..tools.tools import wavg


@property
def _duration(i: pd.core.indexes.datetimes.DatetimeIndex) -> pd.Series:
    """
    Return duration [h] of each timestamp.
    """
    if i.tz is None:
        raise AttributeError("Index is missing timezone information.")

    # Get duration in h for each except final datapoint.
    duration = (i[1:] - i[:-1]).total_seconds() / 3600

    # Get duration in h of final datapoint.
    if i.freq is not None:
        final_duration = ((i[-1] + i.freq) - i[-1]).total_seconds() / 3600
    else:
        final_duration = np.median(duration)

    # Add duration of final datapoint.
    return pd.Series(np.append(duration, final_duration), i)


@property
def _quantity(fr: pd.core.generic.NDFrame) -> pd.Series:
    """
    Return quantity [MWh] series, if power [MW] series is given.
    """
    try:
        return fr["q"]  # Return quantity column, if it is present.
    except KeyError:
        pass
    if isinstance(fr, pd.Series):
        # if fr.name != 'w' and not fr.name.startswith('w_'):
        #     raise ValueError("Series name is not 'w' and does not start with 'w_'.")
        w = fr  # Series
    else:
        if not "w" in fr.columns:
            raise ValueError("Dataframe does not have a column named 'w'.")
        w = fr["w"]  # Series

    q = w * fr.index.duration
    q = q.rename("q")
    return q


@property
def _revenue(fr: pd.core.frame.DataFrame) -> pd.Series:
    """
    Return revenue [Eur] series, if price and quantity series are present in DataFrame.
    """
    try:
        return fr["r"]  # Return revenue column, if it is present.
    except KeyError:
        pass

    r = fr.q * fr.p
    r = r.rename("r")
    return r


# Extend attributes of DateTimeIndex
pd.core.indexes.datetimes.DatetimeIndex.duration = _duration
# Extend attributes of Series and DataFrames
pd.core.generic.NDFrame.q = _quantity
pd.DataFrame.r = _revenue

pd.core.generic.NDFrame.wavg = wavg
