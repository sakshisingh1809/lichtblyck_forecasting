"""
Custom classes that are thin wrappers around the pandas objects.
"""

from __future__ import annotations
import pandas as pd
from . import attributes

FREQUENCIES = ["AS", "QS", "MS", "D", "H", "15T"]



def _aggpf(pf: PfFrame) -> pd.Series:
    """
    Aggregation function for PfFrames.

    Parameters
    ----------
    pf : PfFrame
        Dataframe with (at least) 2 of the following columns: (w or q), p, r.

    Returns
    -------
    pd.Series
        The aggregated series with the aggregated values for w and p.
    """
    if not isinstance(pf, PfFrame):
        pf = PfFrame(pf)
    duration = pf.duration.sum()
    q = pf.q.sum()
    r = pf.r.sum()
    return pd.Series({"w": q / duration, "p": r / q})


def _changefreq(pf: PfFrame, freq: str = "MS") -> PfFrame:
    """
    Resample and aggregate the DataFrame at a new frequency.

    Parameters
    ----------
    freq : str, optional
        The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
        for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
        quarterhour; None to aggregate over the entire time period. The default is 'MS'.

    Returns
    -------
    PfFrame
        Same data at different timescale.
    """
    # By default, resampling labels are sometimes right-bound. Change to make left-bound.
    if freq == "M" or freq == "A" or freq == "Q":
        freq += "S"
    if freq is not None and freq not in FREQUENCIES:
        raise ValueError(
            "Parameter `freq` must be None or one of {"
            + ",".join(FREQUENCIES)
            + "}."
        )

    # Don't resample, just aggregate.
    if freq is None:  
        duration = pf.duration.sum()
        q = pf.q.sum()
        r = pf.r.sum()
        # Must return all data, because time information is lost.
        return pd.Series({"w": q / duration, "q": q, "p": r / q, "r": r})

    # Empty frame.
    if len(pf) == 0:
        return PfFrame(pf.resample(freq).mean())

    down_or_up= FREQUENCIES.index(freq) - FREQUENCIES.index(pf.index.freq)

    # Nothing more needed; dataframe already in desired frequency.
    if down_or_up== 0:
        return pf

    # Must downsample.
    elif down_or_up < 0:
        newpf = PfFrame(pf.resample(freq).apply(_aggpf))
        # Discard rows in new dataframe that are only partially present in original dataframe.
        newpf = newpf[(newpf.index >= pf.index[0]) & (newpf.ts_right <= pf.ts_right[-1])]
        return PfFrame(newpf)

    # Must upsample.
    else:
        # Keep only w and p because these are averages that can be copied over to each child row.
        newpf = PfFrame({"w": pf.w, "p": pf.p}, pf.index)
        # Workaround to avoid missing final values:
        newpf.loc[newpf.ts_right[-1]] = [None, None]  # first, add additional row...
        newpf = newpf.resample(freq).ffill()  # ... then do upsampling ...
        newpf = newpf.iloc[:-1]  # ... and then remove final row.
        return PfFrame(newpf)

    # TODO: change/customize the columns in the returned dataframe.
    


def force_Pf(function):
    """Decorator to ensure a PfFrame (instead of a DataFrame) or a PfSeries (instead of a Series) is returned."""

    def wrapper(*args, **kwargs):
        val = function(*args, **kwargs)
        if type(val) is pd.DataFrame:
            val = PfFrame(val)
        elif type(val) is pd.Series:
            val = PfSeries(val)
        return val

    return wrapper

# Currently not working. Goal: wrap each method of pd.DataFrame and pd.Series
# class PfMeta(type):
#     def __new__(cls, name, bases, dct):
#         klass = super().__new__(cls, name, bases, dct)
#         for base in bases:
#             print (f'base: {base}')
#             for field_name, field in base.__dict__.items():
#                 print (f'field_name: {field_name}, field: {field}')
#                 if callable(field):
#                     print(f'yes, callable {field_name}')
#                     setattr(klass, field_name, force_Pf(field))
#         return klass

class PfSeries(pd.Series):
    """
    PortfolioSeries; pandas series with additional functionality for getting
    duration [h] timeseries.
    """

    duration = property(attributes._duration)
    ts_right = property(attributes._ts_right)
    
    
class PfFrame(pd.DataFrame):
    """
    PortfolioFrame; pandas dataframe with additional functionality for getting
    power [MW], price [Eur/MWh], quantity [MWh] and revenue [Eur] timeseries,
    as well as and right-bound timestamp and duration [h].

    Attributes
    ----------
    w, q, p, r : PfSeries
        Power [MW], quantity [MWh], price [Eur/MWh], revenue [Eur] timeseries.
    ts_right, duration : pandas.Series
        Right timestamp and duration [h] of row.
        
    Methods
    -------
    changefreq()
        Aggregate the data to a new frequency.
    """

    # Time series.
    w = property(force_Pf(attributes._power))
    p = property(force_Pf(attributes._price))
    q = property(force_Pf(attributes._quantity))
    r = property(force_Pf(attributes._revenue))
    duration = property(attributes._duration)
    ts_right = property(attributes._ts_right)

    # Resample and aggregate.
    changefreq = _changefreq




