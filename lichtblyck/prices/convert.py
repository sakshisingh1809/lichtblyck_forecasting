"""
Convert power price timeseries using base, peak, offpeak times.

. Conversions without loss of information:
.. Base and peak prices <--> Peak and offpeak prices
.. Peak and offpeak prices in dataframe with yearly/quarterly/monthly index <--> 
   peak and offpeak prices in series with hourly index

. Conversions with information loss:
.. Hourly varying prices --> Peak and offpeak prices in dataframe with yearly/quarterly/monthly index
"""

from typing import Union, Iterable, Dict
from .utils import ts_leftright, is_peak_hour, duration_bpo
from ..core.utils import changefreq_avg
import pandas as pd
import numpy as np

# General functions.


def group_function(freq: str, bpo: bool = False):
    """Function to group all rows that belong to same 'product'."""
    if freq == "MS":
        if bpo:
            return lambda ts: (ts.year, ts.month, is_peak_hour(ts))
        else:
            return lambda ts: (ts.year, ts.month)
    elif freq == "QS":
        if bpo:
            return lambda ts: (ts.year, ts.quarter, is_peak_hour(ts))
        else:
            return lambda ts: (ts.year, ts.quarter)
    elif freq == "AS":
        if bpo:
            return lambda ts: (ts.year, is_peak_hour(ts))
        else:
            return lambda ts: ts.year
    else:
        raise ValueError("Argument 'freq' must be in {'MS', 'QS', 'AS'}.")


def p_offpeak(p_base, p_peak, ts_left, ts_right) -> float:
    """Return offpeak price from base and peak price and time interval they apply to."""
    if isinstance(p_base, Iterable):
        return np.vectorize(p_offpeak)(p_base, p_peak, ts_left, ts_right)
    b, p, o = duration_bpo(ts_left, ts_right)
    if o == 0:
        return np.nan
    return (p_base * b - p_peak * p) / o


def p_peak(p_base, p_offpeak, ts_left, ts_right) -> float:
    """Return peak price from base and offpeak price and time interval they apply to."""
    if isinstance(p_base, Iterable):
        return np.vectorize(p_peak)(p_base, p_offpeak, ts_left, ts_right)
    b, p, o = duration_bpo(ts_left, ts_right)
    if p == 0:
        return np.nan
    return (p_base * b - p_offpeak * o) / p


def p_base(p_peak, p_offpeak, ts_left, ts_right) -> float:
    """Return base price from peak and offpeak price and time interval they apply to."""
    if isinstance(p_peak, Iterable):
        return np.vectorize(p_base)(p_peak, p_offpeak, ts_left, ts_right)
    b, p, o = duration_bpo(ts_left, ts_right)
    return (p_peak * p + p_offpeak * o) / b


def complete_bpoframe(partial_bpoframe: pd.DataFrame) -> pd.DataFrame:
    """ 
    Add missing information to bpoframe (Dataframe with base, peak, offpeak prices).

    Parameters
    ----------
        partial_bpoframe : DataFrame 
            Dataframe with columns that include 'p_base', 'p_peak' and/or 'p_offpeak'. 
            Datetimeindex with frequency in {'MS', 'QS', 'AS'}.

    Returns
    -------
    DataFrame
        If exactly one of {'p_base', 'p_peak', 'p_offpeak'} is missng, calculate value
        of missing column from other two columns and return complete dataframe.

    Notes
    -----

    In: 
    
                                p_peak      p_offpeak
    ts_left
    2020-01-01 00:00:00+01:00   42.530036   30.614701
    2020-02-01 00:00:00+01:00   33.295167   15.931557
                                ...         ...
    2020-11-01 00:00:00+01:00   49.110873   33.226004
    2020-12-01 00:00:00+01:00   57.872246   35.055449
    12 rows × 2 columns

    Out:

                                p_base      p_peak      p_offpeak
    ts_left
    2020-01-01 00:00:00+01:00   35.034906   42.530036   30.614701
    2020-02-01 00:00:00+01:00   21.919009   33.295167   15.931557
                                ...         ...         ...
    2020-11-01 00:00:00+01:00   38.785706   49.110873   33.226004
    2020-12-01 00:00:00+01:00   43.519745   57.872246   35.055449
    12 rows × 3 columns
    """
    found = sum([col in partial_bpoframe for col in ["p_base", "p_peak", "p_offpeak"]])
    if found == 3:
        return partial_bpoframe
    if found < 2:
        raise ValueError(
            "At least 2 of {'p_base', 'p_peak', 'p_offpeak'} must be present as columns"
        )
    df = partial_bpoframe.copy()
    b, p, o = np.vectorize(duration_bpo)(df.index, df.ts_right)
    if "p_offpeak" not in df:
        df["p_offpeak"] = (df["p_base"] * b - df["p_peak"] * p) / o
    elif "p_peak" not in df:
        df["p_peak"] = (df["p_base"] * b - df["p_offpeak"] * o) / p
    else:
        df["p_base"] = (df["p_peak"] * p + df["p_offpeak"] * o) / b
    return df


def tseries2singlebpo(p: pd.Series) -> pd.Series:
    """
    Aggregate timeseries with varying prices to a single base, peak and offpeak price.

    Parameters
    ----------
        p : Series
            Price timeseries with hourly or quarterhourly frequency.

    Returns
    -------
    Series
        Index: p_base, p_peak, p_offpeak.

    Notes
    -----

    In: 
    
    ts_left
    2020-01-01 00:00:00+01:00    41.88
    2020-01-01 01:00:00+01:00    38.60
    2020-01-01 02:00:00+01:00    36.55
                                 ...  
    2020-12-31 21:00:00+01:00    52.44
    2020-12-31 22:00:00+01:00    51.86
    2020-12-31 23:00:00+01:00    52.26
    Freq: H, Name: p, Length: 8784, dtype: float64

    Out:

    p_base       31.401369
    p_peak       51.363667
    p_offpeak    20.311204
    dtype: float64
    """
    grouped = p.groupby(
        is_peak_hour
    ).mean()  # simple mean works because all rows equal duration
    return pd.Series(
        {
            "p_base": p.mean(),
            "p_peak": grouped.get(True, np.nan),
            "p_offpeak": grouped.get(False, np.nan),
        }
    )


def tseries2bpoframe(p: pd.Series, freq: str = "MS") -> pd.DataFrame:
    """
    Aggregate timeseries with varying prices to a dataframe with base, peak and offpeak 
    price timeseries, grouped by provided time interval.

    Parameters
    ----------
        p : Series
            Price timeseries with hourly or quarterhourly frequency.
        freq : str
            Target frequency. One of {'MS' (default) 'QS', 'AS'} for month,
            quarter, or year prices.

    Returns
    -------
    DataFrame
        Dataframe with base, peak and offpeak prices (as columns). Index: downsampled
        timestamps at provided frequency.

    Notes
    -----

    In: 
    
    ts_left
    2020-01-01 00:00:00+01:00    41.88
    2020-01-01 01:00:00+01:00    38.60
    2020-01-01 02:00:00+01:00    36.55
                                 ...  
    2020-12-31 21:00:00+01:00    52.44
    2020-12-31 22:00:00+01:00    51.86
    2020-12-31 23:00:00+01:00    52.26
    Freq: H, Name: p, Length: 8784, dtype: float64

    Out:

                                p_base      p_peak      p_offpeak
    ts_left
    2020-01-01 00:00:00+01:00   35.034906   42.530036   30.614701
    2020-02-01 00:00:00+01:00   21.919009   33.295167   15.931557
                                ...         ...         ...
    2020-11-01 00:00:00+01:00   38.785706   49.110873   33.226004
    2020-12-01 00:00:00+01:00   43.519745   57.872246   35.055449
    12 rows × 3 columns
    """
    if freq not in ("MS", "QS", "AS"):
        raise ValueError("Argument 'freq' must be in {'MS', 'QS', 'AS'}.")
    return p.resample(freq).apply(tseries2singlebpo).unstack()


def bpoframe2tseries(bpoframe: pd.DataFrame, freq: str = "H") -> pd.Series:
    """
    Convert a dataframe with base, peak and/or offpeak prices, to a single (quarter)hourly
    timeseries.

    Parameters
    ----------
        bpoframe : DataFrame 
            Dataframe with prices. Columns must include at least 2 of {'p_peak', 
            'p_offpeak', 'p_base'}. Datetimeindex with frequency in {'MS', 'QS', 'AS'}.
        freq : str
            Target frequency. One of {'H' (default) '15T'} for hourly or quarterhourly
            index. 

    Returns
    -------
    Series
        Timeseries with prices as provided in `bpoframees`.

    Notes
    -----

    In: 
    
                                p_peak      p_offpeak
    ts_left
    2020-01-01 00:00:00+01:00   42.530036   30.614701
    2020-02-01 00:00:00+01:00   33.295167   15.931557
                                ...         ...
    2020-11-01 00:00:00+01:00   49.110873   33.226004
    2020-12-01 00:00:00+01:00   57.872246   35.055449
    12 rows × 2 columns

    Out:

    ts_left
    2020-01-01 00:00:00+01:00    30.614701
    2020-01-01 01:00:00+01:00    30.614701
    2020-01-01 02:00:00+01:00    30.614701
                                 ...  
    2020-12-31 21:00:00+01:00    35.055449
    2020-12-31 22:00:00+01:00    35.055449
    2020-12-31 23:00:00+01:00    35.055449
    Freq: H, Name: p, Length: 8784, dtype: float64
    """
    if freq not in ("H", "15T"):
        raise ValueError("Argument 'freq' must be in {'H', '15T'}.")
    df = complete_bpoframe(bpoframe)
    df = changefreq_avg(df[["p_peak", "p_offpeak"]], freq)
    ispeak = is_peak_hour(df.index)
    df["p"] = np.nan
    df.loc[ispeak, "p"] = df.loc[ispeak, "p_peak"]
    df.loc[~ispeak, "p"] = df.loc[~ispeak, "p_offpeak"]
    return df["p"]


def tseries2tseries(p: pd.Series, freq: str = "MS") -> pd.Series:
    """
    Transform price timeseries (with possibly variable prices) to one with uniform peak 
    and offpeak prices during certain interval.

    Parameters
    ----------
        p : Series
            Price timeseries with hourly or quarterhourly frequency.
        freq : str
            Target frequency for which peak and offpeak prices will be uniform. One of 
            {'MS' (default) 'QS', 'AS'} for month, quarter, or year prices.

    Returns
    -------
    Series
        Price timeseries where each peak hour within the target frequency has the same
        value. Idem for offpeak hours. Index: as original series.
    
    Notes
    -----
    
    In:

    ts_left
    2020-01-01 00:00:00+01:00    41.88
    2020-01-01 01:00:00+01:00    38.60
    2020-01-01 02:00:00+01:00    36.55
                                 ...  
    2020-12-31 21:00:00+01:00    52.44
    2020-12-31 22:00:00+01:00    51.86
    2020-12-31 23:00:00+01:00    52.26
    Freq: H, Name: p, Length: 8784, dtype: float64

    Out:

    ts_left
    2020-01-01 00:00:00+01:00    30.614701
    2020-01-01 01:00:00+01:00    30.614701
    2020-01-01 02:00:00+01:00    30.614701
                                ...    
    2020-12-31 21:00:00+01:00    35.055449
    2020-12-31 22:00:00+01:00    35.055449
    2020-12-31 23:00:00+01:00    35.055449
    Freq: H, Name: p, Length: 8784, dtype: float64
    """
    if p.index.freq not in ("H", "15T"):
        raise ValueError(
            "Frequency of provided timeseries must be hourly or quarterhourly."
        )

    # Return normal mean, because all rows have same duration.
    return p.groupby(group_function(freq, True)).transform(np.mean)


def bpoframe2bpoframe(bpoframe: pd.DataFrame, freq: str = "AS") -> pd.DataFrame:
    """
    Convert a dataframe with base, peak and/or offpeak prices to a similar dataframe
    with a different frequency.

    Parameters
    ----------
        bpoframe : DataFrame 
            Dataframe with prices. Columns must include at least 2 of {'p_peak', 
            'p_offpeak', 'p_base'}. Datetimeindex with frequency in {'MS', 'QS', 'AS'}.
        freq : str
            Target frequency. One of {'MS', 'QS', 'AS' (default)} for month, quarter, or
            year prices. 

    Returns
    -------
    DataFrame
        Dataframe with base, peak and offpeak prices (as columns). Index: timestamps at
        provided frequency.

    Notes
    -----
    When upsampling, (e.g., from year to month prices), the prices are duplicated
    to each individual month.

    In: 

                                p_base      p_peak      p_offpeak
    ts_left
    2020-01-01 00:00:00+01:00   35.034906   42.530036   30.614701
    2020-02-01 00:00:00+01:00   21.919009   33.295167   15.931557
                                ...         ...         ...
    2020-11-01 00:00:00+01:00   38.785706   49.110873   33.226004
    2020-12-01 00:00:00+01:00   43.519745   57.872246   35.055449
    12 rows × 3 columns

    Out:

                                p_base      p_peak      p_offpeak
    ts_left
    2020-01-01 00:00:00+01:00   30.490036   38.003536   26.312894
    2020-04-01 00:00:00+02:00   25.900919   35.295167   20.681892
    2020-07-01 00:00:00+02:00   32.706785   44.033511   26.371498
    2020-10-01 00:00:00+02:00   39.455197   54.468722   31.063728
    """
    if freq.upper() not in ("MS", "QS", "AS"):
        raise ValueError("Argument 'freq' must be in {'MS', 'QS', 'AS'}.")

    return tseries2bpoframe(bpoframe2tseries(bpoframe, "H"), freq)

