"""
Convert volume [MW] and price [Eur/MWh] timeseries using base, peak, offpeak times.

. Conversions without loss of information:
.. Base and peak values <--> Peak and offpeak values
.. Peak and offpeak values in dataframe with yearly/quarterly/monthly index <--> 
   peak and offpeak values in series with hourly (or shorter) index

. Conversions with information loss:
.. Hourly varying values --> Peak and offpeak values in dataframe with yearly/quarterly/monthly index
"""

from typing import Union, Iterable
from .utils import is_peak_hour, duration_bpo
from ..core.utils import changefreq_avg
from ..tools.nits import Q_
from ..tools.types import Value, Stamp
from ..tools.stamps import freq_up_or_down
from ..tools.frames import trim_frame
import pandas as pd
import numpy as np
from warnings import warn


BPO = ("base", "peak", "offpeak")


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
        raise ValueError("Argument 'freq' must be on of {'MS', 'QS', 'AS'}.")


def offpeak(
    base: Union[Value, Iterable[Value]],
    peak: Union[Value, Iterable[Value]],
    ts_left: Union[Stamp, Iterable[Stamp]],
) -> Union[Value, Iterable[Value]]:
    """Return offpeak value from base and peak value and time interval they apply to."""
    if isinstance(base, Iterable):
        return np.vectorize(offpeak)(base, peak, ts_left)
    b, p, o = duration_bpo(ts_left)
    if o == 0:
        return np.nan
    return (base * b - peak * p) / o


def peak(
    base: Union[Value, Iterable[Value]],
    offpeak: Union[Value, Iterable[Value]],
    ts_left: Union[Stamp, Iterable[Stamp]],
) -> Union[Value, Iterable[Value]]:
    """Return peak value from base and offpeak value and time interval they apply to."""
    if isinstance(base, Iterable):
        return np.vectorize(peak)(base, offpeak, ts_left)
    b, p, o = duration_bpo(ts_left)
    if p == 0:
        return np.nan
    return (base * b - offpeak * o) / p


def base(
    peak: Union[Value, Iterable[Value]],
    offpeak: Union[Value, Iterable[Value]],
    ts_left: Union[Stamp, Iterable[Stamp]],
) -> Union[Value, Iterable[Value]]:
    """Return base value from peak and offpeak value and time interval they apply to."""
    if isinstance(peak, Iterable):
        return np.vectorize(base)(peak, offpeak, ts_left)
    b, p, o = duration_bpo(ts_left)
    return (peak * p + offpeak * o) / b


def complete_bpoframe(partial_bpoframe: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """ 
    Add missing information to bpoframe (Dataframe with base, peak, offpeak values).

    Parameters
    ----------
    partial_bpoframe : DataFrame 
        Dataframe with at least 2 columns with names in {'base', 'peak', 'offpeak'}. 
        Datetimeindex with frequency in {'MS', 'QS', 'AS'}.
    prefix : str, optional (default: '')
        If specified, add this to the column names to search for in the provided dataframe
        (and to the column names in the returned dataframes).

    Returns
    -------
    DataFrame
        If exactly one of {'base', 'peak', 'offpeak'} is missing, calculate value
        of missing column from other two columns and return complete dataframe.

    Notes
    -----
    Can only be used for values that are 'averagable' over a time period, like power [MW]
    and price [Eur/MWh]. Not for e.g. energy [MWh], revenue [Eur], and duration [h].

    In: 
    
                                peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   42.530036   30.614701
    2020-02-01 00:00:00+01:00   33.295167   15.931557
                                ...         ...
    2020-11-01 00:00:00+01:00   49.110873   33.226004
    2020-12-01 00:00:00+01:00   57.872246   35.055449
    12 rows × 2 columns

    Out:

                                base        peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   35.034906   42.530036   30.614701
    2020-02-01 00:00:00+01:00   21.919009   33.295167   15.931557
                                ...         ...         ...
    2020-11-01 00:00:00+01:00   38.785706   49.110873   33.226004
    2020-12-01 00:00:00+01:00   43.519745   57.872246   35.055449
    12 rows × 3 columns
    """
    col2bpo = {f"{prefix}{bpo}": bpo for bpo in BPO}
    series = {col2bpo[c]: s for c, s in partial_bpoframe.iteritems() if c in col2bpo}

    if len(series) > 2:  # i.e., 3
        return partial_bpoframe
    if len(series) < 2:
        raise ValueError(
            f"At least 2 of {', '.join(col2bpo.keys())} must be present as columns."
        )
    df = partial_bpoframe.copy()
    durations = duration_bpo(df.index)
    b, p, o = durations["base"], durations["peak"], durations["offpeak"]
    if "offpeak" not in series:
        df[f"{prefix}offpeak"] = (series["base"] * b - series["peak"] * p) / o
    elif "peak" not in series:
        df[f"{prefix}peak"] = (series["base"] * b - series["offpeak"] * o) / p
    else:
        df[f"{prefix}base"] = (series["peak"] * p + series["offpeak"] * o) / b

    return df[col2bpo.keys()]  # correct order


def tseries2singlebpo(s: pd.Series, prefix: str = "") -> pd.Series:
    """
    Aggregate timeseries with varying values to a single base, peak and offpeak value.

    Parameters
    ----------
    s : Series
        Timeseries with hourly or quarterhourly frequency.
    prefix : str, optional (default: '')
        If specified, add this to the index of the returned Series.

    Returns
    -------
    Series
        Index: base, peak, offpeak.

    Notes
    -----
    Can only be used for values that are 'averagable' over a time period, like power [MW]
    and price [Eur/MWh]. Not for e.g. energy [MWh], revenue [Eur], and duration [h].

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

    base       31.401369
    peak       51.363667
    offpeak    20.311204
    dtype: float64
    """
    # Handle possible units.
    sin, units = (s.pint.magnitude, s.pint.units) if hasattr(s, "pint") else (s, None)

    # Do calculations. Use normal mean, because all rows have same duration.
    grouped = sin.groupby(is_peak_hour).mean()
    sout = pd.Series(
        {
            f"{prefix}base": sin.mean(),
            f"{prefix}peak": grouped.get(True, np.nan),
            f"{prefix}offpeak": grouped.get(False, np.nan),
        }
    )

    # Handle possible units.
    if units is not None:
        sout = sout.astype(f"pint[{units}]")
    return sout


def tseries2bpoframe(s: pd.Series, freq: str = "MS", prefix: str = "") -> pd.DataFrame:
    """
    Aggregate timeseries with varying values to a dataframe with base, peak and offpeak 
    timeseries, grouped by provided time interval.

    Parameters
    ----------
    s : Series
        Timeseries with hourly or quarterhourly frequency.
    freq : str
        Target frequency. One of {'MS' (default) 'QS', 'AS'} for month, quarter, or year 
        values.
    prefix : str, optional (default: '')
        If specified, add this to the column names of the returned dataframe.

    Returns
    -------
    DataFrame
        Dataframe with base, peak and offpeak values (as columns). Index: downsampled
        timestamps at provided frequency.

    Notes
    -----
    Can only be used for values that are 'averagable' over a time period, like power [MW]
    and price [Eur/MWh]. Not for e.g. energy [MWh], revenue [Eur], and duration [h].

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

                                base        peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   35.034906   42.530036   30.614701
    2020-02-01 00:00:00+01:00   21.919009   33.295167   15.931557
                                ...         ...         ...
    2020-11-01 00:00:00+01:00   38.785706   49.110873   33.226004
    2020-12-01 00:00:00+01:00   43.519745   57.872246   35.055449
    12 rows × 3 columns
    """
    if freq not in ("MS", "QS", "AS"):
        raise ValueError("Argument 'freq' must be on of {'MS', 'QS', 'AS'}.")

    # Remove partial data
    s = trim_frame(s, freq)

    # Handle possible units.
    sin, units = (s.pint.magnitude, s.pint.units) if hasattr(s, "pint") else (s, None)

    # Do calculations. Use normal mean, because all rows have same duration.
    sout = sin.resample(freq).apply(lambda s: tseries2singlebpo(s, prefix))

    # Handle possible units.
    if units is not None:
        sout = sout.astype(f"pint[{units}]")
    return sout.unstack()


def bpoframe2tseries(
    bpoframe: pd.DataFrame, freq: str = "H", prefix: str = ""
) -> pd.Series:
    """
    Convert a dataframe with base, peak and/or offpeak values, to a single (quarter)hourly
    timeseries.

    Parameters
    ----------
    bpoframe : DataFrame 
        Dataframe with values. Columns must include at least 2 of {'peak', 'offpeak', 
        'base'}. Datetimeindex with frequency in {'MS', 'QS', 'AS'}.
    freq : str
        Target frequency. One of {'H' (default) '15T'} for hourly or quarterhourly
        index. 
    prefix : str, optional (default: '')
        If specified, add this to the column names to search for in the provided dataframe.

    Returns
    -------
    Series
        Timeseries with values as provided in `bpoframe`.

    Notes
    -----
    Can only be used for values that are 'averagable' over a time period, like power [MW]
    and price [Eur/MWh]. Not for e.g. energy [MWh], revenue [Eur], and duration [h].

    In: 
    
                                peak        offpeak
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
    Freq: H, Length: 8784, dtype: float64
    """
    if freq not in ("H", "15T"):
        raise ValueError("Argument 'freq' must be on of {'H', '15T'}.")

    df = bpoframe.rename({f"{prefix}{bpo}": bpo for bpo in BPO}, axis=1)  # remove prefx
    df = complete_bpoframe(df)  # make sure we have peak and offpeak columns
    df = changefreq_avg(df[["peak", "offpeak"]], freq)
    df["ispeak"] = df.index.map(is_peak_hour)

    return df["offpeak"].where(df["ispeak"], df["peak"])


def tseries2tseries(s: pd.Series, freq: str = "MS") -> pd.Series:
    """
    Transform timeseries (with possibly variable values) to one with (at certain
    frequency) uniform peak and offpeak values.

    Parameters
    ----------
    s : Series
        Timeseries with hourly or quarterhourly frequency.
    freq : str
        Target frequency for which peak and offpeak values will be uniform. One of 
        {'MS' (default) 'QS', 'AS'} for uniformity within each month, quarter, or year.

    Returns
    -------
    Series
        Timeseries where each peak hour within the target frequency has the same
        value. Idem for offpeak hours. Index: as original series.
    
    Notes
    -----
    Can only be used for values that are 'averagable' over a time period, like power [MW]
    and price [Eur/MWh]. Not for e.g. energy [MWh], revenue [Eur], and duration [h].
    
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
    if s.index.freq not in ("H", "15T"):
        raise ValueError(
            f"Frequency of provided timeseries must be hourly or quarterhourly (got {s.index.freq} instead)."
        )

    # Handle possible units.
    sin, units = (s.pint.magnitude, s.pint.units) if hasattr(s, "pint") else (s, None)

    # Return normal mean, because all rows have same duration.
    sout = sin.groupby(group_function(freq, True)).transform(np.mean)

    # Handle possible units.
    if units is not None:
        sout = sout.astype(f"pint[{units}]")
    return sout


def bpoframe2bpoframe(
    bpoframe: pd.DataFrame, freq: str = "AS", prefix: str = ""
) -> pd.DataFrame:
    """
    Convert a dataframe with base, peak and/or offpeak values to a similar dataframe
    with a different frequency.

    Parameters
    ----------
    bpoframe : DataFrame 
        Columns must include at least 2 of {'peak', 'offpeak', 'base'}. Datetimeindex 
        with frequency in {'MS', 'QS', 'AS'}.
    freq : str
        Target frequency. One of {'MS', 'QS', 'AS' (default)} for month, quarter, or
        year values. 
    prefix : str, optional (default: '')
        If specified, add this to the column names to search for in the provided dataframe
        (and to the column names in the returned dataframes).

    Returns
    -------
    DataFrame
        Dataframe with base, peak and offpeak values (as columns). Index: timestamps at
        provided frequency.

    Notes
    -----
    Can only be used for values that are 'averagable' over a time period, like power [MW]
    and price [Eur/MWh]. Not for e.g. energy [MWh], revenue [Eur], and duration [h].

    In: 

                                base        peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   35.034906   42.530036   30.614701
    2020-02-01 00:00:00+01:00   21.919009   33.295167   15.931557
                                ...         ...         ...
    2020-11-01 00:00:00+01:00   38.785706   49.110873   33.226004
    2020-12-01 00:00:00+01:00   43.519745   57.872246   35.055449
    12 rows × 3 columns

    Out:

                                base        peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   30.490036   38.003536   26.312894
    2020-04-01 00:00:00+02:00   25.900919   35.295167   20.681892
    2020-07-01 00:00:00+02:00   32.706785   44.033511   26.371498
    2020-10-01 00:00:00+02:00   39.455197   54.468722   31.063728
    """
    if freq not in ("MS", "QS", "AS"):
        raise ValueError("Argument 'freq' must be one of {'MS', 'QS', 'AS'}.")
    if freq_up_or_down(bpoframe.index.freq, freq) == 1:
        warn("This conversion includes upsampling, e.g. from yearly to monthly values.")

    return tseries2bpoframe(bpoframe2tseries(bpoframe, "H", prefix), freq, prefix)

