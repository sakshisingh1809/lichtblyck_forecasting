"""Create somewhat realistic curves."""
from typing import Tuple
import pandas as pd
import numpy as np
from ...prices.utils import is_peak_hour


def w_offtake(
    i: pd.DatetimeIndex,
    avg: float = 100,
    year_amp: float = 0.20,
    week_amp: float = 0.10,
    day_amp: float = 0.30,
    rand_amp: float = 0.02,
    has_unit: bool = True,
) -> pd.Series:
    """Create a more or less realistic-looking offtake timeseries.

    Parameters
    ----------
    i : pd.DatetimeIndex
        Timestamps for which to create offtake.
    avg : float, optional (default: 100)
        Average offtake in MW.
    year_amp : float, optional (default: 0.2)
        Yearly amplitude as fraction of average. If positive: winter offtake > summer offtake.
    week_amp : float, optional (default: 0.1)
        Weekly amplitude as fraction of average. If positive: midweek offtake > weekend offtake.
    day_amp : float, optional (default: 0.3)
        Day amplitude as fraction of average. If positive: midday offtake > night offtake.
    rand_amp : float, optional (default: 0.02)
        Random amplitude as fraction of average.
    has_unit : bool, optional (default: True)
        If True, return Series with pint unit in MW.

    Returns
    -------
    pd.Series
        Offtake timeseries.
    """
    if year_amp + day_amp + week_amp + rand_amp > 1:
        raise ValueError(
            f"Sum of fractional amplitudes ({year_amp:.1%} and {day_amp:.1%} and {week_amp:.1%} and {rand_amp:.1%}) should not exceed 100%."
        )
    # year angle: 1jan0:00..1jan0:00 -> 0..2pi
    ya = i.map(lambda ts: ts.dayofyear) / 365 * np.pi * 2
    # week angle: Sun0:00..Sun0:00 -> 0..2pi
    wa = (i.map(lambda ts: (ts.weekday() + 1) * 24 + ts.hour) / 168) * np.pi * 2
    # day angle: 0:00..0:00 -> 0..2pi
    da = i.map(lambda ts: (ts.hour * 60 + ts.minute)) / 1440 * np.pi * 2
    # Values: max mid-Jan, mid-week, at 15:00
    yv = year_amp * np.cos(ya - 0.28)
    wv = week_amp * (0.5 - 0.8 * np.cos(wa) - np.cos(2 * wa) / 2 - np.cos(3 * wa) / 6)
    dv = day_amp * (-0.7 * np.cos(da - 0.79) - 0.3 * np.sin(da * 2))
    rv = rand_amp * (1 + 2 * np.random.rand(len(i)))  # TODO: Random mean-reverting walk
    s = pd.Series(avg * (1 + yv + dv + wv + rv), i, name="w")
    return s if not has_unit else s.astype("pint[MW]")


def p_marketprices(
    i: pd.DatetimeIndex,
    avg: float = 100,
    year_amp: float = 0.30,
    week_amp: float = 0.05,
    peak_amp: float = 0.30,
    has_unit: bool = True,
) -> pd.Series:
    """Create a more or less realistic-looking forward price curve timeseries.

    Parameters
    ----------
    i : pd.DatetimeIndex
        Timestamps for which to create prices.
    avg : float, optional (default: 100)
        Average price in Eur/MWh.
    year_amp : float, optional (default: 0.3)
        Yearly amplitude as fraction of average. If positive: winter prices > summer prices.
    week_amp : float, optional (default: 0.05)
        Weekly amplitude as fraction of average. If positive: midweek prices > weekend prices.
    peak_amp : float, optional (default: 0.3)
        Peak-offpeak amplitude as fraction of average. If positive: peak prices > offpeak prices.
    has_unit : bool, optional (default: True)
        If True, return Series with pint unit in Eur/MWh.

    Returns
    -------
    pd.Series
        Price timeseries.
    """
    if year_amp + week_amp + peak_amp > 1:
        raise ValueError(
            f"Sum of fractional amplitudes ({year_amp:.1%} and {week_amp:.1%} and {peak_amp:.1%}) should not exceed 100%."
        )
    # year angle: 1jan0:00..1jan0:00 -> 0..2pi. But: uniform within month
    ya = i.map(lambda ts: ts.month) / 12 * np.pi * 2
    # week angle: Sun0:00..Sun0:00 -> 0..2pi. But: uniform within day.
    wa = i.map(lambda ts: ts.weekday() + 1) / 7 * np.pi * 2
    # peak fraction: -1 (middle of offpeak hours) .. 1 (middle of peak hours)
    if i.freq in ["H", "15T"]:
        b = np.array([0.5, 0.8, 1, 0.8, 0.5])
        if i.freq == "15T":  # repeat every value 4 times
            b = np.array([[bb, bb, bb, bb] for bb in b]).flatten()
        pa = np.convolve(-1 + 2 * i.map(is_peak_hour), b / sum(b), mode="same")
    else:
        pa = np.zeros(len(i))
    # Values
    yv = year_amp * np.cos(ya - 0.35)  # max in feb
    wv = week_amp * np.cos(wa - 1.07)  # max on tuesday
    pv = peak_amp * pa
    s = pd.Series(avg * (1 + yv + wv + pv), i, name="p")
    return s if not has_unit else s.astype("pint[Eur/MWh]")


def wp_sourced(
    w_offtake: pd.Series,
    freq: str = "MS",
    w_avg: float = 0.6,
    p_avg: float = 100,
    rand_amp: float = 0.2,
    has_unit: bool = True,
) -> Tuple[pd.Series]:
    """Create a more or less realistic-looking sourcing volume and sourcing price timeseries.

    Parameters
    ----------
    w_offtake : pd.Series
        Offtake volume timeseries for which to create sourced volume and price timeseries.
    freq : str, optional (default: 'MS')
        Frequency within which sourcing volume and price are uniform.
    w_avg : float, optional (default: 0.6)
        Average sourced fraction.
    p_avg : float, optional (default: 100)
        Average hedge price in Eur/MWh.
    rand_amp : float, optional (default: 0.2)
        Random amplitude, both of sourced fraction (absolute) and of price (as fraction).
    has_unit : bool, optional (default: True)
        If True, return Series with unit.
        - volume series: same unit as ``w_offtake`` or else with pint unit in MW.
        - price series: with pint unit in Eur/MWh.

    Returns
    -------
    (pd.Series, pd.Series)
        Sourced volume timeseries and sourced price timeseries.
    """
    # Prepare series for resampling.
    if hasattr(w_offtake, "pint"):
        w_unit = w_offtake.pint.units
        sin = -1 * w_offtake.pint.magnitude
    else:
        w_unit = None
        sin = -1 * w_offtake

    # Do resampling.
    def fn(sub_s):
        wval = sub_s.mean() * (w_avg + rand_amp * np.random.uniform(-1, 1))
        pval = p_avg * (1 + rand_amp * np.random.uniform(-1, 1))
        return pd.DataFrame({"p": pval, "w": wval}, sub_s.index)

    if sin.index.freq in ["15T", "H"]:
        df = sin.groupby(is_peak_hour).apply(lambda s: s.resample("MS").apply(fn))
    else:
        df = sin.resample(freq).apply(fn)
    w, p = df.w, df.p

    # Add unit if wanted.
    if has_unit:
        wunit = f"pint[{w_unit}]" if w_unit is not None else "pint[MW]"
        w = w.astype(wunit)
        p = p.astype("pint[Eur/MWh]")
    return w, p
