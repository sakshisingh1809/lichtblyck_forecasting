# -*- coding: utf-8 -*-
"""
Attributes for custom dataframes, series, and indices.
"""

import pandas as pd
import numpy as np
from ..tools.tools import wavg


# def _w_ofblock(fr: pd.DataFrame) -> pd.Series:
#     """
#     Returns the power [MW] timeseries of a dataframe by aggregating from below.

#     Parameters
#     ----------
#     fr : pd.DataFrame

#     Returns
#     -------
#     pd.Series
#         The power [MW] timeseries of this dataframe.

#     Notes
#     -----
#     If dataframe has key (column) "w", return that.
#     Else, recursively add all `.w` attributes of child-dataframes.
#     """
#     try:
#         return fr["w"]
#     except KeyError:
#         pass
#     return sum([fr[c].w for c in fr.columns.get_level_values(0).unique()])


# def _p_ofblock(fr: pd.DataFrame) -> pd.Series:
#     """
#     Returns the price [Eur/MWh] timeseries of a dataframe by aggregating from below.

#     Parameters
#     ----------
#     fr : pd.DataFrame

#     Returns
#     -------
#     pd.Series
#         The price [Eur/MWh] timeseries of this dataframe.

#     Notes
#     -----
#     If dataframe has key (column) "p", return that.
#     Else, average all `.p` attributes of child-dataframes, weighted with their
#     `.w` attribute.
#     """
#     try:
#         return fr["p"]
#     except KeyError:
#         pass

#     return (
#         sum([fr[c].p * fr[c].w for c in fr.columns.get_level_values(0).unique()]) / fr.w
#     )


def _power(fr: pd.DataFrame) -> pd.Series:
    """
    Return power [MW] series of DataFrame.

    Column 'w' if present, or else calculate from quantity [MWh].
    """
    try:  # return correct column, if present
        return fr["w"]
    except KeyError:
        pass
    duration = fr.duration
    try:  # else, calculate. (NB: .q to allow quantity to be calculated)
        return (fr.q / duration).rename("w")
    except (KeyError, AttributeError):
        raise AttributeError(
            "No `w` column present, and not enough information to calculate it."
        )


def _price(fr: pd.DataFrame) -> pd.Series:
    """
    Return price [Eur/MWh] series of DataFrame.

    Column 'p' if present, or else calculate from revenue [Eur] and quantity [MWh].
    """
    try:  # return correct column, if present
        return fr["p"]
    except KeyError:
        pass
    try:  # else, calculate. (NB: .q to allow quantity to be calculated.)
        return (fr["r"] / fr.q).rename("p")
    except (KeyError, AttributeError):
        raise AttributeError(
            "No `p` column present, and not enough information to calculate it."
        )


def _quantity(fr: pd.DataFrame) -> pd.Series:
    """
    Return quantity [MWh] series of DataFrame.

    Column 'q' if present, or else calculate from power [MW].
    """
    try:  # return correct column, if present
        return fr["q"]
    except KeyError:
        pass
    try:  # else, calculate
        return (fr["w"] * fr.duration).rename("q")
    except (KeyError, AttributeError):
        pass
    try:  # else, calculate
        return (fr["r"] / fr["p"]).rename("q")
    except KeyError:
        raise AttributeError(
            "No `q` column present, and not enough information to calculate it."
        )


def _revenue(fr: pd.DataFrame) -> pd.Series:
    """
    Return revenue [Eur] series of DataFrame.

    Column 'r' if present, or else calculate from price [Eur/MWh] and quantity [MWh].
    """
    try:  # return correct column, if present
        return fr["r"]
    except KeyError:
        pass
    try:  # else, calculate. (NB: .q to allow quantity to be calculated.)
        return (fr.q * fr["p"]).rename("r")
    except (KeyError, AttributeError):
        raise AttributeError(
            "No `r` column found, and not enough information to calculate it."
        )


def _duration(fr: pd.core.generic.NDFrame) -> pd.Series:
    """
    Return duration [h] of each timestamp in index of DataFrame or Series.
    """
    return (_ts_right(fr) - fr.index).apply(lambda td: td.total_seconds()/3600)
    i = fr.index
    if i.tz is None:
        raise AttributeError("Index is missing timezone information.")

    # Get duration in h for each except final datapoint.
    # . This one breaks for 'MS':
    # duration = ((i + pd.DateOffset(nanoseconds=i.freq.nanos)) - i).total_seconds()/3600
    # . This drops a value at some DST transitions:
    # duration = (i.shift(1) - i).total_seconds() / 3600
    # . This one doesn't give enough values, and adding one with '+ i.freq' gives wrong timestamp at DST transitions:
    # duration = (i[1:] - i[:-1]).total_seconds() / 3600
    if i.freq == "15T":
        duration = [0.25] * len(i)
    elif i.freq == "H":
        duration = [1] * len(i)
    else:
        if i.freq == "D":
            kwargs = {"days": 1}
        elif i.freq == "MS":
            kwargs = {"months": 1}
        elif i.freq == "QS":
            kwargs = {"months": 3}
        elif i.freq == "AS":
            kwargs = {"years": 1}
        else:
            ValueError(f"Invalid frequency: {i.freq}.")
        duration = ((i + pd.DateOffset(**kwargs)) - i).total_seconds() / 3600
    # Get duration in h of final datapoint.
    # if i.freq is not None:
    #     final_duration = ((i[-1] + i.freq) - i[-1]).total_seconds() / 3600
    # else:
    #     final_duration = np.median(duration)

    # Add duration of final datapoint.
    return pd.Series(duration, i)

def _ts_right(fr: pd.core.generic.NDFrame) -> pd.Series:
    i = fr.index
    if i.tz is None:
        raise AttributeError("Index is missing timezone information.")

    # Get right timestamp for each index value, based on the frequency.
    # . This one breaks for 'MS':
    # (i + pd.DateOffset(nanoseconds=i.freq.nanos))
    # . This drops a value at some DST transitions:
    # (i.shift(1))
    # . This one gives wrong value at DST transitions:
    # i + i.freq
    if i.freq == "15T":
        ts_right = i + pd.Timedelta(hours=0.25)
    elif i.freq == "H":
        ts_right = i + pd.Timedelta(hours=1)
    else:
        if i.freq == "D":
            kwargs = {"days": 1}
        elif i.freq == "MS":
            kwargs = {"months": 1}
        elif i.freq == "QS":
            kwargs = {"months": 3}
        elif i.freq == "AS":
            kwargs = {"years": 1}
        else:
            ValueError(f"Invalid frequency: {i.freq}.")
        ts_right = i + pd.DateOffset(**kwargs)
    return pd.Series(ts_right, i)