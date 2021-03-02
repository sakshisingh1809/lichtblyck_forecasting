# -*- coding: utf-8 -*-
"""
Module with tools to modify and standardize dataframes.
"""
from typing import Iterable, Callable, Union
import pandas as pd
import numpy as np


def set_ts_index(
    df: pd.DataFrame, column: str = None, bound: str = "try", tz: str = "Europe/Berlin"
) -> pd.DataFrame:
    """
    Create and add a standardized timestamp index to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str, optional
        Column to create the timestamp from. Use existing index if none specified.
    bound : {'try', 'left', 'right'}, optional
        If 'left' ('right'), specifies that input timestamps are left-(right-)bound.
        If 'try', will first try 'left', then 'right' if exception occurs. Will give
        false results if the data contains no summertime to wintertime changeover and
        times are rightbound.
    tz : str, optional
        Timezone of the input dataframe; used only if input dataframe contains
        timezone-agnostic timestamps. The default is "Europe/Berlin".

    Returns
    -------
    pd.DataFrame
        Dataframe with left-bound timestamp index in the Europe/Berlin timezone.
        Column 'column' is removed, and index renamed to 'ts_left'.
    """
    if column:
        df = df.set_index(column)
    else:
        df = df.copy()  # don't change passed-in df

    df.index = pd.DatetimeIndex(df.index)  # turn / try to turn into datetime

    if bound == "left":
        pass
    elif bound == "right":
        minutes = (df.index[1] - df.index[0]).seconds / 60
        df.index += pd.Timedelta(minutes=-minutes)
    else:
        try:
            return set_ts_index(df, None, "left", tz)
        except:
            return set_ts_index(df, None, "right", tz)
    df.index.name = "ts_left"

    if df.index.tz is None:
        try:
            df = df.tz_localize(tz, ambiguous="infer")
        except:
            df = df.tz_localize(tz, ambiguous="NaT")
    df = df.tz_convert("Europe/Berlin")

    if df.index.freq is None:
        df.index.freq = pd.infer_freq(df.index)
    if df.index.freq is None:
        df = df.resample((df.index[1:] - df.index[:-1]).median()).asfreq()
    if df.index.freq is None:
        # (infer_freq does not work during summer-to-wintertime changeover)
        df.index.freq = (df.index[1:] - df.index[:-1]).median()

    return df


def fill_gaps(s: pd.Series, maxgap=2) -> pd.Series:
    """Fill gaps in series by linear interpolation."""
    is_gap = s.isna()
    next_gap = is_gap.shift(-1)
    prev_gap = is_gap.shift(1)
    index_beforegap = s[~is_gap & next_gap].index
    index_aftergap = s[~is_gap & prev_gap].index
    # remove orphans at beginning and end
    if index_beforegap.empty:
        return s
    elif index_beforegap[-1] > index_aftergap[-1]:
        index_beforegap = index_beforegap[:-1]
    if index_aftergap.empty:
        return s
    elif index_aftergap[0] < index_beforegap[0]:
        index_aftergap = index_aftergap[1:]
    s = s.copy()
    for i_before, i_after in zip(index_beforegap, index_aftergap):
        section = s.loc[i_before:i_after]
        if len(section) > maxgap + 2:
            continue
        x0, y0, x1, y1 = i_before, s[i_before], i_after, s[i_after]
        dx, dy = x1 - x0, y1 - y0
        s.loc[i_before:i_after] = y0 + (section.index - x0) / dx * dy
    return s


def wavg(
    fr: Union[pd.DataFrame, pd.Series],
    weights: Union[Iterable, pd.Series] = None,
    axis: int = 0,
) -> Union[pd.Series, float]:
    """
    Weighted average of dataframe.

    Parameters
    ----------
    fr : Union[pd.DataFrame, pd.Series]
        The input values.
    weights : Union[Iterable, pd.Series], optional
        The weights. If provided as a Series, the weights and values are aligned along its index. If no weights are provided, the normal (unweighted) average is returned instead.
    axis : int, optional
        Calculate each column's average over all rows (if axis==0, default) or
        each row's average over all columns (if axis==1). Ignored for Series.

    Returns
    -------
    Union[pd.Series, float]
        The weighted average. A single float if `fr` is a Series; a Series if
        `fr` is a Dataframe.
    """
    if axis == 1:  # correct allignment
        fr = fr.T
    if weights is None:  # return non-weighted average if no weights are provided
        return fr.mean()
    return fr.mul(weights, axis=0).sum(skipna=False) / sum(weights)


def __is(letter: str) -> Callable[[str], bool]:
    """Returns function that tests if its argument is 'letter' or starts with
    'letter_'."""

    @np.vectorize
    def check(name):
        return name == letter or name.startswith(letter + "_")

    return check


_is_price = __is("p")
_is_quantity = __is("q")
_is_temperature = __is("t")
_is_revenue = __is("r")
_is_power = __is("w")
