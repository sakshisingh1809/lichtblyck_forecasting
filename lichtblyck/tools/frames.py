# -*- coding: utf-8 -*-
"""
Module with tools to modify and standardize dataframes.
"""

from .stamps import FREQUENCIES, freq_up_or_down, trim_index
from pandas.core.frame import NDFrame
from typing import Iterable, Callable, Union
import pandas as pd
import numpy as np


def set_ts_index(
    fr: NDFrame,
    column: str = None,
    bound: str = "left",
    tz: str = "Europe/Berlin",
    continuous: bool = True,
) -> NDFrame:
    """
    Create and add a standardized, continuous timestamp index to a dataframe.

    Parameters
    ----------
    fr : NDFrame
        Pandas series or dataframe.
    column : str, optional
        Column to create the timestamp from. Used only if `fr` is DataFrame; ignored
        otherwise. Use existing index if none specified.
    bound : str, {'left' (default), 'right'}
        If 'left' ('right'), specifies that input timestamps are left-(right-)bound.
    tz : str, optional (default: "Europe/Berlin")
        Timezone of the input frame; used only if input contains timezone-agnostic
        timestamps.
    continuous : bool, optional (default: True)
        If true, raise error if data has missing values (i.e., gaps).

    Returns
    -------
    NDFrame
        Same type as `fr`, with left-bound timestamp index in the Europe/Berlin timezone.
        Column `column` (if applicable) is removed, and index renamed to 'ts_left'.
    """
    if column and isinstance(fr, pd.DataFrame):
        fr = fr.set_index(column)
    else:
        fr = fr.copy()  # don't change passed-in fr

    # Make leftbound.

    fr.index = pd.DatetimeIndex(fr.index)  # turn / try to turn into datetime

    if bound == "left":
        pass
    elif bound.startswith("right"):
        if bound == "right":
            # At start of DST:
            # . Leftbound timestamps contain 3:00 but not 2:00.
            # . Rightbound timestamps: ambiguous. May contain 3:00 but not 2:00 (A), or vice versa (B): try both
            try:
                return set_ts_index(fr, None, "rightA", tz)
            except:
                return set_ts_index(fr, None, "rightB", tz)
        minutes = (fr.index[1] - fr.index[0]).seconds / 60
        if bound == "rightA":
            fr.loc[fr.index[0] + pd.Timedelta(minutes=-minutes)] = np.nan
            fr = pd.concat([fr.iloc[-1:], fr.iloc[:-1]]).shift(-1).dropna()
        if bound == "rightB":
            fr.index += pd.Timedelta(minutes=-minutes)
    else:
        raise ValueError("`bound` must be one of {'left' (default), 'right'}.")
    fr.index.name = "ts_left"

    # Set Europe/Berlin timezone.

    if fr.index.tz is None:
        try:
            fr = fr.tz_localize(tz, ambiguous="infer")
        except:
            fr = fr.tz_localize(tz, ambiguous="NaT")
    fr = fr.tz_convert("Europe/Berlin")

    # Set frequency.

    if fr.index.freq is None:
        fr.index.freq = pd.infer_freq(fr.index)
    if fr.index.freq is None:
        # (infer_freq does not always work, e.g. during summer-to-wintertime changeover)
        tdelta = (fr.index[1:] - fr.index[:-1]).median()
        for freq, (tdelta_min, tdelta_max) in {
            "D": (pd.Timedelta(hours=23), pd.Timedelta(hours=25)),
            "MS": (pd.Timedelta(days=27), pd.Timedelta(days=32)),
            "QS": (pd.Timedelta(days=89), pd.Timedelta(days=93)),
            "AS": (pd.Timedelta(days=364), pd.Timedelta(days=367)),
        }.items():
            if tdelta >= tdelta_min and tdelta <= tdelta_max:
                break
        else:
            freq = tdelta
        fr2 = fr.resample(freq).asfreq()
        # If the new dataframe has additional rows, the original dataframe was not gapless.
        if continuous and len(fr2) > len(fr):
            missing = [i for i in fr2.index if i not in fr.index]
            raise ValueError(
                f"`fr` does not have continuous data; missing data for: {missing}."
            )
        fr = fr2

    # Check if frequency all ok.

    if fr.index.freq is None:
        raise ValueError("Cannot find a frequency in `fr`.")
    elif fr.index.freq not in FREQUENCIES:
        for freq in ["MS", "QS"]:  # Edge case: month-/quarterly but starting != Jan.
            if freq_up_or_down(fr.index.freq, freq) == 0:
                fr.index.freq = freq
                break
        else:
            raise ValueError(
                f"Found unsupported frequency ({fr.index.freq}). Must be one of: {FREQUENCIES}."
            )

    return fr


def fill_gaps(fr: NDFrame, maxgap: int = 2) -> NDFrame:
    """Fill gaps in series by linear interpolation.

    Parameters
    ----------
    fr : NDFrame
        Pandas Series or DataFrame.
    maxgap : int, optional
        Maximum number of rows that are filled. Larger gaps are kept. Default: 2.

    Returns
    -------
    pd.Series
        Series with gaps filled up.
    """
    if isinstance(fr, pd.DataFrame):
        return pd.DataFrame({c: fill_gaps(s, maxgap) for c, s in fr.items()})

    is_gap = fr.isna()
    next_gap = is_gap.shift(-1)
    prev_gap = is_gap.shift(1)
    index_beforegap = fr[~is_gap & next_gap].index
    index_aftergap = fr[~is_gap & prev_gap].index
    # remove orphans at beginning and end
    if index_beforegap.empty:
        return fr
    elif index_beforegap[-1] > index_aftergap[-1]:
        index_beforegap = index_beforegap[:-1]
    if index_aftergap.empty:
        return fr
    elif index_aftergap[0] < index_beforegap[0]:
        index_aftergap = index_aftergap[1:]
    fr = fr.copy()
    for i_before, i_after in zip(index_beforegap, index_aftergap):
        section = fr.loc[i_before:i_after]
        if len(section) > maxgap + 2:
            continue
        x0, y0, x1, y1 = i_before, fr[i_before], i_after, fr[i_after]
        dx, dy = x1 - x0, y1 - y0
        fr.loc[i_before:i_after] = y0 + (section.index - x0) / dx * dy
    return fr


def wavg(
    fr: Union[pd.Series, pd.DataFrame],
    weights: Union[Iterable, pd.Series, pd.DataFrame] = None,
    axis: int = 0,
) -> Union[pd.Series, float]:
    """
    Weighted average of dataframe.

    Parameters
    ----------
    fr : Union[pd.Series, pd.DataFrame]
        The input values.
    weights : Union[Iterable, pd.Series, pd.DataFrame], optional
        The weights. If provided as a Series, the weights and values are aligned along
        its index. If no weights are provided, the normal (unweighted) average is returned
        instead.
    axis : int, optional
        Calculate each column's average over all rows (if axis==0, default) or
        each row's average over all columns (if axis==1). Ignored for Series.

    Returns
    -------
    Union[pd.Series, float]
        The weighted average. A single float if `fr` is a Series; a Series if
        `fr` is a Dataframe.
    """
    # Orient so that we can always sum over rows.
    if axis == 1:
        if isinstance(fr, pd.DataFrame):
            fr = fr.T
        if isinstance(weights, pd.DataFrame):
            weights = weights.T

    if weights is None:
        # Return normal average.
        return fr.mean(axis=0)

    # Turn weights into Series if it's an iterable.
    if not isinstance(weights, pd.DataFrame) and not isinstance(weights, pd.Series):
        weights = pd.Series(weights, fr.index)

    summed = fr.mul(weights, axis=0).sum(skipna=False)  # float or float-Series
    totalweight = weights.sum()  # float or float-Series
    result = summed / totalweight

    # Correction: if total weight is 0, and all original values are the same, keep the original value.
    correctable = np.isclose(totalweight, 0) & (
        fr.nunique() == 1
    )  # bool or bool-Series
    if isinstance(fr, pd.Series):
        return result if not correctable else fr.iloc[0]
    result[correctable] = fr.iloc[0, :][correctable]
    return result


def trim_frame(fr: NDFrame, freq: str) -> NDFrame:
    """Trim index of frame to only keep full periods of certain frequency.

    Parameters
    ----------
    fr : NDFrame
        The (untrimmed) pandas series or dataframe.
    freq : str
        Frequency to trim to. E.g. 'MS' to only keep full months.

    Returns
    -------
    NDFrame
        Subset of `fr`, with same frequency.
    """
    i = trim_index(fr.index, freq)
    return fr.loc[i]


def series_allclose(s1, s2):
    """Compare if all values in series are equal/close. Works with series that have units."""
    without_units = [not (hasattr(s, "pint")) for s in [s1, s2]]
    if all(without_units):
        return np.allclose(s1, s2)
    elif any(without_units):
        return False
    elif s1.pint.dimensionality != s2.pint.dimensionality:
        return False
    # Both have units, and both have same dimensionality (e.g. 'length'). Check values.
    s1_vals = s1.pint.m
    s2_vals = s2.astype(f"pint[{s1.pint.u}]").pint.m
    return np.allclose(s1_vals, s2_vals)
