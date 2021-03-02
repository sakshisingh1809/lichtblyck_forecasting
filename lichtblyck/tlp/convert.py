# -*- coding: utf-8 -*-
"""
Convert between various ways that tlp profiles are stored / communicated. Con-
vert to function consumption[MW] = f(temperature[degC], timestamp).
"""

from ..core import pfseries_pfframe  # pandas extensions
from typing import Callable
from datetime import datetime, date
import pandas as pd
import numpy as np
import functools


def series2function(tlp_s: pd.Series) -> Callable[[pd.Series], pd.Series]:
    """
    Convert tlp-series into tlp-function.

    Parameters
    ----------
    tlp_s : pd.Series
        Series with one or more of following index level names:
        . 't' for temperature [degC];
        . 'time_left_local' for time-of-day;
        . 'weekday' for weekday (monday == 0 .. sunday == 6);
        . 'month' for month number (1-12)
        and consumption [MW] as values.

    Returns
    -------
    Callable[[pd.Series], pd.Series]
        Function that takes a temperature [degC] timeseries as input and
        returns the consumption [MW] timeseries as output.
        
    Notes
    -----
    The following assumptions are made:
    . If tlp_s has >1 index level, that all combinations (i.e., cartesian 
      product) of the individual level values are present in index.
    . That values of 'time_left_local', if present, are equally spaced and 
      cover an entire day.
    . If not all weekday values are specified, that the next-lower value must
      be used (so: if only 0, 5, 6 are specified, the Monday (0) value is used 
      for Friday (4)).
    . Same goes for month values.
    """

    right_on = []  # Join keys
    left_on = []
    vals_avail = {}  # unique values of each level

    for l in range(tlp_s.index.nlevels):
        index = tlp_s.index.get_level_values(l)
        right_on.append(index.name)
        vals_avail[index.name] = index.unique().sort_values()

        if index.name == "t":
            left_on.append("t_avail")

            @functools.lru_cache(1000)
            def t_nearest_avail(t) -> float:
                try:
                    idx = np.nanargmin(np.abs(vals_avail["t"] - t))
                    return vals_avail["t"][idx]
                except ValueError:  # t is nan-value
                    return np.nan

        elif index.name == "time_left_local":
            left_on.append("time")
            time_avail = index.unique().sort_values()
            freq = pd.infer_freq(
                time_avail.map(lambda time: datetime.combine(date.today(), time))
            )
            if freq is None:
                raise ValueError(
                    f"Can't infer frequency from tlp index values ({time_avail})."
                )

        elif index.name == "weekday":
            left_on.append("weekd_avail")

            @functools.lru_cach(7)
            def weekd_nearest_avail(weekd) -> int:
                idx = vals_avail["weekday"].searchsorted(weekd, "right") - 1
                if idx < 0:
                    raise ValueError(
                        f"Can't find weekday {weekd} or lower in tlp index."
                    )
                return vals_avail["weekday"][idx]

        elif index.name == "month":
            left_on.append("month_avail")

            @functools.lru_cach(12)
            def month_nearest_avail(month) -> int:
                idx = vals_avail["month"].searchsorted(month, "right") - 1
                if idx < 0:
                    raise ValueError(f"Can't find month {month} or lower in tlp index.")
                return vals_avail["month"][idx]

        else:
            raise ValueError(f'Index level with unknown name ("{index.name}") found.')

    if "t" not in right_on:
        raise ValueError('No index level with temperatures (name "t") found.')

    def tlp(t: pd.Series):
        # Find nearest available temperature (to look up in index of tlp).
        df = pd.DataFrame({"t": t})
        df["t_avail"] = (
            df["t"].round(1).apply(t_nearest_avail)
        )  # round to (potentially) reduce number of unique values.
        # Stretch to frequency in tlp index.
        if "time" in left_on:
            new_ts = pd.date_range(
                t.index[0],
                t.ts_right[-1],
                freq=freq,
                closed="left",
                tz=t.index.tz,
                name="ts_left",
            )
            df = df.reindex(new_ts).ffill()
            # Only gives good results if all old_ts are elements of new_ts, i.e.,
            # . if len(new_ts) >= len(old_ts); and
            # . if new_ts is further subdivision of each old_ts element.
            df["time"] = df.index.map(
                lambda ts: ts.time
            )  # each element is in index of tlp.

        if "weekd_avail" in left_on:
            df["weekd_avail"] = df.index.map(lambda ts: ts.weekday).map(
                weekd_nearest_avail
            )

        if "month_avail" in left_on:
            df["month_avail"] = df.index.map(lambda ts: ts.month).map(
                month_nearest_avail
            )

        # Do merge to find load.
        merged = df.merge(
            tlp_s.rename("w"), left_on=left_on, right_on=right_on, right_index=True
        )
        load = merged["w"].rename("w")
        load.sort_index(inplace=True)
        load.index.freq = pd.infer_freq(load.index)
        # TODO: don't just use current temperature, but rather also yesterday's as described in standard process.
        return load

    # Add some attributes that might be helpful.
    tlp.vals_avail = vals_avail

    return tlp


def function2function(f: Callable[[float], float]) -> Callable[[pd.Series], pd.Series]:
    """
    Convert offtake function into tlp-function.

    Parameters
    ----------
    f: Callable[[float], float])
        Function that takes one input: the temperature [degC], and calculates 
        one output: the consumption [MW].

    Returns
    -------
    Callable[[pd.Series], pd.Series]
        Function that takes a temperature [degC] timeseries as input and
        returns the consumption [MW] timeseries as output.
    """

    def tlp(t: pd.Series) -> pd.Series:
        return t.apply(f)

    return tlp
