"""
Visualize portfolio lines, etc.
"""

from ..tools.stamps import freq_shortest
from typing import Dict, List, Optional, Iterable
from collections import namedtuple
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import colorsys


class Color(namedtuple("RGB", ["r", "g", "b"])):
    """Class to create an rgb color tuple, with additional methods."""

    def lighten(self, value):
        """Lighten the color by fraction `value`. If `value` < 0, darken."""
        h, l, s = np.array(colorsys.rgb_to_hls(*mpl.colors.to_rgb(self)))
        l += value * ((1 - l) if value > 0 else l)
        return Color(*[min(max(comp, 0), 1) for comp in colorsys.hls_to_rgb(h, l, s)])

    darken = lambda self, value: self.lighten(self, -value)

    light = property(lambda self: self.lighten(0.3))
    xlight = property(lambda self: self.lighten(0.6))
    dark = property(lambda self: self.lighten(-0.3))
    xdark = property(lambda self: self.lighten(-0.6))


class Colors:
    class General:
        PURPLE = Color(0.549, 0.110, 0.706)
        GREEN = Color(0.188, 0.463, 0.165)
        BLUE = Color(0.125, 0.247, 0.600)
        ORANGE = Color(0.961, 0.533, 0.114)
        RED = Color(0.820, 0.098, 0.114)
        YELLOW = Color(0.945, 0.855, 0.090)
        LBLUE = Color(0.067, 0.580, 0.812)
        LGREEN = Color(0.325, 0.773, 0.082)
        BLACK = Color(0, 0, 0)
        WHITE = Color(1, 1, 1)

    class LbCd:  # Lichtblick corporate design
        YELLOW = Color(*mpl.colors.to_rgb("#FA9610"))
        SMOKE = Color(*mpl.colors.to_rgb("#465B68"))
        BERRY = Color(*mpl.colors.to_rgb("#D4496A"))
        AQUA = Color(*mpl.colors.to_rgb("#0BA3C1"))
        GREEN = Color(*mpl.colors.to_rgb("#077144"))
        MOSS = Color(*mpl.colors.to_rgb("#658A7C"))

    class Wqpr:  # Standard colors when plotting a portfolio
        w = Color(*mpl.colors.to_rgb("#0066CC"))
        q = Color(*mpl.colors.to_rgb("#0099FF"))
        r = Color(*mpl.colors.to_rgb("#339933")).light
        p = Color(*mpl.colors.to_rgb("#758C2C"))


def _index2labels(index: pd.DatetimeIndex) -> List[str]:
    """Create labels corresponding to the timestamps in the index."""
    if index.freq == "AS":

        def label(ts, i):
            return f"{ts.year}"

    elif index.freq == "QS":

        def label(ts, i):
            num = ts.quarter
            return f"Q{num}\n" + (f"{ts.year}" if i == 0 or num == 1 else "")

    elif index.freq == "MS":

        def label(ts, i):
            num, name = ts.month, ts.month_name()[:3]
            return f"{name}\n" + (f"{ts.year}" if i == 0 or num == 1 else "")

    else:
        raise ValueError("Daily (or shorter) data should not be plotted as categories.")
    return [label(ts, i) for i, ts in enumerate(index)]


def _categories(ax: plt.Axes, s: pd.Series, cat: bool = None) -> Optional[Iterable]:
    """Category labels for `s`. Or None, if s should be plotted on time axis."""
    if (  # We use categorical data if it's...
        ((ax.lines or ax.collections or ax.containers) and ax.xaxis.have_units())  # set
        or (cat is None and freq_shortest(s.index.freq, "MS") == "MS")  # the default
        or cat  # the user's wish
    ):
        return _index2labels(s.index)
    return None


docstringliteral_plotparameters = """
Other parameters
----------------
cat : bool, optional
    If False, plots x-axis as timeline with timestamps spaced according to their 
    duration. If True, plots x-axis categorically, with timestamps spaced equally. 
    Disregarded if `ax` already has values (then: use whatever is already set). 
    Otherwise, if missing, use True if `s` has a monthly frequency or longer, False
    if the frequency is shorter than monthly.
**kwargs : any formatting are passed to the Axes plot method being used."""


def append_to_doc(text):
    def decorator(func):
        func.__doc__ += f"\n{text}"
        return func

    return decorator


@append_to_doc(docstringliteral_plotparameters)
def plot_timeseries_as_bar(
    ax: plt.Axes, s: pd.Series, cat: bool = None, **kwargs
) -> None:
    """Plot timeseries `s` to axis `ax`, as bar graph."""
    categories = _categories(ax, s, cat)
    if categories:
        ax.bar(categories, s.values, 0.8, **kwargs)
    else:
        x = s.index + 0.5 * (s.index.ts_right - s.index)
        width = (s.index.duration / 24).median() * 0.8
        ax.bar(x, s.values, width, **kwargs)


@append_to_doc(docstringliteral_plotparameters)
def plot_timeseries_as_hline(
    ax: plt.Axes, s: pd.Series, cat: bool = None, **kwargs
) -> None:
    """Plot timeseries `s` to axis `ax`, as horizontal lines."""
    categories = _categories(ax, s, cat)
    if categories:
        x = np.arange(len(categories))
        ax.hlines(s.values, x - 0.4, x + 0.4, label=categories, **kwargs)
    else:
        ax.hlines(s.values, s.index, s.index.ts_right, **kwargs)


@append_to_doc(docstringliteral_plotparameters)
def plot_timeseries_as_line(
    ax: plt.Axes, s: pd.Series, cat: bool = None, **kwargs
) -> None:
    """Plot timeseries `s` to axis `ax`, as jagged line."""
    categories = _categories(ax, s, cat)
    if categories:
        ax.plot(categories, s.values, **kwargs)
    else:
        ax.plot(s.index, s.values, **kwargs)


@append_to_doc(docstringliteral_plotparameters)
def plot_timeseries_as_step(
    ax: plt.Axes, s: pd.Series, cat: bool = None, **kwargs
) -> None:
    """Plot timeseries `s` to axis `ax`, as stepped line (horizontal and vertical lines)."""
    categories = _categories(ax, s, cat)
    if categories:  # Cannot create a step graph on a categories axis.
        x = np.arange(len(categories) + 1) - 0.5  # add final datapoint
        y = [*s.values, s.values[-1]]  # repeat final datapoint
        ax.step(x, y, where="post", **kwargs)
    else:
        x = [*s.index, s.index.ts_right[-1]]  # add final datapoint
        y = [*s.values, s.values[-1]]  # repeat final datapoint
        ax.step(x, y, where="post", **kwargs)


def plot_timeseries(
    ax: plt.Axes, s: pd.Series, cat: bool = None, how: str = "hline", **kwargs
) -> None:
    """Plot timeseries `s` to axis `ax`, as (`how`) 'hline' (default), 'step', 'line', 
    or 'bar'. 
    """
    if how == "hline":
        plot_timeseries_as_hline(ax, s, cat, **kwargs)
    elif how == "line":
        plot_timeseries_as_line(ax, s, cat, **kwargs)
    elif how == "step":
        plot_timeseries_as_step(ax, s, cat, **kwargs)
    elif how == "bar":
        plot_timeseries_as_bar(ax, s, cat, **kwargs)
    else:
        raise ValueError("`how` must be one of {'hline', 'step', 'line', 'bar'}.")


mpl.style.use("seaborn")
