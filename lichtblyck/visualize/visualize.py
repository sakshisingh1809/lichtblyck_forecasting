"""
Visualize portfolio lines, etc.
"""

from typing import Dict
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
    class LbCd:
        YELLOW = Color(*mpl.colors.to_rgb("#FA9610"))
        SMOKE = Color(*mpl.colors.to_rgb("#465B68"))
        BERRY = Color(*mpl.colors.to_rgb("#D4496A"))
        AQUA = Color(*mpl.colors.to_rgb("#0BA3C1"))
        GREEN = Color(*mpl.colors.to_rgb("#077144"))
        MOSS = Color(*mpl.colors.to_rgb("#658A7C"))

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


def plot_timeseries(ax: plt.Axes, s: pd.Series, how: str = "step", **kwargs) -> None:
    """Plot timeseries `s` to axis `ax`. 

    Other Parameters
    ----------------
    how : str, optional
        One of {'step' (default), 'line', 'bar'}.
    **kwargs : any formatting passed to the plot function.
    """
    how = how.lower()[0:1]
    if how == "s":
        ax.hlines(s, s.index.start_time, (s.index + 1).start_time, **kwargs)
    elif how == "l":
        ax.plot(s.index, s, **kwargs)
    elif how == "b":
        ax.bar(s.index, s, 365 * 0.9, **kwargs)
    else:
        raise ValueError("Parameter `how` must be one of {'step', 'line', 'bar'}.")


mpl.style.use("seaborn")
