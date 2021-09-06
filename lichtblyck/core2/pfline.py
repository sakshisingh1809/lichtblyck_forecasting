"""
Dataframe-like class to hold general energy-related timeseries.
"""

from __future__ import annotations
from ..tools.frames import set_ts_index
from ..visualize import visualize as vis
from . import utils
from matplotlib import pyplot as plt
from typing import Iterable, Union
import pandas as pd
import numpy as np
import warnings

# Developer notes: we would like to be able to handle 2 cases with volume AND financial
# information. We would like to...
# ... handle the situation where the volume q == 0 but the revenue r != 0, because this
#   occasionally arises for the sourced volume, e.g. after buying and selling the same
#   volume at unequal price. So: we want to be able to store q and r.
# ... keep price information even if the volume q == 0, because at a later time this price
#   might still be needed, e.g. if a perfect hedge becomes unperfect. So: we want to be
#   able to store q and p.
# It is unpractical to cater to both cases, as it raises questions, e.g. when adding
# them, how is the result stored?.
# The first case one is the most important one, and is therefore used. The second case
# must be handled by storing market prices seperately from volume data.


def _unit(attr: str) -> str:
    units = {"q": "MWh", "w": "MW", "p": "Eur/MWh", "r": "Eur", "t": "degC"}
    return units.get(attr, "")


def _unitsline(headerline: str) -> str:
    """Return a line of text with units that line up with the provided header."""
    text = headerline
    for att in ("w", "q", "p", "r"):
        unit = _unit(att)
        to_add = f" [{unit}]"
        text = text.replace(att.rjust(len(to_add)), to_add)
        while to_add not in text and len(unit) > 1:
            unit = unit[:-3] + ".." if len(unit) > 2 else "."
            to_add = f" [{unit}]"
            text = text.replace(att.rjust(len(to_add)), to_add)
    return text


def _make_df(data) -> pd.DataFrame:
    """From data, create a DataFrame with column `q`, column `p`, or columns `q` and `r`.
    Also, do some data verification."""

    # Do checks on indices.
    indices = [
        *[getattr(data.get(key, None), "index", None) for key in "wqpr"],
        getattr(data, "index", None),
    ]  # (final element necessary in case data is a single series)
    indices = [i for i in indices if i is not None]
    if not indices:
        raise ValueError("No index can be found in the data.")
    if len(set([i.freq for i in indices])) != 1:
        raise ValueError("Passed timeseries have unequal frequency; resample first.")

    # Get timeseries.
    def series_or_none(df, col):  # remove series that are passed but only contain na
        s = df.get(col)
        return s if s is not None and not s.isna().all() else None

    if not isinstance(data, PfLine):
        data = set_ts_index(pd.DataFrame(data))  # make df in case info passed as float
    q, w, r, p = [series_or_none(data, key) for key in "qwrp"]

    # Get price information.
    if p is not None and w is None and q is None and r is None:
        # We only have price information. Return immediately.
        return set_ts_index(pd.DataFrame({"p": p}))  # kind == 'p'

    # Get quantity information (and check consistency).
    if q is None and w is None:
        if r is None or p is None:
            raise ValueError("Must supply (a) volume, (b) price, or (c) both.")
        q = r / p
    if q is None:
        q = w * w.duration
    elif w is not None and not np.allclose(q, w * w.duration):
        raise ValueError("Passed values for `q` and `w` not consistent.")

    # Get revenue information (and check consistency).
    if p is None and r is None:
        return set_ts_index(pd.DataFrame({"q": q}))  # kind == 'q'
    if r is None:  # must calculate from p
        r = p * q
        i = r.isna()  # edge case p==nan. If q==0, assume r=0. If q!=0, raise error
        if i.any() and (abs(q[i]) < 0.001).all():
            r[i] = 0
        elif i.any():
            raise ValueError("Found timestamps with `p`==na, `q`!=0. Unknown `r`.")
    elif p is not None and not np.allclose(r, p * q):
        # Edge case: remove lines where p==nan and q==0 before judging consistency.
        i = p.isna()
        if not (abs(q[i]) < 0.001).all() or not np.allclose(r[~i], p[~i] * q[~i]):
            raise ValueError("Passed values for `q`, `p` and `r` not consistent.")
    return set_ts_index(pd.DataFrame({"q": q, "r": r}).dropna())  # kind == 'all'


class PfLine:
    """Class to hold a related energy timeseries. This can be volume timeseries q
    [MWh] and w [MW], or a price timeseries p [Eur/MWh] or both.

    Parameters
    ----------
    data : object
        Generally: object with one or more attributes or items `w`, `q`, `r`, `p`; all
        timeseries. Most commonly a DataFrame but may also be a dictionary or other
        PfLine object.

    Attributes
    ----------
    w, q, p, r : pd.Series
        Power [MW], quantity [MWh], price [Eur/MWh], revenue [Eur] timeseries, when
        available. Can also be accessed by key (e.g., with ['w']).
    kind : str
        Kind of information/timeseries included in instance. One of {'q', 'p', 'all'}.
    available : str
        Columns available in instance. One of {'wq', 'p', 'wqpr'}.
    index : pandas.DateTimeIndex
        Left timestamp of row.
    ts_right, duration : pandas.Series
        Right timestamp, and duration [h] of row.

    Notes
    -----
    When kind == 'all', updating the PfLine means that we must choose how to recalculate
    the individual timeseries to keep the data consistent. In general, keeping the
    existing price is given priority. So, when multiplying the PfLine by 2, `w`, `q` and
    `r` are doubled, while `p` stays the same. And, when updating the volume (with
    `.set_w` or `.set_q`) the revenue is recalculated, and vice versa. Only when the
    price is updated, the existing volume is kept.'
    """

    def __init__(self, data):
        # The only internal data of this class.
        self._df = _make_df(data)

    # Methods/Properties that return most important information.

    index: pd.DatetimeIndex = property(lambda self: self._df.index)
    duration: pd.Series = property(lambda self: self._df.duration)
    ts_right: pd.Series = property(lambda self: self._df.ts_right)

    @property
    def q(self) -> pd.Series:
        if "q" in self._df:
            return self._df["q"]
        return pd.Series(np.nan, self.index, name="q")

    @property
    def w(self) -> pd.Series:
        if "q" in self._df:
            return pd.Series(self._df["q"] / self._df.duration, name="w")
        return pd.Series(np.nan, self.index, name="w")

    @property
    def r(self) -> pd.Series:
        if "r" in self._df:
            return self._df["r"]
        if "q" in self._df and "p" in self._df:
            return pd.Series(self._df["q"] * self._df["p"], name="r")
        return pd.Series(np.nan, self.index, name="r")

    @property
    def p(self) -> pd.Series:
        if "p" in self._df:
            return self._df["p"]
        if "q" in self._df and "r" in self._df:
            return pd.Series(self._df["r"] / self._df["q"], name="p")
        return pd.Series(np.nan, self.index, name="p")

    @property
    def kind(self) -> str:
        """Kind of data that is stored in the object. Possible values and implications:
            'q': volume data only. Properties .q [MWh] and .w [MW] are available.
            'p': price data only. Property .p [Eur/MWh] is available.
            'all': price and volume data. Properties .q [MWh], .w [MW], .p [Eur/MWh],
                .r [Eur] are available.
        """
        if "q" in self._df:
            return "all" if ("r" in self._df or "p" in self._df) else "q"
        if "p" in self._df:
            return "p"
        raise ValueError("Unexpected value for ._df.")

    @property
    def _summable(self) -> str:  # which time series can be added to others
        return {"p": "p", "q": "q", "all": "qr"}[self.kind]

    @property
    def available(self) -> str:  # which time series have values
        return {"p": "p", "q": "qw", "all": "qwrp"}[self.kind]

    def df(self, cols: str = None) -> pd.DataFrame:
        """DataFrame for this PfLine.

        Parameters
        ----------
        cols : str, optional
            The columns to include. Default: include all that are available.

        Returns
        -------
        pd.DataFrame
        """
        if cols is None:
            cols = self.available
        return pd.DataFrame({col: self[col] for col in cols})

    # Methods/Properties that return new PfLine instance.

    volume: PfLine = property(lambda self: PfLine({"q": self.q}))  # possibly nan-Series
    price: PfLine = property(lambda self: PfLine({"p": self.p}))  # possibly nan-Series

    set_q: PfLine = lambda self, val: self._set_key_val("q", val)
    set_w: PfLine = lambda self, val: self._set_key_val("w", val)
    set_r: PfLine = lambda self, val: self._set_key_val("r", val)
    set_p: PfLine = lambda self, val: self._set_key_val("p", val)

    def _set_key_val(self, key: str, val: Union[PfLine, pd.Series]) -> PfLine:
        """Set or update a timeseries and return the modified PfLine."""
        if isinstance(val, PfLine):
            val = val[key]  # Get pd.Series
        if self.kind == "all" and key == "r":
            raise NotImplementedError(
                "Cannot set `r`; first select `.volume` or `.price`."
            )
        data = {key: val}
        if key in ["w", "q", "r"] and self.kind in ["p", "all"]:
            data["p"] = self["p"]
        elif key in ["p", "r"] and self.kind in ["q", "all"]:
            data["q"] = self["q"]
        return PfLine(data)

    def changefreq(self, freq: str = "MS") -> PfLine:
        """Resample the PfLine to a new frequency.

        Parameters
        ----------
        freq : str, optional
            The frequency at which to resample. 'AS' for year, 'QS' for quarter, 'MS'
            (default) for month, 'D for day', 'H' for hour, '15T' for quarterhour.

        Returns
        -------
        PfLine
            Resampled at wanted frequency.
        """
        if self.kind == "p":
            raise ValueError(
                "Cannot change frequency of price information (because this will introduce errors if volume is not flat)."
            )
            # More correct: allow downsampling, and upsampling if all values are equal.
        return PfLine(utils.changefreq_sum(self.df(self._summable), freq))

    # Visualisation methods.

    def plot_to_ax(self, ax: plt.Axes, col: str = None, **kwargs):
        """Plot a timeseries of the PfLine to a specific axes.

        Parameters
        ----------
        ax : plt.Axes
            The axes object to which to plot the timeseries.
        col : str, optional
            The column to plot. Default: plot volume `w` [MW] (if available) or else
            price `p` [Eur/MWh].
        Any additional kwargs are passed to the pd.Series.plot function.
        """
        if not col:
            col = "w" if "w" in self.available else "p"
        how = {"r": "bar", "q": "bar", "p": "step", "w": "line"}.get(col)
        if not how:
            raise ValueError("`col` must be one of {'w', 'q', 'p', 'r'}")
        s = self[col]
        vis.plot_timeseries(ax, s, how=how, **kwargs)

    def plot(self, cols: str = "wp") -> plt.Figure:
        """Plot one or more timeseries of the PfLine.

        Parameters
        ----------
        cols : str, optional
            The columns to plot. Default: plot volume `w` [MW] and price `p` [Eur/MWh]
            (if available).

        Returns
        -------
        plt.Figure
            The figure object to which the series was plotted.
        """
        cols = [col for col in cols if col in self.available]
        fig, axes = plt.subplots(
            len(cols), 1, True, False, squeeze=False, figsize=(10, len(cols) * 3)
        )

        for col, ax in zip(cols, axes.flatten()):
            color = getattr(vis.Colors.Wqpr, col)
            unit = _unit(col)
            self.plot_to_ax(ax, col, color=color)
            ax.set_ylabel(unit)
        return fig

    # Dunder methods.

    def __getitem__(self, name):
        return getattr(self, name)

    def __len__(self):
        return len(self.index)

    def __bool__(self):
        return len(self) != 0

    def __eq__(self, other):
        if not isinstance(other, PfLine):
            return False
        return self._df.equals(other._df)

    def __add__(self, other):
        if not other:
            return self
        if self.kind == "p" and (isinstance(other, float) or isinstance(other, int)):
            return PfLine(self.df("p") + other)
        if not isinstance(other, PfLine):
            raise NotImplementedError("This addition is not defined.")
        if self.kind != other.kind:
            raise ValueError("Cannot add portfolio lines of unequal kind.")
        # Upsample to shortest frequency.
        freq = utils.freq_shortest(self.index.freq, other.index.freq)
        dfs = [pfl.changefreq(freq).df(pfl._summable) for pfl in [self, other]]
        # Get addition and keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
        df = sum(dfs).dropna().resample(freq).asfreq()
        return PfLine(df)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + -1 * other if other else self

    def __rsub__(self, other):
        return -1 * self + other

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            if self.kind in ["p", "q"]:
                return PfLine(self._df * other)
        elif isinstance(other, PfLine):
            if self.kind == "p" and other.kind == "q":
                return PfLine({"q": other.q, "p": self.p})
            elif self.kind == "q" and other.kind == "p":
                return PfLine({"q": self.q, "p": other.p})
            raise ValueError("Can only multiply volume with price information.")
        raise NotImplementedError("This multiplication is not defined.")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self * (1 / other)
        raise NotImplementedError("This division is not defined.")

    def __repr__(self):
        what = {"p": "price", "q": "volume", "all": "price and volume"}[self.kind]
        header = f"Lichtblick PfLine object containing {what} information."
        body = repr(self.df(self.available))
        units = _unitsline(body.split("\n")[0])
        loc = body.find("\n\n") + 1
        if not loc:
            return f"{header}\n{body}\n{units}"
        else:
            return f"{header}\n{body[:loc]}{units}{body[loc:]}"

