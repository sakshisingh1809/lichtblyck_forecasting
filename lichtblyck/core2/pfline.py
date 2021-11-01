"""
Dataframe-like class to hold general energy-related timeseries; either volume ([MW] or
[MWh]), price ([Eur/MWh]) or both.
"""

from __future__ import annotations
from ..tools.frames import set_ts_index
from ..tools import stamps
from .textoutput import PfLineTextOutput
from .plotoutput import PfLinePlotOutput
from .utils import changefreq_sum
from typing import Union
import pandas as pd
import numpy as np

# Developer notes: we would like to be able to handle 2 cases with volume AND financial
# information. We would like to...
# ... handle the situation where the volume q == 0 but the revenue r != 0, because this
#   occasionally arises for the sourced volume, e.g. after buying and selling the same
#   volume at unequal price. So: we want to be able to store q and r.
# ... keep price information even if the volume q == 0, because at a later time this price
#   might still be needed, e.g. if a perfect hedge becomes unperfect. So: we want to be
#   able to store q and p.
# It is unpractical to cater to both cases, as we'd need to constantly check which case
# we are dealing with, and it also raises questions without a natural answer, e.g. when
# adding them, how is the result stored?
# The first case one is the most important one, and is therefore used. The second case
# must be handled by storing market prices seperately from volume data.


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
    w, q, p, r = [series_or_none(data, key) for key in "wqpr"]

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


class PfLine(PfLineTextOutput, PfLinePlotOutput):
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
        return {"p": "p", "q": "qw", "all": "wqpr"}[self.kind]

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

    # Methods/Properties that return new class instance.

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

    def concat(self, other) -> PfLine:
        """Concatenate two PfLines (along index)."""
        if self.kind != other.kind:
            raise ValueError("Cannot concatenate pflines of unequal kind.")
        pass  # TODO

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
            # More correct: allow upsampling, and allow downsampling if all values are equal.
        return PfLine(changefreq_sum(self.df(self._summable), freq))

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
        if self.index.freq != other.index.freq:
            raise NotImplementedError("Cannot add portfolio lines of unequal frequency.")
        # Get addition and keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
        dfs = [pfl.df(pfl._summable) for pfl in [self, other]]
        df = sum(dfs).dropna().resample(self.index.freq).asfreq()
        return PfLine(df)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + -1 * other if other else self

    def __rsub__(self, other):
        return -1 * self + other

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            # multiply price p (kind == 'p'), volume q (kind == 'q'), or volume q and revenue r (kind == 'all').
            return PfLine(self._df * other)
        if not isinstance(other, PfLine):
            raise NotImplementedError("This multiplication is not defined.")
        if self.index.freq != other.index.freq:
            raise NotImplementedError("Cannot multiply portfolio lines of unequal frequency.")
        if self.kind == "p" and other.kind == "q":
            df = pd.DataFrame({"q": other.q, "p": self.p})
        elif self.kind == "q" and other.kind == "p":
            df = pd.DataFrame({"q": self.q, "p": other.p})
        else:
            raise ValueError("Can only multiply volume with price information.")
        # Keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
        df = df.dropna().resample(self.index.freq).asfreq()
        return PfLine(df)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        if not isinstance(other, float) and not isinstance(other, int):
            raise NotImplementedError("This division is not defined.")
        return self * (1 / other)

