"""
Dataframe-like class to hold general energy-related timeseries.
"""

from __future__ import annotations
from ..tools.tools import set_ts_index
from . import utils
from typing import Iterable
import pandas as pd
import numpy as np


def _unit(attr: str) -> str:
    return {"q": "MWh", "w": "MW", "p": "Eur/MWh", "r": "Eur", "t": "degC"}.get(
        attr, ""
    )


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
    """From data, create a DataFrame with column `q`, columns `q` and `r`, or column `p`.
    Also, do some data verification."""

    errormsg = "Must supply (a) only price information, (b) only volume information, or (c) both."

    def get_by_attr_or_key(obj, a):
        try:
            return getattr(obj, a)
        except AttributeError:
            pass
        try:
            return obj[a]
        except (KeyError, TypeError):
            return None

    # Extract values from data.
    q = get_by_attr_or_key(data, "q")
    w = get_by_attr_or_key(data, "w")
    r = get_by_attr_or_key(data, "r")
    p = get_by_attr_or_key(data, "p")

    # Index.
    indices = [get_by_attr_or_key(obj, "index") for obj in (w, q, p, r)]
    indices = [i for i in indices if i is not None]
    if not indices:
        raise ValueError("No index can be found in the data.")
    if len(set([i.freq for i in indices])) != 1:
        raise ValueError("Passed timeseries do not share same frequency.")

    # Get price information.
    if p is not None and w is None and q is None and r is None:
        # We only have price information. Return immediately.
        return set_ts_index(pd.DataFrame({"p": p}))  # kind == 'p'

    # Get quantity information (and check consistency).
    if q is None and w is None:
        if r is None or p is None:
            raise ValueError(errormsg)
        q = r / p
    if q is None:
        q = w * w.duration
    elif q is not None and w is not None and not np.allclose(q, w * w.duration):
        raise ValueError("Passed values for `q` and `w` not consistent.")
    if p is None and r is None:
        return set_ts_index(pd.DataFrame({"q": q}))  # kind == 'q'

    # Get revenue information (and check consistency).
    if r is None:
        r = p * q
    elif r is not None and p is not None and not np.allclose(r, p * q, equal_nan=True):
        raise ValueError("Passed values for `q`, `p` and `r` not consistet.")
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
    index : pandas.DateTimeIndex
        Left timestamp of row.
    ts_right, duration : pandas.Series
        Right timestamp, and duration [h] of row.

    Notes
    -----
    When setting an attribute (`w`, `q`, `p`, `r`) and kind == 'all': the revenue has 
    least priority and is recalculated from the existing and the updated attributes. 
    If the revenue itself is updated, the price is recalculated and the volume is kept.'  
    """

    def __init__(self, data):
        self._df = _make_df(data)

    # Properties.

    index = property(lambda self: self._df.index)
    duration = property(lambda self: self._df.duration)
    ts_right = property(lambda self: self._df.ts_right)

    @property
    def q(self):
        if self.kind in ["q", "all"]:
            return self._df["q"]
        return pd.Series(np.nan, self.index, name="q")

    @property
    def w(self):
        return pd.Series(self.q / self.duration, name="w")

    @property
    def r(self):
        if self.kind == "all":
            return self._df["r"]
        return pd.Series(np.nan, self.index, name="r")

    @property
    def p(self):
        if self.kind == "p":
            return self._df["p"]
        if self.kind == "all":
            return pd.Series(self.r / self.q, name="p")
        return pd.Series(np.nan, self.index, name="p")

    set_q = lambda self, val: self._set_key_val("q", val)
    set_w = lambda self, val: self._set_key_val("w", val)
    set_r = lambda self, val: self._set_key_val("r", val)
    set_p = lambda self, val: self._set_key_val("p", val)

    def _set_key_val(self, key, val) -> PfLine:
        """Set or update a timeseries and return the modified PfLine."""  
        data = {key: val}
        if key == "p" and self.kind in ["q", "all"]:
            data["q"] = self.q
        elif (key == "w" or key == "q") and self.kind in ["p", "all"]:
            data["p"] = self.p
        elif key == "r":
            if self.kind == "p":
                data["p"] = self.p
            elif self.kind in ["q", "all"]:
                data["q"] = self.q
        return PfLine(data)

    @property
    def kind(self) -> str:
        """Kind of data that is stored in the object. Possible values and implications:
            'q': volume data only. Properties .q [MWh] and .w [MW] are available.
            'p': price data only. Property .p [Eur/MWh] is available.
            'all': price and volume data. Properties .q [MWh], .w [MW], .p [Eur/MWh],
                .r [Eur] are available.
        """
        if "q" in self._df:
            return "all" if "r" in self._df else "q"
        elif "p" in self._df:
            return "p"
        raise ValueError("Object contains no information.")

    @property
    def _summable(self) -> Iterable[str]:  # which time series can be added to others
        return {"p": "p", "q": "q", "all": "qr"}[self.kind]

    @property
    def _available(self) -> Iterable[str]:  # which time series have values
        return {"p": "p", "q": "qw", "all": "qwrp"}[self.kind]

    # Methods.

    def df(self, cols: Iterable[str] = None) -> pd.DataFrame:
        """pd.DataFrame for this object."""
        if cols is None:
            cols = self._available
        return pd.DataFrame({attr: getattr(self, attr) for attr in cols})

    def changefreq(self, freq: str = "MS") -> PfLine:
        """Resample the object to a new frequency.
        
        Parameters
        ----------
        freq : str, optional
            The frequency at which to resample. 'AS' for year, 'QS' for quarter, 'MS' 
            (default) for month, 'D for day', 'H' for hour, '15T' for quarterhour.
        """
        if self.kind == "p":
            raise ValueError(
                "Cannot change frequency of price information (because this will introduce errors if volume is not flat)."
            )
            # More correct: allow downsampling, and upsampling if all values are equal.
        return PfLine(utils.changefreq_sum(self.df(self._summable), freq))

    # Dunder methods.

    def __getitem__(self, name):
        return getattr(self, name)

    def __len__(self):
        return len(self.index)

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
        dfs = [pp.changefreq(freq).df(pp._summable) for pp in [self, other]]
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
            return PfLine(self.df(self._summable) * other)
        if not isinstance(other, PfLine):
            raise NotImplementedError("This multiplication is not defined.")
        if self.kind == "p" and other.kind == "q":
            return PfLine({"q": other.q, "p": self.p})
        elif self.kind == "q" and other.kind == "p":
            return PfLine({"q": self.q, "p": other.p})
        raise ValueError("Can only multiply volume with price information.")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self * (1 / other)
        if not isinstance(other, PfLine):
            raise NotImplementedError("This division is not defined.")

    def __bool__(self):
        try:
            _ = self.kind
        except:
            return False
        return True

    def __eq__(self, other):
        if not isinstance(other, PfLine):
            return False
        return self._df == other._df  # the same if their dataframes are the same

    def __repr__(self):
        what = {"p": "price", "q": "volume", "all": "price and volume"}[self.kind]
        header = f"Lichtblick PfLine object containing {what} information."
        body = repr(self.df(self._available))
        units = _unitsline(body.split("\n")[0])
        loc = body.find("\n\n") + 1
        if not loc:
            return f"{header}\n{body}\n{units}"
        else:
            return f"{header}\n{body[:loc]}{units}{body[loc:]}"

