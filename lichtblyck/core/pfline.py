"""
Dataframe-like class to hold general energy-related timeseries; either volume ([MW] or
[MWh]), price ([Eur/MWh]) or both.
"""

from __future__ import annotations
from ..tools.frames import set_ts_index
from ..tools.nits import name2unit, ureg
from .output_text import PfLineTextOutput
from .output_plot import PfLinePlotOutput
from .output_other import OtherOutput
from .dunder_arithmatic import PfLineArithmatic
from .hedge_functionality import PfLineHedge
from .utils import changefreq_sum
from typing import Callable, Union
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
    """From data, create a DataFrame with column `q`, column `p`, or columns `q` and `r`,
    with relevant units set to them. Also, do some data verification."""

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
        if s is None or s.isna().all():
            return None
        unit = name2unit(col)
        return s.astype(f"pint[{unit}]")  # set (i.e., assume) unit, or convert to unit.

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
        q = w * w.index.duration
    elif w is not None and not np.allclose(q, w * w.index.duration):
        raise ValueError("Passed values for `q` and `w` not consistent.")

    # Get revenue information (and check consistency).
    if p is None and r is None:
        return set_ts_index(pd.DataFrame({"q": q}))  # kind == 'q'
    if r is None:  # must calculate from p
        r = p * q
        i = r.isna()  # edge case p==nan. If q==0, assume r=0. If q!=0, raise error
        if i.any():
            if (abs(q.pint.m[i]) > 0.001).any():
                raise ValueError("Found timestamps with `p`==na, `q`!=0. Unknown `r`.")
            r[i] = 0
    elif p is not None and not np.allclose(r, p * q):
        # Edge case: remove lines where p==nan and q==0 before judging consistency.
        i = p.isna()
        if not (abs(q[i]) < 0.001).all() or not np.allclose(r[~i], p[~i] * q[~i]):
            raise ValueError("Passed values for `q`, `p` and `r` not consistent.")
    return set_ts_index(pd.DataFrame({"q": q, "r": r}).dropna())  # kind == 'all'


class PfLine(
    PfLineTextOutput, PfLinePlotOutput, OtherOutput, PfLineArithmatic, PfLineHedge,
):
    """Class to hold a related energy timeseries. This can be volume timeseries with q
    [MWh] and w [MW], a price timeseries with p [Eur/MWh] or both.

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

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._df.index

    @property
    def w(self) -> pd.Series:
        if "q" in self._df:
            return pd.Series(self._df["q"] / self._df.index.duration, name="w").pint.to(
                "MW"
            )
        return pd.Series(np.nan, self.index, name="w", dtype="pint[MW]")

    @property
    def q(self) -> pd.Series:
        if "q" in self._df:
            return self._df["q"]
        return pd.Series(np.nan, self.index, name="q", dtype="pint[MWh]")

    @property
    def p(self) -> pd.Series:
        if "p" in self._df:
            return self._df["p"]
        if "q" in self._df and "r" in self._df:
            return pd.Series(self._df["r"] / self._df["q"], name="p").pint.to("Eur/MWh")
        return pd.Series(np.nan, self.index, name="p", dtype="pint[Eur/MWh]")

    @property
    def r(self) -> pd.Series:
        if "r" in self._df:
            return self._df["r"]
        if "q" in self._df and "p" in self._df:
            return pd.Series(self._df["q"] * self._df["p"], name="r").pint.to("Eur")
        return pd.Series(np.nan, self.index, name="r", dtype="pint[Eur]")

    def df(self, cols: str = None) -> pd.DataFrame:
        """DataFrame for this PfLine.

        Parameters
        ----------
        cols : str, optional (default: all that are available)
            The columns to include in the dataframe.

        Returns
        -------
        pd.DataFrame
        """
        if cols is None:
            cols = self.available
        return pd.DataFrame({col: self[col] for col in cols})

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
        return {"p": "p", "q": "wq", "all": "wqpr"}[self.kind]

    # Methods/Properties that return new class instance.

    volume: PfLine = property(lambda self: PfLine({"q": self.q}))  # possibly nan-Series
    price: PfLine = property(lambda self: PfLine({"p": self.p}))  # possibly nan-Series

    set_q: Callable[..., PfLine] = lambda self, val: self._set_col_val("q", val)
    set_w: Callable[..., PfLine] = lambda self, val: self._set_col_val("w", val)
    set_r: Callable[..., PfLine] = lambda self, val: self._set_col_val("r", val)
    set_p: Callable[..., PfLine] = lambda self, val: self._set_col_val("p", val)

    def _set_col_val(
        self, col: str, val: Union[pd.Series, PfLine, float, int, ureg.Quantity]
    ) -> PfLine:
        """Set or update a timeseries and return the modified PfLine."""
        # Get pd.Series, in correct unit.
        if isinstance(val, PfLine):
            val = val[col]
        elif isinstance(val, float) or isinstance(val, int):
            val = pd.Series(val, self.index)
        elif isinstance(val, ureg.Quantity):
            val = pd.Series(val.m, self.index).astype(f"pint[{val.u}]")
        val = val.astype(f"pint[{name2unit(col)}]")

        if self.kind == "all" and col == "r":
            raise NotImplementedError(
                "Cannot set `r`; first select `.volume` or `.price`."
            )
        data = {col: val}
        if col in ["w", "q", "r"] and self.kind in ["p", "all"]:
            data["p"] = self["p"]
        elif col in ["p", "r"] and self.kind in ["q", "all"]:
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
