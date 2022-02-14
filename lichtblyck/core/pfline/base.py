"""
Abstract Base Classes for PfLine and PfState.
"""

from __future__ import annotations

# from . import single, multi  #<-- moved to end of file
from .. import base
from ..mixins import PfLineText, PfLinePlot, OtherOutput
from ...prices.utils import duration_bpo
from ...prices import convert
from ...tools.nits import ureg, name2unit

from abc import abstractmethod
from typing import Dict, Mapping, Union, TYPE_CHECKING
import pandas as pd

# Developer notes: we would like to be able to handle 2 cases with volume AND financial
# information. We would like to...
# ... handle the situation where, for some timestamp, the volume q == 0 but the revenue
#   r != 0, because this occasionally arises for the sourced volume, e.g. after buying
#   and selling the same volume at unequal price. So: we want to be able to store q and r.
# ... keep price information even if the volume q == 0, because at a later time this price
#   might still be needed, e.g. if a perfect hedge becomes unperfect. So: we want to be
#   able to store q and p.
# Both cases can be catered to. The first as a 'SinglePfLine', where the timeseries for
# q and r are used in the instance creation. The price is not defined at the timestamp in
# the example, but can be calculated for other timestamps, and downsampling is also still
# possble.
# The second is a bit more complex. It is possible as a 'MultiPfLine'. This has then 2
# 'SinglePfLine' instances as its children: one made from each of the timeseries for q
# and p.


if TYPE_CHECKING:
    from .single import SinglePfLine
    from .multi import MultiPfLine


class PfLine(base.NDFrameLike, Mapping, PfLineText, PfLinePlot, OtherOutput):
    """Class to hold a related energy timeseries. This can be volume timeseries with q
    [MWh] and w [MW], a price timeseries with p [Eur/MWh] or both.
    """

    def __new__(cls, data):
        # Catch case where user actually called PfLine().
        if cls is PfLine:
            subclasses = [single.SinglePfLine, multi.MultiPfLine]
            # If data is instance of subclass: return copy of the object.
            for subcls in subclasses:
                if isinstance(data, subcls):
                    return data  # TODO: make copy instead
            # Try passing data to subclasses to see if they can handle it.
            for subcls in subclasses:
                try:
                    return subcls(data)
                except (ValueError, TypeError):
                    pass
            raise NotImplementedError(
                f"None of the subclasses ({', '.join([subcls.__name__ for subcls in subclasses])}) knows how to handle this data."
            )
        # Otherwise, do normal thing.
        return super().__new__(cls)

    # Additional abstract methods to be implemented by descendents.

    @property
    @abstractmethod
    def children(self) -> Dict[str, PfLine]:
        """Children of this instance, if any."""
        ...

    @property
    @abstractmethod
    def w(self) -> pd.Series:
        """(Flattened) power timeseries in [MW]."""
        ...

    @property
    @abstractmethod
    def q(self) -> pd.Series:
        """(Flattened) energy timeseries in [MWh]."""
        ...

    @property
    @abstractmethod
    def p(self) -> pd.Series:
        """(Flattened) price timeseries in [Eur/MWh]."""
        ...

    @property
    @abstractmethod
    def r(self) -> pd.Series:
        """(Flattened) revenue timeseries in [Eur]."""
        ...

    @property
    @abstractmethod
    def kind(self) -> str:
        """Kind of data that is stored in the instance. Possible values:
        - 'q': volume data only; properties .q [MWh] and .w [MW] are available.
        - 'p': price data only; property .p [Eur/MWh] is available.
        - 'all': price and volume data; properties .q [MWh], .w [MW], .p [Eur/MWh], .r
          [Eur] are available.
        """
        ...

    # Implemented directly here.

    @property
    def summable(self) -> str:
        """Which attributes/colums of this PfLine can be added to those of other PfLines
        to get consistent/correct new PfLine."""
        return {"p": "p", "q": "q", "all": "qr"}[self.kind]

    @property
    def available(self) -> str:  # which time series have values
        """Attributes/columns that are available. One of {'wq', 'p', 'wqpr'}."""
        return {"p": "p", "q": "wq", "all": "wqpr"}[self.kind]

    def flatten(self) -> SinglePfLine:
        """Return flat instance, i.e., without children."""
        return single.SinglePfLine(self)

    @property
    def volume(self) -> SinglePfLine:
        """Return (flattened) volume-only PfLine."""
        return single.SinglePfLine({"q": self.q})

    @property
    def price(self) -> SinglePfLine:
        """Return (flattened) price-only PfLine."""
        return single.SinglePfLine({"p": self.p})

    def _set_col_val(
        self, col: str, val: Union[pd.Series, float, int, ureg.Quantity]
    ) -> SinglePfLine:
        """Set or update a timeseries and return the modified instance."""

        # Get pd.Series of other, in correct unit.
        if isinstance(val, float) or isinstance(val, int):
            val = pd.Series(val, self.index)
        elif isinstance(val, ureg.Quantity):
            val = pd.Series(val.magnitude, self.index).astype(f"pint[{val.u}]")

        if self.kind == "all" and col == "r":
            raise NotImplementedError(
                "Cannot set `r`; first select `.volume` or `.price` before applying `.set_r()`."
            )
        # Create pd.DataFrame.
        data = {col: val.astype(f"pint[{name2unit(col)}]")}
        if col in ["w", "q", "r"] and self.kind in ["p", "all"]:
            data["p"] = self["p"]
        elif col in ["p", "r"] and self.kind in ["q", "all"]:
            data["q"] = self["q"]
        df = pd.DataFrame(data)
        return single.SinglePfLine(df)

    def set_w(self, w: Union[pd.Series, float, int, ureg.Quantity]) -> SinglePfLine:
        """Set or update power timeseries [MW]; returns modified (and flattened) instance."""
        return self._set_col_val("w", w)

    def set_q(self, q: Union[pd.Series, float, int, ureg.Quantity]) -> SinglePfLine:
        """Set or update energy timeseries [MWh]; returns modified (and flattened) instance."""
        return self._set_col_val("q", q)

    def set_p(self, p: Union[pd.Series, float, int, ureg.Quantity]) -> SinglePfLine:
        """Set or update price timeseries [Eur/MWh]; returns modified (and flattened) instance."""
        return self._set_col_val("p", p)

    def set_r(self, r: Union[pd.Series, float, int, ureg.Quantity]) -> SinglePfLine:
        """Set or update revenue timeseries [MW]; returns modified (and flattened) instance."""
        return self._set_col_val("r", r)

    def set_volume(self, other: PfLine) -> SinglePfLine:
        """Set or update volume information; returns modified (and flattened) instance."""
        if not isinstance(other, PfLine) or other.kind != "q":
            raise ValueError(
                "Can only set volume from a PfLine instance with kind=='q'. Use .volume to obtain from given instance."
            )
        return self.set_q(other.q)

    def set_price(self, other: PfLine) -> SinglePfLine:
        """Set or update price information; returns modified (and flattened) instance."""
        if not isinstance(other, PfLine) or other.kind != "p":
            raise ValueError(
                "Can only set price from a PfLine instance with kind=='p'. Use .price to obtain from given instance."
            )
        return self.set_p(other.p)

    def po(self: PfLine, freq: str = "MS") -> pd.DataFrame:
        """Decompose the portfolio line into peak and offpeak values. Takes simple averages
        of volume [MW] and price [Eur/MWh] - does not hedge!

        Parameters
        ----------
        freq : {'MS' (months, default), 'QS' (quarters), 'AS' (years)}
            Frequency of resulting dataframe.

        Returns
        -------
        pd.DataFrame
            The dataframe shows a composition into peak and offpeak values.
        """
        if self.index.freq not in ["15T", "H"]:
            raise ValueError(
                "Only PfLines with (quarter)hourly values can be turned into peak and offpeak values."
            )
        if freq not in ["MS", "QS", "AS"]:
            raise ValueError(
                f"Value of paramater ``freq`` must be one of {'MS', 'QS', 'AS'} (got: {freq})."
            )

        prods = ("peak", "offpeak")

        # Get values.
        dfs = []
        if "w" in self.available:
            vals = convert.tseries2bpoframe(self.w, freq)
            vals.columns = pd.MultiIndex.from_product([vals.columns, ["w"]])
            dfs.append(vals)
        if "p" in self.available:
            vals = convert.tseries2bpoframe(self.p, freq)
            vals.columns = pd.MultiIndex.from_product([vals.columns, ["p"]])
            dfs.append(vals)
        df = pd.concat(dfs, axis=1)

        # Add duration.
        durs = duration_bpo(df.index)
        durs.columns = pd.MultiIndex.from_product([durs.columns, ["duration"]])
        df = pd.concat([df, durs], axis=1)

        # Add additional values and sort.
        if "q" in self.available:
            for prod in prods:
                df[(prod, "q")] = df[(prod, "w")] * df[(prod, "duration")]
        if "r" in self.available:
            for prod in prods:
                df[(prod, "r")] = df[(prod, "q")] * df[(prod, "p")]
        i = pd.MultiIndex.from_product([prods, ("duration", *self.available)])
        return df[i]

    # Iterating over children.

    def items(self):
        return self.children.items()

    def __iter__(self):
        return iter(self.children.keys())

    def __len__(self):
        return len(self.children)

    def __getitem__(self, name):
        return getattr(self, name)

    # Dunder methods.

    def __getattr__(self, name):
        if name in self.children:
            return self.children[name]
        raise AttributeError(f"No such attribute '{name}'.")

    def __bool__(self):
        # To ensure True even if no .children and therefore len()==0
        return True


# Must be at end, because they depend on PfLine existing.
from . import single, multi, enable_arithmatic, enable_hedging

enable_arithmatic.apply()
enable_hedging.apply()
