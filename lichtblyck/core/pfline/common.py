from __future__ import annotations
from typing import TYPE_CHECKING, Union

from . import single  # cannot be from .single import SinglePfLine -> circular imports
from .base import PfLine
from ..mixins import (
    PfLinePlotOutput,
    PfLineArithmatic,
    PfLineTextOutput,
    PfLineHedge,
    OtherOutput,
)
from ...tools.nits import ureg, name2unit
import pandas as pd


if TYPE_CHECKING:
    from .single import SinglePfLine


class PfLineCommon(
    PfLineTextOutput,
    PfLinePlotOutput,
    PfLineArithmatic,
    PfLineHedge,
    OtherOutput,
):
    """Concrete methods and mixins that apply to ALL PfLine descendent classes."""

    # _subclasses = {}

    # def __init_subclass__(cls, **kwargs) -> None:
    #     cls._subclasses[cls.__name__] = cls
    #     return super().__init_subclass__(**kwargs)

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
