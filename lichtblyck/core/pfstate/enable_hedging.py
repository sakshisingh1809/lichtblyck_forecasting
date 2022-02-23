"""Add hedging methods to PfLine classes."""

from __future__ import annotations

from . import pfstate
from ..pfline import PfLine

from typing import TYPE_CHECKING


if TYPE_CHECKING:  # needed to avoid circular imports
    from .pfstate import PfState


class PfStateHedging:
    def hedge_of_unsourced(
        self: PfState, how: str = "val", freq: str = "MS", po: bool = None
    ) -> PfLine:
        """Hedge the unsourced volume, at unsourced prices in the portfolio.

        See also
        --------
        PfLine.hedge

        Returns
        -------
        PfLine
            Hedge (volumes and prices) of unsourced volume.
        """
        return self.unsourced.volume.hedge_with(self.unsourcedprice, how, freq, po)

    def source_unsourced(
        self: PfState, how: str = "val", freq: str = "MS", po: bool = None
    ) -> PfState:
        """Simulate PfState if unsourced volume is hedged and sourced at market prices.

        See also
        --------
        .hedge_of_unsourced()

        Returns
        -------
        PfState
            which is fully hedged at time scales of `freq` or longer.
        """
        tosource = self.hedge_of_unsourced(how, freq, po)
        return self.__class__(
            self._offtakevolume, self._unsourcedprice, self.sourced + tosource
        )

    def m2m_of_sourced(self) -> PfLine:
        """Mark-to-Market value of sourced volume."""
        return self.sourced.volume * (self.unsourcedprice - self.sourced.price)


def apply():
    for attr in dir(PfStateHedging):
        if not attr.startswith("_"):
            setattr(pfstate.PfState, attr, getattr(PfStateHedging, attr))
