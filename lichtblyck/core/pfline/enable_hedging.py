"""Add hedging methods to PfLine classes."""

from __future__ import annotations

from . import base, single, multi
from ...prices import hedge

from typing import List, Callable, Dict, Tuple, TYPE_CHECKING


if TYPE_CHECKING:  # needed to avoid circular imports
    from .base import PfLine


class PfLineHedging:
    def hedge_with(
        self: PfLine, p: PfLine, how: str = "val", freq: str = "MS", po: bool = None
    ) -> PfLine:
        """Hedge the volume in the portfolio line with a price curve.

        Parameters
        ----------
        p : PfLine
            Portfolio line with prices to be used in the hedge.
        how : str, optional (Default: 'val')
            Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
        freq : {'D' (days), 'MS' (months, default), 'QS' (quarters), 'AS' (years)}
            Frequency of hedging products. E.g. 'QS' to hedge with quarter products.
        po : bool, optional
            Type of hedging products. Set to True to split hedge into peak and offpeak.
            (Default: split if volume timeseries has hourly values or shorter.)

        Returns
        -------
        PfLine
            Hedged volume and prices. Index with same frequency as original, but every
            timestamp within a given hedging frequency has the same volume [MW] and price.
            (or, one volume-price pair for peak, and another volume-price pair for offpeak.)

        Notes
        -----
        - If the PfLine contains prices, these are ignored.
        - If ``p`` contains volumes, these are ignored.
        """
        if self.kind == "p":
            raise ValueError(
                "Cannot hedge a PfLine that does not contain volume information."
            )
        if self.index.freq not in ["15T", "H", "D"]:
            raise ValueError(
                "Can only hedge a PfLine with daily or (quarter)hourly information."
            )
        if not isinstance(p, base.PfLine):
            raise TypeError(
                f"Parameter ``p`` must be a PfLine instance; got {type(p)}."
            )
        if po is None:
            po = self.index.freq in ["15T", "H"]  # default: peak/offpeak if possible
        if po and self.index.freq not in ["15T", "H"]:
            raise ValueError(
                "Can only hedge with peak and offpeak products if PfLine has (quarter)hourly information."
            )

        wout, pout = hedge.hedge(self.w, p.p, how, freq, po)
        return single.SinglePfLine({"w": wout, "p": pout})


def apply():
    for attr in dir(PfLineHedging):
        if not attr.startswith("_"):
            setattr(base.PfLine, attr, getattr(PfLineHedging, attr))
