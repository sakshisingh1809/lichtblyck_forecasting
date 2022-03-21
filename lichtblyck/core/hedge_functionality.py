"""Adding hedging methods to PfLine and PfState classes."""


from __future__ import annotations

from ..prices import convert, hedge
from ..prices.utils import duration_bpo
from typing import List, Dict, Tuple, TYPE_CHECKING
import pandas as pd


if TYPE_CHECKING:  # needed to avoid circular imports
    from .pfstate import PfState
    from .pfline import PfLine


class PfLineHedge:
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
        If the PfLine contains prices, these are ignored.
        """
        if self.kind == "p":
            raise ValueError(
                "Cannot hedge a PfLine that does not contain volume information."
            )
        if self.index.freq not in ["15T", "H", "D"]:
            raise ValueError(
                "Can only hedge a PfLine with daily or (quarter)hourly information."
            )
        if po is None:
            po = self.index.freq in ["15T", "H"]  # default: peak/offpeak if possible
        if po and self.index.freq not in ["15T", "H"]:
            raise ValueError(
                "Can only hedge with peak and offpeak products if PfLine has (quarter)hourly information."
            )

        wout, pout = hedge.hedge(self.w, p.p, how, freq, po)
        return self.__class__({"w": wout, "p": pout})

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
                f"The frequency must be one of {'MS', 'QS', 'AS'} (got: {freq})."
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


class PfStateHedge:
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
