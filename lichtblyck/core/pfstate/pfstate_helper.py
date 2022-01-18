"""Prepare/verify input data for PfState initialisation."""


from typing import Iterable
from ..pfline import PfLine, SinglePfLine, MultiPfLine
import pandas as pd
import warnings


def make_pflines(offtakevolume, unsourcedprice, sourced) -> Iterable[PfLine]:
    """Take offtake, unsourced, sourced information. Do some data massaging and return
    3 PfLines: for offtake volume, unsourced price, and sourced price and volume."""

    # Make sure unsourced and offtake are specified.
    if offtakevolume is None or unsourcedprice is None:
        raise ValueError("Must specify offtake volume and unsourced prices.")

    # Offtake volume.
    if isinstance(offtakevolume, pd.Series) or isinstance(offtakevolume, pd.DataFrame):
        offtakevolume = PfLine(offtakevolume)  # using column names or series names
    if isinstance(offtakevolume, PfLine):
        if offtakevolume.kind == "p":
            raise ValueError("Must specify offtake volume.")
        elif offtakevolume.kind == "all":
            warnings.warn("Offtake also contains price infomation; this is discarded.")
            offtakevolume = offtakevolume.volume

    # Unsourced prices.
    if isinstance(unsourcedprice, pd.Series):
        if unsourcedprice.name and unsourcedprice.name in "qwr":
            ValueError("Name implies this is not a price timeseries.")
        elif unsourcedprice.name != "p":
            warnings.warn("Will assume prices, even though series name is not 'p'.")
            unsourcedprice.name = "p"
        unsourcedprice = PfLine(unsourcedprice)
    elif isinstance(unsourcedprice, pd.DataFrame):
        unsourcedprice = PfLine(unsourcedprice)  # using column names or series names

    if isinstance(unsourcedprice, PfLine):
        if unsourcedprice.kind == "q":
            raise ValueError("Must specify unsourced prices.")
        elif unsourcedprice.kind == "all":
            warnings.warn(
                "Unsourced also contains volume infomation; this is discarded."
            )
            unsourcedprice = unsourcedprice.price

    # Sourced volume and prices.
    if sourced is None:
        i = offtakevolume.index.union(unsourcedprice.index)  # largest possible index
        sourced = PfLine(pd.DataFrame({"q": 0, "r": 0}, i))

    # Do checks on indices. Lengths may differ, but frequency should be equal.
    indices = [
        obj.index for obj in (offtakevolume, unsourcedprice, sourced) if obj is not None
    ]
    if len(set([i.freq for i in indices])) != 1:
        raise ValueError("PfLines have unequal frequency; resample first.")

    return offtakevolume, unsourcedprice, sourced
