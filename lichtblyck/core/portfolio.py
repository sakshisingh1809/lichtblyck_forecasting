"""
Extension of pandas DataFrame with additional logic for dataframes holding
portfolio timeseries.
"""

from __future__ import annotations
from .pfseries_pfframe import PfSeries, PfFrame, force_Pf
from .functions import concat, add_header
from typing import Union, Tuple, Iterable
import pandas as pd
import numpy as np
import textwrap
import colorama


class Portfolio:
    """
    Class to hold electricity and gas portfolio data as a collection of timeseries.

    Parameters
    ----------
    name : str
        Name of this portfolio.
    own : subscriptable or iterable, optional
        Specifies energy ON THIS LEVEL of the portfolio (i.e., not in a child
        -portfolio), if any. If not None, must specify at least a power [MW], and if
        applicable also a price [Eur/MWh] time series.
        Time series may be specified as an collection that allows indexing by key
        (keys `w` and `p`), like a pandas DataFrame, or as an iterable that contains
        1 or 2 elements.
    parent : Portfolio, optional
        The parent of this portfolio.
    children : Iterable[Portfolio], optional
        Any portfolios that are children of this portfolio.

    Attributes
    ----------
    w, q, p, r : PfSeries
        Power [MW], quantity [MWh], price [Eur/MWh], revenue [Eur] timeseries.
        Aggregated over children (if any).
    <<name>> : Portfolio
        Use name of child portfolio as attribute to access it.

    Notes
    -----
    A Portfolio can *either* hold a PfFrame, specified by the `own` argument,
    *or* child Portfolios, specified by the `children` argument. It cannot hold
    both. In the first case, the Portfolio is a 'leaf node' portfolio, in the
    second, it's an 'internal node' portfolio.    
    """

    def __init__(
        self,
        name: str,
        own=None,
        parent: Portfolio = None,
        children: Iterable[Portfolio] = [],
    ):
        self.name = name
        self.own = own
        self._children = []
        for child in children:
            self.add_child(child)  # add like this to ensure children are notified
        self.parent = parent  # set like this to ensure parent is notified

    @property
    def own(self) -> Union[None, PfFrame]:
        """Return the PfFrame set at this portfolio."""
        return self._own
    @own.setter
    def own(self, own):
        #Set own PfFrame. Can be specified as 
        if own is None:
            self._own = None
            return
        w = p = None
        # Try to get w and p as keys.
        try:
            w = own["w"]
            p = own["p"]
        except (KeyError, TypeError):
            pass
        # Try to get w and p from iterable.
        if w is None:
            for i, item in enumerate(own):
                if i == 0:
                    w = item
                elif i == 1:
                    p = item
                else:
                    break
        # Save.
        if w is None:
            raise ValueError("At least power [MW] must be specified.")
        elif p is None:
            self.own = PfFrame({"w": w})
        else:
            self.own = PfFrame({"w": w, "p": p})

    # Inheritance.

    @property
    def children(self):
        """Names of child portfolios."""
        return [child.name for child in self._children]

    def add_child(self, child: Portfolio, index:int = None) -> None:
        """Insert child at given index position (default: at end)."""
        if child in self._children:
            return
        if index is None:
            self._children.append(child)
        else:
            self._children.insert(index, child)
        child.parent = self

    @property
    def parent(self) -> Portfolio:
        return self._parent

    @parent.setter
    def parent(self, parent: Portfolio) -> None:
        if self._parent == parent:
            return
        if self._parent is not None:
            raise ValueError(
                f"This portfolio ({self.name}) already has parent ({self.parent.name})."
            )
        self._parent = parent
        parent.add_child(self)

    # Time series.

    @property
    @force_Pf
    def w(self) -> PfSeries:
        """Power [MW] timeseries."""
        if not self._children:
            w = self.own.w  # should exist in pfolio without children
        else:
            w = sum([child.w for child in self._children])  # series
            if self.own is not None:
                w += self.own.w
        return w.rename("w")

    @property
    @force_Pf
    def r(self) -> PfSeries:
        """Revenue [Eur] timeseries."""
        if not self._children:
            r = self.own.r  # will raise AttributeError if .own has no .p attribute
        else:
            r = sum([child.r for child in self._children])  # series
            if self.own is not None:
                r += self.own.r
        return r.rename("r")

    @property
    @force_Pf
    def p(self) -> PfSeries:
        """Price [Eur/MWh] timeseries."""
        return (self.r / self.q).rename("p")

    @property
    @force_Pf
    def q(self) -> PfSeries:
        """Quantity [MWh] timeseries."""
        w = self.w
        return (w * w.duration).rename("w")
    
    @property 
    def index(self):
        """Left-bound timestamps of index."""
        return self.w.index
    
    @property
    def duration(self) -> pd.Series:
        """Duration [h] timeseries."""
        return self.w.duration
    
    @property
    def ts_right(self) -> pd.Series:
        """Timeseries with right-bound timestamps."""
        return self.w.ts_right

    # Resample and aggregate.

    def changefreq(self, freq: str = "MS"):
        """
        Resample and aggregate the DataFrame at a new frequency.

        Parameters
        ----------
        freq : str, optional
            The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
            for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
            quarterhour; None to aggregate over the entire time period. The default is 'MS'.

        Returns
        -------
        Portfolio
            Same data at different timescale.
        """
        # TODO: change/customize the columns in the returned dataframe.
        if self.own is None and not self._children:
            raise ValueError(
                f"Portfolio {self.name} has no children; it should have a .own attribute."
            )
        if self.own is None:
            own = None
        else:
            own = self.own.changefreq(freq)
        children = [child.changefreq(freq) for child in self._children]
        return Portfolio(self.name, own, children=children)

    # Turn into PortfolioFrame.

    def pf(self, maxdepth: int = -1, show: str = "wp") -> PfFrame:
        """
        Portfolioframe for this portfolio, including all children.

        Parameters
        ----------
        maxdepth : int, optional
            Number of child levels to show; showing only the aggregate for rest.
            -1 (default) to show all levels.
        show : str, optional
            Columns to show. The default is "wp".

        Returns
        -------
        PfFrame
        """

        def pf(pf_or_pfolio, show: str):
            return PfFrame({attr: pf_or_pfolio.__getattr__(attr) for attr in show})

        if self.own is None and not self._children:
            raise ValueError(
                f"Portfolio {self.name} has no children; it should have a .own attribute."
            )
        if maxdepth == 0:
            return pf(self, show)
        else:
            dfs = []
            if self.own is not None:
                dfs.append(add_header(pf(self.own, show), "own"))
            if self._children:
                dfs.extend(
                    [
                        add_header(child.pf(maxdepth - 1, show), child.name)
                        for child in self._children
                    ]
                )
            return PfFrame(concat(dfs, axis=1))

    def __getattr__(self, name):
        for child in self._children:
            if child.name == name:
                return child

    def __repr__(self):
        i = self.pf().index
        textblock = textwrap.dedent(
            f"""\
        Lichtblick Portfollio object "{self.name}"
        . Timestamps: first: {i[0]}
                       last: {i[-1]}
                       freq: {i.freq}
        . {'No parent portfolio' if self.parent is None else f'Parent portfolio: "{self.parent.name}"'}
        . Children as shown in treeview:
            """
        )
        return textblock + _treetext(self)

    def __str__(self):
        return self.pf().__str__()


def height(pfolio):
    """Maximum path length from node to leaf."""
    if not pfolio.children:
        return 0
    return 1 + max([height(child) for child in pfolio._children])


def depth(pfolio):
    """Path length from node to root."""
    if pfolio.parent is None:
        return 0
    return 1 + depth(pfolio.parent)


def _treetext(portfolio):
    """
    Portfolio tree structure and aggregate information about the nodes.
    """

    SECTION_WIDTHS = [20, 12, 12, 12, 12]

    # Unique color for the inset (line), based on portfolio depth. (important: all elements have len()==9. Therefore, can't use NORMAL, as that has 5 instead of 4 characters)
    insetcolors = [
        colorama.Style.__dict__["BRIGHT"] + colorama.Fore.__dict__[f]
        for f in ["YELLOW", "GREEN", "BLUE", "MAGENTA", "RED", "CYAN", "BLACK", "WHITE"]
    ]
    insetcol = lambda depth: insetcolors[depth]
    insetlen = lambda inset: len(inset) // 10

    def set_inset_len(inset, length, depth=0):
        inset = inset[: length * 10]
        if (diff := length - insetlen(inset)) > 0:
            inset += (insetcol(depth) + "─") * diff
        return inset

    def insetbase(inset, depth, last_child):
        parent_color = insetcol(depth - 1)
        inset = set_inset_len(inset, depth * 2 + 1)  # shorten
        if depth != 0:
            # Continuation of tree lines of higher nodes.
            inset = (
                inset[:-21] + ("│" if inset[-21] in "│├┬" else " ") + parent_color + " "
            )
            # Tree line of this node.
            if last_child:
                inset += parent_color + "└"
            else:
                inset += parent_color + "├"
        return inset + parent_color + "─"

    def headertext():
        text = ""
        for headers in [
            ("avg w", "sum q", "wavg p", "sum r"),
            ("[MW]", "[MWh]", "Eur/MWh", "Eur"),
        ]:
            line = " " * SECTION_WIDTHS[0]
            for i, header in enumerate(headers):
                line += header.rjust(SECTION_WIDTHS[i + 1])
            text += line + "\n"
        return text

    def linetext(inset, depth, pfolio, is_pfolio):
        # Add horizontal line to node.
        if is_pfolio:
            inset += insetcol(depth) + "Σ"
            name = pfolio.name
            add = 2
        else:
            name = "own"
            add = 3
        if depth == 0:  # don't show name on portfolio EIroot
            name = ""
        inset = inset + (insetcol(depth - 1) + "─") * add
        # Add name of node.
        maxnamelen = SECTION_WIDTHS[0] - insetlen(inset) - 1
        if maxnamelen < 2:
            inset = set_inset_len(inset, insetlen(inset) - (2 - maxnamelen))
            maxnamelen = 2
        name = name.ljust(maxnamelen)
        if len(name) > maxnamelen:
            name = name[: maxnamelen - 1] + "…"
        # Add information.
        line = inset + colorama.Style.RESET_ALL + " " + name + insetcol(depth - 1)
        if not is_pfolio and pfolio is None:
            line += "".join(["(empty)".rjust(w) for w in SECTION_WIDTHS[1:]])
        else:
            for i, s in enumerate((pfolio.w, pfolio.q, pfolio.p, pfolio.r)):
                maxx = SECTION_WIDTHS[i + 1] - 1
                line += f" {s.sum():{maxx}.2f}"
        return line

    def pfoliotext(pfolio, inset, depth=0, last_child=True):
        # Draw pfolio summary.
        inset = insetbase(inset, depth, last_child)
        text = linetext(inset, depth, pfolio, True) + "\n"
        # Draw own.
        inset = insetbase(inset, depth + 1, not (bool(pfolio._children)))
        text += linetext(inset, depth + 1, pfolio.own, False) + "\n"
        # Draw children.
        for i, child in enumerate(pfolio._children):
            text += pfoliotext(
                child, inset, depth + 1, (i == len(pfolio._children) - 1)
            )
        return text

    return headertext() + pfoliotext(portfolio, (insetcol(-1) + "─") * 4)[:-1]
