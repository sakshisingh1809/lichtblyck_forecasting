"""
Dataframe-like classes to hold general energy-related timeseries.
"""

from __future__ import annotations
from .pfseries_pfframe import FREQUENCIES
from . import utils
from typing import Union, Iterable
import pandas as pd
import numpy as np
import functools
import textwrap
import colorama


def _make_df(data) -> pd.DataFrame:
    """From data, create a DataFrame with columns `q` (and possibly `r`). Also, do some
    data verification."""

    def get_by_attr_or_key(a, obj=data):
        try:
            return getattr(obj, a)
        except AttributeError:
            pass
        try:
            return obj[a]
        except (KeyError, TypeError):
            return None

    # Extract values from data.
    q = get_by_attr_or_key("q")
    w = get_by_attr_or_key("w")
    r = get_by_attr_or_key("r")
    p = get_by_attr_or_key("p")

    # Index.
    for obj in (data, w, q, p, r):  # in case a dictionary of Series was passed.
        i = get_by_attr_or_key("index", obj)
        if i is not None:
            break
    else:
        raise ValueError("No index can be found in the data.")
    df = pd.DataFrame(index=i)
    df.index = df.index.rename("ts_left")

    # Quantity.
    if q is not None:
        pass
    elif w is not None:
        q = w * df.duration
    elif r is not None and p is not None:
        q = r / p
    else:
        raise ValueError("No values for `q` can be found in the data.")

    # Revenue.
    if r is not None:
        pass
    elif p is not None and q is not None:
        r = q * p
    else:  # no financial data is included, set to nan.
        pass  # r stays None

    # Save.
    df["q"] = np.nan if q is None else q
    df["r"] = np.nan if r is None else r

    # Consistancy check: if r is present at all, it must be present for exactly same rows as q
    if r is not None:
        if sum(
            df["q"].isna() ^ df["r"].isna()
        ):  # r is na when q is not, or vice versa.
            raise ValueError(
                "Price or revenue data is provided, but not for all datapoints."
            )

    # Consistancy checks for redundant data.
    if w is not None:
        if not df.empty and not np.allclose(df["q"], w * df.duration):
            raise ValueError("Passed values for `q` and `w` not compatible.")

    if p is not None:
        if not df.empty and not np.allclose(df["r"], df["q"] * p, equal_nan=True):
            raise ValueError("Passed values for `q`, `p` and `r` not compatible.")

    return df


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


def _treetext(portfolio, attributes="wqpr", as_cols=True):
    """
    Portfolio tree structure and aggregate information about its nodes.


    """
    # Unique colors for the various levels.
    colors = [
        colorama.Style.__dict__["BRIGHT"] + colorama.Fore.__dict__[f]
        for f in ["WHITE", "YELLOW", "GREEN", "BLUE", "MAGENTA", "RED", "CYAN", "BLACK"]
    ]

    def color(level):
        return colors[level % len(colors)]

    def remove_color(inset):
        for c in colors:
            inset = inset.replace(c, "")
        return inset

    def makelen(text, wanted, rjust=True, from_middle=True):
        if wanted < 1:
            return ""
        if (current := len(text)) <= wanted:
            if rjust:
                return text.rjust(wanted)
            else:
                return text.ljust(wanted)
        while (current := len(text)) > wanted:
            if from_middle:
                text = text[: (current - 1) // 2] + "…" + text[(current + 3) // 2 :]
            else:
                text = text[: current - 2] + "…"
        return text

    def combine(text1, text2, wanted, rjust=True, from_middle=True):
        """Concatenate text1 and text2 to a total length of wanted_len. text2
        is shortened if too long; the entire string is padded with spaces. text1
        may have coloring codes; text2 should not."""
        return text1 + makelen(
            text2, wanted - len(remove_color(text1)), rjust, from_middle
        )

    def treepart(pfolio, prev, last_child):
        """Return 2-element list with tree lines and coloring. One for first row, 
        and one for all subsequent rows."""
        # parent-insets
        base = "".join([color(i) + ("│ ", "  ")[p] for i, p in enumerate(prev)])
        # current level
        level = len(prev)
        # make the lines for current level
        block = [base, base]
        block[0] += color(level) + ("└─" if last_child else "├─")
        block[0] += (
            (color(level + 1) + "Σ") if pfolio.children else (color(level) + "─")
        )
        block[0] += color(level) + "─ "
        block[1] += color(level) + ("  " if last_child else "│ ")
        block[1] += color(level + 1) + "│" if pfolio.children else " "
        block[1] += " " + color(level)
        return block

    def getvalue(pfolio, attr=None, ts=None):
        try:
            return getattr(pfolio, attr)[ts]
        except:
            return ".."

    def vs_ts(pfolio, attributes="wp"):
        """Print portfolio structure, with timestamps as columns, and attributes
        as rows."""
        stamps = pfolio.index
        widths = [22, 2, [11, 11, 11, 11], 9]  # tree, head, datacols, tail
        num_of_col = len(stamps)

        # Creating the format strings.
        if num_of_col < len(widths[2]):
            widths[2] = widths[2][:num_of_col]
        bodyfrmt = [
            f"{{val[{i}]:{w}.{{decimals[{i}]}}f}}" for i, w in enumerate(widths[2])
        ]
        headfrmt = [f"{{val[{i}]:>{w}.{w}}}" for i, w in enumerate(widths[2])]
        if num_of_col > (target := len(widths[2])):
            stamps = [*stamps[: target // 2], *stamps[-(target + 1) // 2 :]]
            bodyfrmt.insert(target // 2, "…")
            headfrmt.insert(target // 2, "…")
        # Turn into one string.
        bodyfrmt = "{treeline} {attr} " + " ".join(bodyfrmt) + " {unit}"
        headfrmt = f'{"":{widths[0]}} {"":{widths[1]}} ' + " ".join(headfrmt)

        # Header/Footer.
        header = [
            headfrmt.format(val=[str(f(ts)) for ts in stamps])
            for f in (lambda ts: ts.date(), lambda ts: ts.time())
        ]
        footer = []
        # Body.
        def bodypart(pfolio, treeblock):
            block = []
            for l, a in enumerate(attributes):
                tl, toadd = {True: (0, pfolio.name), False: (1, "")}[l == 0]
                kwargs = {"attr": makelen(a, widths[1])}
                kwargs["treeline"] = combine(treeblock[tl], toadd, widths[0], False)
                kwargs["decimals"] = ([2] if a in "pw" else [0]) * len(stamps)
                kwargs["unit"] = makelen(f"[{_unit(a)}]", widths[-1], False)
                kwargs["val"] = [getattr(pfolio, a)[ts] for ts in stamps]
                block.append(bodyfrmt.format(**kwargs))
            return block

        def bodyblock(pfolio, prev=[], last_child=True):
            body = bodypart(pfolio, treepart(pfolio, prev, last_child))
            # Children.
            prev.append(last_child)
            for i, child in enumerate(childlist := list(pfolio)):
                body.extend(bodyblock(child, prev, (i == len(childlist) - 1)))
            prev.pop()
            return body

        body = bodyblock(pfolio)
        return "\n".join([*header, *body, *footer])

    def vs_attr(pfolio, attributes="wp", num_of_ts=5):
        """Print portfolio structure, with attributes as columns, and timestamps
        as rows."""
        stamps = pfolio.index
        if num_of_ts < len(stamps):
            stamps = [
                *stamps[: (num_of_ts - 1) // 2],
                "...",
                *stamps[-(num_of_ts - 1) // 2 :],
            ]
        widths = [22, 17, [10, 10, 10, 10]]  # tree, head, datacols
        num_of_col = len(attributes)

        # Creating the format strings.
        if num_of_col < len(widths[2]):
            widths[2] = widths[2][:num_of_col]
        elif num_of_col > len(widths[2]):
            attributes = attributes[: len(widths[2])]
        bodyfrmt = [
            f"{{val[{i}]:>{w}.{{decimals[{i}]}}f}}" for i, w in enumerate(widths[2])
        ]
        headfrmt = [f"{{val[{i}]:>{w}.{w}}}" for i, w in enumerate(widths[2])]
        # Turn into one string.
        bodyfrmt = "{treeline} {ts} " + " ".join(bodyfrmt)
        headfrmt = f'{"":{widths[0]}} {"":{widths[1]}} ' + " ".join(headfrmt)

        # Header/Footer.
        header = colorama.Style.RESET_ALL + headfrmt.format(val=attributes)
        footer = _unitsline(header)
        # Body.
        def bodypart(pfolio, treeblock):
            block = []
            for l, ts in enumerate(stamps):
                tl, toadd = {True: (0, pfolio.name), False: (1, "")}[l == 0]
                kwargs = {"ts": makelen(str(ts), widths[1], False, False)}
                kwargs["treeline"] = combine(treeblock[tl], toadd, widths[0], False)
                kwargs["decimals"] = [2 if a in "pw" else 0 for a in attributes]
                kwargs["val"] = [getvalue(pfolio, a, ts) for a in attributes]
                try:
                    block.append(bodyfrmt.format(**kwargs))
                except ValueError:
                    block.append(bodyfrmt.replace("}f}", "}}").format(**kwargs))
            return block

        def bodyblock(pfolio, prev=[], last_child=True):
            body = bodypart(pfolio, treepart(pfolio, prev, last_child))
            # Children.
            prev.append(last_child)
            for i, child in enumerate(childlist := list(pfolio)):
                body.extend(bodyblock(child, prev, (i == len(childlist) - 1)))
            prev.pop()
            return body

        body = bodyblock(pfolio)
        return "\n".join([header, *body, footer])

    if as_cols:
        return vs_attr(portfolio, attributes)
    else:
        return vs_ts(portfolio, attributes)


def _add(*pfs: _PortfolioBase) -> SingePf:
    """
    Add portfolios into a single pf. 

    Notes
    -----
    . If portfolios are incompatible due to distinctive financial data, raise error.
    . If portfolios have different frequencies, upsample to shortest frequency.
    . If portfolios span different time periods, reduce to smallest common period.
    """
    # Turn all into _PortfolioBase instances.
    pfs = [
        pf if isinstance(pf, _PortfolioBase) else SinglePf(pf, name="") for pf in pfs
    ]
    # Check if compatible.
    withoutfinancialdata = [pf.r.isna().all() for pf in pfs]
    if any(withoutfinancialdata) and not all(withoutfinancialdata):
        raise ValueError(
            "Some (but not all) portfolios have price/revenue information. Before adding, use the .value method on those that don't."
        )
    add_r = not any(withoutfinancialdata)
    # Upsample to shortest frequency.
    newfreq = utils.freq_shortest(*[pf.index.freq for pf in pfs])
    pfs = [pf.changefreq(newfreq) for pf in pfs]
    # Add, keep only common rows, and resample to keep freq (possibly re-adds gaps in middle).
    data = {"q": sum([pf.q for pf in pfs]).dropna().resample(newfreq).asfreq()}
    if add_r:
        data["r"] = sum([pf.r for pf in pfs]).dropna().resample(newfreq).asfreq()
    return SinglePf(data, name="added")


class _PortfolioBase:
    def __init__(self, name: str):
        if not name:
            raise ValueError("Must pass non-empty string for parameter `name`.")
        self._name = name
        self._childlist = []

    # .w and .p always calculated from .q, .duration and .r.
    w = property(lambda self: pd.Series(self.q / self.duration, name="w"))
    p = property(lambda self: pd.Series(self.r / self.q, name="p"))

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: str):
        self._name = val

    # Children.

    @property
    def children(self):
        """Names of child portfolios."""
        return [child.name for child in self]

    def remove_child(self, child: Union[str, _PortfolioBase]):
        """Remove child. Pass child or its name as argument."""
        if child in self:
            self._childlist.remove(child)
            # child._set_parent(None)
            return
        for other in self:
            if other.name == child:
                self.remove_child(other)
                return
        raise ValueError(f"Child not found: {child}.")

    # Dunder methods.

    def __iter__(self):  # To iterate over object means to iterate over children.
        return iter(self._childlist)

    def __getitem__(self, name):
        return getattr(self, name)

    def __len__(self):
        return len(self.q)

    def __add__(self, other):
        if other == 0:
            return self
        return _add(self, other)

    def __radd__(self, other):
        return self + other

    def __mul__(self, factor: float = 1):
        return SinglePf({"q": factor * self.q, "r": factor * self.r}, name=self.name)

    def __rmul__(self, factor: float = 1):
        return self * factor

    def __sub__(self, other):
        if other == 0:
            return self
        if not isinstance(other, _PortfolioBase):
            other = SinglePf(other, name="other")
        return self + -1 * other

    def __rsub__(self, other):
        return -1 * self + other


class SinglePf(_PortfolioBase):
    """Class to hold electricity or gas data, for a single portfolio.

    Parameters
    ----------
    data : object
        Generally: object with attribute or item `w` (or `q`), and possibly also 
        attribute or item `r` (or `p`), both timeseries. Most commonly a DataFrame.
    name : str
        Name for the portfolio.

    Attributes
    ----------
    w, q, p, r : pd.Series
        Power [MW], quantity [MWh], price [Eur/MWh], revenue [Eur] timeseries.
        Can also be accessed by key (e.g., with ['w']).
    ts_right, duration : pandas.Series
        Right timestamp and duration [h] of row.
    """

    def __init__(self, data, *, name: str = None):
        if name is None:
            try:
                name = data.name
            except:
                raise ValueError("No value provided for `name`.")

        super().__init__(name)
        self._df = _make_df(data)  # specific to SinglePf

    # Inheritance downwards.

    height = property(lambda self: 0)

    # Time series.

    q = property(lambda self: self._df.q)
    r = property(lambda self: self._df.r)
    index = property(lambda self: self._df.index)
    duration = property(lambda self: self._df.duration)
    ts_right = property(lambda self: self._df.ts_right)

    # Value.

    def value(self, p) -> SinglePf:
        """
        Value the portfolio with a price timeseries.

        Parameters
        ----------
        p : price timeseries

        Returns
        -------
        SinglePf
            Portfolio with same quantity information, but provided price information.
        """
        return SinglePf({"q": self.q, "p": p}, self.name)

    # Resample and aggregate.

    @functools.wraps(utils.changefreq_sum)
    def changefreq(self, freq: str = "MS") -> SinglePf:
        new_df = utils.changefreq_sum(self.df("qr"), freq)

        return SinglePf(new_df, name=self.name)

    def flatten(self, *args, **kwargs) -> SinglePf:
        return self

    # Turn into Dataframe.

    def df(self, show: Iterable[str] = "wp", *args, **kwargs) -> pd.DataFrame:
        """
        pd.DataFrame for this portfolio.

        See also
        --------
        MultiPf.df
        """
        return pd.DataFrame({attr: getattr(self, attr) for attr in show})

    # Dunder methods.

    def __repr__(self):
        header = f'Lichtblick SinglePf object for portfolio "{self.name}"'
        body = repr(self.df("wqpr"))
        units = _unitsline(body.split("\n")[0])
        loc = body.find("\n\n") + 1
        if loc == 0:
            return header + "\n" + body + "\n" + units
        else:
            return header + "\n" + body[:loc] + units + body[loc:]


class MultiPf(_PortfolioBase):
    """
    Class to hold electricity or gas timeseries data, for a collection of portfolios.

    Parameters
    ----------
    data : Iterable[Union[MultiPf, SinglePf]], optional
        Iterable of children of this portfolio.
    name : str
        Name for the portfolio.

    Attributes
    ----------
    w, q, p, r : pd.Series
        Power [MW], quantity [MWh], price [Eur/MWh], revenue [Eur] timeseries.
        Aggregated over children. Can also be accessed by key (e.g., with ['w']).
    ts_right, duration : pandas.Series
        Right timestamp and duration [h] of row.
    <<name>> : Portfolio
        Use name of child portfolio as attribute to access it. Can also be accessed
        by key (e.g., with ['subpf']).

    """

    def __init__(
        self, data: Iterable[Union[SinglePf, MultiPf]] = None, *, name: str = None
    ):
        if name is None:
            try:
                name = data.name
            except AttributeError:
                raise ValueError("No value provided for `name`.")

        super().__init__(name)
        if data:
            for child in data:
                self.add_child(child)  # add like this to ensure children are notified

    # Inheritance downwards.

    def add_child(self, child: Union[SinglePf, MultiPf], index: int = None):
        """Insert child at given index position (default: at end)."""
        # Check.
        if child in self:
            return  # child already present, don't add again
        for o, other in enumerate(self):
            if other.name == child.name:
                raise ValueError(
                    f'(Different) child with this name ("{child.name}") is already present.'
                )
            if o == 0 and (of := other.index.freq) != (cf := child.index.freq):
                raise ValueError(
                    f"Child has different frequency {cf} than this pf {of}; "
                    + f"Run .changefreq({of}) on the child before adding."
                )
        # Add child.
        if index is None:
            self._childlist.append(child)
        else:
            self._childlist.insert(index, child)

    @property
    def height(self):
        """Maximum path length from node to leaf."""
        return 1 + max([child.height for child in self])

    # Time series.

    q = property(lambda self: pd.Series(sum([child.q for child in self]), name="q"))
    r = property(lambda self: pd.Series(sum([child.r for child in self]), name="r"))
    index = property(lambda self: self.q.index)
    duration = property(lambda self: self.q.duration)
    ts_right = property(lambda self: self.q.ts_right)

    # Resample and aggregate.

    def changefreq(self, freq: str = "MS") -> MultiPf:
        """
        Resample and aggregate the portfolio at a new frequency.

        See also
        -----
        SinglePf.changefreq
        """
        return MultiPf([child.changefreq(freq) for child in self], name=self.name)

    def flatten(self, levels: int = 1) -> Union[MultiPf, SinglePf]:
        """
        Flatten the portfolio to fewer levels.

        Parameters
        ----------
        levels: int, optional
            How many levels to keep; lower levels are aggregated. The default is 1.

        Returns
        -------
        Union[MultiPf, SinglePf]
        """
        if levels == 1:
            return SinglePf({"q": self.q, "r": self.r}, name=self.name)
        return MultiPf([child.flatten(levels - 1) for child in self], name=self.name)

    # Turn into DataFrame.

    def df(self, show: Iterable[str] = "wp", levels: int = -1) -> pd.DataFrame:
        """
        pd.DataFrame for this portfolio, including children.

        Parameters
        ----------
        show : Iterable[str], optional
            Columns to include. The default is 'wp'
        levels : int, optional
            How many levels to include; lower levels are aggregated. -1 to
            include all levels. The default is -1.

        Returns
        -------
        pd.DataFrame
        """
        if levels == 1:
            return pd.DataFrame({attr: getattr(self, attr) for attr in show})

        dfs = [
            utils.add_header(child.df(show, levels - 1), child.name)
            for child in self
        ]
        return utils.concat(dfs, axis=1)

    # Tree view.

    @functools.wraps(_treetext)
    def treeview(self, *args, **kwargs):
        return _treetext(self, *args, **kwargs)

    # Dunder methods.

    def __getattr__(self, name):
        for child in self._childlist:
            if child.name == name:
                return child
        raise AttributeError

    def __setitem__(self, name, child):
        if child.name != name:
            child.name = name
        self.add_child(child)

    def __repr__(self):
        i = self.index
        header = textwrap.dedent(
            f"""\
        Lichtblick MultiPf object for portfolio "{self.name}"
        . Timestamps: first: {i[0] }      timezone: {i.tz}
                       last: {i[-1]}          freq: {i.freq}
        . Treeview of children (monthly aggregates):
            """
        )
        return header + _treetext(self.changefreq("MS"))

    def __str__(self):
        return self.df().__str__()


def portfolio(data):
    try:
        return SinglePf(data)
    except:
        pass
    try:
        return MultiPf(data)
    except:
        pass
    raise ValueError(
        "Parameter `data` not suitable to create a single portfolio and not suitable to create a multi portfolio."
    )

