"""
Extension of pandas DataFrame with additional logic for dataframes holding
portfolio timeseries.
"""

from __future__ import annotations
from .pfseries_pfframe import FREQUENCIES
from . import functions
from typing import Union, Tuple, Iterable, Dict, List
import pandas as pd
import numpy as np
import functools
import textwrap
import colorama


def _make_df(data) -> pd.DataFrame:
    """From data, create a DataFrame with columns `q` and `r`, if possible. Also,
    do some data verification."""

    def get_by_attr_or_key(a, obj=data):
        try:
            return getattr(obj, a)
        except:
            pass
        try:
            return obj[a]
        except:
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
        df["q"] = q
    elif w is not None:
        df["q"] = w * df.duration
    elif r is not None and p is not None:
        df["q"] = r / p
    else:
        raise ValueError("No values for `q` can be found in the data.")

    # Revenue.
    if r is not None:
        df["r"] = r
    elif p is not None:
        df["r"] = df["q"] * p
    else:  # no financial data is included, set to nan.
        df["r"] = np.nan

    # Consistancy checks for redundant data.
    if w is not None:
        if not df.empty and not np.allclose(df["q"], w * df.duration):
            raise ValueError("Passed values for `q` and `w` not compatible.")

    if p is not None:
        if not df.empty and not np.allclose(df["r"], df["q"] * p, equal_nan=True):
            raise ValueError("Passed values for `q`, `p` and `r` not compatible.")

    return df


# def _make_childlist(data) -> pd.DataFrame:
#     """From data, create a List with SinglePf and/or MultiPf objects, if possible."""

#     if type(data) is MultiPf:
#         return (
#             data._childlist
#         )  # if MultiPf is passed, extract _childlist attribute.

#     if isinstance(data, Iterable):
#         return data  # assume data is collection of MultiPf and SinglePf instances.

#     if isinstance(
#         data, Dict
#     ):  # assume keys are names that these instances must be given.
#         newlist = []
#         for name, inst in data.items():
#             if type(inst) is MultiPf:
#                 newlist.append(MultiPf(inst, name))
#             elif type(inst) is SinglePf:
#                 newlist.append(SinglePf(inst, name))
#         return newlist

#     raise ValueError("Object can't be created with specified information.")


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

    # def columns(name, values, widths, max_cumul_width=70, keepextremes=True):
    #     value_width = list(zip(values, widths)) # shorten to shortest.
    #     cumul_width = lambda : sum([vw[1] + 1 for vw in value_width])
    #     if cumul_width() > max_cumul_width:
    #         def to_remove():
    #             return (len(value_width)//2) if keepextremes else (len(value_width) - 1)
    #         value_width[to_remove()] = ('..', 2)
    #         while cumul_width() > max_cumul_width:
    #             value_width.pop(to_remove())
    #             value_width[to_remove()] = ('..', 2)
    #     return {'vals': [vw[0] for vw in value_width],
    #             'widths': [vw[1] for vw in value_width],
    #             'kwargs': [{'width': vw[1], name: vw[0]} for vw in value_width]}

    # def rows(name, values, max_num_of_rows=10, keepextremes=True):
    #     if len(values) > max_num_of_rows:
    #         if keepextremes:
    #             values = [*values[:max_num_of_rows//2], '...', *values[-(max_num_of_rows+1)//2:]]
    #         else:
    #             values = [*values[:max_num_of_rows-1], '...']
    #     return {'vals': [v for v in values], 'kwargs': [{name: v} for v in values]}

    # def getformatcode(ts=None, attr=None, width=None):
    #     decimals = 2 if attr in ['p', 'w'] else 0
    #     return f':{width}.{decimals}'

    def getvalue(pfolio, attr=None, ts=None):
        try:
            return getattr(pfolio, attr)[ts]
        except:
            return ".."

    # def get_formatted_values(pfolio, **kwargs):
    #     val = getvalue(pfolio, **kwargs),
    #     formatcode = getformatcode(**kwargs)
    #     try:
    #         return f' {val:{formatcode}f}'
    #     except:
    #         return f' {val:{formatcode}}'

    # def vs_ts(pfolio, attributes='wp'):
    #     widths = [22, 2, [11, 11, 11, 11], 9] # tree, head, datacols, tail

    #     ax1 = columns('ts', portfolio.index, widths[2], 50)

    #     parts = lambda ts: (ts, '') if isinstance(ts, str) else (ts.date(), ts.time())
    #     colheader = [[' {val:>{width}.{width}}'.format(val=str(part), width=width)
    #                for part in parts(ts)]
    #               for ts, width in zip(ax1['vals'], ax1['widths'])]
    #     colheader = np.array(colheader).T

    #     ax0 = rows('attr', attributes)

    #     for attr in ax0['vals']:
    #         makelen(attr, widths[1], )
    #     rowheader = np.array([[for ]])

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


# def _treetext(portfolio):
#     """
#     Portfolio tree structure and aggregate information about its nodes.


#     """

#     SECTION_WIDTHS = [20, 12, 12, 12, 12]

#     # Unique color for the inset (line), based on portfolio depth. (important: all elements have len()==9. Therefore, can't use NORMAL, as that has 5 instead of 4 characters)
#     insetcolorors = [
#         colorama.Style.__dict__["BRIGHT"] + colorama.Fore.__dict__[f]
#         for f in ["YELLOW", "GREEN", "BLUE", "MAGENTA", "RED", "CYAN", "BLACK", "WHITE"]
#     ]
#     insetcolor = lambda depth: insetcolorors[depth]
#     insetlen = lambda inset: len(inset) // 10

#     def set_inset_len(inset, length, depth=0):
#         inset = inset[: length * 10]
#         if (diff := length - insetlen(inset)) > 0:
#             inset += (insetcolor(depth) + "─") * diff
#         return inset

#     def insetbase(inset, depth, last_child):
#         parent_color = insetcolor(depth - 1)
#         inset = set_inset_len(inset, depth * 2 + 1)  # shorten
#         if depth != 0:
#             # Continuation of tree lines of higher nodes.
#             inset = (
#                 inset[:-21] + ("│" if inset[-21] in "│├┬" else " ") + parent_color + " "
#             )
#             # Tree line of this node.
#             if last_child:
#                 inset += parent_color + "└"
#             else:
#                 inset += parent_color + "├"
#         return inset + parent_color + "─"

#     def headerfooter():
#         head = colorama.Style.RESET_ALL + "".join(
#             [att.rjust(width) for att, width in zip(list(" wqpr"), SECTION_WIDTHS)]
#         )
#         foot = _unitsline(head)
#         return head + "\n", "\n" + foot

#     def linetext(inset, depth, pfolio):
#         # Add horizontal line to node.
#         name = "" if depth == 0 else pfolio.name
#         if type(pfolio) is SinglePf:
#             add = 3
#         else:
#             inset += insetcolor(depth) + "Σ"
#             add = 2
#         inset = inset + (insetcolor(depth - 1) + "─") * add
#         # Add name of node.
#         maxnamelen = SECTION_WIDTHS[0] - insetlen(inset) - 1
#         if maxnamelen < 2:
#             inset = set_inset_len(inset, insetlen(inset) - (2 - maxnamelen))
#             maxnamelen = 2
#         name = name.ljust(maxnamelen)
#         if len(name) > maxnamelen:
#             name = name[: maxnamelen - 1] + "…"
#         # Add information.
#         line = inset + colorama.Style.RESET_ALL + " " + name + insetcolor(depth - 1)
#         aggs = pd.Series(pfolio.flatten().changefreq("AS").pf("wprq").iloc[0])
#         for i, val in enumerate((aggs.w, aggs.q, aggs.p, aggs.r)):
#             maxx = SECTION_WIDTHS[i + 1] - 1
#             line += f" {val:{maxx}.2f}"
#         return line

#     def pfoliotext(pfolio, inset, depth=0, last_child=True):
#         # Draw pfolio summary.
#         inset = insetbase(inset, depth, last_child)
#         text = linetext(inset, depth, pfolio) + "\n"
#         # Draw children.
#         for i, child in enumerate(childlist := list(pfolio)):
#             text += pfoliotext(child, inset, depth + 1, (i == len(childlist) - 1))
#         return text

#     header, footer = headerfooter()
#     return header + pfoliotext(portfolio, (insetcolor(-1) + "─") * 4)[:-1] + footer


class _PortfolioBase:
    def __init__(self, name: str):
        if not name:
            raise ValueError("Must pass non-empty string for parameter `name`.")
        self._name = name
        self._parent = None
        self._childlist = []

    # .w and .p alway calculated from .q, .duration and .r.
    w = property(lambda self: pd.Series(self.q / self.duration, name="w"))
    p = property(lambda self: pd.Series(self.r / self.q, name="p"))

    @property
    def name(self) -> str:
        return self._name

    # Inheritance upwards.

    @property
    def parent(self):
        """Parent of the portfolio."""
        return self._parent

    def _set_parent(self, parent) -> None:
        # Method is not public. Parent to be set automatically while adding the child, not manually by user.
        if (
            parent is not None
            and self._parent is not None
            and self._parent is not parent
        ):
            raise ValueError("Portfolio already has parent.")
        self._parent = parent

    @property
    def depth(self):
        """Path length from node to root."""
        if self.parent is None:
            return 0
        return 1 + self.parent.depth

    # Inheritance downwards.

    @property
    def children(self):
        """Names of child portfolios."""
        return [child.name for child in self]

    def remove_child(self, child: Union[str, _PortfolioBase]):
        """Remove child. Pass child or its name as argument."""
        if child in self:
            self._childlist.remove(child)
            child._set_parent(None)
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


class SinglePf(_PortfolioBase):
    """
    Class to hold electricity or gas data, for a single portfolio.

    Attributes
    ----------
    w, q, p, r : pd.Series
        Power [MW], quantity [MWh], price [Eur/MWh], revenue [Eur] timeseries.
        Can also be accessed by key (e.g., with ['w']).
    ts_right, duration : pandas.Series
        Right timestamp and duration [h] of row.
    """

    def __init__(self, data, name: str = None):
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

    # Resample and aggregate.

    @functools.wraps(functions.changefreq_summable)
    def changefreq(self, freq: str = "MS") -> SinglePf:
        new_df = functions.changefreq_summable(self.df("qr"), freq)

        return SinglePf(new_df, self.name)

    def flatten(self, *args, **kwargs) -> SinglePf:
        return self

    # Turn into PortfolioFrame.

    def df(self, show: Iterable[str] = "wp", *args, **kwargs) -> pd.DataFrame:
        """
        pd.DataFrame for this portfolio.

        See also
        --------
        MultiPf.df
        """
        return pd.DataFrame({attr: getattr(self, attr) for attr in show})

    # Dunder methods.

    def __len__(self):
        return len(self._df)

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

    def __init__(self, data: Iterable[Union[SinglePf, MultiPf]], name: str = None):
        if name is None:
            try:
                name = data.name
            except:
                raise ValueError("No value provided for `name`.")

        super().__init__(name)
        # childlist = _make_childlist(edata)  # specific to MultiPf
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
        # Set parent first (raises error if can't proceed).
        child._set_parent(self)
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
        return MultiPf([child.changefreq(freq) for child in self], self.name)

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
            return SinglePf({"q": self.q, "r": self.r}, self.name)
        return MultiPf([child.flatten(levels - 1) for child in self], self.name)

    # Turn into PortfolioFrame.

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
            functions.add_header(child.df(show, levels - 1), child.name)
            for child in self
        ]
        return functions.concat(dfs, axis=1)

    # Tree view.

    @functools.wraps(_treetext)
    def treeview(self, *args, **kwargs):
        return _treetext(self, *args, **kwargs)

    # Dunder methods.

    def __getattr__(self, name):
        for child in self._childlist:
            if child.name == name:
                return child

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
        . {'No parent portfolio' if self.parent is None else f'Parent portfolio: "{self.parent.name}"'}
        . Children as shown in treeview (monthly aggregates):
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
