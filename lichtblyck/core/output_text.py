"""
Module with mixins, to add 'text-functionality' to PfLine and PfState classes.
"""

from __future__ import annotations
from ..tools import nits
from typing import List, Callable, Dict, Tuple, TYPE_CHECKING
import pandas as pd
import colorama
import functools
import textwrap

if TYPE_CHECKING:  # needed to avoid circular imports
    from .pfstate import PfState
    from .pfline import PfLine


# Unique colors for the various levels.
STYLES = [
    colorama.Style.__dict__["BRIGHT"] + colorama.Fore.__dict__[f]
    for f in ["WHITE", "GREEN", "YELLOW", "BLUE", "MAGENTA", "RED", "CYAN", "BLACK"]
]


def _style(level: int) -> str:
    """Style for given tree level."""
    return colorama.Style.RESET_ALL if level == -1 else STYLES[level % len(STYLES)]


def _remove_styles(text: str) -> str:
    """Remove all styles from text."""
    for style in [colorama.Style.RESET_ALL, *STYLES]:
        text = text.replace(style, "")
    return text


def _makelen(txt: str, goal: int, rjust: bool = True, from_middle: bool = False) -> str:
    """Shorten or lengthen a string to wanted length."""
    if goal < 1:
        return ""
    to_add = goal - len(_remove_styles(txt))
    if to_add >= 0:
        return " " * to_add + txt if rjust else txt + " " * to_add
    txt = _remove_styles(txt)  # TODO: retain styles also when shortening
    if from_middle:
        return txt[: goal // 2] + "…" + txt[-(goal - 1) // 2 :]
    else:
        return txt[: goal - 1] + "…"


def _unitsline(headerline: str) -> str:
    """Return a line of text with units that line up with the provided header."""
    text = headerline
    for col in "wqpr":
        unit = f"{nits.name2unit(col):~P}"
        to_add = f" [{unit}]"
        text = text.replace(col.rjust(len(to_add)), to_add)
        while to_add not in text and len(unit) > 1:
            unit = unit[:-3] + ".." if len(unit) > 2 else "."
            to_add = f" [{unit}]"
            text = text.replace(col.rjust(len(to_add)), to_add)
    return text


def _unitsline2(headerline: str, units: Dict[str, nits.ureg.Unit]) -> str:
    """Return a line of text with units that line up with the provided header."""
    text = headerline
    for name, unit in units.items():
        u = f"{unit:~P}"
        to_add = f" [{u}]"
        text = text.replace(name.rjust(len(to_add)), to_add)
        while to_add not in text and len(u) > 1:
            u = u[:-3] + ".." if len(u) > 2 else "."
            to_add = f" [{u}]"
            text = text.replace(name.rjust(len(to_add)), to_add)
    return text


def _treegraphs(drawprev: List[bool], has_children: bool, is_last: bool) -> Tuple[str]:
    """Return 2-element list with tree lines and coloring. One for first row,
    and one for all subsequent rows."""
    # continuation of parent lines
    base = "".join([_style(l) + ("│ ", "  ")[p] for l, p in enumerate(drawprev)])
    # current level
    level = len(drawprev)
    # make the lines for current level
    graphs0 = graphs1 = base
    graphs0 += _style(level) + ("└─" if is_last else "├─")
    graphs0 += (_style(level + 1) + "Σ") if has_children else (_style(level) + "─")
    graphs0 += _style(level) + "─ "
    graphs1 += _style(level) + ("  " if is_last else "│ ")
    graphs1 += _style(level + 1) + "│" if has_children else " "
    graphs1 += " " + _style(level)
    return graphs0, graphs1


def _width(
    treedepth: int, time_as_rows: bool, wanted_datacol_count: int, total_width: int = 88
) -> Dict:
    """How many characters are available for each section of the printout."""
    # Each value includes 1 character for leading space (except tree).
    # Define fixed widths.
    width = {
        "tree": 2 * treedepth + 5,
        "index": 26 if time_as_rows else 2,
        "tail": 0 if time_as_rows else 9,
    }
    # Split up remaining space into datacolumns
    minwidth = 10 if time_as_rows else 11  # 11 works for timestamp columns
    total = total_width - width["tree"] - width["index"] - width["tail"]
    fitcount = total // minwidth
    if fitcount >= wanted_datacol_count:
        num, div = total, wanted_datacol_count
        width["cols"] = [num // div + (1 if i < num % div else 0) for i in range(div)]
    else:
        fitcount = (total - 2) // minwidth  # save two characters for elipsis...
        num, div = total - 2, fitcount
        width["cols"] = [num // div + (1 if i < num % div else 0) for i in range(div)]
        width["cols"].insert(fitcount // 2, 2)  # ...added here (width == 2)
    return width


def _formatval(pfl, col, width, ts):
    decimals = 2 if col in "pw" else 0
    try:
        val = pfl[col][ts]
    except KeyError:
        val = ".."
    if isinstance(val, str):
        return f" {val:>{width-1}}"
    if isinstance(val, nits.Q_):
        val = val.magnitude
    return f" {val:>{width-1},.{decimals}f}".replace(",", " ")


def _datablockfn_time_as_cols(
    cols: str, indexwidth: int, colwidths: List[int], tailwidth: int, stamps: List
) -> Callable:
    """Returns function to create all lines with data for pfline (for all attributes)."""

    def dataline(pfl: PfLine, col) -> str:
        line = " " + _makelen(col, indexwidth - 1, False, False)  # index (column)
        for ts, colwidth in zip(stamps, colwidths):
            line += _formatval(pfl, col, colwidth, ts)  # datavalues
        line += " " + _makelen(
            f"[{nits.name2unit(col)}]", tailwidth, False, True
        )  # unit
        return line

    def datablockfn(pfl: PfLine):
        return [dataline(pfl, col) for col in cols]

    return datablockfn


def _datablockfn_time_as_rows(
    cols: str, indexwidth: int, colwidths: List[int], stamps: List
) -> Callable:
    """Returns function to create all lines with data for pfline (at all timestamps)."""

    def dataline(pfl: PfLine, ts) -> str:
        line = " " + _makelen(str(ts), indexwidth - 1, False, False)  # index (ts)
        for col, colwidth in zip(cols, colwidths):
            line += _formatval(pfl, col, colwidth, ts)  # datavalues
        return line

    def datablockfn(pfl: PfLine):
        return [dataline(pfl, ts) for ts in stamps]

    return datablockfn


def _bodyblockfn(datablockfn: Callable, treewidth: int) -> Callable:
    """Returns function to create all full lines (with tree and data) for pfline."""

    def bodyblockfn(pfl: PfLine, name, treegraphs):
        lines = [treegraphs[0] + " " + name]
        for line in datablockfn(pfl):
            lines += [_makelen(treegraphs[1], treewidth, False) + line]
        return lines

    return bodyblockfn


def _bodyblock(pfl_bodyfn, pfs, parts):
    def body(siblings: Dict, drawprev=[]):
        lines = []
        for s, (part, kids) in enumerate(siblings.items()):
            pfl, is_last, has_kids = pfs[part], (s == len(siblings) - 1), bool(kids)
            lines += pfl_bodyfn(pfl, part, _treegraphs(drawprev, has_kids, is_last))
            drawprev.append(has_kids)
            lines += body(kids, drawprev)
            drawprev.pop()
        return lines

    return body(parts)


def _time_as_rows(pfs: PfState, cols="wqpr", num_of_ts=5, colorful: bool = True) -> str:
    """Print portfolio structure, with attributes as columns, and one row per timestamp."""

    stamps = pfs.offtake.index  # TODO fix
    parts = {"offtake": {}, "pnl_cost": {"sourced": {}, "unsourced": {}}}

    # Partition available horizontal space.
    width = _width(2, True, len(cols))

    # Restrict used vertical space.
    if len(stamps) > num_of_ts:
        i = (num_of_ts - 1) // 2
        stamps = [*stamps[:i], "...", *stamps[-i:]]

    # Header and footer.
    pfs_header = _style(-1) + " " * (width["tree"] + width["index"])
    pfs_header += "".join([f" {c:>{w-1}}" for c, w in zip(cols, width["cols"])])
    pfs_footer = _unitsline(pfs_header)

    # Body.
    pfl_datafn = _datablockfn_time_as_rows(cols, width["index"], width["cols"], stamps)
    pfl_bodyfn = _bodyblockfn(pfl_datafn, width["tree"])
    pfs_body = _bodyblock(pfl_bodyfn, pfs, parts)

    # Return.
    text = "\n".join([pfs_header, *pfs_body, pfs_footer])
    return text if colorful else _remove_styles(text)


def _time_as_cols(pfs: PfState, cols="qp", colorful: bool = True) -> str:
    """Print portfolio structure, with one column per timestamp, and attributes as rows."""

    stamps = pfs.offtake.index  # TODO fix
    parts = {"offtake": {}, "pnl_cost": {"sourced": {}, "unsourced": {}}}

    # Partition available horizontal space.
    width = _width(2, False, len(stamps))
    if 2 in width["cols"]:
        i = width["cols"].index(2)
        stamps = [*stamps[:i], "…", *stamps[(i - len(width["cols"]) + 1) :]]

    # Header.
    pfs_headers = [_style(-1) + " " * (width["tree"] + width["index"])] * 3
    for ts, w in zip(stamps, width["cols"]):
        txts = ["", "…", ""] if w == 2 else ts.strftime("%Y-%m-%d %H:%M:%S %z").split()
        pfs_headers = [h + _makelen(f" {txt}", w) for txt, h in zip(txts, pfs_headers)]

    # Body.
    pfl_datafn = _datablockfn_time_as_cols(
        cols, width["index"], width["cols"], width["tail"], stamps
    )
    pfl_bodyfn = _bodyblockfn(pfl_datafn, width["tree"])
    pfs_body = _bodyblock(pfl_bodyfn, pfs, parts)

    # Return.
    text = "\n".join([*pfs_headers, *pfs_body])
    return text if colorful else _remove_styles(text)


class PfLineTextOutput:
    FORMAT = {"w": "{:7,.2f}", "q": "{:11,.0f}", "p": "{:11,.2f}", "r": "{:13,.0f}"}

    def __repr__(self: PfLine):
        what = {"p": "price", "q": "volume", "all": "price and volume"}[self.kind]
        header = f"Lichtblick PfLine object containing {what} information."
        # Split dataframe into magnitude and unit, and format magnitude.
        stringseries = {}
        units = {}
        for name, s in self.df().items():
            units[name] = s.pint.units
            formatting = self.FORMAT.get(name, "{}").format
            stringseries[name] = s.pint.magnitude.apply(formatting).str.replace(
                ",", " "
            )
        body = repr(pd.DataFrame(stringseries))

        unitsline = _unitsline2(body.split("\n")[0], units)
        loc = body.find("\n\n") + 1
        if not loc:
            return f"{header}\n{body}\n{unitsline}"
        else:
            return f"{header}\n{body[:loc]}{unitsline}{body[loc:]}"


class PfStateTextOutput:
    def _as_str(
        self: PfState,
        time_axis: int = 0,
        colorful: bool = True,
        cols: str = "qp",
        num_of_ts: int = 7,
    ) -> str:
        """Treeview of the portfolio state.

        Parameters
        ----------
        time_axis : int, optional (default: 0)
            Put timestamps along vertical axis (0), or horizontal axis (1).
        colorful : bool, optional (default: True)
            Make tree structure clearer by including colors. May not work on all output
            devices.
        cols : str, optional (default: "qp")
            The values to show when time_axis == 1 (ignored if 0).
        num_of_ts : int, optional (default: 7)
            How many timestamps to show when time_axis == 0 (ignored if 1).

        Returns
        -------
        str
        """
        if time_axis == 1:
            return _time_as_cols(self, cols, colorful)
        else:
            return _time_as_rows(self, num_of_ts=num_of_ts, colorful=colorful)

    @functools.wraps(_as_str)
    def print(self: PfState, *args, **kwargs) -> None:
        i = self.offtake.index  # TODO: fix
        txt = textwrap.dedent(
            f"""\
        . Timestamps: first: {i[0] }      timezone: {i.tz}
                       last: {i[-1]}          freq: {i.freq}
        . Treeview:
        """
        )
        print(txt + self._as_str(*args, **kwargs))

    def __repr__(self: PfState):
        return "Lichtblick PfState object.\n" + self._as_str(0, False)
