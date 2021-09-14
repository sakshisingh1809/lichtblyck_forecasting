"""
Module to create a string of the portfolio state as a tree structure.
"""

from .pfline import PfLine, _unitsline
from ..tools import units
from typing import List, Callable, Dict, Tuple
import colorama

# Unique colors for the various levels.
STYLES = [
    colorama.Style.__dict__["BRIGHT"] + colorama.Fore.__dict__[f]
    for f in ["WHITE", "YELLOW", "GREEN", "BLUE", "MAGENTA", "RED", "CYAN", "BLACK"]
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
    else:
        return f" {val:>{width-1}.{decimals}f}"


def _datablockfn_time_as_cols(
    cols: str, indexwidth: int, colwidths: List[int], tailwidth: int, stamps: List
) -> Callable:
    """Returns function to create all lines with data for pfline (for all attributes)."""

    def dataline(pfl: PfLine, col) -> str:
        line = " " + _makelen(col, indexwidth - 1, False, False)  # index (column)
        for ts, colwidth in zip(stamps, colwidths):
            line += _formatval(pfl, col, colwidth, ts)  # datavalues
        line += " " + _makelen(f"[{units.BU(col)}]", tailwidth, False, True)  # unit
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


def time_as_rows(pfs, cols="wqpr", num_of_ts=5, colorful: bool = True) -> str:
    """Print portfolio structure, with attributes as columns, and timestamps as rows."""

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


def time_as_cols(pfs, cols="qp", colorful: bool = True) -> str:
    """Print portfolio structure, with timestamps as columns, and attributes as rows."""

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
