"""String representation of PfLine and PfState objects."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, Tuple, Dict
import pandas as pd
import numpy as np
import colorama


if TYPE_CHECKING:
    from ..pfstate import PfState
    from ..pfline import PfLine

COLORS = ["WHITE", "YELLOW", "CYAN", "GREEN", "RED", "BLUE", "MAGENTA", "BLACK"]
TREECOLORS = [colorama.Style.BRIGHT + getattr(colorama.Fore, f) for f in COLORS]
_UNITS = {"w": "MW", "q": "MWh", "p": "Eur/MWh", "r": "Eur"}
VALUEFORMAT = {"w": "{:,.1f}", "q": "{:,.0f}", "p": "{:,.2f}", "r": "{:,.0f}"}
DATETIMEFORMAT = "%Y-%m-%d %H:%M:%S %z"
COLWIDTHS_TAXIS0 = {"w": 12, "q": 11, "p": 11, "r": 13}
MAX_DEPTH = 6


def _remove_color(text: str) -> str:
    """Remove all color from text."""
    for color in [colorama.Style.RESET_ALL, *TREECOLORS]:
        text = text.replace(color, "")
    return text


def _df_with_strvalues(df: pd.DataFrame, units: Dict = _UNITS):
    """Turn dataframe with single column names ('w', 'p', etc) into text strings."""
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Dataframe must have single column index; has MultiIndex.")
    str_series = {}
    for name, s in df.items():
        sin = s.pint.to(units.get(name)).pint.magnitude
        formt = VALUEFORMAT.get(name).format
        sout = sin.apply(formt).str.replace(",", " ", regex=False)
        str_series[name] = sout.mask(s.isna(), "")
    return pd.DataFrame(str_series)


def _df_with_strindex(df: pd.DataFrame, num_of_ts: int):
    """Turn datetime index of dataframe into text, and reduce number of rows."""
    df.index = df.index.map(lambda ts: ts.strftime(DATETIMEFORMAT))
    if len(df.index) > num_of_ts:
        i1, i2 = num_of_ts // 2, (num_of_ts - 1) // 2
        inter = pd.DataFrame([[".."] * len(df.columns)], [".."], df.columns)
        df = pd.concat([df.iloc[:i1], inter, df.iloc[-i2:]])
    return df


def _what(pfl) -> str:
    return {"p": "price", "q": "volume", "all": "price and volume"}[pfl.kind]


def _index_info(i: pd.DatetimeIndex) -> Iterable[str]:
    """Info about the index."""
    return [
        f". Timestamps: first: {i[0] }     timezone: {i.tz}",
        f"               last: {i[-1]}         freq: {i.freq} ({len(i)} datapoints)",
    ]


def _children_info(pfl: PfLine) -> Iterable[str]:
    """Info about the children of the portfolio line."""
    childtxt = [f"'{name}' ({_what(child)})" for name, child in pfl.children.items()]
    return [". Children: " + ("none" if not childtxt else ", ".join(childtxt))]


def _treedict(depth: int, is_last_child: bool, has_children: bool) -> Dict[str, str]:
    """Dictionary with 4 strings that are used in drawing the tree."""
    colors = {"0": TREECOLORS[depth], "1": TREECOLORS[depth + 1]}
    tree = {}
    # 00 = first chars on header text line, #10 = first chars on other text lines
    if depth == 0:
        tree["00"] = colors["0"] + "─"
    else:
        tree["00"] = colors["0"] + ("└" if is_last_child else "├")
    tree["10"] = " " if is_last_child else (colors["0"] + "│")
    # 01 = following chars on header line, #11 = following chars on other text lines
    tree["01"] = (colors["1"] + "●" + colors["0"]) if has_children else "─"
    tree["01"] += "─" * (MAX_DEPTH - depth) + " "
    tree["11"] = ((colors["1"] + "│") if has_children else " ") + colors["0"]
    tree["11"] += " " * (MAX_DEPTH - depth + 3)
    return tree


def _pfl_dataheader_taxis0(
    cols: Iterable[str] = "wqpr", units: Dict = _UNITS
) -> Iterable[str]:
    out = [" " * 25] * 2  # width of timestamps
    for c in cols:
        width = COLWIDTHS_TAXIS0[c] + 1
        out[0] += f"{c:>{width}}"
        out[1] += f"{units[c]:>{width}}"
    return out


def _pfl_flatdatablock_taxis0(
    pfl: PfLine, cols: Iterable[str], num_of_ts: int
) -> Iterable[str]:
    # Obtain dataframe with index = timestamp as string and columns = one or more of 'qwpr'.
    df = _df_with_strvalues(pfl.flatten().df(cols))
    df = _df_with_strindex(df, num_of_ts)
    col_space = {k: v for k, v in COLWIDTHS_TAXIS0.items() if k in df}
    df_str = df.to_string(col_space=col_space, index_names=False, header=False)
    return df_str.split("\n")


def _pfl_nestedtree_taxis0(
    pfl_dict: Dict[str, PfLine], cols: Iterable[str], num_of_ts: float, depth: int
) -> Iterable[str]:
    """Treeview of the portfolio line."""
    out = []
    for c, (name, pfl) in enumerate(pfl_dict.items()):
        tree = _treedict(depth, bool(c == len(pfl_dict) - 1), bool(pfl.children))
        # Name.
        out.append(tree["00"] + tree["01"] + name)
        # Top-level body block.
        for txtline in _pfl_flatdatablock_taxis0(pfl, cols, num_of_ts):
            out.append(tree["10"] + tree["11"] + colorama.Style.RESET_ALL + txtline)
        # Children.
        for txtline in _pfl_nestedtree_taxis0(pfl.children, cols, num_of_ts, depth + 1):
            out.append(tree["10"] + txtline)
    return out


def pfl_as_string(pfl: PfLine, flatten: bool, num_of_ts: int, color: bool) -> str:
    lines = [f"PfLine object with {_what(pfl)} information."]
    lines.extend(_index_info(pfl.index))
    lines.extend(_children_info(pfl))
    lines.extend([""])
    cols = pfl.available
    if flatten:
        lines.extend(_pfl_dataheader_taxis0(cols))
        lines.extend([""])
        lines.extend(_pfl_flatdatablock_taxis0(pfl, cols, num_of_ts))
    else:
        spaces = " " * (MAX_DEPTH + 5)
        lines.extend([spaces + txtline for txtline in _pfl_dataheader_taxis0(cols)])
        lines.extend(_pfl_nestedtree_taxis0({"(this pfline)": pfl}, cols, num_of_ts, 0))
    txt = "\n".join(lines)
    return txt if color else _remove_color(txt)


def pfs_as_string(pfs: PfState, num_of_ts: int, color: bool) -> str:
    lines = ["PfState object."]
    lines.extend(_index_info(pfs.index))
    spaces = " " * (MAX_DEPTH + 5)
    lines.extend([spaces + txtline for txtline in _pfl_dataheader_taxis0("wqpr")])
    lines.extend(_pfl_nestedtree_taxis0({"offtake": pfs.offtake}, "wqpr", num_of_ts, 0))
    lines.extend(
        _pfl_nestedtree_taxis0({"pnl_cost": pfs.pnl_cost}, "wqpr", num_of_ts, 0)
    )
    txt = "\n".join(lines)
    return txt if color else _remove_color(txt)


class PfLineText:
    __repr__ = lambda self: pfl_as_string(self, True, 20, False)

    def print(
        self: PfLine, flatten: bool = False, num_of_ts: int = 5, color: bool = True
    ) -> None:
        """Treeview of the portfolio line.

        Parameters
        ----------
        flatten : bool, optional (default: False)
            if True, show only the top-level (aggregated) information.
        num_of_ts : int, optional (default: 5)
            How many timestamps to show for each PfLine.
        color : bool, optional (default: True)
            Make tree structure clearer by including colors. May not work on all output
            devices.

        Returns
        -------
        None
        """
        print(pfl_as_string(self, flatten, num_of_ts, color))


class PfStateText:
    __repr__ = lambda self: pfs_as_string(self, 5, False)

    def print(self: PfState, num_of_ts: int = 5, color: bool = True) -> None:
        """Treeview of the portfolio state.

        Parameters
        ----------
        num_of_ts : int, optional (default: 5)
            How many timestamps to show for each PfLine.
        color : bool, optional (default: True)
            Make tree structure clearer by including colors. May not work on all output
            devices.

        Returns
        -------
        None
        """
        print(pfs_as_string(self, num_of_ts, color))
