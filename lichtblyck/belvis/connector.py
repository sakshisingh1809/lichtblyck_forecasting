"""
Retrieve data from Belvis.
"""

from ..core.portfolio import Portfolio
from anytree import Node, RenderTree
import pandas as pd
import numpy as np
import datetime as dt
import subprocess
import pathlib

EXE = pathlib.Path("C:\\") / "Kisters" / "BelVis" / "kibasic.exe"
MEM = pathlib.Path(__file__).parent / "memoized"
TSNAMES = {
    "offtake.w": "#LB Saldo aller Prognosegeschäfte +UB",
    "procured.FW.w": "#LB Saldo aller Termingeschäfte +UB",
    "procured.FW.r": "#LB Wert aller Termingeschäfte",
    "procured.DA.w": "#LB Saldo aller Spotgeschäfte",
    "procured.DA.r": "#LB Wert aller Spotgeschäfte",
    "procured.ID.w": "#LB Saldo aller Intradaygeschäfte",
    "procured.ID.r": "#LB Wert aller Intradaygeschäfte",
}


def _age(path) -> float:
    """Time lapsed sind last time file was modified, in days (float). np.inf if
    file does not exist"""
    if not path.exists():
        return np.inf
    delta = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
    return delta.total_seconds() / 24 / 3600


def _fetch_books(target):
    bas = pathlib.Path(__file__).parent / "book.bas"
    # TODO: pass target as parameter to script
    p1 = subprocess.Popen([EXE, "-nogui", bas], stdout=subprocess.PIPE)
    u = p1.communicate()  # TODO: wie weiss ich, ob hier ein Fehler passiert ist?
    message = bytes.decode(u[0])
    return message


def _fetch_timeseries(target):
    bas = pathlib.Path(__file__).parent / "timeseries.bas"
    # TODO: pass target as parameter to script
    p1 = subprocess.Popen([EXE, "-nogui", bas], stdout=subprocess.PIPE)
    u = p1.communicate()
    message = bytes.decode(u[0])
    return message


def _fetch_values(target):
    bas = pathlib.Path(__file__).parent / "values.bas"
    # TODO: pass tsid, datefrom, dateuntil, target as parameter to script
    p1 = subprocess.Popen([EXE, "-nogui", bas], stdout=subprocess.PIPE)
    u = p1.communicate()
    message = bytes.decode(u[0])
    return message


def _booknode(bookname: str, maxage: float = 14) -> Node:
    """
    Return Node with specified name.

    Parameters
    ----------
    bookname : str
        The short name of the root portfolio, i.e., of the portfolio at the highest
        level of interest.
    maxage : float
        Maximum age (in days) of file with book structure information before it is refreshed
        (= re-fetched from Belvis). The default is 14.

    Returns
    -------
    Node
        with .id attribute corresponding to the Belvis id of this book, and other
        Nodes in its .children attribute.
    """
    # Get book structure...
    bookpath = MEM / "book.csv"
    if _age(bookpath) > maxage:
        message = _fetch_books(bookpath)
        age = _age(bookpath)
        if age > maxage:
            if age == np.inf:
                error = "File does not exist."
            else:
                error = "File older ({age:.2f} days) than wanted."
        raise RuntimeError(error + " Output of Belvis script:\n" + message)
    # ...as tree...
    books = pd.read_csv(bookpath, encoding="ansi")
    books["node"] = books.apply(
        lambda row: Node(
            row["SHORTNAME_BOOK_S"], id=row["IDENT_BOOK_L"], pid=row["IDENT_PARENT_L"]
        ),
        axis=1,
    )
    for node in books.node:
        node.children = [n for n in books.node if round(n.pid, 0) == round(node.id, 0)]
    # ...and find correct node.
    nodes = books[books["SHORTNAME_BOOK_S"] == bookname]
    if bookname is None or len(nodes) == 0:
        if bookname is None:
            error = 'Argument "bookname" not specified.'
        else:
            error = f'Book "{bookname}" not found.'
        trees = [
            RenderTree(node).by_attr()
            for node in books[books["IDENT_PARENT_L"].isna()].node
        ]
        raise ValueError(error + " Please pick one from the following:\n".join(trees))
    return nodes[1]


def _add_timeseries(booknode: Node, maxage: float = 14) -> None:
    """
    Adds IDs of relevant timeseries to node and its descendents, as attribute .tsids.

    Parameters
    ----------
    booknode : Node
        Node of the root portfolio.
    maxage : float, optional
        Maximum age (in days) of file with timeseries information before it is refreshed
        (= re-fetched from Belvis). The default is 14.

    Returns
    -------
    None
    """
    # Get timeseries...
    tspath = MEM / "timeseries.csv"
    if _age(tspath) > maxage:
        message = _fetch_timeseries(tspath)
        age = _age(tspath)
        if age > maxage:
            if age == np.inf:
                error = "File does not exist."
            else:
                error = "File older ({age:.2f} days) than wanted."
        raise RuntimeError(error + " Output of Belvis script:\n" + message)
    ts = pd.read_csv(tspath, encoding="ansi")
    # ...and add relevant IDs to Nodes.
    def add_tsids(node: Node):
        node.tsids = {}
        for key, tsname in TSNAMES.items():
            found = ts[(ts["IDENT_BOOK_L"] == node.id) & (ts["NAME_ZR_S"] == tsname)]
            if len(found) == 0:
                errors.append(f'Not found in book "{node.name}": "{tsname}"')
                tsid = "error: missing"
            elif len(found) > 1:
                errors.append(f'Found multiple in book "{node.name}": "{tsname}"')
                tsid = "error: multiple"
            else:
                tsid = found["IDENT_BOOK_L"].iloc[0]  # timeseries ID
            node.tsids[key] = tsid
        for child in node.children:
            add_tsids(child)

    errors = []
    add_tsids(booknode)
    if len(errors):
        raise ValueError("While finding needed timeseries:\n" + "\n".join(errors))


def power(bookname: str = None, maxage: float = 2) -> Portfolio:
    """
    Load Belvis data into Portfolio object.

    Parameters
    ----------
    bookname : str, optional
        The short name of the root portfolio, i.e., of the portfolio at the highest
        level of interest. Common choices: 'PKG', 'GKG', 'LUD', 'PK_Neu'.
    maxage : float, optional
        Data found locally (in folder 'memoized') is refreshed (i.e., fetches from
        Belvis) if it is older than this many days. The default is 2.0.

    Returns
    -------
    Portfolio object.
    """
    book = _booknode(bookname, maxage)  # Relevant attributes: .id and .children
    _add_timeseries(book)  # Adds attribute: .tsids
