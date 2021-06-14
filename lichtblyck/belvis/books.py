"""
Get information about the portfolio structure in Belvis.

Uses kibasic language to fetch data but only needs to be repeated if PF-structure 
changes. If not, .find_node uses memoized information.
"""

from anytree import Node, RenderTree
from urllib import parse
from typing import List
import pandas as pd
import datetime as dt
import subprocess
import pathlib


# %% Using the kibasic language. Only needs to be repeated if PF-structure changes.

EXE = pathlib.Path("C:\\") / "Kisters" / "BelVis" / "kibasic.exe"
MEM = pathlib.Path(__file__).parent / "memoized"


def _fetch_books(target):
    bas = pathlib.Path(__file__).parent / "book.bas"
    # TODO: pass target as parameter to script
    p1 = subprocess.Popen([EXE, "-nogui", bas], stdout=subprocess.PIPE)
    u = p1.communicate()  # TODO: wie weiss ich, ob hier ein Fehler passiert ist?
    message = bytes.decode(u[0])
    return message


def _load_books() -> pd.DataFrame:
    bookpath = MEM / "book.csv"
    books = pd.read_csv(bookpath, encoding="ansi")
    books["node"] = books.apply(
        lambda row: Node(
            row["SHORTNAME_BOOK_S"], id=row["IDENT_BOOK_L"], pid=row["IDENT_PARENT_L"]
        ),
        axis=1,
    )
    for node in books.node:
        node.children = [n for n in books.node if round(n.pid, 0) == round(node.id, 0)]
    return books


def find_node(pf: str) -> Node:
    """
    Return Node with specified name.

    Parameters
    ----------
    pf : str
        The short name of the root portfolio, i.e., of the portfolio at the highest
        level of interest.

    Returns
    -------
    Node
        with .id and .name attributes corresponding to the Belvis id and name of this
        book, and sub-portfolios as Nodes in its .children attribute.
    """
    # Get book structure as tree...
    books = _load_books()
    # ...and find correct node.
    nodes = books[books["SHORTNAME_BOOK_S"] == pf].node
    if not pf or not len(nodes):
        if pf is None:
            error = 'Argument "pf" not specified.'
        else:
            error = f'Book "{pf}" not found.'
        raise ValueError(
            error
            + " Please pick a node from the following: \n"
            + "\n".join(show_structure())
        )
    return nodes.iloc[0]


def show_structure() -> List[str]:
    books = _load_books()
    return [
        RenderTree(node).by_attr()
        for node in books[books["IDENT_PARENT_L"].isna()].node
    ]
