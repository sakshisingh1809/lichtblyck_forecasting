"""Verify input data and turn into object needed in MultiPfLine instantiation."""

from __future__ import annotations
from .abc import PfLine
from .multi import MultiPfLine
from .create import create_pfline
from typing import Counter, Mapping, Dict


def data_to_childrendict(data) -> Dict[str, PfLine]:
    """Check data, and turn into a dictionary."""

    if isinstance(data, MultiPfLine):
        return data.children
    if not isinstance(data, Mapping):
        raise TypeError(
            "`data` must be dict or other Mapping (or a MultiPfLine instance)."
        )

    children = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError("Keys must be strings.")
        if isinstance(value, PfLine):
            children[key] = value
        else:
            children[key] = create_pfline(value)  # try to cast to PfLine instance

    return children


def assert_pfline_kindcompatibility(children: Dict) -> None:
    """Check pflines in dictionary, and raise error if their kind is not compatible."""

    if len(children) == 0:
        raise ValueError("Must provide at least 1 child.")

    if len(children) == 1:
        return  # No possible compatibility errors if only 1 child.

    # Check kind.

    kindcounter = Counter(child.kind for child in children.values())

    if len(kindcounter) == 1:
        return  # No compatibility error if all children of same kind.

    if kindcounter["p"] == kindcounter["q"] == 1 and kindcounter["all"] == 0:
        return  # Children of distinct can only be combined in this exact setting.

    raise ValueError(
        "All children must be of the same kind, or there must be exactly one volume-only child (i.e., with .kind == 'q') and one price-only child (i.e., with .kind == 'p')."
    )


def intersect_indices(children: Dict[str, PfLine]) -> Dict[str, PfLine]:
    """Keep only the overlapping part of each PfLine's index."""

    indices = [child.index for child in children.values()]
    
    # Check frequency.

    if len(indices) > 1 and len(set([i.freq for i in indices])) != 1:
        raise ValueError("PfLines have unequal frequency; resample first.")

    # Check/fix indices.

    if len(children)
    idx = indices[0]
    for idx2 in indices[1:]:
        idx = idx.intersection(idx2)
    return children


def make_childrendict(data) -> Dict[str, PfLine]:
    """From data, create a dictionary of PfLine instances. Also, do some data verification."""
    children = data_to_childrendict(data)
    assert_pfline_kindcompatibility(children)
    children = intersect_indices(children)
    return children
