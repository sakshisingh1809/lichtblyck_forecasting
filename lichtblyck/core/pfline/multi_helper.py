"""Verify input data and turn into object needed in MultiPfLine instantiation."""

from __future__ import annotations

from . import multi
from .base import PfLine

from typing import Counter, Mapping, Dict


def make_childrendict(data) -> Dict[str, PfLine]:
    """From data, create a dictionary of PfLine instances. Also, do some data verification."""
    children = _data_to_childrendict(data)
    _assert_pfline_kindcompatibility(children)
    children = _intersect_indices(children)
    return children


def _data_to_childrendict(data) -> Dict[str, PfLine]:
    """Check data, and turn into a dictionary."""

    if isinstance(data, multi.MultiPfLine):
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
            children[key] = PfLine(value)  # try to cast to PfLine instance

    return children


def _assert_pfline_kindcompatibility(children: Dict) -> None:
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


def _intersect_indices(children: Dict[str, PfLine]) -> Dict[str, PfLine]:
    """Keep only the overlapping part of each PfLine's index."""

    if len(children) < 2:
        return children  # No index errors if only 1 child.

    indices = [child.index for child in children.values()]

    # Check frequency.

    if len(set([i.freq for i in indices])) != 1:
        raise ValueError("PfLines have unequal frequency; resample first.")

    # Check/fix indices.

    idx = indices[0]
    for idx2 in indices[1:]:
        idx = idx.intersection(idx2)
    if len(idx) == 0:
        raise ValueError("PfLine indices describe non-overlapping periods.")

    children = {name: child.loc[idx] for name, child in children.items()}
    return children
