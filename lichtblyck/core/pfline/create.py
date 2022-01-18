"""Create PfLine instance."""


# from .single import SinglePfLine
# from .multi import MultiPfLine
from . import multi, single
from .base import PfLine

from typing import Iterable


def create_pfline(data) -> PfLine:
    """Create SinglePfLine or MultiPfLine instance from input data."""

    if isinstance(data, single.SinglePfLine) or isinstance(data, multi.MultiPfLine):
        return data

    if isinstance(data, Iterable) and any(e in data for e in "wqpr"):
        return single.SinglePfLine(data)

    if any(hasattr(data, e) for e in "wqpr"):
        return single.SinglePfLine(data)

    return multi.MultiPfLine(data)
