"""Create PfLine instance."""

from .abc import PfLine
from .single import SinglePfLine
from .multi import MultiPfLine
from typing import Iterable


def create_pfline(data) -> PfLine:
    """Create SinglePfLine or MultiPfLine instance from input data."""

    if isinstance(data, SinglePfLine) or isinstance(data, MultiPfLine):
        return data

    if isinstance(data, Iterable) and any(e in data for e in "wqpr"):
        return SinglePfLine(data)

    if any(hasattr(data, e) for e in "wqpr"):
        return SinglePfLine(data)

    return MultiPfLine(data)
