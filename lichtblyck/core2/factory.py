"""Register factories of portfolio objects."""

from __future__ import annotations
from typing import Callable, TYPE_CHECKING

# if TYPE_CHECKING:  # needed to avoid circular imports
from .pfstate import PfState
from .pfline import PfLine


def register_pfstate_source(methodname:str, function:Callable[..., PfState]) -> None:
    setattr(PfState, methodname, function)

def register_pfline_source(methodname:str, function:Callable[..., PfLine]) -> None:
    setattr(PfLine, methodname, function)