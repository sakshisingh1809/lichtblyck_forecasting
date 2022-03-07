"""
Module with mixins, to add other output-methods to PfLine and PfState classes.
"""

from __future__ import annotations
from ..tools import nits
from typing import List, Callable, Dict, Tuple, TYPE_CHECKING
import pandas as pd
import functools

if TYPE_CHECKING:  # needed to avoid circular imports
    from .pfstate import PfState
    from .pfline import PfLine


class OtherOutput:  # for both PfLine and PfState
    @functools.wraps(pd.DataFrame.to_clipboard)
    def to_clipboard(self: PfLine, *args, **kwargs) -> None:
        self.df().pint.dequantify().to_clipboard(*args, **kwargs)

    @functools.wraps(pd.DataFrame.to_excel)
    def to_excel(self: PfLine, *args, **kwargs) -> None:
        self.df().pint.dequantify().tz_localize(None).to_excel(*args, *kwargs)
