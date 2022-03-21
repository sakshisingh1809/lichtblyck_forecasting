"""
Module with mixins, to add other output-methods to PfLine and PfState classes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union
import pandas as pd
import functools

if TYPE_CHECKING:  # needed to avoid circular imports
    from ..pfstate import PfState
    from ..pfline import PfLine


def _prepare_df(pfl_or_pfs: Union[PfLine, PfState]) -> pd.DataFrame:
    return pfl_or_pfs.df(flatten=False).pint.dequantify().tz_localize(None)


class OtherOutput:  # for both PfLine and PfState
    @functools.wraps(pd.DataFrame.to_clipboard)
    def to_clipboard(self: Union[PfLine, PfState], *args, **kwargs) -> None:
        _prepare_df(self).to_clipboard(*args, **kwargs)

    @functools.wraps(pd.DataFrame.to_excel)
    def to_excel(self: Union[PfLine, PfState], *args, **kwargs) -> None:
        _prepare_df(self).to_excel(*args, *kwargs)
