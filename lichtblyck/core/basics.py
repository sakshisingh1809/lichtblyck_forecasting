"""
Extend pandas classes; add new attributes.
"""

from ..tools import stamps, frames
import pandas as pd


def apply():
    pd.core.frame.NDFrame.wavg = frames.wavg
    pd.DatetimeIndex.duration = property(stamps.duration)
    pd.DatetimeIndex.ts_right = property(stamps.ts_right)
    pd.Timestamp.duration = property(stamps.duration)
    pd.Timestamp.ts_right = property(stamps.ts_right)
