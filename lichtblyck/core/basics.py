"""
Extend pandas classes; add new attributes.
"""

from ..tools.stamps import duration, ts_right
from ..tools.frames import wavg
import pandas as pd


pd.core.frame.NDFrame.wavg = wavg
pd.DatetimeIndex.duration = property(duration)
pd.DatetimeIndex.ts_right = property(ts_right)
pd.Timestamp.duration = property(duration)
pd.Timestamp.ts_right = property(ts_right)
