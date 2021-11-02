"""
Extend pandas classes; add new attributes.
"""

from . import attributes
from ..tools.frames import wavg
import pandas as pd

pd.core.frame.NDFrame.duration = property(attributes._duration_frame)
pd.core.frame.NDFrame.ts_right = property(attributes._ts_right_frame)

pd.core.frame.NDFrame.wavg = wavg
pd.DatetimeIndex.duration = property(attributes._duration)
pd.DatetimeIndex.ts_right = property(attributes._ts_right)
