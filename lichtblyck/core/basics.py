"""
Extend pandas classes; add new attributes.
"""

from . import attributes
import pandas as pd

FREQUENCIES = ["AS", "QS", "MS", "D", "H", "15T"]

pd.core.frame.NDFrame.duration = property(attributes._duration)
pd.core.frame.NDFrame.ts_right = property(attributes._ts_right)
pd.core.frame.NDFrame.wavg = attributes.wavg
