"""
Extend pandas classes; add new attributes.
"""

from . import attributes
import pandas as pd

FREQUENCIES = ["AS", "QS", "MS", "D", "H", "15T"]

# Perfect containment; a short-frequency time period always entirely falls within a single high-frequency time period.
# AS -> 4 QS; QS -> 3 MS; MS -> 28-31 D; D -> 23-25 H; H -> 4 15T

pd.core.frame.NDFrame.duration = property(attributes._duration)
pd.core.frame.NDFrame.ts_right = property(attributes._ts_right)
pd.core.frame.NDFrame.wavg = attributes.wavg
