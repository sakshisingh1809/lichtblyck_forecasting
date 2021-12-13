"""Type aliases."""

import datetime as dt
import pandas as pd
from typing import Union
from .nits import Q_


Stamp = Union[dt.datetime, pd.Timestamp]
Value = Union[float, int, Q_]