# -*- coding: utf-8 -*-
"""
Module to read price data from disk.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lichtblick.tools.tools import set_ts_index


def spot() -> pd.Series:
    """Return spot price timeseries."""
    filepath = Path('lichtblick/prices/sourcedata/spot.tsv')
    data = pd.read_csv(
        filepath, header=None, sep='\t', 
        names=['date', 'time', 'price', 'anmerkungen', 'empty'])
    data['ts_right'] = pd.to_datetime(data["date"] + " " + data["time"], format="%d.%m.%Y %H:%M:%S")
    data['p_spot'] = data['price'].apply(lambda x: np.NaN if x == '---' else float(x.replace(',', '.'))) #Replace missing values with NaN, convert others to float.
    spot = set_ts_index(data, 'ts_right', 'right')
    return spot['p_spot']