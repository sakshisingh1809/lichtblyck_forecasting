"""Verify input data and turn into object needed in SinglePfLine instantiation."""

from __future__ import annotations
from .abc import PfLine
from ...tools.frames import set_ts_index
from ...tools import nits
from typing import Dict, Optional
import pandas as pd
import numpy as np


def dict_to_dataframe(dict) -> pd.DataFrame:
    """Check data in dictionary, and turn into a general dataframe."""
    indices = [value.index for value in dict.values() if hasattr(value, "index")]
    if len(indices) == 0:
        raise ValueError("No index can be found in the data.")
    if len(indices) > 1 and len(set([i.freq for i in indices])) != 1:
        raise ValueError("Timeseries have unequal frequency; resample first.")
    idx = indices[0]
    for idx2 in indices[1:]:
        idx = idx.intersection(idx2)
    newdict = {}
    for key, val in dict.items():
        newdict[key] = val[idx] if isinstance(val, pd.Series) else pd.Series(val, idx)
    return pd.DataFrame(newdict)


def data_to_wqpr_series(data) -> pd.DataFrame:
    """Turn data into 'standardized series'."""

    # Turn data into object accessible by name.
    if not isinstance(data, PfLine):
        # Turn into dataframe...
        data = dict_to_dataframe(data) if isinstance(data, Dict) else pd.DataFrame(data)
        # ... in certain standard form.
        data = set_ts_index(data)

    # Get timeseries and add unit.
    def series_or_none(obj, col):  # remove series that are passed but only contain na
        s = obj.get(col)
        if s is None or s.isna().all():
            return None
        unit = nits.name2unit(col)
        return s.astype(f"pint[{unit}]")  # set (i.e., assume) unit, or convert to unit.

    # Turn into 4 series.
    w, q, p, r = (series_or_none(data, k) for k in "wqpr")
    return w, q, p, r


def wqpr_series_to_singlepfline_dataframe(
    w: Optional[pd.Series],
    q: Optional[pd.Series],
    p: Optional[pd.Series],
    r: Optional[pd.Series],
) -> pd.DataFrame:
    """Check data in series, and turn into dataframe with q, p, or qr columns."""

    # Get price information.
    if p is not None and w is None and q is None and r is None:
        # We only have price information. Return immediately.
        return set_ts_index(pd.DataFrame({"p": p}))  # kind == 'p'

    # Get quantity information (and check consistency).
    if q is None and w is None:
        if r is None or p is None:
            raise ValueError("Must supply (a) volume, (b) price, or (c) both.")
        q = r / p
    if q is None:
        q = w * w.index.duration
    elif w is not None and not np.allclose(q, w * w.index.duration):
        raise ValueError("Passed values for `q` and `w` not consistent.")

    # Get revenue information (and check consistency).
    if p is None and r is None:
        return set_ts_index(pd.DataFrame({"q": q}))  # kind == 'q'
    if r is None:  # must calculate from p
        r = p * q
        i = r.isna()  # edge case p==nan. If q==0, assume r=0. If q!=0, raise error
        if i.any():
            if (abs(q.pint.m[i]) > 0.001).any():
                raise ValueError("Found timestamps with `p`==na, `q`!=0. Unknown `r`.")
            r[i] = 0
    elif p is not None and not np.allclose(r, p * q):
        # Edge case: remove lines where p==nan and q==0 before judging consistency.
        i = p.isna()
        if not (abs(q[i]) < 0.001).all() or not np.allclose(r[~i], p[~i] * q[~i]):
            raise ValueError("Passed values for `q`, `p` and `r` not consistent.")
    return set_ts_index(pd.DataFrame({"q": q, "r": r}).dropna())  # kind == 'all'


def make_df(data) -> pd.DataFrame:
    """From data, create a DataFrame with column `q`, column `p`, or columns `q` and `r`,
    with relevant units set to them. Also, do some data verification."""
    w, q, p, r = data_to_wqpr_series(data)
    df = wqpr_series_to_singlepfline_dataframe(w, q, p, r)
    return df
