"""Testing of pandas objects, taking into account they may have units."""
import functools
from ..tools.nits import Q_
import pandas as pd


@functools.wraps(pd.testing.assert_frame_equal)
def assert_frame_equal(left, right, *args, **kwargs):
    if hasattr(left, "pint"):
        left = left.pint.to_base_units().pint.dequantify()
    if hasattr(right, "pint"):
        right = right.pint.to_base_units().pint.dequantify()
    # Dataframes equal even if *order* of columns is not the same.
    left, right = left.sort_index(axis=1), right.sort_index(axis=1)
    pd.testing.assert_frame_equal(left, right, *args, **kwargs)


@functools.wraps(pd.testing.assert_series_equal)
def assert_series_equal(left, right, *args, **kwargs):
    # If series of quantities, make units equal.
    if hasattr(left, "pint"):
        left = left.pint.to_base_units().pint.m
    if hasattr(right, "pint"):
        right = right.pint.to_base_units().pint.m

    try:
        pd.testing.assert_series_equal(left, right, *args, **kwargs)
    except AssertionError:
        raise
    except Exception:
        # If left or right is a Series of Quantities: split Quantities into (magn, unit)-tuples.
        def split(val):
            if isinstance(val, Q_):
                val = val.to_base_units()
                return (val.m, val.u)
            return (val, None)

        assert_series_equal(left.apply(split), right.apply(split), *args, **kwargs)


assert_index_equal = pd.testing.assert_index_equal
