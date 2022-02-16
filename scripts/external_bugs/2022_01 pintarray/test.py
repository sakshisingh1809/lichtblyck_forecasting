# %%

import pandas as pd
import numpy as np
from pint_pandas import PintArray

pa = PintArray([1, 45, -4.5], "m")
s1 = pd.Series(pa)
s2 = s1.astype("pint[cm]")
s = pd.Series(pa.quantity.magnitude)
df = pd.DataFrame({"d": pa})


def series_allclose(s1, s2):
    without_units = [not (hasattr(s, "pint")) for s in [s1, s2]]
    if all(without_units):
        return np.allclose(s1, s2)
    elif any(without_units):
        return False
    elif s1.pint.dimensionality != s2.pint.dimensionality:
        return False
    # Both have units, and both have same dimensionality (e.g. 'length'). Check values.
    s1_vals = s1.pint.m
    s2_vals = s2.astype(f"pint[{s1.pint.u}]").pint.m
    return np.allclose(s1_vals, s2_vals)


def pintarray_allclose(pa1, pa2):
    if pa1.quantity.dimensionality != pa2.quantity.dimensionality:
        return False
    a1 = pa1.quantity.magnitude
    a2 = pa2.astype(f"pint[{pa1.quantity.units}]").quantity.magnitude
    return np.allclose(a1, a2)


# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
#       16 pa = PintArray([1, 45, -4.5], 'm')
# ----> 17 np.allclose(pa, pa)

# <__array_function__ internals> in allclose(*args, **kwargs)

# ~\Anaconda3\envs\lb38\lib\site-packages\numpy\core\numeric.py in allclose(a, b, rtol, atol, equal_nan)
#    2247
#    2248     """
# -> 2249     res = all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
#    2250     return bool(res)
#    2251

# <__array_function__ internals> in isclose(*args, **kwargs)

# ~\Anaconda3\envs\lb38\lib\site-packages\numpy\core\numeric.py in isclose(a, b, rtol, atol, equal_nan)
#    2353         y = asanyarray(y, dtype=dt)
#    2354
# -> 2355     xfin = isfinite(x)
#    2356     yfin = isfinite(y)
#    2357     if all(xfin) and all(yfin):

# TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

# %%
