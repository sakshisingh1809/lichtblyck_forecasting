import pandas as pd
import numpy as np
import pint
import pint_pandas

# For reference: __neg__ on Series without units.
# -----------------------------------------------

# Input:
s_plain = pd.Series(np.random.rand(2), index=['a', 'b'])
# a    0.380723
# b    0.777449
# dtype: float64

# Output:
-s_plain
# a   -0.380723
# b   -0.777449
# dtype: float64


# Issue: __neg__ on Series with pint dtype.
# -----------------------------------------

# Input:
s_units = s_plain.astype('pint[MWh]')
# ...\Software\Anaconda\envs\lb38\lib\site-packages\pint_pandas\pint_array.py:648: UnitStrippedWarning: The unit of the quantity is stripped when downcasting to ndarray.
#   return np.array(qtys, dtype="object", copy=copy)
# a      0.380723445919278
# b     0.7774489057875323
# dtype: pint[megawatt_hour]

# Won't work:
-s_units # TypeError: Unary negative expects numeric dtype, not pint[megawatt_hour]

# Workaround:
pd.Series(-s_units.pint.m).astype(f'pint[{s_units.pint.units}]')

# versions:
{'numpy': '1.20.1', 'pandas': '1.2.4', 'pint': '0.18', 'pint_pandas': '0.2'}