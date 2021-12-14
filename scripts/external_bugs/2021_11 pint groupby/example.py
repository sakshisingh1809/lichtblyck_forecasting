import pandas as pd
import numpy as np
import pint
import pint_pandas

# For reference: groupby on Series without units.
# -----------------------------------------------

# Input:
s_plain = pd.Series(np.random.rand(6), index=['a', 'b', 'c'] * 2)
# a    0.032244
# b    0.073113
# c    0.601260
# a    0.953453
# b    0.821208
# c    0.543708
# dtype: float64

# Output:
s_plain.groupby(level=0, axis=0).mean()
# a    0.492848
# b    0.447161
# c    0.572484
# dtype: float64


# Issue: groupby on Series with pint dtype.
# -----------------------------------------

# Input:
s_units = s_plain.astype('pint[MWh]')
# a    0.032243727749885376
# b     0.07311299213301359
# c      0.6012603010230052
# a      0.9534530925540614
# b      0.8212084539670164
# c      0.5437076762858951
# dtype: pint[megawatt_hour]

# Won't work:
s_units.groupby(level = 0, axis=0).mean() # DataError: No numeric types to aggregate


# I have been looking for work-arounds which I share here:

# Workaround, step 1. Not yet ideal, because:
# returns Series of *quantities*, i.e. Series without pint dtype.
s_units.groupby(level= 0, axis=0).apply(np.mean)
# a    0.4928484101519734 megawatt_hour
# b     0.447160723050015 megawatt_hour
# c    0.5724839886544502 megawatt_hour
# dtype: object

# Workaround try 2. Returns expected result.
s_units.groupby(level= 0, axis=0).apply(np.mean).astype('pint[MWh]')
s_units.groupby(level= 0, axis=0).apply(np.mean).astype(f'pint[{s_units.pint.units}]')
# 0    0.4928484101519734
# 1     0.447160723050015
# 2    0.5724839886544502
# dtype: pint[megawatt_hour]