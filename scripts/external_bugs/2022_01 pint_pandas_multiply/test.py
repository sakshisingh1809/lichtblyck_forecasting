import pandas as pd
import numpy as np
import pint_pandas

i = pd.date_range("2022", periods=8760, freq="H", tz="Europe/Berlin")
s_unit = pd.Series(np.random.rand(8760), i, dtype="pint[W]").rename("po")
s_float = pd.Series(np.random.rand(8760), i).rename("qo")

s_unit * s_float
