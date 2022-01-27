import pint
import pint_pandas
import pandas as pd
import numpy as np

s = pd.Series(np.random.rand(2), dtype="pint[J]")

df_good = pd.DataFrame()
df_good["energy"] = s


df_bad = pd.DataFrame(columns=[[], []])
df_bad[("toy", "energy")] = s
