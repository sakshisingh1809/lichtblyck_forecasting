import pandas as pd
import numpy as np

df = pd.DataFrame(
    np.random.rand(365, 2),
    pd.date_range("2022", freq="D", periods=365, tz="Europe/Berlin"),
    pd.MultiIndex.from_product([["a"], ["i", "ii"]]),
)
df.resample("MS").mean()
