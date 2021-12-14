
import pandas as pd
index = pd.date_range('2020', freq='MS', periods=3)
_ = index.map(print)
# %%

import pandas as pd
index = pd.Index([1, 12, 192])
_ = index.map(print)
# %%
