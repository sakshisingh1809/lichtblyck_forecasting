#%%
import lichtblyck as lb
import pandas as pd

u1 = lb.dev.get_singlepfline(kind="p")
u2 = lb.dev.get_singlepfline(u1.index, kind="all")
i = pd.date_range("2022", freq="15T", periods=10000)
u3 = lb.dev.get_pfstate()
#

# %%

u3
# %%
