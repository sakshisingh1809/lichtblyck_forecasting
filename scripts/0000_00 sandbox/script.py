#%%
from matplotlib import pyplot as plt
import lichtblyck as lb
import pandas as pd

#%%
# lb.belvis.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")
i = pd.date_range("2020", "2020-05", freq="H", closed="left", tz="Europe/Berlin")
u = lb.dev.get_pfstate(i)
u.plot()
fig = u.asfreq("MS").plot()
#%%
fig, ax = plt.subplots(2, 5, figsize=(20, 10))
offtake = u.offtake + lb.tools.nits.Q_(120, "MW")
for i, how in enumerate(["jagged", "bar", "area", "step", "hline"]):
    lb.visualize.visualize.plot_timeseries(
        ax[0, i], offtake.asfreq("MS").w, how, "{:.1f}", True
    )
    lb.visualize.visualize.plot_timeseries(
        ax[1, i], offtake.asfreq("MS").w, how, "{:.1f}", False
    )
# %%

d = lb.dev.get_pfstates()
lb.plot_pfstates(d)
d2 = {k: v.asfreq("MS") for k, v in d.items()}
lb.plot_pfstates(d2)


# %%
