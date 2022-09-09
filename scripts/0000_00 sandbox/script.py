#%%
from matplotlib import pyplot as plt
import lichtblyck as lb
import pandas as pd
import portfolyo as pf

lb.belvis.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")

# %%


o6 = lb.belvis.data.offtakevolume("gas", "SBK6_G", "2022")
s6 = lb.belvis.data.sourced("gas", "SBK6_G", "2022")
o1 = lb.belvis.data.offtakevolume("gas", "SBK1_G", "2022")
s1 = lb.belvis.data.sourced("gas", "SBK1_G", "2022")
u = lb.belvis.data.unsourcedprice("gas", "2022")
pfs1 = lb.portfolios.pfstate("gas", "SBK1", "2022")
pfs6 = lb.portfolios.pfstate("gas", "SBK6", "2022")
pfs = lb.portfolios.pfstate("gas", "B2C_LEGACY", "2022")

# %%
pfs = pfs1.__add__(pfs6)

# %%
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
