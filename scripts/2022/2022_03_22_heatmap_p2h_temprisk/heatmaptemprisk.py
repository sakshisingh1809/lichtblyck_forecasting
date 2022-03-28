"""Create temprisk heatmap - what-if-analysis for 2022H2."""


#%%
from dataclasses import dataclass
import lichtblyck as lb
import numpy as np
from tqdm import tqdm
import pandas as pd
import datetime as dt

lb.belvis.auth_with_passwordfile("username_password.txt")

ref = lb.portfolios.pfstate("power", "B2C_P2H_LEGACY", "2022-07")

#%%

table = pd.read_csv("p2h_offtakescalingfactors_per_calmonth_and_degC.csv", index_col=0)
sensitivity = table["1 degC"] - 1
table.columns = [int(c.replace(" degC", "")) for c in table.columns]

#%%


@dataclass
class Simresult:
    pbase: float  # base price
    pfs: lb.PfState
    ro: float  # total cost
    qo: float  # offtake volume
    po: float  # total price


# %%

simresults = {}
for delta_p in tqdm([-200, -150, -100, -50, 0, 50, 100, 150, 200]):
    ref2 = ref.set_unsourcedprice(ref.unsourcedprice + delta_p)
    pbase = ref2.unsourcedprice.p.mean()
    for delta_t in tqdm(np.arange(-4, 5)):
        factors = table.loc[6:, delta_t]
        asseries = factors.loc[ref.index.month].set_axis(ref.index)
        pfs = ref2.set_offtakevolume(ref._offtakevolume * asseries).asfreq("MS")
        pfl = pfs.pnl_cost
        r, q = pfl.r.sum().magnitude, pfl.q.sum().magnitude
        simresults[(delta_p, delta_t)] = Simresult(pbase, pfs, r, q, r / q)

# %%

df_r = pd.Series({key: sr.ro for key, sr in simresults.items()}).unstack()
df_r_vsref = df_r - df_r.loc[0, 0]


df_p = pd.Series({key: sr.po for key, sr in simresults.items()}).unstack()
df_p_vsref = df_p - df_p.loc[0, 0]

df_q = pd.Series({key: sr.qo for key, sr in simresults.items()}).unstack()

writer = pd.ExcelWriter(
    f"heatmap_2022H2_on_{dt.date.today()}.xlsx", engine="xlsxwriter"
)
df_p.to_excel(writer, "p")
df_r.to_excel(writer, "r")
df_p_vsref.to_excel(writer, "delta_p")
df_r_vsref.to_excel(writer, "delta_r")
df_q.to_excel(writer, "q")


writer.save()
writer.close()  # will give warning ('already closed') but still necessary to open in excel

# %%
