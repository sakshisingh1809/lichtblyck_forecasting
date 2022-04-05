"""Create temprisk heatmap - what-if-analysis for 2022H2."""


#%%
from dataclasses import dataclass
import lichtblyck as lb
import numpy as np
from tqdm import tqdm
import pandas as pd
import datetime as dt


lb.belvis.auth_with_passwordfile("username_password.txt")

# %%

commodity, pfname = [("power", "B2C_P2H_LEGACY"), ("gas", "B2C_LEGACY")][1]

if pfname == "B2C_P2H_LEGACY":
    incomeprices = [-4.00, -151.82, -89.28, -74.14, 40.99, 20.56, 74.45, 90.12]
    scalingfile = "p2h_offtakescalingfactors_per_calmonth_and_degC.csv"
elif pfname == "B2C_LEGACY":
    incomeprices = [19.78, 11.92, 10.28, 10.83, 14.03, 28.48, 33.82, 37.19]
    scalingfile = "gas_offtakescalingfactors_per_calmonth_and_degC.csv"
else:
    raise ValueError("unexpected pfname")

#%%

ref = lb.portfolios.pfstate(commodity, pfname, "2022-05", recalc=False)
plan_price = lb.PfLine(pd.Series(incomeprices, index=ref.asfreq("MS").index, name="p"))
table = pd.read_csv(scalingfile, index_col=0)
sensitivity = table["1 degC"] - 1
table.columns = [int(c.replace(" degC", "")) for c in table.columns]

#%%


@dataclass
class Simresult:
    pbase: float  # base price
    pfs: lb.PfState
    qo: float  # offtake volume
    ro: float  # total cost
    po: float  # total price
    r_tr: float  # actual temprisk [Eur]
    p_tr: float  # actual temprisk [Eur/MWh]


# %%

simresults = {}
for delta_p_fraction in tqdm([-0.75, -0.50, -0.25, 0, 0.25, 0.5, 0.75, 1]):
    ref2 = ref.set_unsourcedprice(ref.unsourcedprice * (1 + delta_p_fraction))
    pbase = ref2.unsourcedprice.p.mean()
    for delta_t in tqdm(np.arange(-4, 5)):
        factors = table.loc[5:, delta_t]
        asseries = factors.loc[ref.index.month].set_axis(ref.index)
        pfs = ref2.set_offtakevolume(ref._offtakevolume * asseries).asfreq("MS")
        pfl = pfs.pnl_cost
        income = plan_price * pfl.q
        qo, ro, ri = (
            pfl.q.sum().magnitude,
            pfl.r.sum().magnitude,
            income.r.sum().magnitude,
        )
        simresults[(delta_p_fraction, delta_t)] = Simresult(
            pbase, pfs, qo, ro, ro / qo, ro - ri, (ro - ri) / qo
        )

# %%

df_ro = pd.Series({key: sr.ro for key, sr in simresults.items()}).unstack()
df_r_tr = pd.Series({key: sr.r_tr for key, sr in simresults.items()}).unstack()

df_po = pd.Series({key: sr.po for key, sr in simresults.items()}).unstack()
df_p_tr = pd.Series({key: sr.p_tr for key, sr in simresults.items()}).unstack()

df_q = pd.Series({key: sr.qo for key, sr in simresults.items()}).unstack()

writer = pd.ExcelWriter(
    f"heatmap_temprisk_{commodity}_{pfname}_on_{dt.date.today()}.xlsx",
    engine="xlsxwriter",
)

df_ro.to_excel(writer, "r_procurement")
df_r_tr.to_excel(writer, "r_temprisk")
df_po.to_excel(writer, "p_procurement")
df_p_tr.to_excel(writer, "p_temprisk")

df_q.to_excel(writer, "q")

writer.save()
writer.close()  # will give warning ('already closed') but still necessary to open in excel

# %%
