"""Create temprisk heatmap - what-if-analysis for 2022H2."""


#%%
from dataclasses import dataclass
import lichtblyck as lb
import numpy as np
from tqdm import tqdm
import pandas as pd
import datetime as dt


@dataclass
class Simresult:
    pbase: pd.Series  # base price [Eur/MWh] per quarter
    pfs: lb.PfState
    pfl: lb.PfLine  # Materialized offtake volume, price, cost per quarter.
    po_tr: pd.Series  # Materialized temprisk [Eur/MWh] per quarter. Is the increase in portfolio price vs reference portfolio price.
    ro_tr: pd.Series  # Materialized temprisk [Eur] per quarter. Is the increase in portfolio cost *that is not covered by the reference price*.


lb.belvis.auth_with_passwordfile("username_password.txt")

# %% Settings

commodity, pfname = [("power", "B2C_P2H_LEGACY"), ("gas", "B2C_LEGACY")][0]

if pfname == "B2C_P2H_LEGACY":
    scalingfile = "p2h_offtakescalingfactors_per_calmonth_and_degC.csv"
elif pfname == "B2C_LEGACY":
    scalingfile = "gas_offtakescalingfactors_per_calmonth_and_degC.csv"
else:
    raise ValueError("unexpected pfname")

# %%

ref = lb.portfolios.pfstate(commodity, pfname, "2022-06", "2023-04", recalc=False)
ref = ref.add_sourced(ref.hedge_of_unsourced("MS"))  # fully hedge at today's prices
ref_costs_quarter = ref.asfreq("QS").pnl_cost
table = pd.read_csv(scalingfile, index_col=0)
sensitivity = table["1 degC"] - 1
table.columns = [int(c.replace(" degC", "")) for c in table.columns]


# %%

simresults = {}
for delta_p_fraction in tqdm([-0.75, -0.50, -0.25, 0, 0.25, 0.5, 0.75, 1]):
    intermediate = ref.set_unsourcedprice(ref.unsourcedprice * (1 + delta_p_fraction))
    pbase = intermediate.unsourcedprice.asfreq("QS").p
    for delta_t in tqdm(np.arange(-2, 3)):
        factors = table.loc[:, delta_t]
        asseries = factors.loc[ref.index.month].set_axis(ref.index)
        sim = intermediate.set_offtakevolume(ref._offtakevolume * asseries).asfreq("QS")
        sim_costs_quarter = sim.pnl_cost
        po_tr = sim_costs_quarter.p - ref_costs_quarter.p
        ro_tr = po_tr * sim_costs_quarter.q
        simresults[(delta_p_fraction, delta_t)] = Simresult(
            pbase, sim, sim_costs_quarter, po_tr, ro_tr
        )

# %%

series_ro = pd.DataFrame({key: sr.pfl.r for key, sr in simresults.items()}).unstack()
series_ro_tr = pd.DataFrame({key: sr.ro_tr for key, sr in simresults.items()}).unstack()

series_po = pd.DataFrame({key: sr.pfl.p for key, sr in simresults.items()}).unstack()
series_po_tr = pd.DataFrame({key: sr.po_tr for key, sr in simresults.items()}).unstack()

series_q = pd.DataFrame({key: sr.pfl.q for key, sr in simresults.items()}).unstack()
series_pbase = pd.DataFrame({key: sr.pbase for key, sr in simresults.items()}).unstack()

writer = pd.ExcelWriter(
    f"heatmap_temprisk_{commodity}_{pfname}_on_{dt.date.today()}.xlsx",
    engine="xlsxwriter",
)

series_ro.pint.m.unstack(1).tz_localize(None, 0, 1).to_excel(writer, "r_procurement")
series_ro_tr.pint.m.unstack(1).tz_localize(None, 0, 1).to_excel(writer, "r_temprisk")
series_po.pint.m.unstack(1).tz_localize(None, 0, 1).to_excel(writer, "p_procurement")
series_po_tr.pint.m.unstack(1).tz_localize(None, 0, 1).to_excel(writer, "p_temprisk")

series_q.pint.m.unstack(1).tz_localize(None, 0, 1).to_excel(writer, "q")
series_pbase.pint.m.unstack(1).tz_localize(None, 0, 1).to_excel(writer, "pbase")

writer.save()
writer.close()  # will give warning ('already closed') but still necessary to open in excel

# %%
