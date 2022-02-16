"""Analyse relative contribution of several factors to loss in 2021Q4."""

# %% Imports and prep.

import lichtblyck as lb
import pandas as pd

lb.belvis.auth_with_password("Ruud.Wijtvliet", "Ammm1mmm2mmm3mmm")

# As-is portfolio.
pfs = lb.portfolios.pfstate("power", "B2C_P2H", "2021-10", "2022")
# Ingredients for analysis:
# . QHPFC in may:
mayprices = pd.read_csv("QHPFC.csv", ";", decimal=",")
mayprices = mayprices[mayprices.columns[-1]]
mayprices.index = pfs.offtake.index
mayprices = lb.PfLine({"p": mayprices})
# . Offtake under norm temperatures:
# if norm then nov 0.5degC warmer so 1.5% less offtake
i = pfs.offtake.index
novmask = (i >= "2021-11") & (i < "2021-12")
nov = pfs.offtake.q.loc[novmask] / 1.015
octdec = pfs.offtake.q.loc[~novmask]
normofftake = lb.PfLine({"q": pd.concat([nov, octdec], axis=0)})


# %% Reference frames.

# A) As-is: no additional hedging
pfs_0h = pfs
result_0h = pfs_0h.changefreq("MS")

# B) As-should 'months': additional hedging of months in may
# volumes as predicted in may and prices as predicted in may
pfs_may_before = lb.PfState(normofftake, mayprices, pfs.sourced)
to_buy = pfs_may_before.hedge_of_unsourced("val", "MS")
pfs_may_afterm = pfs_may_before.add_sourced(to_buy)
# resulting situation in december with actual volumes and actual prices
pfs_mh = lb.PfState(pfs.offtake, pfs.unsourcedprice, pfs_may_afterm.sourced)
result_mh = pfs_mh.changefreq("MS")

# C) As-should 'quarters': additional hedging of the quarter, in may
# volumes as predicted in may and prices as predicted in may
pfs_may_before = lb.PfState(normofftake, mayprices, pfs.sourced)
to_buy = pfs_may_before.hedge_of_unsourced("val", "QS")
pfs_may_afterq = pfs_may_before.add_sourced(to_buy)
# resulting situation in december with actual volumes and actual prices
pfs_qh = lb.PfState(pfs.offtake, pfs.unsourcedprice, pfs_may_afterq.sourced)
result_qh = pfs_qh.changefreq("MS")


# %% what-if norm temperatures

pfs_0h_ifnorm = pfs_0h.set_offtakevolume(normofftake)
result_0h_ifnorm = pfs_0h_ifnorm.changefreq("MS")
pfs_mh_ifnorm = pfs_mh.set_offtakevolume(normofftake)
result_mh_ifnorm = pfs_mh_ifnorm.changefreq("MS")
pfs_qh_ifnorm = pfs_qh.set_offtakevolume(normofftake)
result_qh_ifnorm = pfs_qh_ifnorm.changefreq("MS")

# %% what-if only parallel price change since may

avg_may = mayprices.p.mean()
avg_actual = pfs.unsourcedprice.p.mean()
prices_parallel = mayprices * float(avg_actual / avg_may)

pfs_0h_ifparallel = pfs_0h.set_unsourcedprice(prices_parallel)
result_0h_ifparallel = pfs_0h_ifparallel.changefreq("MS")
pfs_mh_ifparallel = pfs_mh.set_unsourcedprice(prices_parallel)
result_mh_ifparallel = pfs_mh_ifparallel.changefreq("MS")
pfs_qh_ifparallel = pfs_qh.set_unsourcedprice(prices_parallel)
result_qh_ifparallel = pfs_qh_ifparallel.changefreq("MS")


# %% add all to excel file

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter("data1.xlsx", engine="xlsxwriter")
result_0h.to_excel(writer, "0h")
result_mh.to_excel(writer, "mh")
result_qh.to_excel(writer, "qh")
result_0h_ifnorm.to_excel(writer, "0h, if norm")
result_mh_ifnorm.to_excel(writer, "mh, if norm")
result_qh_ifnorm.to_excel(writer, "qh, if norm")
result_0h_ifparallel.to_excel(writer, "0h, if parallel")
result_mh_ifparallel.to_excel(writer, "mh, if parallel")
result_qh_ifparallel.to_excel(writer, "qh, if parallel")
writer.save()
writer.close()
# %%
