# -*- coding: utf-8 -*-

from proof_of_concepts.classdef import C2

instA = C2(55)
instB = instA.added_to_val(22)
instB.val

#%%


import pandas as pd
import numpy as np

import lichtblyck as lb


#%%


import pandas as pd
import numpy as np
from lichtblyck import Portfolio, PfFrame, PfSeries


def get_index(tz="Europe/Berlin", freq="D"):
    count = {"M": 10, "D": 100, "H": 1000, "15T": 1000}[freq]
    periods = np.random.randint(count, count * 10)
    shift = np.random.randint(0, 3)
    i = pd.date_range("2020-01-01", freq=freq, periods=periods, tz=tz)
    return i + shift * i.freq


def get_pfframe(i, columns="wp"):
    return PfFrame(np.random.rand(len(i), len(columns)), i, list(columns))


i = get_index()
u = get_pfframe(i)
x = Portfolio("B2B", get_pfframe(i))
y = Portfolio("Sim", parent=x)
z = Portfolio("Uns", get_pfframe(i), x)
a = Portfolio("SimA", get_pfframe(i), y)
b = Portfolio("SimB", get_pfframe(i), y)

#%% Debugging / Proof of principle.


import pandas as pd
import numpy as np

# Possibility to access Series with .p, .q, .r:


class PriQuaRev1(pd.DataFrame):
    @property
    def r(self):
        try:
            return self["r"]
        except KeyError:
            return (self["q"] * self["p"]).rename("r")

    @property
    def q(self):
        try:
            return self["q"]
        except KeyError:
            return (self["r"] / self["p"]).rename("q")

    @property
    def p(self):
        try:
            return self["p"]
        except KeyError:
            return (self["r"] / self["q"]).rename("p")


pf1 = PriQuaRev1(np.random.rand(10, 2), columns=["q", "p"])
# Standard
pf1.p, pf1.q, pf1["p"], pf1["q"]
pf1[["p", "q"]]
# New
pf1.r  # working
pf1["r"]  # not working
pf1[["p", "r"]]  # not working


# Possibility to access Series also with ['p'], ['q'], ['r']:


class PriQuaRev2(pd.DataFrame):
    @property
    def r(self):
        try:
            return super().__getitem__("r")
        except KeyError:
            return (self["q"] * self["p"]).rename("r")

    @property
    def q(self):
        try:
            return super().__getitem__("q")
        except KeyError:
            return (self["r"] / self["p"]).rename("q")

    @property
    def p(self):
        try:
            return super().__getitem__("p")
        except KeyError:
            return (self["r"] / self["q"]).rename("p")

    def __getitem__(self, name):
        print("getitem " + str(name))
        if name in list("pqr"):
            return self.__getattribute__(name)
        return super().__getitem__(name)


pf2 = PriQuaRev2(np.random.rand(10, 2), columns=["q", "p"])
# Standard
pf2.p, pf2.q, pf2["p"], pf2["q"]
pf2[["p", "q"]]
# New
pf2.r  # working
pf2["r"]  # now also working
pf2[["p", "r"]]  # still not working


#%%

import lichtblyck as lb
from lichtblyck import SinglePf, MultiPf, LbPf
from lichtblyck.core.dev import (
    get_index,
    get_dataframe,
    get_singlepf,
    get_multipf_standardcase,
    get_multipf_allcases,
    get_lbpf_nosubs,
    get_lbpf_subs_standardcase,
    get_lbpf_subs_allcases,
    OK_FREQ,
    OK_COL_COMBOS,
)

i = get_index("Europe/Berlin", "D")
offtake = get_singlepf(i)
sourced = get_multipf_allcases(i)

lbpf = LbPf(offtake=offtake, sourced=sourced, name="LUD")


a = get_singlepf(i, columns="wp")
b = get_singlepf(i, columns="wp")
c = get_singlepf(i, columns="w")

u1 = a + b
u2 = a + c
u3 = a - b
u4 = a - c
# %% Sample usage

lud = lb.read_belvis("LUD")  # LbPf with children


lud = lb.belvis.loadpf("LUD")
heatold = lb.belvis.loadpf("heat_old")

rh = (lud.rh + heatold.rh).rename("rh")
hp = (lud.hp + heatold.hp).rename("hp")

offtake = rh.Offtake

#%%

import pandas as pd
import numpy as np

a = pd.Series([1, 2, 3, 4, 5, 6, 7]).values
b = pd.Series([7, 5, 4, 8, 12, 22, -1]).values
pd.Series(a, b).plot()
# %%

import pandas as pd
import numpy as np

df = pd.DataFrame({"a": np.arange(5, 10), "b": np.arange(2, 12, 2)})
df = df[df.sum(axis=1) < 16]
for col, s in df.items():
    replace = s < 6
    df.loc[replace, col] = 99


# %%

import pandas as pd
import numpy as np


def replacewith99ifatmost1na(s):
    s = s[s > 2]
    replace = s <= 6
    s.loc[replace] = 99
    return s


s = pd.Series(np.arange(0, 12, 2))
s = replacewith99ifatmost1na(s)


# %%
def replacewith99ifatmost1na(df):
    df = df[df.sum(axis=1) < 16]
    for col, s in df.items():
        replace = s < 6
        df.loc[replace, col] = 99
    return df


df = pd.DataFrame({"a": np.arange(4, 10), "b": np.arange(0, 12, 2)})
df = replacewith99ifatmost1na(df)


#%%

import pandas as pd

tz = "Europe/Berlin"
stamps = [
    pd.Timestamp("2020-01-01 00:00", tz=tz),
    pd.Timestamp("2020-01-01 03:00", tz=tz),
    pd.Timestamp("2020-01-27 18:00", tz=tz),
]


# %%

