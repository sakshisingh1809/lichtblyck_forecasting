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
