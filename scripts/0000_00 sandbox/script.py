import lichtblyck as lb
import pandas as pd
import cProfile

pr = cProfile.Profile()

#%%
x = lb.PfLine(s)

#%%

pr.enable()
x.plot("q")
pr.disable()

#%%

pr.print_stats(sort="cumtime")
# pr.print_stats(sort="tottime")

#%%
