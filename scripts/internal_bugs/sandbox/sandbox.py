#%%
import lichtblyck as lb
import cProfile

lb.belvis.auth_with_passwordfile("cred.txt")
# %%

pfs = lb.portfolios.pfstate("power", "PKG", "2022", "2022-02")

# %%


#%%


pr = cProfile.Profile()
pr.enable()
# code to execute
pfs.print()
pr.disable()

# %%
pr.print_stats(sort="cumtime")
pr.print_stats(sort="tottime")
# other sorting methods here:
# https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats

#%%
result = cProfile.run("pfs.print()")
result2 = cProfile.run("pfs.print()", sort=1)
