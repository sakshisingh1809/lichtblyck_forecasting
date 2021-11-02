"""
Create overview of the portfolios. 
First step: overview of current situation. 
"""
#%%
import lichtblyck as lb
import time 

lb.belvis.auth_with_password('Ruud.Wijtvliet', 'Ammm1mmm2mmm3mmm')

pfs = lb.PfState.from_belvis('power', 'LUD', '2022', '2025')

#%%
start = time.perf_counter()
for i in range(3):
    print(pfs)
print(f'That took {time.perf_counter()-start:.2f} seconds.')