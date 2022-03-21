
#%%
import lichtblyck as lb
lb.belvis.auth_with_password('API-User-FRM', '')
pfs = lb.portfolios.pfstate('power', 'B2C_P2H', '2021-11', '2022')
d = pfs.asfreq('D')

#%%
q = pfs.asfreq('QS')
h = pfs.unsourced.hedge('QS', 'val')
h2 = h.volume.po('QS')
#%%