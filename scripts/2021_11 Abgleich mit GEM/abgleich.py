
#%%
import lichtblyck as lb
lb.belvis.auth_with_password('Ruud.Wijtvliet', 'Ammm1mmm2mmm3mmm')
pfs = lb.portfolios.pfstate('power', 'B2C_P2H', '2021-11', '2022')
d = pfs.changefreq('D')

#%%
q = pfs.changefreq('QS')
h = pfs.unsourced.hedge('QS', 'val')
h2 = h.volume.po('QS')
#%%