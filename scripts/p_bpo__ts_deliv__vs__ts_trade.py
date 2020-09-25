# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:36:35 2020

@author: ruud.wijtvliet
"""
import pandas as pd
import lichtblyck as lb


writer = pd.ExcelWriter('preise.xlsx')
for prod in ['q', 'm']:
    f = lb.prices.futures(prod)[['p_base', 'p_peak', 'p_offpeak']]
    f.index = f.index.droplevel(0)
    f = f.unstack(0).swaplevel(axis=1).sort_index(axis=1)
    f.tz_localize(None).tz_localize(None, axis=1, level=0).to_excel(writer, prod)
writer.close()

