"""
Create overview of the portfolios. 
First step: overview of current situation. 
"""
#%%
import pickle
from pathlib import Path

try:
    with open(Path(__file__).parent / 'pfs.pkl', 'rb') as f:
        pfs = pickle.load(f)
except NameError:
    with open('pfs.pkl', 'rb') as f:
        pfs = pickle.load(f)
#%%
print(pfs)