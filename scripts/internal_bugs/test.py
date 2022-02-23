
from requests.exceptions import ConnectionError
from typing import Tuple, Dict, List, Union, Iterable
from urllib import parse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime as dt
import jwt
import time
import json
import requests

session = requests.Session()

def request(path: str, *queryparts: str):
    string = f"http://lbbelvis01:8040{path}"
    if queryparts:
        queryparts = [parse.quote(qp, safe=":=") for qp in queryparts]
        string += "?" + "&".join(queryparts)
    response = session.get(string)
    if response.status_code == 200:
        print(json.loads(response.text))
    else:
        print('error')


request("/rest/session", f"usr=Ruud.Wijtvliet", f"pwd=Ammm1mmm2mmm3mmm", f"tenant=PFMSTROM")
request("/rest/energy/belvis/PFMSTROM/timeSeries/22592444") 
request("/rest/energy/belvis/PFMSTROM/timeSeries/42818384") 

#%%

class A:
    def __init__(self, val:int):
        self.val = val

    def __add__(self, other):
        print(other)
        return A(self.val + other.val)

aas = [A(v) for v in (3,8,9,-3)]

sum(aas)
# %%
