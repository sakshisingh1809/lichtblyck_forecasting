"""
Module to find correct file to given climate zone.
"""
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import numpy as np
from geopy.distance import great_circle
 
CLIMATEZONEFILE = 'lichtblick/temperatures/sourcedata/climate_zones.csv'
HISTORICDATAFOLDER = 'lichtblick/temperatures/sourcedata/historic/'
FUTUREDATAFOLDER = 'lichtblick/temperatures/sourcedata/future/'

def info(climate_zone:int, info:str='id'):
    """Return information about specified climate zone, based on 'info'. If 
    info == 'id', return weather station id. If info == 'latlon', return tuple
    with (lat, lon) in degrees."""
    df = pd.read_csv(CLIMATEZONEFILE, sep=';')
    df = df.set_index(df.columns[0])
    if climate_zone not in df.index:
        raise ValueError("Value for argument 'climate_zone' must be one of " + ", ".join(df.index.values))
    if info.lower() == 'id':
        return df.loc[climate_zone, 'Stations_ID']
    if info.lower() == 'latlon':
        return (df.loc[climate_zone, 'Breite'], df.loc[climate_zone, 'Laenge'])
    raise ValueError("Value for argument 'info' must be one of {'cz', 'latlon'}.")


def historicdata(climate_zone:int) -> bytes:
    """Return bytes object, i.e., file contents of historic climate data for
    specified climate zone."""
    # Find the correct station id...
    sid = info(climate_zone, 'id')
    # ... then, find the zip archive corresponding to that station...
    archives = (entry for entry in Path(HISTORICDATAFOLDER).iterdir() 
                if entry.is_file() and entry.suffix == '.zip')
    for archive in archives:
        if f'KL_{sid:05}' in archive.name:
            break
    else:
        raise FileNotFoundError(f"Could not find climate date for station with id {sid}.")
    # ... then extract the correct file from that zip archive...
    with ZipFile(archive) as zf:
        for filename in zf.namelist():
            if 'produkt_klima_tag' in filename:
                break
        else:
            raise FileNotFoundError(f"Could not find the correct file in the archive for station with id {sid}.")
        bytes_data = zf.read(filename)
    # ... and return its content.
    return bytes_data


def futuredata(climate_zone:int) -> Path:
    """Return reference to file of future climate data for
    specified climate zone."""
    # Find the coordinates of the station...
    sought_loc = info(climate_zone, 'latlon')
    # ...then, find the location with future data that's closest to it...
    folders = [entry for entry in Path(FUTUREDATAFOLDER).iterdir() 
               if entry.is_dir()]
    found_locs = [np.array([n[4:10], n[10:]], float) / 10000 
              for n in (folder.name for folder in folders)]
    dists = np.array([great_circle(found_loc, sought_loc).km for found_loc in found_locs])
    idx = dists.argmin()
    if (mindist:=dists[idx]) > 10:
        raise ValueError(f'The nearest station is far away: {mindist:.0f} km.')
    # ...and find the corresponding file.
    folder = folders[idx]   
    files = (entry for entry in folder.iterdir() if entry.is_file())
    for file in files:
        if 'TRY2045' in file.name and '_Jahr' in file.name:
            break
    else:
        raise FileNotFoundError(f"Can't find a file named 'TRY2045..._Jahr...' in {folder.name}.")
    return file
        