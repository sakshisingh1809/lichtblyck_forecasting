"""
Module handle climate zones and return correct (future or historic) data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from zipfile import ZipFile
from geopy.distance import great_circle
from typing import Union, Any, Callable


CLIMATEZONEFILE = Path(__file__).parent / "climate_zones.csv"
HISTORICDATAFOLDER = Path(__file__).parent / "historic"
FUTUREFOURIERFILE = Path(__file__).parent / "future" / "fouriercoefficients.xlsx"


def info(climate_zone: int, info: str = "name") -> Union[pd.Series, Any]:
    """Return information about specified climate zone, based on 'info'. If
    info == 'id', return weather station id. If info == 'latlon', return tuple
    with (lat, lon) in degrees, etc."""
    df = pd.read_csv(CLIMATEZONEFILE, sep=";")
    df = df.set_index(df.columns[0])
    if climate_zone not in df.index:
        raise ValueError(
            "Value for argument 'climate_zone' must be one of "
            + ", ".join(df.index.values)
        )
    if info.lower() == "id":
        return df.loc[climate_zone, "Stations_ID"]
    if info.lower() == "latlon":
        return (df.loc[climate_zone, "Breite"], df.loc[climate_zone, "Laenge"])
    if info.lower() == "name":
        return df.loc[climate_zone, "Name"]
    raise ValueError(
        "Value for argument 'info' must be one of {'id', 'latlon', 'name'}."
    )


def historicdata(climate_zone: Union[int, Path]) -> bytes:
    """Return bytes object, i.e., file contents of historic climate data for
    specified climate zone (if int) or from specified file (if path)."""
    if isinstance(climate_zone, int):
        # Find the correct station id...
        sid = info(climate_zone, "id")
        # ... then, find the zip archive corresponding to that station...
        archives = (
            entry
            for entry in Path(HISTORICDATAFOLDER).iterdir()
            if entry.is_file() and entry.suffix == ".zip"
        )
        for archive in archives:
            if f"KL_{sid:05}" in archive.name:
                break
        else:
            raise FileNotFoundError(
                f"Could not find climate date for station with id {sid}."
            )
    else:
        archive = climate_zone
    # ... then extract the correct file from that zip archive...
    with ZipFile(archive) as zf:
        for filename in zf.namelist():
            if "produkt_klima_tag" in filename:
                break
        else:
            raise FileNotFoundError(
                f"Could not find the correct file in the archive for station with id {sid}."
            )
        bytes_data = zf.read(filename)
    # ... and return its content.
    return bytes_data


def futurefourierdata(climate_zone: int) -> pd.Series:
    """Return Fourier coefficients to calculate future temperatures for specified climate zone."""
    # Open file with all the coefficients...
    df = pd.read_excel(FUTUREFOURIERFILE, "values", index_col=0)
    # ...and return the coefficients for this climate zone.
    return df[f"t_{climate_zone}"]


def forallzones(function: Callable[[int], pd.Series]) -> pd.DataFrame:
    """Execute 'function' for each climate zone and return in single dataframe."""
    series = []
    for cz in range(1, 16):
        s = function(cz)
        series.append(s.rename(s.name + f"_{cz}"))
    return pd.concat(series, axis=1)
