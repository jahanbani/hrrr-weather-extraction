"""For this code to work you need to change the core.py of Herbie
The file names must be changed. search for localfilename
However, if we don't change this it seems that there are
two files that are being downloaded, why?
It is located in /local/alij/anaconda3/envs/herbie/lib/python3.11/site-packages/herbie
"""

import os

# I need to calculate the solar output here
import pandas as pd

import utils
from prereise.gather.const import (
    DATADIR,
    END,
    SEARCHSTRING,
    SELECTORS,
    START,
    TZ,
    YEAR,
)
from prereise.gather.winddata.hrrr.calculations import (
    extract_data_parallel,
    extract_data_xarray,
)

from myutils.FILECONSTANTS import load_yaml


# download section
DOWNLOAD = False
if DOWNLOAD:
    utils.download_data(START, END, DATADIR, SEARCHSTRING)


scendict = load_yaml("/home/alij/psse/InputData/RENCAP/WINDSOLARSCEN.yaml")

WINDSCEN = scendict["WINDSCEN"]
SOLARSCEN = scendict["SOLARSCEN"]
WINDSCEN = "special"
SOLARSCEN = "special"

print("*" * 50)
print(
    f"The year is {YEAR} and the wind scenario is {WINDSCEN} and the solar scenario is {SOLARSCEN}"
)
print("*" * 50)


# POINTSFN = "../psse/InputData/In_Iowa_Wind_Turbines.xlsx"
POINTSFNS = {
    "wind": [
        "wind.csv",
    ],
    "solar": [f"solar.csv"],
}
# mapping: the mapping of the dropped duplicate rows
points, wind_farms, solar_plants, mapping = utils.get_points(POINTSFNS)

READGRIB = True
CALCWIND = True
CALCSOL = True

DEFAULT_HOURS_FORECASTED = ["0", "1"]
if READGRIB:
    # Use the new xarray-based extraction function
    data = extract_data_parallel(
    points,
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    WINDSCEN,
    SOLARSCEN
    )
    

FILENAMEENDING = f"_{WINDSCEN}_{SOLARSCEN}_{str(START.year)}_{str(START.month)}_to_{str(END.year)}_{str(END.month)}.parquet"
folder_path = "./"
fns = [
    file.split(".")[0]
    for file in os.listdir(folder_path)
    if (file.endswith(FILENAMEENDING))
]


def read_and_process_data(filename, mapping):
    df = pd.read_parquet(filename + ".parquet")
    dupcols = pd.DataFrame({vv: df[k] for k, v in mapping.items() for vv in v})
    return pd.concat([df, dupcols], axis=1)


# Initialize an empty dictionary to store the data
data = {}

# Read and process the data locally
for fn in fns:
    if CALCWIND and "Wind80" in fn:
        print(f"Reading {fn}")
        data[fn] = read_and_process_data(fn, mapping)
    if CALCSOL and any(x in fn for x in ["rad", "vbd", "vdd", "2tmp", "Wind10"]):
        print(f"Reading {fn}")
        data[fn] = read_and_process_data(fn, mapping)

print("Read the data successfully")

if CALCWIND:
    # find the wind related files; there is Wind80 in their names
    windfns = [fn for fn in fns if fn.startswith("Wind80")]
    # convert the wind speed dat to wind_farms
    wind_output_power = utils.calculate_wind_pout(
        data[windfns[0]][
            list(set(data[windfns[0]].columns).intersection(set(wind_farms["pid"])))
        ],
        wind_farms,
        START,
        END,
        TZ,
        WINDSCEN,
    ).round(2)

if CALCSOL:
    # find the solar related files; there is Wind80 in their names
    solw = [fn for fn in fns if fn.startswith("Wind10")]
    sol2tmp = [fn for fn in fns if fn.startswith("2tmp")]
    solvdd = [fn for fn in fns if fn.startswith("vdd")]
    solvbd = [fn for fn in fns if fn.startswith("vbd")]
    # solar part
    solar_output_power = utils.prepare_calculate_solar_power(
        data[solw[0]][
            list(set(data[solw[0]].columns).intersection(set(solar_plants["pid"])))
        ],
        data[sol2tmp[0]][
            list(set(data[sol2tmp[0]].columns).intersection(set(solar_plants["pid"])))
        ],
        data[solvdd[0]][
            list(set(data[solvdd[0]].columns).intersection(set(solar_plants["pid"])))
        ],
        data[solvbd[0]][
            list(set(data[solvbd[0]].columns).intersection(set(solar_plants["pid"])))
        ],
        solar_plants,
        YEAR,
        TZ,
        SOLARSCEN,
    )


__import__("ipdb").set_trace()
