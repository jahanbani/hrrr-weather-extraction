from herbie.fast import FastHerbie
import pandas as pd

# from config_unified import
from config_unified import HRRRConfig

config = HRRRConfig()


def download_data(START, END, DATADIR, SEARCHSTRING):
    # Create a range of dates
    FHDATES = pd.date_range(
        start=START,
        end=END,
        freq="1h",
    )
    print("the search string is")

    print(SEARCHSTRING)
    print(f"start date is {START} and end date is {END}")

    print("Create a range of forecast lead times")
    fxx = [0, 1]
    FH = FastHerbie(
        FHDATES,
        model="hrrr",
        fxx=fxx,
        product="subh",
        max_threads=100,
        save_dir=DATADIR,
    )
    print("downloading")
    FH.download(
        search=SEARCHSTRING,
        save_dir=DATADIR,
        max_threads=200,
        verbose=True,
    )


download_data(
    config.download_start_date,
    config.download_end_date,
    config.DOWNLOADDATADIR,
    config.SEARCHSTRING,
)
