"""
This script visualises the 4-week ahead forecast of the influenza model starting from the most recent NHSN HSN data
"""

__author__      = "T.W. Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group (JHUBSPH) & Bento Lab (Cornell CVM). All Rights Reserved."

import re
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# hierarchSIR functions
from hierarchSIR.utils import make_data_pySODM_compatible

##############
## Settings ##
##############

hyperparameters = 'exclude_None'
model_name = 'SIR-1S'
skip_fips = []
start_calibration_month = 9 
plot_on = 'centroid'

##########################
## Find latest forecast ##
##########################

def extract_timestamp(fname, pattern):
    match = pattern.search(fname.name)
    if match:
        return datetime.strptime(match.group(0)[-10:], "%Y-%m-%d")
    return None

from pathlib import Path
forecast_folder = Path(os.path.join(os.path.dirname(__file__), f'../../data/interim/calibration/forecast/{model_name}/hyperparameters-{hyperparameters}/'))
pattern = re.compile(r"forecast_reference_date-\d{4}-\d{2}-\d{2}")                                     # regex to capture gathered timestamp
files_with_time = [(f, extract_timestamp(f, pattern)) for f in forecast_folder.glob("*.csv")]          # collect files and their timestamps
files_with_time = [(f, t) for f, t in files_with_time if t is not None]
latest_forecast_file, reference_date = max(files_with_time, key=lambda x: x[1])                                           # get the latest file

############################
## Load forecast and data ##
############################

# get the latest data (dummy)
data, _, _, _ = make_data_pySODM_compatible(datetime(2000,1,1), datetime(3000,2,1), 1, preliminary=True)
end_date = max(data[0].index)
# helper function
def get_influenza_season_label(date: datetime) -> str:
    """
    Given a datetime, return the influenza season label in the format 'YYYY-YYYY'.
    Season runs from September 1 to August 31.
    """
    year = date.year
    if date.month >= 9:  # September or later → start of new season
        start_year = year
        end_year = year + 1
    else:  # January–August → still in previous season
        start_year = year - 1
        end_year = year
    return f"{start_year}-{end_year}"
# retrieve latest season
season = get_influenza_season_label(end_date)
# get start date
start_date = datetime(int(season[0:4]), start_calibration_month, 1)

# get the latest forecast (For now, assuming there is only )
forecast = pd.read_csv(latest_forecast_file, index_col=0)
fips_state_list =  forecast['location'].unique().tolist()
fips_state_list = [x for x in fips_state_list if x not in skip_fips]
fips_mappings = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/interim/demography/demography.csv'), dtype={'fips_state': int})
name_state_list = [fips_mappings.loc[fips_mappings['fips_state'] == x]['abbreviation_state'].squeeze() for x in fips_state_list]
forecast["target_end_date"] = pd.to_datetime(forecast["target_end_date"])

# get the shapefiles
gdf = gpd.read_file(os.path.join(os.path.dirname(__file__),f'../../data/raw/geography/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'))
gdf["representative_point"] = gdf.geometry.representative_point()
gdf["centroid"] = gdf.geometry.representative_point()
gdf["STATEFP"] = gdf["STATEFP"].astype(int)

# get the demography to normalize case counts
demography = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/interim/demography/demography.csv'), dtype={'fips_state': int})

###################
## Build the map ##
###################

# base map
fig, ax = plt.subplots(figsize=(24.1*0.75, 13.9*0.75))
gdf.plot(ax=ax, color="black", alpha=0.15, edgecolor="black")

# plot forecasts per state
for name_state, fips_state in zip(name_state_list, fips_state_list):

    # get x and y of the state's location
    gdf_row = gdf[gdf['STATEFP'] == fips_state]
    cx, cy = gdf_row[plot_on].x.values[0], gdf_row[plot_on].y.values[0]
    
    # get state forecast quantiles
    fc = forecast[forecast["location"] == fips_state]

    # normalize to incidence per 100K
    pop = demography.loc[demography['fips_state'] == fips_state, 'population'].values[0]
    fc['value'] = fc['value'] / pop * 10E5

    # get data
    data, _, _, _ = make_data_pySODM_compatible(start_date, end_date+timedelta(days=1), fips_state, preliminary=True)

    # normalize data
    pop = demography.loc[demography['fips_state'] == fips_state, 'population'].values[0]
    data = data[0]*7 / pop * 10E5

    # inset axes
    iax = inset_axes(ax, width=1, height=0.7, loc="center",
                     bbox_to_anchor=(cx, cy),
                     bbox_transform=ax.transData,
                     borderpad=0)
    
    # plot forecast intervals
    iax.fill_between(fc["target_end_date"].unique(), fc.loc[fc['output_type_id'] == 0.25, 'value'], fc.loc[fc['output_type_id'] == 0.75, 'value'], color="green", alpha=0.2)
    iax.fill_between(fc["target_end_date"].unique(), fc.loc[fc['output_type_id'] == 0.025, 'value'], fc.loc[fc['output_type_id'] == 0.975, 'value'], color="green", alpha=0.1)
    iax.scatter(data.index, data.values, color='black', alpha=1, linestyle='None', facecolors='black', s=10, linewidth=1)

    # inside your loop, after plotting into iax
    iax.xaxis.set_major_locator(mdates.MonthLocator())
    iax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b = abbreviated month name
    iax.tick_params(axis='x', labelsize=5, rotation=0)
    iax.tick_params(axis='y', labelsize=5)
    iax.set_xlim([start_date, end_date+timedelta(weeks=5)])
    iax.set_ylim([-10,200])

    # put state in
    iax.text(
    0.05, 0.95,                   # position (x,y) in axes fraction coordinates
    f'{name_state}',                   # text string
    transform=iax.transAxes,      # use axes coordinates (0–1)
    fontsize=5,
    va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=0.5)
    )

ax.set_xlim([-130, -65])
ax.set_ylim([23, 53])
ax.set_axis_off()

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), f'../../data/interim/calibration/forecast/{model_name}/hyperparameters-{hyperparameters}/forecast_reference_date-{reference_date.strftime("%Y-%m-%d")}.png'), dpi=400)
#plt.show()
plt.close()