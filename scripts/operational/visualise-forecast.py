"""
This script visualises the 4-week ahead forecast of the influenza model starting from the most recent NHSN HSN data
"""

__author__      = "T.W. Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group (JHUBSPH) & Bento Lab (Cornell CVM). All Rights Reserved."

import sys,os
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# hierarchSIR functions
from hierarchSIR.utils import make_data_pySODM_compatible

reference_date = '2025-02-01' #TODO: find latest algorithmically
model_name = 'SIR-1S'
hyperparameters = 'exclude_None'
skip_fips = []
start_calibration_month = 9 
plot_on = 'centroid'

############################
## Load forecast and data ##
############################

# get the latest data (dummy)
data, _, _, _ = make_data_pySODM_compatible(datetime(2000,1,1), datetime(2025,2,1), 1)
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
forecast = pd.read_csv(os.path.join(os.path.dirname(__file__), f'../../data/interim/calibration/forecast/{model_name}/hyperparameters-{hyperparameters}/reference_date-{reference_date}/forecast_reference_date-{reference_date}.csv'))
fips_state_list =  forecast['location'].unique().tolist()
fips_mappings = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/interim/demography/demography.csv'), dtype={'fips_state': int})
name_state_list = [fips_mappings.loc[fips_mappings['fips_state'] == x]['abbreviation_state'].squeeze() for x in fips_state_list]
forecast["target_end_date"] = pd.to_datetime(forecast["target_end_date"])

# get the shapefiles
gdf = gpd.read_file(os.path.join(os.path.dirname(__file__),f'../../data/raw/geography/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'))
gdf["representative_point"] = gdf.geometry.representative_point()
gdf["centroid"] = gdf.geometry.representative_point()
gdf["STATEFP"] = gdf["STATEFP"].astype(int)

###################
## Build the map ##
###################

# base map
fig, ax = plt.subplots(figsize=(11.7, 8.3))
gdf.plot(ax=ax, color="whitesmoke", edgecolor="gray")

# plot forecasts per state
for name_state, fips_state in zip(name_state_list, fips_state_list):

    # get x and y of the state's location
    gdf_row = gdf[gdf['STATEFP'] == fips_state]
    cx, cy = gdf_row[plot_on].x.values[0], gdf_row[plot_on].y.values[0]
    
    # get state forecast quantiles
    fc = forecast[forecast["location"] == fips_state]
    # Compute quantiles
    quantiles = (
        fc
        .groupby(["target_end_date"])["value"]
        .quantile([0.025, 0.25, 0.5, 0.75, 0.975])          # 25%, median, 75%
        .unstack()                                          # pivot out quantile levels into columns
        .reset_index()
    )

    # get data
    data, _, _, _ = make_data_pySODM_compatible(start_date, end_date, fips_state)

    # inset axes
    iax = inset_axes(ax, width=1.3, height=0.7, loc="center",
                     bbox_to_anchor=(cx, cy),
                     bbox_transform=ax.transData,
                     borderpad=0)
    
    # plot forecast intervals
    iax.fill_between(quantiles["target_end_date"], quantiles[0.25], quantiles[0.75], color="green", alpha=0.2)
    iax.fill_between(quantiles["target_end_date"], quantiles[0.025], quantiles[0.975], color="green", alpha=0.1)
    iax.scatter(data[0].index, data[0].values*7, color='black', alpha=1, linestyle='None', facecolors='black', s=10, linewidth=1)

    # inside your loop, after plotting into iax
    iax.xaxis.set_major_locator(mdates.MonthLocator())
    iax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b = abbreviated month name
    iax.tick_params(axis='x', labelsize=5, rotation=0)
    iax.tick_params(axis='y', labelsize=5)
    iax.set_xlim([start_date, end_date+timedelta(weeks=5)])

    # put state in
    iax.text(
    0.05, 0.95,                   # position (x,y) in axes fraction coordinates
    f'{name_state}',                   # text string
    transform=iax.transAxes,      # use axes coordinates (0–1)
    fontsize=5,
    va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=0.5)
    )

ax.set_xlim([-180, -67])

plt.show()
plt.close()