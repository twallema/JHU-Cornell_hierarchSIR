import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from hierarchSIR.utils import get_NC_influenza_data

smoothness = 75000
delay = 14

# Load & smooth data
# >>>>>>>>>>>>>>>>>>

# load dataset
ds = xr.open_dataset("../../data/interim/thermal_comfort_indices/utci_mean_NC.nc")

# resample to daily maximum
max = ds.resample(time="1D").max() - 273.15

# convert to a pandas dataframe
max_smooth = max["utci"].to_pandas()

# smooth maximum (using splines)
# Example input series (your UTCI series)
x = np.arange(len(max_smooth))
y = max_smooth.values
# Fit smoothing spline
# s parameter controls smoothing: larger = smoother
spline = UnivariateSpline(x, y, s=smoothness)  
# Predict smoothed values
y_smooth = spline(x)
max_smooth= pd.Series(index=max_smooth.index, data=spline(x), name='utci')

# smooth maximum (using ewm)
# max_smooth = max_smooth.ewm(span=62).mean()
max_smooth = max_smooth.shift(delay)

# back to xarray
max_smooth = xr.DataArray(max_smooth, coords={"time": max.time}, dims="time")



# Convert to a labeled seasonal dataset
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# make a seasonal plot of the daily maximum
utci = max_smooth #["utci"]
# Convert to pandas for easier datetime handling
time_index = utci["time"].to_index()
# Define season year: if month >= 10, season starts that year, else previous year
season_year = time_index.year.where(time_index.month >= 10, time_index.year - 1)
# Build a season label, e.g. "2014-2015"
season_label = season_year.astype(str) + "-" + (season_year + 1).astype(str)
# Compute "day of season" (days since October 1 of that season)
season_start = pd.to_datetime(season_year.astype(str) + "-10-01")
day_of_season = (time_index - season_start).days
# Make a new dataframe
df = pd.DataFrame({
    "utci": utci.values,
    "season": season_label,
    "day": day_of_season
})
# Average over all seasons
q1 = df.groupby("day")["utci"].quantile(0.025)
q2 = df.groupby("day")["utci"].median()
q3 = df.groupby("day")["utci"].quantile(0.975)
# Throw out seasons 2013-2014, 2020-2021, 2021-2022, 2022-2023
df = df[~df['season'].isin(['2013-2014', '2020-2021', '2021-2022', '2022-2023'])]


# Compute a double-sigmoid modifier
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Example parameters (to calibrate)
T_low = 14   # lower comfort threshold
T_high = 28  # upper comfort threshold
k1 = 0.15    # steepness (cold side)
k2 = 0.15     # steepness (hot side)

def double_sigmoid(T, T_low, T_high, k1, k2):
    cold_term = 1 / (1 + np.exp(-k1 * (T - T_low)))
    hot_term  = 1 / (1 + np.exp(k2 * (T - T_high)))
    return np.exp(- cold_term * hot_term)

# Visualise modifier
T = np.linspace(-10,50,100)
fig,ax=plt.subplots()
ax.plot(T, double_sigmoid(T, T_low, T_high, k1, k2), color='black')
ax.set_xlabel('UTCI (degrees C)')
ax.set_ylabel('Behavioral model')
plt.savefig(f'../../data/interim/thermal_comfort_indices/figs/modifier.pdf')
#plt.show()
plt.close()

# Assuming your dataframe is df with column "utci"
df["L"] = double_sigmoid(df["utci"], T_low, T_high, k1, k2)

# Normalize to make a multiplicative modifier centered around 1
df["modifier"] = df["L"] / df["L"].mean()

# Visualise incidence and utci on one plot
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Approximate day-of-season for month starts
month_starts = [0, 31, 61, 92, 123, 151, 182, 212, 243]
month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]

# Plot each season
for focus_season in df['season'].unique().tolist():

    # init figure
    fig,ax = plt.subplots(nrows=3, ncols=1, figsize=(8.3,11.7/2), sharex=True)
    # split climate data in focal/non-focal
    focus = df[df['season'] == focus_season]
    not_focus = df[df['season'] != focus_season]
    # plot all other seasons weather data finely in background
    for season, group in not_focus.groupby("season"):
        ax[0].plot(group["day"], group["utci"], linewidth=1, alpha=0.2, color='black')
    # plot focal season
    ax[0].plot(focus['day'], focus['utci'], color='black', linewidth=2.5, label=focus_season)
    # plot trend
    ax[0].plot(q2, color='red', label='median (2014-2025)', linewidth=2.5)
    ax[0].fill_between(q1.index, q1, q3, color='red', label='95% QR (2014-2025)', alpha=0.1)
    # axes decorators
    ax[0].set_xlim([0,244])
    ax[0].set_ylabel("UTCI (degrees C)")
    ax[0].set_title(f"Season: {focus_season}")
    ax[0].legend(fontsize=8)
    # visualise focal modifier
    ax[1].plot(focus['day'], focus['modifier'], color='black', linewidth=2.5)
    ax[1].set_ylabel('Modifier $\\Delta \\beta_t$')
    # visualise other modifiers
    for season, group in not_focus.groupby("season"):
        ax[1].plot(group['day'], group['modifier'], color='black', linewidth=1, alpha=0.2)
    # plot incidence of focal season
    flu_df = 7*get_NC_influenza_data(datetime(int(focus_season[0:4]), 10, 1), datetime(int(focus_season[0:4])+1, 6, 1), season)
    flu_df['day'] = list((flu_df.index - pd.to_datetime(focus_season[0:4] + "-10-01")).days)
    ax[2].plot(flu_df['day'], flu_df['H_inc'], marker='o', markersize=5, color='black')
    # visualise all other seasons
    for season, group in not_focus.groupby("season"):
        flu_df = 7*get_NC_influenza_data(datetime(int(season[0:4]), 10, 1), datetime(int(season[0:4])+1, 6, 1), season)
        flu_df['day'] = list((flu_df.index - pd.to_datetime(season[0:4] + "-10-01")).days)
        ax[2].plot(flu_df["day"], flu_df["H_inc"], linewidth=1, alpha=0.2, color='black')
    # axes decorators
    ax[2].set_xlim([0,244])
    ax[2].set_xlabel("Day of influenza season (Oct–Jun)")
    ax[2].set_ylabel("Hospital incidence (-)")
    
    plt.xticks(month_starts, month_labels)  # custom ticks
    plt.tight_layout()
    plt.savefig(f'../../data/interim/thermal_comfort_indices/figs/{focus_season}.pdf')
    #plt.show()
    plt.close()