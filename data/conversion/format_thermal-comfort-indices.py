import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import UnivariateSpline


# merge all raw data & average across NC bounding box
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# load all files and concatenate them in a single dataset
directory = "../raw/thermal_comfort_indices"
ds = xr.open_mfdataset(f"{directory}/*.nc", combine="by_coords")

# Average over all lon/lat
ds = ds.mean(dim=["lon", "lat"])['utci']

# Save a copy of the unified dataset
ds.to_netcdf("../interim/thermal_comfort_indices/utci_mean_NC.nc")

# smooth data using splines
# >>>>>>>>>>>>>>>>>>>>>>>>>

smoothness = 75000

# resample to daily maximum
max = ds.resample(time="1D").max() - 273.15
# convert to a pandas dataframe
max = max.to_pandas()

# smooth maximum (using splines)
# Example input series (your UTCI series)
x = np.arange(len(max))
y = max.values
# Fit smoothing spline
# s parameter controls smoothing: larger = smoother
spline = UnivariateSpline(x, y, s=smoothness)  
# Predict smoothed values
y_smooth = spline(x)
max_smooth = pd.Series(index=max.index, data=spline(x), name='utci')
max_smooth.index.name = 'date'


# attach the seasonal trend
# >>>>>>>>>>>>>>>>>>>>>>>>>

max_smooth = max_smooth.to_frame(name="utci")

# Extract month-day string for grouping
max_smooth["month_day"] = max_smooth.index.strftime("%m-%d")

# Compute climatology (mean across years for each month-day)
climatology = max_smooth.groupby("month_day")["utci"].mean()

# Map it back to full daily dataframe
max_smooth["seasonal_mean"] = max_smooth["month_day"].map(climatology)

# Drop helper column if you don’t want it
max_smooth = max_smooth.drop(columns="month_day")

# Save the smoothed result
max_smooth.to_csv("../interim/thermal_comfort_indices/utci_mean_NC.csv", index=True)