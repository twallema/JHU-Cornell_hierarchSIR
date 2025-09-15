import xarray as xr

# load all files and concatenate them in a single dataset
directory = "../raw/thermal_comfort_indices"
ds = xr.open_mfdataset(f"{directory}/*.nc", combine="by_coords")

# Average over all lon/lat
ds = ds.mean(dim=["lon", "lat"])['utci']

# Save to a new NetCDF file (single consolidated dataset)
ds.to_netcdf("../interim/thermal_comfort_indices/utci_mean_NC.nc")