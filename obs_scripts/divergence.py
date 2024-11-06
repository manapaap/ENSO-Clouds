# -*- coding: utf-8 -*-
"""
Compare low-level divergence with SST and column heating from CERES

if the CERES quantity tracks poorly we can use optical depths

if that doesnt work we need retrievals at different pressure levels

The era5 grids also must be rescaled to the ceres grid (lower res)
"""


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import pandas as pd
import cartopy.crs as ccrs

os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
# We need to do the same thing as with the CERES data where annual
# composites are generated and used to construct El Nino/La Nina anomaly fields
from obs_scripts.vis_clouds import load_oni_idx, is_enso_oni, progress_bar
from obs_scripts.atmos_data import plot_winds
from obs_scripts.ceres_comps import plot_radiation
from CZ_model.standard_funcs import interp_grid


cz_domain = [-30, 30, 120, -80]


def mean_year(xr_array):
    """
    Computes the mean climatological year by averaging each month's data 
    over all years in a memory-efficient way.
    """
    # Extract the unique years
    years = np.unique(xr_array.time.dt.year.values)
    total_years = len(years)
    
    # Start with the first 12 months as the initial array for accumulation
    climatology = xr_array.isel(time=slice(0, 12)).copy()

    # Sum each 12-month slice (each year) incrementally to avoid memory overload
    for n in range(12, len(xr_array.time) - 9, 12):
        # Sibtract 9 to prefent computation of the incomplete year (2024)
        # progress_bar(n, len(xr_array.time), 'Calcuting Climatlogy')
        climatology = iterated_sum(climatology, xr_array.isel(time=slice(n, n + 12)))

    # Divide by the total number of years to get the mean climatological year
    return climatology / total_years


def iterated_sum(agg_array, element):
    """
    Incrementally adds each variable in 'element' to 'agg_array' in-place.
    """
    for var in agg_array.data_vars:
        agg_array[var].values += element[var].values
    return agg_array


def reorder_year_dim(ds, mode="May"):
    """
    Assigns coordinate to xarray year starting in May and ending in April
    of the next year. Can also be used for regular month assignment
    """
    # Extract month from the original time axis
    ds = ds.assign_coords(month=("time", ds['time.month'].data))
    # Reindex the months so the order is May (5) to April (4)
    if mode=="May":
        month_order = [5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]
    else:
        month_order = list(range(1, 13))
    ds = ds.swap_dims({"time": "month"}).reindex(month=month_order).drop_vars('time')
    return ds


def enso_composite(xr_array, nino_idx, print_num=False, small=False):
    """
    Creates three composites- one for Neutral conditions, one for El Nino,
    and one for La Nina. Then subtracts the climatological year from this 
    to get anomaly fields for these three phases. 
    """
    # Faster and simpler calculation if ceres as it works
    if small:
        climatology = xr_array.groupby('time.month').mean(dim='time')
    else:
        climatology = mean_year(xr_array.copy(deep=True))
        climatology = reorder_year_dim(climatology, mode="Jan") # Align Jan-Dec
    years = np.unique(xr_array.time.dt.year.values)
    tot = len(years)
    
    # Initialize composites
    el_nino, la_nina, neutral = None, None, None
    num_el_nino, num_la_nina, num_neutral = 0, 0, 0

    # Loop over years, but align composites from May to April
    for n, yr in enumerate(range(years[0], years[-1])):
        progress_bar(n, tot, f'; ENSO calc, Year: {yr}')
        # Check the ENSO phase in December (yr)
        enso_state = is_enso_oni(nino_idx, f'{yr}.12')

        # Select data from May (yr) to April (yr+1)
        may_to_april = xr_array.sel(time=slice(f'{yr}-04-01', f'{yr+1}-03-30')).copy(deep=True)
        # Overwrite time information that of climatology
        may_to_april = reorder_year_dim(may_to_april, mode='May') # Align May-April
        # Update composites based on ENSO state in December
        if enso_state == 'El Nino':
            if el_nino is None:
                el_nino = may_to_april
            else:
                el_nino = el_nino + may_to_april
            num_el_nino += 1
        elif enso_state == 'La Nina':
            if la_nina is None:
                la_nina = may_to_april
            else:
                la_nina = la_nina + may_to_april
            num_la_nina += 1            
        else:  # Neutral
            if neutral is None:
                neutral = may_to_april
            else:
                neutral = neutral + may_to_april
            num_neutral += 1
            
    # Safeguard against zero divisions
    if el_nino is not None:
        el_nino /= num_el_nino
        el_nino = el_nino.sortby("month") - climatology
        # el_nino.groupby('time.month').mean(dim='time')
    if la_nina is not None:
        la_nina /= num_la_nina
        la_nina = la_nina.sortby("month") - climatology
        # la_nina.groupby('time.month').mean(dim='time')
    if neutral is not None:
        neutral /= num_neutral
        neutral = neutral.sortby("month") - climatology
        # neutral.groupby('time.month').mean(dim='time')
    
    if print_num:
        print(f'\n{num_el_nino} El Nino years, {num_la_nina} La Nina years,' +\
              f' {num_neutral} Neutral years')
    
    # Store this as a dict so it is managable
    comps = {'el_nino': el_nino, 'la_nina': la_nina, 'neutral': neutral}
    
    return comps


def save_composites(comp_pres, comp_sing, comp_rad, directory='era5_reanal/anomalies/'):
    """
    Saves the composite dictionaries (comp_pres, comp_sing, comp_rad) to NetCDF files.
    Each composite (El Nino, La Nina, Neutral) will be saved with the naming convention:
    <pres/sing/rad>_<el_nino/la_nina/neutral>.nc in the specified directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    composite_dict = {'pres': comp_pres, 'sing': comp_sing, 'rad': comp_rad}
    
    for comp_type, comp_dict in composite_dict.items():
        for phase, data in comp_dict.items():
            filename = os.path.join(directory, f'{comp_type}_{phase}.nc')
            data.to_netcdf(filename)
            print(f'Saved {phase} composite for {comp_type} to {filename}')


def load_composites(directory='era5_reanal/anomalies/'):
    """
    Loads the composite dictionaries (comp_pres, comp_sing, comp_rad) from NetCDF files.
    Returns the dictionaries with the same structure as in the script.
    """
    comp_pres, comp_sing, comp_rad = {}, {}, {}
    
    composite_dict = {'pres': comp_pres, 'sing': comp_sing, 'rad': comp_rad}
    phases = ['el_nino', 'la_nina', 'neutral']
    
    for comp_type, comp_dict in composite_dict.items():
        for phase in phases:
            filename = os.path.join(directory, f'{comp_type}_{phase}.nc')
            if os.path.exists(filename):
                comp_dict[phase] = xr.open_dataset(filename)
                print(f'Loaded {phase} composite for {comp_type} from {filename}')
            else:
                print(f'Warning: {filename} does not exist. Skipping load for {comp_type} {phase}.')
                comp_dict[phase] = None
    
    return comp_pres, comp_sing, comp_rad


def crop_era5(xr_array):
    """
    Crops to CZ modeldimentions
    """
    min_lat, max_lat, east_lon, west_lon = cz_domain
    
    ds_east = xr_array.sel(latitude=slice(max_lat, min_lat), 
                           longitude=slice(east_lon, 180))
    ds_west = xr_array.sel(latitude=slice(max_lat, min_lat),
                           longitude=slice(-180, west_lon))
    
    return xr.concat([ds_east, ds_west], dim="longitude")


def plot_scalar(era5, var, title='', lims=cz_domain, cbar='HCC (Frac)'):
    """
    Contour plot of a scalar field (e.g., hcc) across the globe or a specified region.
    """
    # Define central longitude to correctly handle global data
    proj = ccrs.PlateCarree(central_longitude=180)
    
    # Ensure longitude wraps correctly if in 0-360 range
    if era5.longitude.max() > 180:
        era5 = era5.assign_coords(longitude=(((era5.longitude + 180) % 360) - 180))
        era5 = era5.sortby('longitude')
    
    # Extract data for plotting
    data = era5[var]
    lon = era5.longitude.values
    lat = era5.latitude.values
    
    # Print diagnostics to confirm data ranges
    print(f"Data variable '{var}' - min: {data.min().item()}, max: {data.max().item()}")
    print(f"Longitude range: {lon.min()} to {lon.max()}")
    print(f"Latitude range: {lat.min()} to {lat.max()}")

    # Create meshgrid if needed
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # Create plot
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 5))
    ax.set_global()
    ax.set_title(title)
    
    # Use pcolormesh for a continuous plot
    pcm = ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree(), shading='auto', cmap='viridis')
    
    # Add coastlines and gridlines
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.05,
                        shrink=0.65)
    cbar.set_label(f'{var} (Frac)')
    
    if lims is not None:
        ax.set_ylim(lims[0], lims[1])
        ax.set_xlim(lims[3] + 20, lims[2] - 20)
    plt.show()


def main():
    global comp_pres, comp_sing, comp_rad
    directory = 'era5_reanal/anomalies/'    
    # Check if files exist
    files_exist = all(os.path.exists(os.path.join(directory, f'{comp}_{phase}.nc'))
                      for comp in ['pres', 'sing', 'rad'] 
                      for phase in ['el_nino', 'la_nina', 'neutral'])
    
    if files_exist:
        # Load existing composites
        print("Files exist. Loading composites from NetCDF files.")
        comp_pres, comp_sing, comp_rad = load_composites(directory)
    else:
        # Generate composites and save them
        print("Files not found. Generating composites and saving to NetCDF files.")
        era5_sing = xr.open_dataset('era5_reanal/era5_reanal_modern.nc').drop_vars('expver')
        era5_pres = xr.open_dataset('era5_reanal/era5_reanal_pres.nc').drop_vars('expver')
        # Crop to region
        era5_sing = crop_era5(era5_sing)
        era5_pres = crop_era5(era5_pres)
        ceres = xr.load_dataset('misc_data/CERES_radiation.nc')
        
        # Create column data for CERES
        ceres['atms_net_all'] = ceres['toa_net_all_mon'] - ceres['sfc_net_tot_all_mon']
        # nino_idx = load_nino_idx('misc_data/nino_all.csv')
        oni_idx = load_oni_idx('misc_data/oni_index.txt')
        
        # Standardize time format
        ceres['time'] = pd.to_datetime(ceres.time)
        # Create time axis
        era5_pres['time'] = pd.to_datetime(era5_pres.date, format='%Y%m%d')
        era5_sing['time'] = pd.to_datetime(era5_sing.date, format='%Y%m%d')
        # Assign coordinate
        era5_pres = era5_pres.assign_coords(time=("date", era5_pres.time.data))
        era5_sing = era5_sing.assign_coords(time=("date", era5_sing.time.data))
        # Swap coordinate and drop old one
        era5_pres = era5_pres.swap_dims({"date": "time"}).drop_vars('date')
        era5_sing = era5_sing.swap_dims({"date": "time"}).drop_vars('date')
        
        # Generate anomaly fields
        print('Calculating ERA5 Pressure Levels data')
        comp_pres = enso_composite(era5_pres, oni_idx, print_num=True)
        print('Calculating ERA5 Single Levels data')
        comp_sing = enso_composite(era5_sing, oni_idx, print_num=False)
        print('\nCalculating CERES data')
        comp_rad = enso_composite(ceres, oni_idx, print_num=True, small=True)
        
        # Save the generated composites
        save_composites(comp_pres, comp_sing, comp_rad, directory)

    
if __name__ == '__main__':
    main()
