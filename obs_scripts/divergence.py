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
import calendar
# To make xesmf load correctly
# os.environ['ESMFMKFILE'] = 'C:/Users/aakas/anaconda3/envs/[envname]/lib/esmf.mk'
# import xesmf as xe

os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
# We need to do the same thing as with the CERES data where annual
# composites are generated and used to construct El Nino/La Nina anomaly fields
from obs_scripts.vis_clouds import load_oni_idx, is_enso_oni, progress_bar
from obs_scripts.atmos_data import plot_winds
from obs_scripts.ceres_comps import plot_radiation
from CZ_model.standard_funcs import interp_grid


cz_domain = [-30, 30, 120, -80]
ep_domain = [-30, 10, -120, -80]


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
    for n in range(12, 12 * (len(xr_array.time) // 12), 12):
        # Floor division then multiplication to only capture full years
        progress_bar(n, len(xr_array.time), 'Calcuting Climatlogy')
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


def calc_eis(era5_eis):
    """
    Calculates estimated inversion strength of dataarray and returns the same
    
    also returns theta_700
    
    copied from cloud_corr. I should create a shared function file
    """
    # Remove last month since our single levels data doesn't have that
    era5_eis = era5_eis[{'time':slice(0, len(era5_eis.time) - 1)}]
    t_700 = era5_eis.sel(pressure_level=700)['t']
    t_1000 = era5_eis.sel(pressure_level=1000)['t']
    
    # R/cp from wikipedia
    theta_700 = t_700 * (1000 / 700)**(0.286)
    # Since theta = T at surface reference pressure
    return (theta_700 - t_1000), theta_700


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


def polyfit_detrend(dataarray, dim='time'):
    # Fit a polynomial and subtract the trend
    trend = dataarray.polyfit(dim=dim, deg=1)
    fit = xr.polyval(dataarray[dim], trend.polyfit_coefficients)
    return dataarray - fit


def enso_composite(xr_array, nino_idx, print_num=False, deas=True):
    """
    Creates three composites- one for Neutral conditions, one for El Nino,
    and one for La Nina. Returns each of these composites along with 
    the climatological year
    """
    # DETREND LINEAR
    # xr_array = xr_array.map(lambda da: polyfit_detrend(da, 'time'))
    # Faster and simpler calculation working?
    climatology = xr_array.groupby('time.month').mean(dim='time')
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
        # el_nino = el_nino.sortby("month") - climatology
        # el_nino.groupby('time.month').mean(dim='time')
    if la_nina is not None:
        la_nina /= num_la_nina
        # la_nina = la_nina.sortby("month") - climatology
        # la_nina.groupby('time.month').mean(dim='time')
    if neutral is not None:
        neutral /= num_neutral
        # neutral = neutral.sortby("month") - climatology
        # neutral.groupby('time.month').mean(dim='time')
    
    if print_num:
        print(f'\n{num_el_nino} El Nino years, {num_la_nina} La Nina years,' +\
              f' {num_neutral} Neutral years')
    
    # Store this as a dict so it is managable
    comps = {'el_nino': el_nino.sortby("month"),
             'la_nina': la_nina.sortby("month"), 
             'neutral': neutral.sortby("month"),
             'clim': climatology}
    
    return comps


def enso_composite_deas(xr_array, nino_idx, print_num=False, weird=False):
    """
    Creates three composites- one for Neutral conditions, one for El Nino,
    and one for La Nina. Returns each of these composites along with 
    the climatological year
    
    Deaseasonalized FIRST
    """
    if weird:
        xr_array = xr_array.drop_vars('number')
    # Faster and simpler calculation if ceres as it works
    climatology = xr_array.groupby('time.month').mean(dim='time')
    climatology = climatology
    years = np.unique(xr_array.time.dt.year.values)
    tot = len(years)
    
    # Initialize composites
    el_nino, la_nina, neutral = None, None, None
    num_el_nino, num_la_nina, num_neutral = 0, 0, 0
    
    xr_array_ds = xr_array.copy(deep=True).sel(time=slice("2000-01", "2023-12"))
    years = np.unique(xr_array_ds.time.dt.year)
    for n, year in enumerate(years):
        # no idea what "number" is
        year_data = xr_array_ds.sel(time=slice(f'{year}-01', f'{year}-12'))
        time_axis = year_data.time
        # Reassign time axis so it is consistent with the selected slice
        climatology['time'] = time_axis.data
        if len(time_axis) < 12 and n > 0:
            climatology = climatology[{'time':slice(0, len(time_axis))}]
        climatology = climatology.assign_coords(time=("month",
                                                     climatology.time.data))
        climatology = climatology.swap_dims({"month": "time"})
        # Subtract climatology year by year       
        # index climatology to not lose extra year
        xr_array_ds[{"time":slice(12 * n, 12 * (n + 1))}] -= climatology
    
    xr_array_ds = xr_array_ds.map(lambda da: polyfit_detrend(da, 'time'))
    xr_array = xr_array_ds

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
        # el_nino = el_nino.sortby("month") - climatology
        # el_nino.groupby('time.month').mean(dim='time')
    if la_nina is not None:
        la_nina /= num_la_nina
        # la_nina = la_nina.sortby("month") - climatology
        # la_nina.groupby('time.month').mean(dim='time')
    if neutral is not None:
        neutral /= num_neutral
        # neutral = neutral.sortby("month") - climatology
        # neutral.groupby('time.month').mean(dim='time')
    
    if print_num:
        print(f'\n{num_el_nino} El Nino years, {num_la_nina} La Nina years,' +\
              f' {num_neutral} Neutral years')
    
    # Store this as a dict so it is managable
    comps = {'el_nino': el_nino.sortby("month"),
             'la_nina': la_nina.sortby("month"), 
             'neutral': neutral.sortby("month")}
    
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


def load_composites(directory='era5_reanal/anomalies/', clim=True):
    """
    Loads the composite dictionaries (comp_pres, comp_sing, comp_rad) from NetCDF files.
    Returns the dictionaries with the same structure as in the script.
    """
    comp_pres, comp_sing, comp_rad = {}, {}, {}
    
    composite_dict = {'pres': comp_pres, 'sing': comp_sing, 'rad': comp_rad}
    phases = ['el_nino', 'la_nina', 'neutral']
    if clim:
        phases.append('clim')
    
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


def crop_era5(xr_array, rename=True, coord='180', domain=cz_domain,
              mode='inside'):
    """
    Crops to CZ modeldimentions
    
    Returns the region OUTSIDE our box if mode=='outside'
    """
    min_lat, max_lat, east_lon, west_lon = cz_domain
    
    if rename:
        xr_array = xr_array.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    if mode=='inside':
        if coord=='180':
            ds_east = xr_array.sel(lat=slice(max_lat, min_lat), 
                                   lon=slice(east_lon, 180))
            ds_west = xr_array.sel(lat=slice(max_lat, min_lat),
                                   lon=slice(-180, west_lon))
            df = xr.concat([ds_east, ds_west], dim="lon")
            
        elif coord=='360':
            df = xr_array.sel(lat=slice(min_lat, max_lat), 
                              lon=slice(east_lon, 360 + west_lon))
        else:
            print('How do you bork your own code this bad...')
    else:
        if coord == '180':
            # Latitude mask (everything outside min_lat to max_lat)
            lat_mask = ((xr_array['lat'] > max_lat) | (xr_array['lat'] < min_lat))
        
            # Longitude mask (everything outside east_lon to west_lon)
            lon_mask_east = (xr_array['lon'] > east_lon) & (xr_array['lon'] <= 180)
            lon_mask_west = (xr_array['lon'] < west_lon) & (xr_array['lon'] >= -180)
            lon_mask = ~(lon_mask_east | lon_mask_west)  # Invert tropical Pacific region
        
            # Combine the masks
            mask = lat_mask | lon_mask
        
            # Apply the mask
            df = xr_array.where(mask, drop=True)
        else:
            print('Need to implement for 360...')

    return df


def plot_scalar(comp_dict, var, month=12, phase='la_nina', title='',
                lims=cz_domain, cbar='HCC (Frac)'):
    """
    Contour plot of a scalar field (e.g., hcc) across the globe or a specified region.
    """
    era5 = comp_dict[phase].sel(month=month)
    # Define central longitude to correctly handle global data
    proj = ccrs.PlateCarree(central_longitude=180)
    
    # Ensure longitude wraps correctly if in 0-360 range
    if era5.lon.max() > 180:
        era5 = era5.assign_coords(lon=(((era5.lon + 180) % 360) - 180))
        era5 = era5.sortby('lon')
    
    # Extract data for plotting
    data = era5[var]
    lon = era5.lon.values
    lat = era5.lat.values
    
    # Print diagnostics to confirm data ranges
    print(f"Data variable '{var}' - min: {data.min().item():.3f}, max: {data.max().item():.3f}")
    # print(f"Longitude range: {lon.min()} to {lon.max()}")
    # print(f"Latitude range: {lat.min()} to {lat.max()}")

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


def single_level_div(comp_sing, comp_pres):
    """
    Calculates divergence (mean and 950 hpa) from comp_pres and assigns
    it as a level to comp_sing
    """
    for phase, df in comp_pres.items():
        mean_d = comp_pres[phase]['d'].sel(pressure_level=slice(1000, 850))
        mean_d = mean_d.mean(dim='pressure_level')
        comp_sing[phase]['d_mean'] = mean_d
        comp_sing[phase]['d_950'] = comp_pres[phase]['d'].sel(pressure_level=950)
        
    return comp_sing        


def coarsen_era5(comp_sing, comp_pres, comp_rad):
    """
    Coarsens the ERA5 grids to match CERES grid dimentions
    """
    lat_factor = comp_sing['el_nino'].dims['lat'] //\
        comp_rad['el_nino'].dims['lat']
    lon_factor = comp_sing['el_nino'].dims['lon'] //\
        comp_rad['el_nino'].dims['lon']
    
    for phase, df in comp_pres.items():
        comp_pres[phase] = df.coarsen(lat=lat_factor, lon=lon_factor,
                               boundary="trim").mean()
    for phase, df in comp_sing.items():
        comp_sing[phase] = df.coarsen(lat=lat_factor, lon=lon_factor,
                               boundary="trim").mean()
    return comp_sing, comp_pres


def regrid_era5(comp_sing, comp_pres, comp_rad):
    """
    Regrids ERA5 data in comp_sing and comp_pres dictionaries to CERES grid dimensions.
    Uses the latitude and longitude from comp_rad['el_nino'] as the target CERES grid.
    """
    # Define CERES grid based on comp_rad
    ds_ceres_grid = xr.Dataset({
        "lat": (["lat"], comp_rad['el_nino'].lat),  # CERES latitudes
        "lon": (["lon"], comp_rad['el_nino'].lon),  # CERES longitudes
    })
    
    # Create the regridder once for efficiency
    regridder = xe.Regridder(list(comp_sing.values())[0], ds_ceres_grid,
                             method="bilinear")
    
    # Regrid each dataset in comp_sing
    for key, ds_era5 in comp_sing.items():
        comp_sing[key] = regridder(ds_era5)
    
    # Regrid each dataset in comp_pres
    for key, ds_era5 in comp_pres.items():
        comp_pres[key] = regridder(ds_era5)
        
    return comp_sing, comp_pres


def create_anomaly(comp_dict):
    """
    Creates the anomaly fields for el nino/la nina/neutral by
    subtracting the climatology field from the same dict
    """
    comp_anom = dict()
    
    for key, value in comp_dict.items():
        if key == 'clim':
            continue
        comp_anom[key] = comp_dict[key] - comp_dict['clim']
    
    return comp_anom


def plot_enso_comp_multi(comp_sing, month, variables=['hcc', 'mcc', 'lcc'], lims=cz_domain,
                         title='', names=['High CC', 'Mid CC', 'Low CC'],
                         scale=100, unit='%'):
    """
    Plot a 2x3 grid of scalar fields for El Niño and La Niña conditions across specified variables.
    
    Parameters:
    - comp_sing: Dictionary containing data arrays for different ENSO conditions (El Niño and La Niña).
    - month: Integer representing the month index to be plotted.
    - variables: List of variable names to plot (default ['hcc', 'mcc', 'lcc']).
    - lims: domain limit (just the cz domain)
    - title: title to preceed calendar month name
    - names: labels for variables
    - scale: scale multiplier for percentage
    - unit: unit of label
    """
    width = len(comp_sing.keys())
    figsize = (24, 10) if width==3 else (20, 11)
    
    fig, axs = plt.subplots(width, 3, subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}, 
                            figsize=figsize)
    # Position title
    height = 0.95 if width==3 else 0.95
    shrink = 0.69 if width==3 else 0.9
    sep = 0.03 if width==3 else 0.03
    fig.suptitle(title + f' for {calendar.month_name[month]}', fontsize=20,
                 y=height)
    
    # Conditions and row labels
    conditions = list(comp_sing.keys())
    row_labels = ['El Niño', 'La Niña', 'Neutral', 'Climatology']
    plt.tight_layout()
    
    # Iterate over rows and columns for the ENSO conditions and variables
    for row, condition in enumerate(conditions):
        for col, var in enumerate(variables):
            ax = axs[row, col]
            ax.set_global()
            ax.set_title(f'{row_labels[row]} - {names[col]}')
            
            # Extract data for the specific condition, month, and variable
            data = comp_sing[condition].sel(month=month)[var]
            lon = comp_sing[condition].lon.values
            lat = comp_sing[condition].lat.values
            
            # Color bar that is white at zero
            # norm = TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
            
            # Adjust longitude range if needed
            #if lon.max() > 180:
             #   lon = (((lon + 180) % 360) - 180)
            #    lon, data = np.sort(lon), data.sel(lon=lon)
            
            # Create meshgrid for plotting
            lon2d, lat2d = np.meshgrid(lon, lat)
            
            # Plot with pcolormesh
            pcm = ax.pcolormesh(lon2d, lat2d, scale * data, transform=ccrs.PlateCarree(), 
                                shading='auto', cmap='RdBu_r')
            
            # Add coastlines and gridlines
            ax.coastlines()
            ax.gridlines(dms=True, draw_labels=True,
                         x_inline=False, y_inline=False)
            # Add color bar to each plot
            cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', 
                                pad=0.05, shrink=shrink, aspect=20)
            cbar.set_label(unit)
            
            # Set limits if needed (optional customization)
            ax.set_ylim(lims[0], lims[1])
            ax.set_xlim(lims[3] + 20, lims[2] - 20)
            # ax.gridlines(draw_labels=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    plt.subplots_adjust(wspace=sep)
    plt.show()


def plot_enso_comp(comp_sing, month, var='sst', lims=None, title='', unit='',
                   name='SST', neutral=False):
    """
    Plot a 1x2/3 grid of scalar fields for El Niño and La Niña conditions across specified variables.
    
    Parameters:
    - comp_sing: Dictionary containing data arrays for different ENSO conditions (El Niño and La Niña).
    - month: Integer representing the month index to be plotted.
    - variables: List of variable names to plot (default ['hcc', 'mcc', 'lcc']).
    """
    if not neutral and len(comp_sing.keys()) > 2:
        comp_sing.pop('neutral')
    width = len(comp_sing.keys())
    # dumb double check to make same function work
    height = 0.45 if width==3 else 0.4
    height = 0.6 if width==2 else height
    fig, axs = plt.subplots(1, width, subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)}, 
                            figsize=(24, 11), layout='constrained')
    fig.suptitle(title + f' for {calendar.month_name[month]}', fontsize=20,
                 y=height)
    
    # Conditions and row labels
    conditions = list(comp_sing.keys())
    row_labels = ['El Niño', 'La Niña', 'Neutral', 'Climatology']
    
    # Iterate over rows and columns for the ENSO conditions and variables
    for col, condition in enumerate(conditions):
        ax = axs[col]
        ax.set_global()
        ax.set_title(f'{row_labels[col]} - {var.upper()}')
        
        # Extract data for the specific condition, month, and variable
        data = comp_sing[condition].sel(month=month)[var]
        lon = comp_sing[condition].lon.values
        lat = comp_sing[condition].lat.values
        
        # Color bar that is white at zero
        # norm = TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
        
        # Adjust longitude range if needed
        if lon.max() > 180:
            lon = (((lon + 180) % 360) - 180)
            data['lon'] = (((data.lon + 180) % 360) - 180)
        # Create meshgrid for plotting
        lon2d, lat2d = np.meshgrid(lon, lat)
        
        # Plot with pcolormesh
        pcm = ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree(), 
                            shading='auto', cmap='RdBu_r')
        
        # Add coastlines and gridlines
        ax.coastlines()
        ax.gridlines(dms=True, 
                     x_inline=False, y_inline=False)
        
        # Add color bar to each plot
        # cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', 
        #                    pad=0.05, shrink=shrink, aspect=20)
        # cbar.set_label(name + unit)
        
        # Set limits if needed (optional customization)
        ax.set_ylim(lims[0], lims[1])
        ax.set_xlim(lims[3] + 20, lims[2] - 20)
        
    # plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    cbar = fig.colorbar(pcm, ax=axs.ravel().tolist(), orientation='horizontal',
                        aspect=50)
    cbar.set_label(name + ' (' + unit + ')')
    plt.show()


def main():
    global comp_sing, anom_sing, anom_rad
    dir_anom = 'era5_reanal/anomalies/'    
    dir_clim = 'era5_reanal/climatology/'
    # Check if files exist
    files_exist_anom = all(os.path.exists(os.path.join(dir_anom, f'{comp}_{phase}.nc'))
                      for comp in ['pres', 'sing', 'rad'] 
                      for phase in ['el_nino', 'la_nina', 'neutral'])
    files_exist_clim = all(os.path.exists(os.path.join(dir_clim, f'{comp}_{phase}.nc'))
                      for comp in ['pres', 'sing', 'rad'] 
                      for phase in ['el_nino', 'la_nina', 'neutral', 'clim'])
    
    if files_exist_anom and files_exist_clim:
        # Load existing composites
        print("Files exist. Loading composites from NetCDF files.")
        comp_pres, comp_sing, comp_rad = load_composites(dir_clim)
        anom_pres, anom_sing, anom_rad = load_composites(dir_anom, clim=False)
    else:
        # Generate composites and save them
        print("Files not found. Generating composites and saving to NetCDF files.")
        era5_sing = xr.open_dataset('era5_reanal/era5_reanal_modern.nc').drop_vars('expver')
        era5_pres = xr.open_dataset('era5_reanal/era5_reanal_pres.nc').drop_vars('expver') # 'era5_reanal/era5_reanal_pres.nc
        # Load in the EIS variable too
        era5_eis = xr.open_dataset('era5_reanal/era5_lts.nc').drop_vars('expver')
        # Crop to region (also renames to lat/lon for conveinence)
        era5_sing = crop_era5(era5_sing, rename=True)
        era5_pres = crop_era5(era5_pres, rename=True)
        era5_eis = crop_era5(era5_eis, rename=True)
                
        ceres = xr.load_dataset('ceres_data/ceres_ebaf_all.nc')
        
        # Create column data for CERES
        ceres['atms_net_all'] = ceres['toa_net_all_mon'] - ceres['sfc_net_tot_all_mon']
        # nino_idx = load_nino_idx('misc_data/nino_all.csv')
        oni_idx = load_oni_idx('misc_data/oni_index.txt')
        
        # Standardize time format
        ceres['time'] = pd.to_datetime(ceres.time)
        # Create time axis
        era5_pres['time'] = pd.to_datetime(era5_pres.date, format='%Y%m%d')
        era5_sing['time'] = pd.to_datetime(era5_sing.date, format='%Y%m%d')
        era5_eis['time'] = pd.to_datetime(era5_eis.date, format='%Y%m%d')
        # Assign coordinate
        era5_pres = era5_pres.assign_coords(time=("date", era5_pres.time.data))
        era5_sing = era5_sing.assign_coords(time=("date", era5_sing.time.data))
        era5_eis = era5_eis.assign_coords(time=("date", era5_eis.time.data))
        # Swap coordinate and drop old one
        era5_pres = era5_pres.swap_dims({"date": "time"}).drop_vars('date')
        era5_sing = era5_sing.swap_dims({"date": "time"}).drop_vars('date')
        era5_eis = era5_eis.swap_dims({"date": "time"}).drop_vars('date')
        
        # Pass theta 700 information to the single level data
        era5_sing['eis'], era5_sing['theta_700'] = calc_eis(era5_eis)
        
        # Generate anomaly fields
        print('Calculating ERA5 Pressure Levels data')
        comp_pres = enso_composite(era5_pres, oni_idx, print_num=True)
        anom_pres = enso_composite_deas(era5_pres, oni_idx, weird=True)
        print('\nCalculating ERA5 Single Levels data')
        comp_sing = enso_composite(era5_sing, oni_idx, print_num=False)
        anom_sing = enso_composite_deas(era5_sing, oni_idx, print_num=False, weird=True)
        print('\nCalculating CERES data')
        comp_rad = enso_composite(ceres, oni_idx, print_num=True)
        # Start with the full year for CERES
        anom_rad = enso_composite_deas(ceres.sel(time=slice("2001-01", "2023-12")), oni_idx)
        
        # Save the generated composites
        save_composites(comp_pres, comp_sing, comp_rad, dir_clim)
        save_composites(anom_pres, anom_sing, anom_rad, dir_anom)
    
    # Create anomaly fields
    # anom_sing = create_anomaly(comp_sing)
    # anom_pres = create_anomaly(comp_pres)
    # anom_rad = create_anomaly(comp_rad)
    # Actual data analysis time!
    # Add divergence
    anom_sing = single_level_div(anom_sing, anom_pres)
    anom_sing, anom_pres = coarsen_era5(anom_sing, anom_pres, anom_rad)
    
    # Plotting
    month = 12
    
    # Plot cloud cover
    plot_enso_comp_multi(comp_sing, month=month, lims=cz_domain, title='Composite')

    plot_enso_comp_multi(anom_sing, month=month, lims=cz_domain, title='Anomaly')
    
    # Plot Radiation
    variables = ['toa_net_all_mon', 'sfc_net_sw_all_mon', 'atms_net_all']
    labels = ['TOA Net', 'SFC Net', 'ATM Net']
    plot_enso_comp_multi(comp_rad, month=month, variables=variables, lims=cz_domain, 
                         title='Radiation Composites', names=labels, 
                         scale=1, unit='W/m2')
    plot_enso_comp_multi(anom_rad, month=month, variables=variables,
                         lims=cz_domain, title='Radiation Anomalies', 
                         names=labels, scale=1, unit='W/m2')
    
    # SST Anomalies
    plot_enso_comp(comp_sing, month=month, var='sst', lims=cz_domain, 
                   title='Sea Surface Temperature', unit='K',
                       name='SST')
    plot_enso_comp(anom_sing, month=month, var='sst', lims=cz_domain,
                   title='SST Anomalies', unit='K',
                       name='SST')
    plot_enso_comp(anom_sing, month=month, var='eis', lims=cz_domain,
                   title='EIS Anomalies', unit='K',
                       name='EIS')
    plot_enso_comp(anom_sing, month=month, var='theta_700', lims=cz_domain,
                   title='700 HPa Pot. Temp. Anomalies', unit='K',
                       name='Theta 700')
    
    
if __name__ == '__main__':
    main()
