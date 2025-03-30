0# -*- coding: utf-8 -*-
"""
GOAL: Load clouds data and create representative maps of high/low/middle
clouds during neutral/el nino/la nina conditions over the CZ model
domain
"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
import obs_scripts.shared_funcs as share

    
def load_cloud_file(fpath, domain=share.cz_domain_360):
    """
    Loads the cloud data file, cropping it to the CZ model region
    """
    min_lat, max_lat, min_lon, max_lon = domain
    data = xr.load_dataset(fpath)
    
    lats = data['eqlat']
    lons = data['eqlon']
    mask = (lats >= min_lat) & (lats <= max_lat) & (lons >= min_lon) &\
        (lons <= max_lon)
    
    data = data.where(mask, drop=True)
    
    # also do the normal lat/lon slice for safety
    data = data.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
    
    return data


def isccp_cloud_dict():
    """
    Returns the dictionary representing the mapping ebtween cloud bins
    and ISCCP cloud types
    
    https://www.ncei.noaa.gov/sites/g/files/anmtlf171/files/2022-11/
    ISCCP%20User%E2%80%99s%20Guide%20v5_0.pdf
    """
    # All, second column is ice
    ref = {13: 'cirrus', 16: 'cirrus_i',
           14: 'cirrostratus', 17: 'cirrostratus_i',
           15: 'deep convection', 18: 'deep convection i',
           7: 'altocumulus', 10: 'altocumulus_i',
           8: 'altostratus', 11: 'altostratus_i',
           9: 'nimbostratus', 12: 'nimbostratus_i',
           1: 'cumulus', 4: 'cumulus_i',
           2: 'stratocumulus', 5: 'stratocumulus_i',
           3: 'stratus', 6: 'stratus_i'}
    
    # Simplified
    simp = {13: 'high', 16: 'high', 14: 'high', 17: 'high',
            15: 'deep', 18: 'deep', 7: 'mid', 10: 'mid', 
            8: 'mid', 11: 'mid', 9: 'mid', 12: 'mid',
            1: 'cumulus', 4: 'cumulus', 2: 'stratus', 5: 'stratus',
            3: 'stratus', 6: 'stratus'}
    
    # High cirrus clouds???    
    return ref, simp


def filter_cloud_types(data, cloud_category, cloud_dict):
    """
    Filters the cloud data for a specific cloud category using the cloud_dict.
    
    Parameters:
    data (xarray.Dataset): The cloud dataset.
    cloud_category (str): The cloud category to filter for (e.g., 'high', 
                                                            'deep', 'mid',
                                                            'cumulus',
                                                            'stratus').
    cloud_dict (dict): The dictionary mapping cloud type indices to categories.
    
    Returns:
    xarray.Dataset: The filtered cloud dataset.
    """
    # Find the cloud type indices that correspond to the desired category
    cloud_indices = [key - 1 for key, value in cloud_dict.items() if
                     value == cloud_category]
    
    # Filter the dataset using .sel with the list of cloud indices
    filtered_data = data.sel(cloud_type=cloud_indices)
    
    return filtered_data


def plot_map(cloud_type, var='cldamt', central_longitude=180, title='deep',
             plot='diff', domain=share.cz_domain_360):
    """
    Plots a map of the cloud type in question, showing cloud amount
    in each cell
    """
    min_lat, max_lat, min_lon, max_lon = domain
    
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    
    fig = plt.figure(figsize=(14, 5))
    ax = plt.axes(projection=proj)
   
    # Adjust extent based on central_longitude
    if central_longitude==0:
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], 
                      crs=ccrs.PlateCarree())
    else:
        ax.set_extent([min_lon - 180, max_lon - 180, 
                       min_lat, max_lat], crs=ccrs.PlateCarree())
        ax.set_xlim(min_lon - 180, max_lon - 180)
        ax.set_ylim(min_lat, max_lat)
        ax.gridlines(draw_labels=True)
    
    ax.coastlines()
    # Remove nan and inf and replace with domain mean
    data = np.asarray(cloud_type[var])
    clean = data[~np.isnan(data)]
    mean = np.mean(clean[~np.isinf(clean)])
    data[np.isnan(data)] = mean
    data[np.isinf(data)] = mean
    
    if plot == 'diff':
        # In case the difference doesn't hit zero
        if data.min() >= 0:
            cent = (data.min() + data.max()) / 2
            low = 0
        else:
            # normal case
            cent = 0
            low = data.min()
        
        norm = TwoSlopeNorm(vmin=low, vcenter=cent, vmax=data.max())
        sc = ax.tricontourf(cloud_type['eqlon'],
                         cloud_type['eqlat'],
                         data, norm=norm, cmap="RdBu_r", 
                         transform=ccrs.PlateCarree())
        scc = ax.tricontour(cloud_type['eqlon'],
                         cloud_type['eqlat'],
                         data, norm=norm, colors='black', 
                         transform=ccrs.PlateCarree(), alpha=0.7)
        cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label(f'{var} %')
        
    else:
        sc = ax.tricontourf(cloud_type['eqlon'],
                         cloud_type['eqlat'],
                         data,
                         cmap='viridis',
                         transform=ccrs.PlateCarree())
        scc = ax.tricontour(cloud_type['eqlon'],
                         cloud_type['eqlat'],
                         data, colors='black', 
                         transform=ccrs.PlateCarree(), alpha=0.7)
        cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label(f'{var} %')
        for im in ax.get_images():
            im.set_clim(0, 100)

    plt.title('Cloud Amount During ' + title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.show()
    

def is_enso(nino_idx, date, out=False, cutoff=0.5):
    """
    Given a date, tells you if the date is in neutral, El Nino, or La Nina
    phase of ENSO. Uses the Nino 3.4 index with a 5-month consecutive cutoff.
    
    Parameters:
    - nino_idx : DataFrame containing the 'time' and '3.4_anom' columns
    - date : str, date in format '%Y-%m' to check ENSO status for
    - out : bool, if True, prints the ENSO state
    - cutoff : float, anomaly threshold for El Nino and La Nina
    
    Returns:
    - state : str, 'El Nino', 'La Nina', or 'Neutral'
    """
    # Convert the date to a datetime object and get the 5-month window
    date = pd.to_datetime(date, format='%Y.%m')
    date_start = date - pd.DateOffset(months=4)  # Use 4 months back to include the current month

    # Select the relevant 5-month window
    vals = nino_idx.loc[(nino_idx['time'] <= date) & (nino_idx['time'] >= date_start)]
    
    # Check if we have 5 months of data (to avoid errors near the start of dataset)
    if len(vals) < 5:
        if out:
            print("Insufficient data for 5-month analysis")
        return None  # or raise an exception, or handle as you prefer

    # Determine ENSO phase based on cutoff value
    if (vals['3.4_anom'] >= cutoff).all():
        state = 'El Nino'
    elif (vals['3.4_anom'] <= -cutoff).all():
        state = 'La Nina'
    else: 
        state = 'Neutral'
    
    # Optionally print the ENSO state
    if out:
        print(state)
    
    return state


def get_fnames(dirpath, season):
    """
    Gets the relevant file names to help create the enso composite
    only includes files within the season of intrest
    """
    all_files = os.listdir(dirpath)
    
    if season == 'all':
        return all_files
    elif season == 'winter':
        months = [12, 1, 2]
    elif season == 'fall':
        months = [9, 10, 11]
    elif season == 'summer':
        months = [6, 7, 8]
    elif season == 'spring':
        months = [3, 4, 5]
    elif type(season) is int:
        months = [season]
    else:
        print("INVALID SEASON SELECTION")
        return None
    
    sel_files = [x for x in all_files if int(x[5:7]) in months]
    return sel_files


def enso_composite(dirpath, nino_idx, season='all', var='cldamt', 
                   domain=share.cz_domain_360):
    """
    Creates a composite map of ENSO around the equatorial pacific, returning
    three arrays for neutral, el nino, and la nina
    
    If season is specified as 'fall', 'spring', 'winter', or 'summer', will
    create the composite for those months only. Can also specify season
    as a number in which case only that month's data will be used to create
    the composite
    """
    to_load = get_fnames(dirpath, season)
    total = len(to_load)
    
    # Initialize loop vars
    el_nino = None
    num_el_nino = 0
    la_nina = None
    num_la_nina = 0
    neutral = None
    num_neutral = 0
    clim = None
    
    for n, fname in enumerate(to_load):
        share.progress_bar(n, total)
        enso_state = share.is_enso_oni(nino_idx, fname[:7])
        # Create composite by ENSO state
        if clim is None:
            clim = load_cloud_file(dirpath + fname, domain=domain)
        if enso_state == 'El Nino':
            if el_nino is None:
                el_nino = load_cloud_file(dirpath + fname, domain=domain)
                num_el_nino += 1
            else:
                file = load_cloud_file(dirpath + fname, domain=domain)
                el_nino[var] += file[var]
                clim[var] += file[var]
                num_el_nino += 1
        elif enso_state == 'La Nina':
            if la_nina is None:
                la_nina = load_cloud_file(dirpath + fname, domain=domain)
                num_la_nina += 1
            else:
                file = load_cloud_file(dirpath + fname, domain=domain)
                la_nina[var] += file[var]
                clim[var] += file[var]
                num_la_nina += 1 
        elif enso_state == 'Neutral':
            if neutral is None:
                neutral = load_cloud_file(dirpath + fname, domain=domain)
                num_neutral += 1
            else:
                file = load_cloud_file(dirpath + fname, domain=domain)
                neutral[var] += file[var]
                clim[var] += file[var]
                num_neutral += 1
    # Safeguard in case a selection of time is chosen without an event
    if el_nino != None:
        el_nino[var] /= num_el_nino
    if la_nina != None:
        la_nina[var] /= num_la_nina
    if neutral != None:
        neutral[var] /= num_neutral
    if clim != None:
        clim[var] /= (num_el_nino + num_la_nina + num_neutral)
    
    return el_nino, la_nina, neutral, clim
            

def create_xarray(dirpath, to='era5_reanal/timeseries/isccp_comb.nc'):
    """
    Combines the individual ISCCP files into a single xarray dataarray
    
    Do this in sets of 10 years because it becomes too slow alll at once
    
    Also saves the file
    """
    files = get_fnames(dirpath, 'all')
    num = len(files)
    # start with one file
    decades = np.arange(1980, 2020, 10)
    os.mkdir('ISCCP_clouds/temp')    
    for decade in decades:
        rel_files = [file for file in files if str(decade)[:3] in file]
        isccp = xr.load_dataset(dirpath + rel_files[0])
        for n, file in enumerate(rel_files[1:]):
            share.progress_bar(n, num, f'combining files...{decade}')
            next_entry = xr.load_dataset(dirpath + file)
            # combine files sequentially to reduce memory overhead
            isccp = xr.concat([isccp, next_entry], dim='time')  
        # Write to temp folder
        isccp.to_netcdf(f'ISCCP_clouds/temp/{decade}.nc')
    # Combine the decade files!
    files = []
    for decade in decades:
        files.append(xr.load_dataset(f'ISCCP_clouds/temp/{decade}.nc'))
    isccp = xr.concat(files, dim='time')
    # Fix dates (since they number on 15th rather than 1st)
    new_time = pd.to_datetime(isccp.time.to_index()).to_period('M').to_timestamp()
    isccp = isccp.assign_coords(time=new_time)
    isccp.to_netcdf(to)
    # delete the temp files now
    for decade in decades:
        os.remove(f'ISCCP_clouds/temp/{decade}.nc')
    os.rmdir('ISCCP_clouds/temp')
    return isccp


def deseasonalize_isccp(isccp):
    """
    De-seasonalizes the data by subtracting the mean year from every entry
    done month-by-month to reduce memory overhead
    """
    years = isccp.time.dt.year
    months = isccp.time.dt.month
    num = len(years)
    # Calc clim
    clim = isccp.groupby('time.month').mean(dim='time')
    # Drop all non-numeric variables as those won't be in the climatology
    isccp = isccp.drop_vars(list(set(isccp.keys()) - set(clim.keys())))
    for n, (year, month) in enumerate(zip(years, months)):
        share.progress_bar(n, num, f'Deseasonalizing...{int(year)}-{int(month)}')
        isccp[{'time': n}] -= clim.sel(month=month)
    return isccp


def cloud_types(isccp_anom, cloud_dict):
    """
    Creates separate data variables for high, medium, Sc, and Cu cloud types
    in isccp baased on the cldamt_type variable. Simplifies analysis and
    code
    """
    clouds = list(cloud_dict.values())
    for cloud in clouds:
        # Subtract 1 as ISCCP is zero-indexedf
        bins = [x - 1 for x, y in cloud_dict.items() if y == cloud]
        data = isccp_anom.cldamt_types.sel({'cloud_type': bins})
        isccp_anom[cloud] = data.sum(dim='cloud_type')
    return isccp_anom


def main():
    global isccp_anom, era5_data, pc_enso
    oni_idx = share.load_oni_idx(fpath='misc_data/oni_index.txt')
    oni_rel = oni_idx.query('"1983-07" <= time <= "2017-06"').reset_index(drop=True)
    # plot_enso(nino_idx.query('year >= 2000'), idx='Nino 3.4')
    oni_idx = share.load_oni_idx('misc_data/oni_index.txt')
    share.plot_enso(oni_idx.query('year >= 1983'), 'anom', 0.5, idx='ONI ')
    _, cloud_dict = isccp_cloud_dict()
    
    isccp_file = 'era5_reanal/timeseries/isccp_anom.nc'
    if os.path.exists(isccp_file):
        isccp_anom = xr.load_dataset(isccp_file)
    else:    
        # Removed the old plots and ENSO calculation since it didn't use
        # cloud types. Composites are also less useful than real data
        isccp = create_xarray('ISCCP_clouds/', 
                              to='era5_reanal/timeseries/isccp_comb.nc')
        isccp_anom = deseasonalize_isccp(isccp)
        isccp_anom = cloud_types(isccp_anom, cloud_dict)
        isccp_anom.to_netcdf('era5_reanal/timeseries/isccp_anom.nc')
    
    # Let's now calculate our C/E indives to compare to the ISCCP variables
    # This is ERA5 from all_cloud_corr.py
    era5_data = xr.load_dataset('era5_all/timeseries/era5_anom_all.nc')
    era5_data = era5_data.sel({'time': isccp_anom.time})
    # Calculate our EOFs
    _, pc_enso = share.calc_eof(era5_data, 'sst', n_pc=2,
                                plot=False, region='equator', detrend=True)
    pc_enso['PC1'] *= -1
    pc_enso = share.rotate_enso_eof(pc_enso)
    
    _, pc_700 = share.calc_eof(era5_data, var='theta_700', n_pc=1, plot=False,
                                region='tropics', detrend=True)
    pc_enso['PC_theta'] = -pc_700['PC1']
    
    pc_enso['oni'] = oni_rel['oni']
    
    # Correlations?
    corr = share.calc_corr_vect(isccp_anom, 'stratus', pc_enso, 'C')
    share.plot_corr(corr, cbar_lab='R', lims=share.pac_domain, 
              title='Correlation Between ISCCP Sc and C Mode')
    # Consistent! Let's see the low cloud anomaly
    lcc_anom = share.isolate_ep_isccp(isccp_anom, 'stratus')
    lcc_eq_anom = share.isolate_ep_isccp(isccp_anom, 'stratus', 
                                   share.eqp_domain_360)
    
    share.plot_combined(pc_enso['C'], lcc_anom, isccp_anom.time, 
                        'C', 'Isccp SC')
    share.plot_combined(pc_enso['PC2'], lcc_eq_anom, isccp_anom.time, 
                        'PC2', 'Isccp SC')
    
    # Write EOF to file for later
    pc_enso.to_csv('misc_data/enso_pcs_isccp.csv', index=False)
    
if __name__ == "__main__":
    main()
    