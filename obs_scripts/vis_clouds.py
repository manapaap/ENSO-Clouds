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
from os import chdir, listdir
import pandas as pd
import sys
from matplotlib.colors import TwoSlopeNorm


chdir('C:/Users/aakas/Documents/ENSO-Clouds/')

# Entire CZ model domain; min_lat, max_lat, min_lon, max_lon
cz_domain = [-30, 30, 120, -80 + 360]
ep_domain = [-30, -15, 240, 280]


def load_nino_idx(fpath):
    """
    Loads the nino indices and formats it into pandas df with dates
    
    Source:
        https://psl.noaa.gov/data/correlation/nina34.anom.data
        https://psl.noaa.gov/data/timeseries/monthly/NINO34/
    """
    data = pd.read_csv('misc_data/nino_all.csv', header=0,
                       names=['year', 'month', '1_2', '1_2_anom', '3',
                              '3_anom', '4', '4_anom', '3.4', '3.4_anom'])
    
    data['time'] = pd.to_datetime(data['year'].astype(str) + "-" +\
                                  data['month'].astype(str), format='%Y-%m')
    return data


def load_cloud_file(fpath, domain=cz_domain):
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
    
    return ref, simp


@np.vectorize
def convert_pos_degrees(coord):
    """
    Converts an array of positive degrees to nevative (0-360 to -180 to 180)
    """
    if coord > 180:
        return coord - 360
    else:
        return coord


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
             plot='diff', domain=cz_domain):
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


def plot_enso(nino):
    """
    Creates a plot of the nino 3.4 index and cutoffs
    """
    plt.figure()
    plt.plot(nino['time'], nino['3.4_anom'], color='black')
    plt.hlines(0.4, xmin=nino['time'][0], xmax=nino['time'].iloc[-1],
               linestyle='dashed', alpha=0.8, color='grey')
    plt.hlines(-0.4, xmin=nino['time'][0], xmax=nino['time'].iloc[-1],
               linestyle='dashed', alpha=0.8, color='grey')
    plt.xlabel('Years')
    plt.ylabel('Nino 3.4 anomaly')
    plt.title('Nino 3.4 Index')
    plt.grid()    
    ax = plt.gca()
    ax.fill_between(nino['time'], 0.4, nino['3.4_anom'], 
                    where=nino['3.4_anom'] > 0.4,
                    alpha=0.6, color='red')
    ax.fill_between(nino['time'], -0.4, nino['3.4_anom'], 
                    where=nino['3.4_anom'] < -0.4,
                    alpha=0.6, color='blue')
    plt.show()
    

def is_enso(nino_idx, date, out=False):
    """
    Given a date, tells you if the date is in neurral, el nino, or la nina
    phase of enso. Uses nino 3.4 definition
    
    Date format- %Y-%m
    """
    date = pd.to_datetime(date, format='%Y.%m')
    
    vals = nino_idx.loc[(nino_idx['time'] <= date) & (nino_idx['time'] >= date)]

    if vals['3.4_anom'].iloc[0] >= 0.4:
        state = 'El Nino'
    elif vals['3.4_anom'].iloc[0] <= -0.4:
        state = 'La Nina'
    else: 
        state = 'Neutral'
    
    if out:
        print(state)
    return state


def get_fnames(dirpath, season):
    """
    Gets the relevant file names to help create the enso composite
    only includes files within the season of intrest
    """
    all_files = listdir(dirpath)
    
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
   

def progress_bar(n, max_val, cus_str=''):
    """
    I love progress bars in long loops
    """
    sys.stdout.write('\033[2K\033[1G')
    print(f'Computing...{100 * (n + 1) / max_val:.2f}% complete ' + cus_str,
          end="\r")    


def enso_composite(dirpath, nino_idx, season='all', var='cldamt', 
                   domain=cz_domain):
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
    
    for n, fname in enumerate(to_load):
        progress_bar(n, total)
        enso_state = is_enso(nino_idx, fname[:7])
        # Create composite by ENSO state
        if enso_state == 'El Nino':
            if el_nino is None:
                el_nino = load_cloud_file(dirpath + fname, domain=domain)
                num_el_nino += 1
            else:
                file = load_cloud_file(dirpath + fname, domain=domain)
                el_nino[var] += file[var]
                num_el_nino += 1
        elif enso_state == 'La Nina':
            if la_nina is None:
                la_nina = load_cloud_file(dirpath + fname, domain=domain)
                num_la_nina += 1
            else:
                file = load_cloud_file(dirpath + fname, domain=domain)
                la_nina[var] += file[var]
                num_la_nina += 1 
        elif enso_state == 'Neutral':
            if neutral is None:
                neutral = load_cloud_file(dirpath + fname, domain=domain)
                num_neutral += 1
            else:
                file = load_cloud_file(dirpath + fname, domain=domain)
                neutral[var] += file[var]
                num_neutral += 1
    # Safeguard in case a selection of time is chosen without an event
    if el_nino != None:
        el_nino[var] /= num_el_nino
    if la_nina != None:
        la_nina[var] /= num_la_nina
    if neutral != None:
        neutral[var] /= num_neutral
    
    return el_nino, la_nina, neutral
            

def main():
    nino_idx = load_nino_idx('misc_data/nino_all.csv')
    plot_enso(nino_idx)
    
    # cloud_ex = load_cloud_file('ISCCP_clouds/2015.12.nc')   
    _, cloud_dict = isccp_cloud_dict()
    
    domain = cz_domain
    central_longitude = 180
    
    el_nino, la_nina, neutral = enso_composite('ISCCP_clouds/', nino_idx, 
                                               'winter', domain=domain)
    
    # Variable of intrest
    var = 'cldamt' # 'cldamt'
    
    plot_map(el_nino, title='Composite El Nino', var=var, plot='reg',
             central_longitude=central_longitude, domain=domain)
    plot_map(la_nina, title='Composite La Nina', var=var, plot='reg',
             central_longitude=central_longitude, domain=domain)
    plot_map(neutral, title='Composite Neutral', var=var, plot='reg',
             central_longitude=central_longitude, domain=domain)

    # this is intresting, but most important is the differences
    # in cloud state between each ENSO state
    # set to zero for anything less than 10% for clarity
    
    
    nino_nina_diff = el_nino.copy(deep=True)
    nino_nina_diff[var] -= la_nina[var]
    plot_map(nino_nina_diff, title='El Nino - La Nina Difference',
             var=var, domain=domain, plot='diff',
                      central_longitude=central_longitude)
    
    nino_neu_diff = el_nino.copy(deep=True)
    nino_neu_diff[var] -= neutral[var]
    plot_map(nino_neu_diff, title='El Nino - Neutral Difference',
             var=var, domain=domain, plot='diff',
                      central_longitude=central_longitude)
    
    nina_neu_diff = la_nina.copy(deep=True)
    nina_neu_diff[var] -= neutral[var]
    plot_map(nina_neu_diff, title='La Nina - Neutral Difference',
             var=var, domain=domain, plot='diff',
                      central_longitude=central_longitude)
    
    
if __name__ == "__main__":
    main()
    
