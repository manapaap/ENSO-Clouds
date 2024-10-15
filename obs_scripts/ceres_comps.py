# -*- coding: utf-8 -*-
"""
File to process CERES monthly radiation data
"""

import xarray as xr
from os import chdir
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


chdir('C:/Users/aakas/Documents/ENSO-Clouds/')


from obs_scripts.vis_clouds import load_nino_idx, is_enso, plot_enso
from CZ_model.standard_funcs import progress_bar


def plot_radiation(sel_var, title='', units=''):
    """
    PLots the radiation from ceres from the specific ceres dataset we care
    for
    
    sel_var = ceres[var] format
    """
    plt.figure()
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=proj)
    
    con = ax.contourf(sel_var.lon, sel_var.lat, sel_var.data,
                      origin='lower', transform=ccrs.PlateCarree(), 
                      cmap='viridis')
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    fig = plt.gcf()
    cbar = fig.colorbar(con, fraction=0.02, pad=0.04) 
    cbar.ax.set_title(units)
    
    plt.show()


def enso_composite(ceres, nino_idx):
    """
    Creates three composites using the ceres data: El Nino conditions,
    La Nina, and Neutral
    
    Uses is_enso function from previous file to ensure this
    """
    tot = len(ceres.time)
    # Initialize loop vars
    el_nino = None
    num_el_nino = 0
    la_nina = None
    num_la_nina = 0
    neutral = None
    num_neutral = 0
    
    for n, time in enumerate(ceres.time):
        progress_bar(n, tot)
        format_date = str(time.data)[:7].replace('-', '.')
        enso_state = is_enso(nino_idx, format_date)
        
        # Create composite by ENSO state
        if enso_state == 'El Nino':
            if el_nino is None:
                el_nino = ceres.sel(time=time)
                num_el_nino += 1
            else:
                file = ceres.sel(time=time)
                el_nino += file
                num_el_nino += 1
        elif enso_state == 'La Nina':
            if la_nina is None:
                la_nina = ceres.sel(time=time)
                num_la_nina += 1
            else:
                file = ceres.sel(time=time)
                la_nina += file
                num_la_nina += 1 
        elif enso_state == 'Neutral':
            if neutral is None:
                neutral = ceres.sel(time=time)
                num_neutral += 1
            else:
                file = ceres.sel(time=time)
                neutral += file
                num_neutral += 1
    # Safeguard in case a selection of time is chosen without an event
    if el_nino != None:
        el_nino /= num_el_nino
    if la_nina != None:
        la_nina /= num_la_nina
    if neutral != None:
        neutral /= num_neutral
    
    print(f'\n{num_el_nino} El Nino events')
    print(f'{num_la_nina} La Nina events')
    print(f'{num_neutral} Neutral')
    
    return el_nino, la_nina, neutral
    

def main():
    nino_idx = load_nino_idx('misc_data/nino_all.csv')
    plot_enso(nino_idx)
    
    global ceres
    ceres = xr.load_dataset('misc_data/CERES_radiation.nc')
    
    # This is net surface radiation, really the only thing we care for
    var = 'sfc_net_tot_all_mon'
    toa = 'toa_net_all_mon'
    # Create atmospheric column data
    ceres['atms_net_all'] = ceres[toa] - ceres[var]
    
    # Grand mean across time
    ceres_clim = ceres.mean(dim='time', skipna=True)
    plot_radiation(ceres_clim[var], 'Climatological Surface Radiation Flux',
                   'W/m2')
    
    el_nino, la_nina, neutral = enso_composite(ceres, nino_idx)
    
    # Plot these
    plot_radiation(el_nino[var] - ceres_clim[var],
                   'El Nino Surface Radiation Anomaly',
                   'W/m2')
    plot_radiation(la_nina[var] - ceres_clim[var],
                   'La Nina Surface Radiation Anomaly',
                   'W/m2')
    plot_radiation(neutral[var] - ceres_clim[var],
                   'Neutral Surface Radiation Anomaly',
                   'W/m2')
    
    # Let's create more sophisticated composites which contain an
    # annual cycle of radiation

if __name__ == '__main__':
    main()

