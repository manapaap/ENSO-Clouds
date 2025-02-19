# -*- coding: utf-8 -*-
"""
File to process CERES monthly radiation data
"""

import xarray as xr
from os import chdir
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation
import calendar

chdir('C:/Users/aakas/Documents/ENSO-Clouds/')


import obs_scripts.shared_funcs as share


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
        share.progress_bar(n, tot)
        format_date = str(time.data)[:7].replace('-', '.')
        enso_state = share.is_enso_oni(nino_idx, format_date)
        
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


def enso_composite_by_year(ceres, nino_idx, var):
    """
    Creates composites for ENSO phases, grouping data from May to April based
    on ENSO phase in December.
    """
    # tot = len(ceres.time)
    years = ceres.time.dt.year.values
    # months = ceres.time.dt.month.values
    
    # Initialize composites
    el_nino = None
    la_nina = None
    neutral = None
    num_el_nino, num_la_nina, num_neutral = 0, 0, 0

    # Loop over years, but align composites from May to April
    for yr in range(years[0], years[-1]):
        # Check the ENSO phase in December (yr)
        enso_state = share.is_enso_oni(nino_idx, f'{yr}.12')

        # Select data from May (yr) to April (yr+1)
        may_to_april = ceres.sel(time=slice(f'{yr}-05-01', f'{yr+1}-04-30'))
        # Overwrite time information with month
        may_to_april = may_to_april.groupby('time.month').mean(dim='time')
        # Update composites based on ENSO state in December
        if enso_state == 'El Nino':
            if el_nino is None:
                el_nino = may_to_april
            else:
                el_nino += may_to_april
            num_el_nino += 1
        elif enso_state == 'La Nina':
            if la_nina is None:
                la_nina = may_to_april
            else:
                la_nina += may_to_april
            num_la_nina += 1
        else:  # Neutral
            if neutral is None:
                neutral = may_to_april
            else:
                neutral += may_to_april
            num_neutral += 1
    # Safeguard against zero divisions
    if el_nino is not None:
        el_nino /= num_el_nino
        # el_nino.groupby('time.month').mean(dim='time')
    if la_nina is not None:
        la_nina /= num_la_nina
        # la_nina.groupby('time.month').mean(dim='time')
    if neutral is not None:
        neutral /= num_neutral
        # neutral.groupby('time.month').mean(dim='time')
    
    print(f'{num_el_nino} El Nino years, {num_la_nina} La Nina years,' +\
          f' {num_neutral} Neutral years')
    
    return el_nino, la_nina, neutral


def animate_radiation(sel_var, title='', units=''):
    """
    Animates the evolution of a selected radiation variable (or SST anomalies)
    over the annual cycle, making the plot agnostic to the year.
    
    sel_var = ceres[var] format (xarray DataArray)
    title = Plot title
    units = Units for the color bar
    """
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    # Set up initial plot
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=proj)

    # Create the coastline and gridlines
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Initialize contour plot with the first time slice
    con = ax.contourf(sel_var.lon, sel_var.lat, sel_var.isel(month=0).data, 
                      origin='lower', transform=ccrs.PlateCarree(), cmap='viridis')

    cbar = plt.colorbar(con, ax=ax, fraction=0.02, pad=0.04)
    cbar.ax.set_title(units)

    # Function to update each frame in the animation
    def update(frame):
        month = frame % 12  # Restrict the frame to within 0-11 (months of the year)

        # Instead of ax.clear(), remove previous contour and update
        nonlocal con
        for coll in con.collections:
            coll.remove()
        
        # Update the contour plot for the current month
        new_data = sel_var.isel(month=month).data
        con = ax.contourf(sel_var.lon, sel_var.lat, new_data, 
                          origin='lower', transform=ccrs.PlateCarree(), cmap='viridis')

        # Extract the month name and update the title with it
        ax.set_title(f"{title} - Month: {calendar.month_name[month+1]}")  # +1 to get correct month

    # Create animation object using FuncAnimation
    ani = FuncAnimation(fig, update, frames=12, repeat=True,
                        interval=3000)  # 12 frames for 12 months

    plt.show()
    return ani


def main():
    global ceres, el_nino, la_nina
    nino_idx = share.load_nino_idx('misc_data/nino_all.csv')
    share.plot_enso(nino_idx)
    oni_idx = share.load_oni_idx('misc_data/oni_index.txt')
    share.plot_enso(oni_idx, var='oni', cutoff=0.5)
    ceres = xr.load_dataset('ceres_data/ceres_ebaf_all.nc')
    
    # This is net surface radiation, really the only thing we care for
    var = 'sfc_net_tot_all_mon'
    toa = 'toa_net_all_mon'
    # Create atmospheric column data
    ceres['atms_net_all'] = ceres[toa] - ceres[var]
    
    # Grand mean across time
    ceres_clim = ceres.mean(dim='time', skipna=True)
    plot_radiation(ceres_clim[var], 'Climatological Surface Radiation Flux',
                   'W/m2')
       
    # Let's create more sophisticated composites which contain an
    # annual cycle of radiation
    clim_year = ceres.groupby('time.month').mean(dim='time')
    
    el_nino, la_nina, neutral = enso_composite_by_year(ceres, oni_idx, var)
    
    # Convert these to anual anomalies
    el_nino = el_nino - clim_year
    la_nina = la_nina - clim_year
    neutral = neutral - clim_year
    # We can plot these however we want, but we also can make an animation!
    # global ani_nino_sw
    # ani_nino_sw = animate_radiation(el_nino[var], 'El Nino Sfc. Rad. Anom.', 'W/m2')
    # Incredible. Let's calculate the total energy imbalance across time
    el_nino_bal = el_nino['sfc_net_tot_all_mon'].sum()
    la_nina_bal = la_nina['sfc_net_tot_all_mon'].sum()
    
    print(f'El Nino Energy Imbalance: {float(el_nino_bal):.3f} W/m2')
    print(f'La Nina Energy Imbalance: {float(la_nina_bal):.3f} W/m2')
    
    # Let's isolate the Eastern Pacific and create a time series
    # of radiation anomalies across the same
    ceres_anom = ceres - ceres_clim
    ceres_ep = ceres_anom.sel(lat=slice(-30, 0), lon=slice(240, 280))
    

if __name__ == '__main__':
    main()

