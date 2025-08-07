# -*- coding: utf-8 -*-
"""
Processing CMIP6 models to detrend and try to replicate the reanalysis/
satellite data

Let's try to make this modular...take SST and 
"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import detrend

os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
import obs_scripts.shared_funcs as share


def combine_files(folder='CMIP6/NOAA-GFDL/cllcalipso/'):
    """
    Reads and combines all the raw files into a
    single xarray file which can be handled
    more easily
    
    also adds pandas datetime
    """
    files = os.listdir(folder)
    # Empty list to hold files
    loaded = [None for _ in files]
    for n, file in enumerate(files):
        try: 
            loaded[n] = xr.load_dataset(folder + file)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            print(file)
    output = xr.merge(loaded)
    # Go to single time from time band for convenience
    try:
        # Dumb check since the time format is sometimes already good
        output['time'] = output.indexes['time'].to_datetimeindex()
    except:
        do_nothing = True
    # Set to monthly so days are -01
    output = output.\
        assign_coords(time=output.indexes['time'].to_period('M').to_timestamp())
    
    if 'time_bnds' in list(output.data_vars) and 'plev_bnds' in list(output.data_vars):
        # E3SM 
        output = output.drop_vars(['plev', 'plev_bnds', 'time_bnds',
                                   'lat_bnds', 'lon_bnds'],
                                  errors='ignore')
        if 'bnds' in list(output.data_vars):
            # canESM
            output = output.drop_vars('bnds')
    elif 'time_bnds' in list(output.data_vars):
        # Only in GFDL so far
        output = output.drop_vars(['time_bnds', 'lat_bnds', 'lon_bnds'],
                                  errors='ignore')
    elif 'plev_bounds' in list(output.data_vars):
        # This is for IPSL now...
        output = output.drop_vars(['time_bounds', 'plev_bounds', 'plev'],
                                  errors='ignore')
        
    return output


def isccp_to_sc(data):
    """
    Processes the clisccp variable in climate model output to isolate
    stratus and stratocumulus clouds and returns a smaller dataarray
    containing just that variable
    """
    


def deseasonalize(data):
    """
    De-seasonalizes the data by subtracting the mean year from every entry
    done month-by-month to reduce memory overhead. Also does the linear
    detrend
    """
    years = data.time.dt.year
    months = data.time.dt.month
    num = len(years)
    # Calc clim
    clim = data.groupby('time.month').mean(dim='time')
    for n, (year, month) in enumerate(zip(years, months)):
        share.progress_bar(n, num, f'Deseasonalizing...{int(year)}-{int(month)}')
        data[{'time': n}] -= clim.sel(month=month)
    # data = share.polyfit_detrend(data)
    return data


def calc_nino_anom(data, rem_trend=True):
    """
    Calculates the average SST anomaly within the nino 3.4 region
    
    0-360 coords
    """
    region = data.sel(lat=slice(-5, 5),
                      lon=slice(190, 240))
    nino = region.mean(dim=('lat', 'lon'))
    if rem_trend:
        nino = detrend(nino.tos.to_numpy())
    return nino


def plot_gcm_corr_subplot(data, to_corr, var, titles, types, 
                          vars2 = ['E', 'C', 'nino_3.4'],
                          lims=share.pac_domain, levels=5, to=''):
    """
    Plots a N data * N to_corr subplot correlating each array with a timeseries
    of entries. Plots the pearson correlation for each plot at 99% sig
    
    data and to_corr are lists containing xarray/pandas dataframes containing
    the information we want to correlate
    
    var and vars2 tell us the native variable names within data and to_corr
    these also correspond to titles/types. to_corr is a list containing
    relevant EOFs
    
    This is a modified version of the function in shared_funcs to make it 
    work with the GCM correlation I'm doing now. Thus, correlation
    is fixed to work with C, E, and nino 3.4 indices
    """
    num_rows = len(data)
    num_cols = 3
    
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, 
                        dpi=600, subplot_kw={'projection': proj},
                        figsize=(10, 9))  # or adjust as needed)
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.95, bottom=0.01)
    # top=0.65
    
    for row in range(num_rows):
        corr_rel = to_corr[row]
        for col in range(num_cols):
            corr = share.calc_corr_vect(data[row], var, corr_rel, vars2[col])
            corr_field = corr.fillna(0)
            # Wrapping for ERA5
            if corr_field.lon.max() > 180:
                corr_field['lon'] = ((corr_field.lon + 180) % 360) - 180
                corr_field = corr_field.sortby('lon')
            # Extract data for plotting
            lon = corr_field.lon.values
            lat = corr_field.lat.values
            # Non sig value mask
            nonsig = np.zeros(corr_field.shape)
            nonsig[np.isnan(corr_field)] = 1
            # Create meshgrid if needed
            lon2d, lat2d = np.meshgrid(lon, lat)
            # Determine the color limits to center around zero
            vmin, vmax = np.percentile(corr_field.values, [0.1, 99.9])  # Robust scaling
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            # Begin plotting
            axs[row, col].set_global()
            axs[row, col].set_title(titles[row] + ' ' + var + ' and ' +\
                                    types[col],
                                    fontsize='small')
            # masked_data = np.ma.masked_invalid(corr_field.data)
            # Use pcolormesh with centered color limits around zero
            pcm = axs[row, col].pcolormesh(lon2d, lat2d, corr_field.data,
                                           transform=ccrs.PlateCarree(),
                                shading='auto', cmap='RdBu_r', norm=norm)
            pcm2 = axs[row, col].pcolormesh(lon2d, lat2d, nonsig.data,
                                            transform=ccrs.PlateCarree(),
                                shading='auto', cmap='Greys', alpha=0.1)
            
            tiers = np.linspace(vmin, vmax, levels)
            lon1d = np.asarray(lon2d).reshape(-1)
            lat1d = np.asarray(lat2d).reshape(-1)
            corr1d = np.asarray(corr_field).reshape(-1)
            contour = axs[row, col].tricontour(lon1d, lat1d, corr1d,
                                               levels=tiers, 
                                    colors='black', linewidths=0.8, 
                                  transform=ccrs.PlateCarree())
            axs[row, col].clabel(contour, inline=True, fontsize=4,
                     fmt="%.1f", inline_spacing=5)
            
            # Add coastlines and gridlines
            axs[row, col].coastlines()
            gl = axs[row, col].gridlines(draw_labels=False, dms=True,
                                         alpha=0.5)
            if row == num_rows - 1:
                gl.bottom_labels = True
                gl.xlabel_style = {'size': 8}
            if col == 0:
                gl.left_labels = True
                gl.ylabel_style = {'size': 8}
            
            # Add colorbar and label
            cbar = plt.colorbar(pcm, ax=axs[row, col], location='right',
                                pad=0.02, shrink=0.75, aspect=20, ticks=tiers)
            cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in cbar.get_ticks()])
            cbar.ax.tick_params(labelsize=8)
            # Set plot limits if specified
            if lims is not None and len(lims) == 4:
                axs[row, col].set_ylim(lims[0], lims[1])
                axs[row, col].set_xlim(lims[3], lims[2])
            if lims is not None and len(lims) == 2:
                # Set to tropics still
                axs[row, col].set_ylim(lims[0], lims[1])
    # Savefig and show
    if to:
        fig.savefig(f'figures\saves\{to}.png', dpi=600,
                    bbox_inches='tight', pad_inches=0)  
    plt.show()


def keep_vars_coords(ds: xr.Dataset, vars_keep=None, coords_keep=None):
    """
    Keep only specified data variables and coordinates in an xarray Dataset.
    Explicitly removes unneeded coordinates (even if they share dimensions).
    """
    vars_keep = set(vars_keep or [])
    coords_keep = set(coords_keep or [])
    
    coords_all = set(ds.coords)
    coords_drop = coords_all - coords_keep

    # Step 1: Keep only requested data variables
    ds_new = ds[list(vars_keep & set(ds.data_vars))]

    # Step 2: Drop all coordinates (reset to plain data variables)
    ds_new = ds_new.drop_vars(coords_drop, errors='ignore')

    return ds_new


def main():
    global ukesm_sst, ukesm_cll, ukesm_eof
    start_date = '1983-07-01'
    end_date = '2017-06-02'

    gfdl_cll = combine_files('CMIP6/NOAA-GFDL-CM4/cllcalipso/').\
        sel(time=slice(start_date, end_date))
        
    # Clean the isccp data
    gfdl_sc = xr.load_dataset('CMIP6/NOAA-GFDL-CM4/clisccp/' +\
                              'clisccp_CFmon_GFDL-CM4_historical' +\
                              '_r1i1p1f1_gr1_195001-201412.nc').\
                                  sel(time=slice(start_date, end_date))
    gfdl_sc = gfdl_sc.sel({'plev': slice(1000 * 100, 680 * 100),
                           'tau': slice(3.6, 360)}).sum(dim=["tau", "plev"])
    gfdl_sc = keep_vars_coords(gfdl_sc, ['clisccp'], ['lat', 'lon', 'time'])
    gfdl_sst = combine_files('CMIP6/NOAA-GFDL-CM4/sst/').\
        sel(time=slice(start_date, end_date))
    gfdl_sc['time'] = gfdl_sst.time
    
    gfdl_cll = deseasonalize(gfdl_cll)
    gfdl_sc = deseasonalize(gfdl_sc)
    gfdl_sst = deseasonalize(gfdl_sst)
    # restrict sst field to pacific domain
    gfdl_crop = gfdl_sst.sel(lon=slice(120, 300))
    _, gfdl_eof = share.calc_eof(gfdl_crop, 'tos', n_pc=2, exclude_land=False,
                              plot=False, region='equator', detrend=True)
    # as always, fix sign 
    gfdl_eof['PC2'] *= -1
    gfdl_eof = share.rotate_enso_eof(gfdl_eof)
    gfdl_eof['nino_3.4'] = calc_nino_anom(gfdl_sst)
    
    # It works! and we fail to see a correlation in the region of intrest
        
    # Let's try this with E3SM
    e3sm_cll = combine_files('CMIP6/E3SM-1-1/cllcalipso/').\
        sel(time=slice(start_date, end_date))
    e3sm_sst = combine_files('CMIP6/E3SM-1-1/sst/').\
        sel(time=slice(start_date, end_date))
        
    e3sm_cll = deseasonalize(e3sm_cll)
    e3sm_sst = deseasonalize(e3sm_sst)
    # restrict sst field to pacific domain
    e3sm_crop = e3sm_sst.sel(lon=slice(120, 300))
    _, e3sm_eof = share.calc_eof(e3sm_crop, 'tos', n_pc=2, exclude_land=False,
                              plot=False, region='equator', detrend=True)
    e3sm_eof['PC2'] *= -1
    e3sm_eof = share.rotate_enso_eof(e3sm_eof)
    e3sm_eof['nino_3.4'] = calc_nino_anom(e3sm_sst)
   
    
    # try this with IPSL CM6
    ipsl_cll = combine_files('CMIP6/IPSL-CM6A/cllcalipso/').\
        sel(time=slice(start_date, end_date))    
    ipsl_sst = combine_files('CMIP6/IPSL-CM6A/sst_clean/').\
        sel(time=slice(start_date, end_date))  
    
    ipsl_cll = deseasonalize(ipsl_cll)
    ipsl_sst = deseasonalize(ipsl_sst)

    ipsl_crop = ipsl_sst.sel(lon=slice(120, 300))
    _, ipsl_eof = share.calc_eof(ipsl_crop, 'tos', n_pc=2, exclude_land=False,
                                 plot=False, region='equator', detrend=True)

    ipsl_eof = share.rotate_enso_eof(ipsl_eof)
    ipsl_eof['nino_3.4'] = calc_nino_anom(ipsl_sst)
    
    # CESM
    cesm_sst = combine_files('CMIP6/NCAR-CESM2/sst/').\
        sel(time=slice(start_date, end_date))  
    cesm_cll = combine_files('CMIP6/NCAR-CESM2/cllcalipso/').\
        sel(time=slice(start_date, end_date))  
    
    cesm_cll = deseasonalize(cesm_cll)
    cesm_sst = deseasonalize(cesm_sst)    
    
    cesm_crop = cesm_sst.sel(lon=slice(120, 300))
    _, cesm_eof = share.calc_eof(cesm_crop, 'tos', n_pc=2, exclude_land=False,
                              plot=False, region='equator', detrend=True)

    cesm_eof = share.rotate_enso_eof(cesm_eof)
    cesm_eof['nino_3.4'] = calc_nino_anom(cesm_sst)
     
    
    # CanESM6
    canesm_sst = combine_files('CMIP6/CanESM5/sst_clean/').\
        sel(time=slice(start_date, end_date))  
    canesm_cll = combine_files('CMIP6/CanESM5/cllcalipso/').\
        sel(time=slice(start_date, end_date))  

    canesm_cll = deseasonalize(canesm_cll)
    canesm_sst = deseasonalize(canesm_sst)

    canesm_crop = canesm_sst.sel(lon=slice(120, 300))
    _, canesm_eof = share.calc_eof(canesm_crop, 'tos', n_pc=2, exclude_land=False,
                              plot=False, region='equator', detrend=True)
    canesm_eof['nino_3.4'] = calc_nino_anom(canesm_sst)
    canesm_eof = share.rotate_enso_eof(canesm_eof)


    # MRI-ESM
    
    mri_sst = combine_files('CMIP6/MRI-ESM2/sst/').\
        sel(time=slice(start_date, end_date)) 
    mri_cll = combine_files('CMIP6/MRI-ESM2/cllcalipso/').\
        sel(time=slice(start_date, end_date)) 
        
    mri_sst = deseasonalize(mri_sst)
    mri_cll = deseasonalize(mri_cll)
    
    mri_crop = mri_sst.sel(lon=slice(120, 300))
    _, mri_eof = share.calc_eof(mri_crop, 'tos', n_pc=2, exclude_land=False,
                              plot=False, region='equator', detrend=True)
    mri_eof['PC2'] *= -1
    mri_eof = share.rotate_enso_eof(mri_eof)
    mri_eof['nino_3.4'] = calc_nino_anom(mri_sst)
    
    # UKESM- broken for some regridding issue
    
    if True:
        ukesm_sst = combine_files('CMIP6/UKESM/sst_clean/').\
            sel(time=slice(start_date, end_date)) 
        ukesm_cll = combine_files('CMIP6/UKESM/cllcalipso/').\
            sel(time=slice(start_date, end_date)) 
            
        ukesm_sst = deseasonalize(ukesm_sst)
        ukesm_cll = deseasonalize(ukesm_cll)
        
        ukesm_crop = ukesm_sst.sel(lon=slice(120, 300))
        _, ukesm_eof = share.calc_eof(ukesm_crop, 'tos', n_pc=2, exclude_land=False,
                                  plot=False, region='equator', detrend=True)
        # mri_eof['PC2'] *= -1
        ukesm_eof = share.rotate_enso_eof(ukesm_eof)
    
  

    # Steps:
    # Combine files for my sanity
    # restrict to ISCCP period
    # remove annual cycle
    # calculate SST PCs and rotate
    # 
    global data, data2, to_corr
    if True:
        data = [gfdl_cll, e3sm_cll, ipsl_cll, 
                cesm_cll, canesm_cll, mri_cll]
        data2 = [gfdl_sst, e3sm_sst, ipsl_sst, 
                cesm_sst, canesm_sst, mri_sst]
        
        to_corr = [gfdl_eof, e3sm_eof, ipsl_eof, 
                   cesm_eof, canesm_eof, mri_eof]
        var = 'cllcalipso'
        titles= ['NOAA-GFDL-CM4', 'DOE-E3SM', 'IPSL-CM6A',
                 'NCAR-CESM2', 'CanESM5', 'MRI-ESM2']
        types = ['E Index', 'C Index', 'Niño 3.4']
        
        plot_gcm_corr_subplot(data, to_corr, var, titles, types)
        


if __name__ == '__main__':
    main()

