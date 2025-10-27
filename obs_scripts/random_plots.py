# -*- coding: utf-8 -*-
"""
random misc. plots that are secondary to the primary analysis
(ex. boxes around regions of intrest)
"""


import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import xarray as xr
from string import ascii_lowercase
from matplotlib.colors import TwoSlopeNorm


os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
import obs_scripts.shared_funcs as share


def plot_box(lims=share.pac_domain):
    """
    Contour plot of a scalar field (e.g., hcc) across the globe or a specified region.
    
    Slope_val determines percentile bounds for slope plot
    """
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 5))
    ax.set_global()
    rect1 = patches.Rectangle((190, -5), 50, 10, linewidth=2,
                             edgecolor='black', facecolor='lightcoral',
                             label='Niño 3.4', alpha=0.8,
                             transform=ccrs.PlateCarree(), zorder=10)
    ax.add_patch(rect1)
    rect2 = patches.Rectangle((240, -20), 40, 20, linewidth=2,
                             edgecolor='black', facecolor='lightsteelblue',
                             label='Southeast Pacific', alpha=0.8,
                             transform=ccrs.PlateCarree(), zorder=10)
    ax.add_patch(rect2)
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                 linestyle='dashed', alpha=0.75, zorder=1)
    ax.legend()
    
    # Set plot limits if specified
    if lims is not None and len(lims) == 4:
        ax.set_ylim(lims[0], lims[1])
        ax.set_xlim(lims[3], lims[2])
    if lims is not None and len(lims) == 2:
        # Set to tropics still
        ax.set_ylim(lims[0], lims[1])
    plt.show()


def plot_gcm(lims=share.pac_domain, levels=5):
    """
    Publication panel corr of the Historical and AMIP experiments
    """
    cesm_amip = xr.open_dataset('misc_data/correlations/cesm_amip.nc').fillna(0)
    ipsl_amip = xr.open_dataset('misc_data/correlations/ipsl_amip.nc').fillna(0)
    miroc_amip = xr.open_dataset('misc_data/correlations/miroc_amip.nc').fillna(0)
    amip = [cesm_amip, ipsl_amip, miroc_amip]
    cesm_hist = xr.open_dataset('misc_data/correlations/cesm_hist.nc').fillna(0)
    ipsl_hist = xr.open_dataset('misc_data/correlations/ipsl_hist.nc').fillna(0)
    miroc_hist = xr.open_dataset('misc_data/correlations/miroc_hist.nc').fillna(0)
    hist = [cesm_hist, ipsl_hist, miroc_hist]
    
    names = ['NCAR-CESM2', 'IPSL-CM6A', 'MIROC6']
    
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, 
                        dpi=600, subplot_kw={'projection': proj},
                        figsize=(10, 6))  # or adjust as needed)
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.475, bottom=0.01)
    i = 0 
    for col in range(3):
        amip_data = amip[col]['__xarray_dataarray_variable__']
        hist_data = hist[col]['__xarray_dataarray_variable__']
        lon_hist = hist_data.lon.values
        lat_hist = hist_data.lat.values
        lon_amip = amip_data.lon.values
        lat_amip = amip_data.lat.values
        # Non sig value mask
        nonsig_amip = np.zeros(amip_data.shape)
        nonsig_amip[np.isnan(amip_data)] = 1
        nonsig_hist = np.zeros(hist_data.shape)
        nonsig_hist[np.isnan(hist_data)] = 1
        # Create meshgrid if needed
        lon2d_hist, lat2d_hist = np.meshgrid(lon_hist, lat_hist)
        lon2d_amip, lat2d_amip = np.meshgrid(lon_amip, lat_amip)
        # Determine the color limits to center around zero
        vmin_hist, vmax_hist = np.percentile(hist_data.values, [0.1, 99.9])
        vmin_amip, vmax_amip = np.percentile(amip_data.values, [0.1, 99.9])
        norm_hist = TwoSlopeNorm(vmin=vmin_hist, vcenter=0, vmax=vmax_hist)
        norm_amip = TwoSlopeNorm(vmin=vmin_amip, vcenter=0, vmax=vmax_amip)
        # Begin plotting
        axs[0, col].set_global()
        axs[0, col].set_title(ascii_lowercase[i] + ') ' + 
                                names[col] + ' Historical Sc + St ' + 'and C',
                                fontsize='x-small')
        axs[1, col].set_title(ascii_lowercase[i + 3] + ') ' + 
                                names[col] + ' AMIP Sc + St ' + 'and C',
                                fontsize='x-small')
        i += 1
        # masked_data = np.ma.masked_invalid(corr_field.data)
        # Use pcolormesh with centered color limits around zero
        pcm_hist = axs[0, col].pcolormesh(lon2d_hist, lat2d_hist, hist_data.data,
                                       transform=ccrs.PlateCarree(),
                                       shading='auto', cmap='RdBu_r', 
                                       norm=norm_hist)
        pcm2_hist = axs[0, col].pcolormesh(lon2d_hist, lat2d_hist, nonsig_hist.data,
                                        transform=ccrs.PlateCarree(),
                            shading='auto', cmap='Greys', alpha=0.1)
        
        pcm_amip = axs[1, col].pcolormesh(lon2d_amip, lat2d_amip, 
                                          amip_data.data,
                                       transform=ccrs.PlateCarree(),
                                       shading='auto', cmap='RdBu_r', 
                                       norm=norm_amip)
        pcm2_amip = axs[1, col].pcolormesh(lon2d_amip, lat2d_amip, nonsig_amip.data,
                                        transform=ccrs.PlateCarree(),
                            shading='auto', cmap='Greys', alpha=0.1)
        
        tiers_hist = np.linspace(vmin_hist, vmax_hist, levels)
        lon1d_hist = np.asarray(lon2d_hist).reshape(-1)
        lat1d_hist = np.asarray(lat2d_hist).reshape(-1)
        lon1d_amip = np.asarray(lon2d_amip).reshape(-1)
        lat1d_amip = np.asarray(lat2d_amip).reshape(-1)
        corr1d_hist = np.asarray(hist_data).reshape(-1)
        contour_hist = axs[0, col].tricontour(lon1d_hist, lat1d_hist, corr1d_hist,
                                              levels=tiers_hist, 
                                colors='black', linewidths=0.8, 
                              transform=ccrs.PlateCarree())
        axs[0, col].clabel(contour_hist, inline=True, fontsize=4,
                 fmt="%.1f", inline_spacing=5)
        tiers_amip = np.linspace(vmin_amip, vmax_amip, levels)
        corr1d_amip = np.asarray(amip_data).reshape(-1)
        contour_amip = axs[1, col].tricontour(lon1d_amip, lat1d_amip, corr1d_amip,
                                              levels=tiers_amip, 
                                colors='black', linewidths=0.8, 
                              transform=ccrs.PlateCarree())
        axs[1, col].clabel(contour_amip, inline=True, fontsize=4,
                 fmt="%.1f", inline_spacing=5)
        # Add coastlines and gridlines
        axs[0, col].coastlines()
        axs[1, col].coastlines()
        gl = axs[0, col].gridlines(draw_labels=False, dms=True,
                                     alpha=0.5)
        gl2 = axs[1, col].gridlines(draw_labels=False, dms=True,
                                     alpha=0.5)

        gl2.bottom_labels = True
        gl2.xlabel_style = {'size': 8}
        if col == 0:
            gl.left_labels = True
            gl.ylabel_style = {'size': 8}
            gl2.left_labels = True
            gl2.ylabel_style = {'size': 8}
        
        # Add colorbar and label
        cbar_hist = plt.colorbar(pcm_hist, ax=axs[0, col], location='right',
                            pad=0.02, shrink=0.75, aspect=20, ticks=tiers_hist)
        cbar_amip = plt.colorbar(pcm_amip, ax=axs[1, col], location='right',
                            pad=0.02, shrink=0.75, aspect=20, ticks=tiers_hist)
        cbar_hist.ax.set_yticklabels(['{:.1f}'.format(x) for x in
                                      cbar_hist.get_ticks()])
        cbar_amip.ax.set_yticklabels(['{:.1f}'.format(x) for x in
                                      cbar_amip.get_ticks()])
        cbar_hist.ax.tick_params(labelsize=8)
        cbar_amip.ax.tick_params(labelsize=8)
        # Set plot limits if specified
        if lims is not None and len(lims) == 4:
            axs[0, col].set_ylim(lims[0], lims[1])
            axs[0, col].set_xlim(lims[3], lims[2])
            axs[1, col].set_ylim(lims[0], lims[1])
            axs[1, col].set_xlim(lims[3], lims[2])
        if lims is not None and len(lims) == 2:
            # Set to tropics still
            axs[0, col].set_ylim(lims[0], lims[1])
    

def main():
    plt.rcParams['figure.dpi'] = 600
    plot_box()
    plot_gcm()

if __name__ == '__main__':
    main()

