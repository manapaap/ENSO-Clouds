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


os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
import obs_scripts.shared_funcs as share


def plot_box(region, lims=share.pac_domain):
    """
    Contour plot of a scalar field (e.g., hcc) across the globe or a specified region.
    
    Slope_val determines percentile bounds for slope plot
    """
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 5))
    ax.set_global()
    rect1 = patches.Rectangle((190, -5), 50, 10, linewidth=2,
                             edgecolor='red', facecolor='lightcoral',
                             label='Niño 3.4', alpha=0.8,
                             transform=ccrs.PlateCarree(), zorder=10)
    ax.add_patch(rect1)
    rect2 = patches.Rectangle((240, -20), 40, 20, linewidth=2,
                             edgecolor='blue', facecolor='lightsteelblue',
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


def plot_boxes(lims):
    """
    Plots all the regions we consider in the study
    """
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, axs = plt.subplots(2, 1, subplot_kw={'projection': proj},
                           figsize=(10, 10))
    plt.subplots_adjust(hspace=0.05)
    # Top Box
    ax = axs[0]
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
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                 linestyle='dashed', alpha=0.75, zorder=1)
    gl.bottom_labels = False
    ax.legend(loc='upper center')
    
    # Set plot limits if specified
    if lims is not None and len(lims) == 4:
        ax.set_ylim(lims[0], lims[1])
        ax.set_xlim(lims[3], lims[2])
    if lims is not None and len(lims) == 2:
        # Set to tropics still
        ax.set_ylim(lims[0], lims[1])
    # Bottom boxes
    ax = axs[1]
    ax.set_global()
    rect1 = patches.Rectangle((240, 0), 40, 10, linewidth=2,
                             edgecolor='black', facecolor='peachpuff',
                             label='NEQP', alpha=0.8,
                             transform=ccrs.PlateCarree(), zorder=10)
    ax.add_patch(rect1)
    rect2 = patches.Rectangle((240, -10), 40, 10, linewidth=2,
                             edgecolor='black', facecolor='lightgreen',
                             label='SEQP', alpha=0.8,
                             transform=ccrs.PlateCarree(), zorder=10)
    ax.add_patch(rect2)
    rect3 = patches.Rectangle((240, -20), 40, 10, linewidth=2,
                             edgecolor='black', facecolor='violet',
                             label='LSEP', alpha=0.8,
                             transform=ccrs.PlateCarree(), zorder=10)
    ax.add_patch(rect3)
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                 linestyle='dashed', alpha=0.75, zorder=1)
    gl.top_labels = False
    ax.legend(loc='upper center')
    
    # Set plot limits if specified
    if lims is not None and len(lims) == 4:
        ax.set_ylim(lims[0], lims[1])
        ax.set_xlim(lims[3], lims[2])
    if lims is not None and len(lims) == 2:
        # Set to tropics still
        ax.set_ylim(lims[0], lims[1])
    plt.show()
    
    


def main():
    # plot_box(None)
    # define boxes
    plot_boxes(share.pac_domain)
    


if __name__ == '__main__':
    main()

