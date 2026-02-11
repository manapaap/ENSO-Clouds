# -*- coding: utf-8 -*-
"""
Explain LCC Anomalies via composites and regression
"""

import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import linregress
from os import chdir
from scipy.stats import t
import matplotlib.pyplot as plt

chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
import obs_scripts.shared_funcs as share


def plot_contributions(lcc_explained, lcc_change):
    """
    Plots the actual change in low cloud cover over the SEP,
    along with the contributions from each statistically significant
    cloud controlling factor
    """
    lcc_explained = lcc_explained.drop('rel_error', axis=1)
    lcc_merge = pd.concat([lcc_change, lcc_explained])
    labels = ['Observed\nSc + St', 'Predicted\nSc + St', 'EIS', 
              '$\\mathrm{RH}_{700}$', 'Cirrus', 'Cold\nAdvection']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Define colors: obs and prediction in darker blue, CCFs in lighter blue
    colors = ['#1f77b4', '#1f77b4'] + ['#6baed6'] * (len(lcc_merge) - 2)
    colors[0] = '#08519c'  # Make observed even darker for emphasis
    bars = ax.bar(range(len(lcc_merge)), lcc_merge.change, 
                   yerr=lcc_merge.error,
                   color=colors,
                   edgecolor='black',
                   linewidth=1.2,
                   capsize=5,
                   error_kw={'linewidth': 1.5, 'ecolor': 'black'})
    # Set x-axis labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    # Add horizontal line at zero
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    # Labels and styling
    ax.set_ylabel('Change in SEP Sc + St (%)', fontsize=12)
    ax.set_ylim(bottom=min(lcc_merge.change.min() - 1, -1))
    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    # After setting xticklabels, add a vertical divider
    ax.axvline(1.5, color='gray', linewidth=2, linestyle='-', alpha=0.4)
    
    # Add labels above the two sections
    ax.text(0.5, -0.95, 'Total Change', 
            ha='center', fontsize=10, color='#08519c')
    ax.text(3.5, -0.95, 'Contribution from each CCF', 
            ha='center', fontsize=10, color='#1f77b4')
    plt.tight_layout()
    return fig, ax
    
    
def main():
    global all_cp, lcc_explained, lcc_change
    # load files- ccf dataframe
    ccf_data = pd.read_csv('misc_data/ccf_timeseries.csv')
    ccf_slopes = pd.read_csv('misc_data/ccf_slopes.csv', index_col=0)
    # load files- composite maps
    ccf_comp = xr.load_dataset('misc_data/composites/era5_cp.nc').\
        drop_vars('pressure_level')
    cloud_comp = xr.load_dataset('misc_data/composites/cirrus_cp.nc')
    
    cp_lcc = share.isolate_ep_isccp(cloud_comp, 'sc_adj')
    cp_hcc = share.isolate_ep_isccp(cloud_comp, 'high')
    # error
    cp_lcc_std = share.isolate_ep_isccp(cloud_comp, 'sc_adj', mode='std')
    cp_hcc_std = share.isolate_ep_isccp(cloud_comp, 'high', mode='std')
    
    cp_eis = share.isolate_ep_era5(ccf_comp, 'eis')
    cp_rh700 = share.isolate_ep_era5(ccf_comp, 'rh_700')
    cp_sstadv = share.isolate_ep_era5(ccf_comp, 'cold_adv')
    cp_sst = share.isolate_ep_era5(ccf_comp, 'sst')
    cp_w700 = share.isolate_ep_era5(ccf_comp, 'w_700')
    cp_ws10 = share.isolate_ep_era5(ccf_comp, 'speed')
    # get associated error
    cp_eis_std = share.isolate_ep_era5(ccf_comp, 'eis', mode='std')
    cp_rh700_std = share.isolate_ep_era5(ccf_comp, 'rh_700', mode='std')
    cp_sstadv_std = share.isolate_ep_era5(ccf_comp, 'cold_adv', mode='std')
    cp_sst_std = share.isolate_ep_era5(ccf_comp, 'sst', mode='std')
    cp_w700_std = share.isolate_ep_era5(ccf_comp, 'w_700', mode='std')
    cp_ws10_std = share.isolate_ep_era5(ccf_comp, 'speed', mode='std')
    
    ccf_cp = {'sc_adj': float(cp_lcc),
              'EIS': float(cp_eis),
              '700 hPa Relative Humidity': float(cp_rh700),
              'Cirrus Fraction': float(cp_hcc),
              'Cold Advection': float(cp_sstadv)}
    ccf_cp_std = {'sc_adj': float(cp_lcc_std),
                  'EIS': float(cp_eis_std),
                  '700 hPa Relative Humidity': float(cp_rh700_std),
                  'Cirrus Fraction': float(cp_hcc_std),
                  'Cold Advection': float(cp_sstadv_std)}
    # get this organized like the slopes
    ccf_order = pd.DataFrame([ccf_cp, ccf_cp_std],
                             index=['change', 'error']).T.drop('sc_adj')
    # 95% ci
    ccf_order['error'] *= t.ppf(1 - (1 - 0.95)/ 2, 100) / np.sqrt(100 - 1)
    # same for slopes
    sig_slopes = ccf_slopes[['params', 'error']][ccf_slopes.p_adj <= 0.05]
    # explain final lcc
    lcc_explained = sig_slopes['params'] * ccf_order['change']
    lcc_explained = pd.DataFrame({'change': lcc_explained})
    # get error in prediction
    sig_slopes['rel_error'] = sig_slopes['error'] / abs(sig_slopes['params'])
    ccf_order['rel_error'] = ccf_order['error'] / abs(ccf_order['change'])
    # add relative errors in quadrature
    lcc_explained['rel_error'] = np.sqrt(sig_slopes['rel_error']**2 +\
                                         ccf_order['rel_error']**2)
    lcc_explained['error'] = abs(lcc_explained['change']) *\
        lcc_explained['rel_error']
    # do this for low cloud cover
    lcc_change = pd.DataFrame([ccf_cp['sc_adj'], ccf_cp_std['sc_adj']],
                              index=['change', 'error'], columns=['obs']).T
    lcc_change['error'] *= t.ppf(1 - (1 - 0.95)/ 2, 100) / np.sqrt(100 - 1)
    # compare to error from all other ccfs
    net_change = lcc_explained['change'].sum()
    error = np.sqrt((lcc_explained['error']**2).sum())
    lcc_change.loc['lcc_pred'] = {'change': net_change,
                                  'error': error}
    
    plot_contributions(lcc_explained, lcc_change)
    
    if False:
        # repeat with all points- this is also worse somehow
        all_cp = {'sc_adj': float(cp_lcc),
                  'SST': float(cp_sst),
                  'EIS': float(cp_eis),
                  '700 hPa Relative Humidity': float(cp_rh700),
                  'Cirrus Fraction': float(cp_hcc),
                  'Cold Advection': float(cp_sstadv),
                  '10m Windspeed': float(cp_ws10),
                  '700 hPa Subsidence': float(cp_w700)}
        all_order = pd.DataFrame(all_cp, index=['change']).T.drop('sc_adj')
        lcc_exp_all = ccf_slopes['params'] * all_order['change']
    
    if False:
        # repeat with data from the whole SEP
        # sadly, caused worse results
        sep_slopes = xr.load_dataset('misc_data/ccf_corr.nc')
        domain = share.ep_domain_360
        ccf_comp['lon'] = (ccf_comp.lon + 360) % 360
        sep_ccf_comp = ccf_comp.sel(lat=slice(*domain[:2][::-1]),
                                    lon=slice(*domain[2:])).\
                        isel(lat=slice(1, None), lon=slice(1, None)).\
                        coarsen(lat=4, lon=4, boundary='trim').mean().\
                        sortby('lat')
        sep_cloud_comp = cloud_comp.sel(lat=slice(*domain[:2]),
                                        lon=slice(*domain[2:]))
        # fix dim offset
        sep_ccf_comp['lat'] = sep_cloud_comp['lat']
        sep_ccf_comp['lon'] = sep_cloud_comp['lon']
        # predict low cloud?
        sep_ccf_comp['high'] = sep_cloud_comp['high']
        cloud_grid = sep_ccf_comp * sep_slopes
        sep_cloud_comp['sc_pred'] = cloud_grid.to_array().sum('variable')
        sep_cloud_comp['diff'] = sep_cloud_comp['sc_adj'] - sep_cloud_comp['sc_pred']

if __name__ == '__main__':
    main()
