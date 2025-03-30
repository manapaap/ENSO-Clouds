# -*- coding: utf-8 -*-
"""
Composite creation for EP/CP El Ninos

in development file
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import pandas as pd

os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
import obs_scripts.shared_funcs as share


def assign_loc(lst):
    """
    Assigns central/eastern/mixed based on occurence during the 4-month
    chunk we are observing
    
    If central/eastern are >=2, and the other two are split, it is considered
    a central/eastern event
    
    If mixed >= 2, it is a mixed event. Or, if central and eastern both == 2
    
    This is kind of a "voting" system if you think about it
    """
    count_cent = lst.count('Central')
    count_east = lst.count('Eastern')
    count_mix = lst.count('Mixed')
    
    if count_cent >= 2 and count_east < 2:
        return 'Central'
    elif count_east >= 2 and count_cent < 2:
        return 'Eastern'
    elif count_cent == count_east == 2:
        return 'Mixed'
    else:
        return 'Mixed'
    

def assign_state(lst):
    """
    Assigns El Nino/ La Nina/ Neutral based on occurence during the 4-month
    chunk we are observing
    
    If El Nino/La Nina are >=2, and the other two are split, it is considered
    a central/eastern event
    
    If Neutral >= 2, it is a neutral event. Or, if central and eastern both == 2
    """
    count_neu = lst.count('Neutral')
    count_la = lst.count('La Nina')
    count_el = lst.count('El Nino')
    
    if count_la >= 2 and count_el < 2:
        return 'La Nina'
    elif count_el >= 2 and count_la < 2:
        return 'El Nino'
    elif count_el == count_la == 2:
        return 'Neutral'
    else:
        return 'Neutral'


def isolate_events(xr_ds, ep_cp, state='El Nino', loc='Central'):
    """
    Uses categorized El Nino/La Nina and Central/Eastern events from ep_cp 
    to isolate events from the xr_ds (intended to be era5/ceres/isccp) and
    return the climatological pattern of the same- i.e. the mean evolution
    of an event in question
    
    Going back to a single mean value to see if results are better there...
    """
    if loc == 'all':
        rel_years = ep_cp.query(f'state == "{state}"')
    else:
        rel_years = ep_cp.query(f'state == "{state}"').query(f'loc == "{loc}"')
    rel_data = xr_ds.sel(time=rel_years.time.values)
    clim = rel_data.mean(dim='time')
    return clim


def plot_comp_dict(comp_dict, var, ep_cp, cbar_lab='%', name=None):
    """
    Loops over the composite dictionaries to create four plots for 
    our variable in cp/ep/mixed/nina conditions
    """
    if name is None:
        name = var
    n_cp = len(ep_cp.query('state == "El Nino"').query('loc == "Central"'))
    n_ep = len(ep_cp.query('state == "El Nino"').query('loc == "Eastern"'))
    n_nina = len(ep_cp.query('state == "La Nina"'))
    # Match the keys to dict
    names = {'ep_nino': f'{name} during Eastern Pacific El Nino, N = {int(n_ep / 4)}',
             'cp_nino': f'{name} during Central Pacific El Nino, N = {int(n_cp / 4)}',
             'nina': f'{name} during La Nina, N = {int(n_nina / 4)}'}
    
    for key, value in comp_dict.items():
        share.plot_scalar_field(value[var], title=names[key],
                                lims=share.pac_domain, cbar_lab=cbar_lab)


def main():
    global ep_cp, era5_states, isccp_states, ebaf_states, ep_cp_2
    # Begin with ISCCP datas
    oni_idx = share.load_oni_idx(fpath='misc_data/oni_index.txt')
    oni_rel = oni_idx.query('"1983-07" <= time <= "2017-06"').reset_index(drop=True)
    # using data from isccp date range
    pc_enso = pd.read_csv('misc_data/enso_pcs_isccp.csv')
    ep_cp = pc_enso.copy().reset_index(drop=True)
    ep_cp['oni'] = oni_rel['oni']
    ep_cp['C_E_ratio'] = np.abs(ep_cp['C'] / ep_cp['E'])
    ep_cp = ep_cp.query('month >= 11 or month <= 2')
    # Assign state
    ep_cp['state'] = 'Neutral'
    ep_cp['state'][ep_cp.oni >= 0.5] = 'El Nino'
    ep_cp['state'][ep_cp.oni <= -0.5] = 'La Nina'
    # Group by spatial expression
    ep_cp['loc'] = 'Eastern'
    ep_cp['loc'][ep_cp.C_E_ratio >= 0.9] = 'Mixed'
    ep_cp['loc'][ep_cp.C_E_ratio >= 1.4] = 'Central'

    # I want to tidy this and ensure the categories are consistent per event
    # Step 1: Define event groups Proceed with assuming we have 4 month groups
    breaks = ep_cp['year'].diff()
    # align with the actual year cutoffs
    breaks[:-2] = breaks[2:]
    breaks[-2:-1] = 0
    ep_cp['event_id'] = breaks.cumsum()
    # Any events with mixed in the middle are classed as mixed
    ep_cp['loc'] = ep_cp.groupby('event_id')['loc'].\
            transform(lambda x: assign_loc(list(x.values)))
    # Analogous thing for actual enso state
    ep_cp['state'] = ep_cp.groupby('event_id')['state'].\
            transform(lambda x: assign_state(list(x.values)))
    
    # Let's load in the ERA5 and ISCCP files now
    isccp_anom = xr.load_dataset('era5_reanal/timeseries/isccp_anom.nc')
    era5_data = xr.load_dataset('era5_all/timeseries/era5_anom_all.nc')
    era5_iscc = era5_data.sel({'time': isccp_anom.time})
    
    # we now want to isolate sections of time within isccp/era5 to 
    # create composites of central/eastern pacific El Ninos
    era5_states = {}
    isccp_states = dict()
    # ERA5
    era5_states['ep_nino'] = isolate_events(era5_iscc, ep_cp, 
                                  state='El Nino', loc='Eastern')
    era5_states['cp_nino'] = isolate_events(era5_iscc, ep_cp, 
                                  state='El Nino', loc='Central')
    era5_states['nina'] = isolate_events(era5_iscc, ep_cp, 
                                         state='La Nina', loc='all')
    # ISCCP
    isccp_states['ep_nino'] = isolate_events(isccp_anom, ep_cp, 
                                  state='El Nino', loc='Eastern')
    isccp_states['cp_nino'] = isolate_events(isccp_anom, ep_cp, 
                                  state='El Nino', loc='Central')
    isccp_states['nina'] = isolate_events(isccp_anom, ep_cp, state='La Nina', 
                                          loc='all')
    # Some demonstration plots
    plot_comp_dict(isccp_states, 'stratus', ep_cp, cbar_lab='%')
    plot_comp_dict(era5_states, 'sst', ep_cp, cbar_lab='K')

    # Let's do the same for radiation to get lw/sw
        
    oni_rel = oni_idx.query('"2000-03" <= time <= "2024-07"').reset_index(drop=True)
    pc_enso = pd.read_csv('misc_data/enso_pcs.csv')
    # Hack
    pc_enso['time'] = oni_rel.time
    ep_cp_2 = pc_enso.copy().reset_index(drop=True)
    ep_cp_2['oni'] = oni_rel['oni']
    ep_cp_2['C_E_ratio'] = np.abs(ep_cp_2['C'] / ep_cp_2['E'])
    ep_cp_2 = ep_cp_2.query('month >= 11 or month <= 2')
    # Assign state
    ep_cp_2['state'] = 'Neutral'
    ep_cp_2['state'][ep_cp_2.oni >= 0.5] = 'El Nino'
    ep_cp_2['state'][ep_cp_2.oni <= -0.5] = 'La Nina'
    # Group by spatial expression
    ep_cp_2['loc'] = 'Eastern'
    ep_cp_2['loc'][ep_cp_2.C_E_ratio >= 0.9] = 'Mixed'
    ep_cp_2['loc'][ep_cp_2.C_E_ratio >= 1.4] = 'Central'
    # Tidy
    breaks = ep_cp_2['year'].diff()
    # align with the actual year cutoffs
    breaks[:-2] = breaks[2:]
    breaks[-2:-1] = 0
    ep_cp_2['event_id'] = breaks.cumsum()
    # Any events with mixed in the middle are classed as mixed
    ep_cp_2['loc'] = ep_cp_2.groupby('event_id')['loc'].\
            transform(lambda x: assign_loc(list(x.values)))
    # Analogous thing for actual enso state
    ep_cp_2['state'] = ep_cp_2.groupby('event_id')['state'].\
            transform(lambda x: assign_state(list(x.values)))
    
    # CERES Data!
    ebaf_anom = xr.open_dataset('era5_reanal/timeseries/ceres_ebaf.nc')
    ebaf_states = {}
    
    ebaf_states['ep_nino'] = isolate_events(ebaf_anom, ep_cp_2, 
                                  state='El Nino', loc='Eastern')
    ebaf_states['cp_nino'] = isolate_events(ebaf_anom, ep_cp_2, 
                                  state='El Nino', loc='Central')
    ebaf_states['nina'] = isolate_events(ebaf_anom, ep_cp_2, 
                                         state='La Nina', loc='all')
    
    

if __name__ == '__main__':
    main()



