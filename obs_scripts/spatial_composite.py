# -*- coding: utf-8 -*-
"""
Composite creation for EP/CP El Ninos

in development file
"""

import numpy as np
import xarray as xr
import os
import pandas as pd
import copy


os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
import obs_scripts.shared_funcs as share


def assign_loc(lst, cut_cent, cut_east):
    """
    Assigns central/eastern/mixed based on occurence during the 4-month
    chunk we are observing
    """
    mean_loc = np.mean(lst[1:])
    
    if mean_loc >= cut_cent:
        return 'Central'
    elif mean_loc < cut_east:
        return 'Eastern'
    else:
        return 'Mixed'
    

def assign_state(lst, cut_nino=0.5, cut_nina=-0.5):
    """
    Assigns El Nino/ La Nina/ Neutral based on occurence during the 4-month
    chunk we are observing
    """
    mean_state = np.mean(lst)
    
    if mean_state >= cut_nino:
        return 'El Nino'
    elif mean_state < cut_nina:
        return 'La Nina'
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


def plot_comp_dict(comp_dict, var, ep_cp, cbar_lab='%', name=None, levels=4,
                   lims=share.pac_domain):
    """
    Loops over the composite dictionaries to create four plots for 
    our variable in cp/ep/mixed/nina conditions
    """
    if name is None:
        name = var
    n_cp = len(ep_cp.query('state == "El Nino"').query('loc == "Central"'))
    n_ep = len(ep_cp.query('state == "El Nino"').query('loc == "Eastern"'))
    n_mix = len(ep_cp.query('state == "El Nino"').query('loc == "Mixed"'))
    n_nina = len(ep_cp.query('state == "La Nina"'))
    # Match the keys to dict
    names = {'ep_nino': f'{name} during Eastern Pacific El Nino, N = {int(n_ep / 4)}',
             'cp_nino': f'{name} during Central Pacific El Nino, N = {int(n_cp / 4)}',
             'mix_nino': f'{name} during Mixed El Nino, N = {int(n_mix / 4)}',
             'nina': f'{name} during La Nina, N = {int(n_nina / 4)}'}
    
    for key, value in comp_dict.items():
        share.plot_scalar_field(value[var], title=names[key], levels=levels,
                                lims=lims, cbar_lab=cbar_lab)

def t_change(forcing):
    """
    Integrates forcing over 4 months to show inferred ocean temp. change
    """
    return (4 * 30 * 24 * 60*60)*(forcing / (50* 1000 * 4184))
    

def main():
    global ep_cp, era5_states, isccp_states, ebaf_states, events_all
    global ep_cp_2, anom_rad, anom_cloud, anom_met, test, test2, test_all
    # Define cutofs
    cut_cp = 2.0
    cut_ep = 2.0
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
    test = ep_cp.copy()
    ep_cp['loc'] = ep_cp.groupby('event_id')['C_E_ratio'].\
            transform(lambda x: assign_loc(x,
                                           cut_cent=cut_cp,
                                           cut_east=cut_ep))
    # Analogous thing for actual enso state
    ep_cp['state'] = ep_cp.groupby('event_id')['oni'].\
            transform(lambda x: assign_state(list(x.values)))
    
    # Let's load in the ERA5 and ISCCP files now
    # date filtering??    
    
    isccp_anom = xr.load_dataset('era5_reanal/timeseries/isccp_anom.nc')
    era5_data = xr.load_dataset('era5_all/timeseries/era5_anom_all.nc')
    era5_iscc = era5_data.sel({'time': isccp_anom.time})
    # Get into per day units
    era5_iscc['cold_adv_day'] = 3600 * 24 * era5_iscc['cold_adv_smooth']
    era5_iscc['w_700_cm'] = 100 * era5_iscc['w_700']
    
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
    plot_comp_dict(isccp_states, 'sc_adj', ep_cp, cbar_lab='%')
    plot_comp_dict(copy.deepcopy(era5_states), 'sst', ep_cp, cbar_lab='K')

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
    test2 = ep_cp_2.copy()
    ep_cp_2['loc'] = ep_cp_2.groupby('event_id')['C_E_ratio'].\
            transform(lambda x: assign_loc(x,
                                           cut_cent=cut_cp,
                                           cut_east=cut_ep))
    # Analogous thing for actual enso state
    ep_cp_2['state'] = ep_cp_2.groupby('event_id')['oni'].\
            transform(lambda x: assign_state(list(x.values)))
    
    # CERES Data!
    ebaf_anom = xr.open_dataset('era5_reanal/timeseries/ceres_ebaf.nc')
    ebaf_states = {}
    
    ebaf_states['ep_nino'] = isolate_events(ebaf_anom, ep_cp_2, 
                                  state='El Nino', loc='Eastern')
    ebaf_states['cp_nino'] = isolate_events(ebaf_anom, ep_cp_2, 
                                  state='El Nino', loc='Central')
    # ebaf_states['mix_nino'] = isolate_events(ebaf_anom, ep_cp_2, 
    #                               state='El Nino', loc='Mixed')
    ebaf_states['nina'] = isolate_events(ebaf_anom, ep_cp_2, 
                                         state='La Nina', loc='all')
    
    if True:
        # Big subplot sample call
        data = [isccp_states, isccp_states, ebaf_states]
        titles = ['Sc + St', 'Cirrus', 'Net Radiation']
        types = ['EP El Niño', 'CP El Niño', 'La Niña']
        names = ['sc_adj', 'high', 'sfc_net_tot_all_mon']
        levels = [4, 6, 5]
        cbar_lab = ['%', '%', 'W/m²']
        
        share.plot_scalar_subplot(data=data, titles=titles, types=types, 
                                  names=names, cbar_lab=cbar_lab, top=0.35,
                                  levels=levels)
    if True:
        # Big subplot sample call
        data = [era5_states] * 6
        titles = ['SST', 'EIS', '10m WS', 'ω$_{700}$', 
                  'RH$_{700}$', 'Cold Adv.']
        types = ['EP El Niño', 'CP El Niño', 'La Niña']
        names = ['sst', 'eis', 'speed', 'w_700_cm', 'rh_700', 'cold_adv_day']
        levels = [5, 4, 6, 5, 6, 5, 5]
        cbar_lab = ['K', 'K', 'm/s', 'cm/s', '%', 'K/day']
        
        share.plot_scalar_subplot(data=data, titles=titles, types=types, 
                                  top=0.7,
                                  names=names, cbar_lab=cbar_lab, levels=levels)
    if True:
        # Big subplot sample call
        data = [ebaf_states] * 4
        titles = ['SW TOA', 
                  'LW TOA', 'SW SFC', 'LW SFC']
        types = ['EP El Niño', 'CP El Niño', 'La Niña']
        names = ['toa_sw_all_mon', 'toa_lw_all_mon',
                 'sfc_net_sw_all_mon', 'sfc_net_lw_all_mon']
        levels = [5, 6, 5, 5]
        cbar_lab = ['W/m²', 'W/m²', 'W/m²', 'W/m²']
        
        share.plot_scalar_subplot(data=data, titles=titles, types=types, 
                                  top=0.475,
                                  names=names, cbar_lab=cbar_lab, levels=levels)

    # Let's isolate the SEP region to calculate radiative fluxes
    domain = share.ep_domain_360 
    lat_bounds = domain[:2]
    lon_bounds = domain[2:]
    # Splice the SEP region
    anom_rad = dict()
    anom_cloud = dict()
    anom_met = dict()
    for state, value in ebaf_states.items():
        anom_rad[state] = value.copy().sel(lat=slice(*lat_bounds), 
                                           lon=slice(*lon_bounds))
    for state, value in isccp_states.items():
        anom_cloud[state] = value.copy().sel(lat=slice(*lat_bounds), 
                                           lon=slice(*lon_bounds))
    
    for state, value in era5_states.items():
        value['lon'] = (value.lon + 360) % 360
        anom_met[state] = value.sel(lat=slice(*lat_bounds[::-1]), 
                                    lon=slice(*lon_bounds))
        
    rad_vars = ['toa_net_all_mon', 'sfc_net_tot_all_mon',
                'toa_sw_all_mon', 'sfc_net_sw_all_mon',
                'toa_lw_all_mon', 'sfc_net_lw_all_mon']
    rad_info = []
    phase = 'cp_nino'
    
    for var in rad_vars:
        rad_info.append(anom_rad[phase][var].mean(dim=['lat', 'lon']))
    for var, info in zip(rad_vars, rad_info):
        print(f'Mean {var} during {phase} over SEP: {info:.2f}')
    
    mean_cloud = anom_cloud['cp_nino']['sc_adj'].mean(dim=['lat', 'lon'])
    print(f'Mean LCC change during CP El Nino over SEP: {mean_cloud:.2f}')
    
    mean_cloud = anom_cloud['ep_nino']['sc_adj'].mean(dim=['lat', 'lon'])
    print(f'Mean LCC change during EP El Nino over SEP: {mean_cloud:.2f}')
    
    mean_cloud = anom_cloud['nina']['sc_adj'].mean(dim=['lat', 'lon'])
    print(f'Mean LCC change during La Nina over SEP: {mean_cloud:.2f}')
    
    mean_cirrus = anom_cloud[phase]['high'].mean(dim=['lat', 'lon'])
    print(f'Mean Cirrus change during {phase} over SEP: {mean_cirrus:.2f}')
    
    mean_sst = anom_met[phase]['sst'].mean(dim=['lat', 'lon'])
    print(f'Mean SST change during {phase} over SEP: {mean_sst:.2f}')


    # Isolating individual events for publication
    events = ep_cp.query('month == 12').\
        reset_index(drop=True)[['year', 'oni','C_E_ratio', 'state', 'loc']]
    
    events2 = ep_cp_2.query('month == 12').\
        reset_index(drop=True)[['year', 'oni','C_E_ratio', 'state', 'loc']]
        
    events_all = pd.concat([events, events2]).\
        drop_duplicates(subset=['year']).\
            reset_index(drop=True)
    
    # Reindex ceres events
    test2['event_id'] += test['event_id'].iloc[-1]
    
    test_all = pd.concat([test[['year', 'C_E_ratio', 'event_id']],
                          test2[['year', 'C_E_ratio', 'event_id']]]).\
            reset_index(drop=True)
    
    test_all['C_E_mean'] = test_all.groupby('event_id')['C_E_ratio'].\
            transform(lambda x: np.mean(x[1:]))
    test_all = test_all.drop_duplicates(subset=['year']).reset_index(drop=True)
    events_all['C_E_ratio'] = test_all['C_E_ratio']
    events_all.to_csv('misc_data/enso_category.csv', index=False)
    
    # Testing anc crying fro Mike's book classification
    cent_events = [5, 12, 20, 22, 24, 27, 32]
    east_events = [4, 15, 33]
    other_events = [9]
    
    # select certain composite variables to be used in the CCF
    # attribution analysis
    era5_states['cp_nino'][['eis', 'rh_700', 'cold_adv',
                            'sst', 'w_700', 'sst', 'speed']].\
        to_netcdf('misc_data/composites/era5.nc')
    isccp_states['cp_nino'][['high', 'sc_adj']].\
        to_netcdf('misc_data/composites/cirrus.nc')
    
    
if __name__ == '__main__':
    main()

