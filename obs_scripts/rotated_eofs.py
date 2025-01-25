# -*- coding: utf-8 -*-
"""
Analysis of the Rotated ENSO EOFs as per Ken et al. 2011
"""


import xarray as xrr
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import calendar
import matplotlib.colors as mcolors

os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')

import obs_scripts.vis_clouds as enso
from obs_scripts.divergence import crop_era5
import obs_scripts.cloud_corr as corr


def classify_enso(oni_idx, start=2000, end=2023):
    """
    Classifies years from start to end as El Nino, La Nina, or Neutral
    depending on ENSO state in December
    """
    pass


def add_enso_state(pc_enso, oni_idx):
    """
    Adds tag for "El Nino", "La Nina", or "Neutral" depending on ENSO
    state in December
    
    Classifiation is for May to following April
    """
    pc_enso['enso_state'] = ''
    
    for n, (year, month) in enumerate(zip(pc_enso['year'], pc_enso['month'])):
        # We categorize for previous ENSO state
        if month < 5:
            year -= 1
        date = str(year) + '.12'
        pc_enso.enso_state[n] = enso.is_enso_oni(oni_idx, date)
        
    return pc_enso


def plot_enso_progression(pc_enso, pc_theta, nino_idx, trop_anom, state='El Nino'):
    """
    Plots the progression of ENSO events with respect to indices C, E,
    and Nino 3.4 and FT anomaly. Labels peaks of each
    
    start date of May and ends in April
    """
    # Filter intended data- ENSO PCs
    filt = pc_enso.query(f"enso_state == '{state}'")
    filt = filt.drop('enso_state', axis=1).groupby('month').mean()
    filt['month'] = filt.index
    # Troposphere PC
    filt_t = pc_theta.query(f"enso_state == '{state}'")
    filt_t = filt_t.drop('enso_state', axis=1).groupby('month').mean()
    filt_t['month'] = filt_t.index
    # Nino 3.4 Anomaly
    filt_nin = nino_idx.query(f"enso_state == '{state}'")
    filt_nin = filt_nin.drop('enso_state', axis=1).groupby('month').mean()
    filt_nin['month'] = filt_nin.index  
    # Trooposphere warming
    filt_th = trop_anom.query(f"enso_state == '{state}'")
    filt_th = filt_th.drop('enso_state', axis=1).groupby('month').mean()
    filt_th['month'] = filt_th.index 
    # Isolate peaks- by ENSO state
    if state == 'El Nino':
        fxn = np.max
    else:
        fxn = np.min
    peak_c = int(np.where(filt['C'] == fxn(filt['C']))[0])
    peak_e = int(np.where(filt['E'] == fxn(filt['E']))[0])
    peak_t = int(np.where(filt_t['PC1'] == fxn(filt_t['PC1']))[0])
    peak_nin = int(np.where(filt_nin['3.4_anom'] == fxn(filt_nin['3.4_anom']))[0])
    peak_th = int(np.where(filt_th['theta_700_anom'] == fxn(filt_th['theta_700_anom']))[0])
    
    plt.figure()
    plt.plot(filt.month, filt.E, label=f'E Mode peaks {calendar.month_name[peak_e + 1]}', 
             color='mediumblue')
    plt.plot(filt.month, filt.C, label=f'C Mode peaks {calendar.month_name[peak_c + 1]}',
             color='aquamarine')
    plt.plot(filt_t.month, filt_t['PC1'], 
             label=f'Θ₇₀₀ PC1 peaks {calendar.month_name[peak_t + 1]}',
             color='red', linestyle='dashdot')
    plt.plot(filt_nin.month, filt_nin['3.4_anom'], 
             label=f'Nino 3.4 peaks {calendar.month_name[peak_nin + 1]}',
             color='magenta', linestyle='dotted')
    # plt.plot(filt_th.month, filt_th['theta_700_anom'], 
    #         label=f'Mean Θ₇₀₀ peaks {calendar.month_name[peak_th + 1]}',
    #         color='coral', linestyle='dashed')

    plt.ylabel('Magnitude')
    plt.xlabel('Month')
    plt.title(f'Mean {state} Progression')
    plt.grid()
    plt.legend()
    plt.show()
    
    
def plot_enso_space(pc_enso, years=None):
    """
    Plots the trajectory of mean enso and a specific ENSO year in the ENSO
    phase space of E and C indices, similar to Ken et al
    
    Start: July (circle)
    End: March (cross)
    """
    # Filter intended data- El Nino ENSO PCs
    filt_el = pc_enso.query("enso_state == 'El Nino'")
    filt_el = filt_el.drop('enso_state', axis=1).groupby('month').mean()
    filt_el['month'] = filt_el.index
    # July start
    for col in filt_el.columns:
        filt_el[col] = np.roll(filt_el[col], 6)
    filt_el = filt_el.iloc[:9]
    # La Nina PCs
    filt_la = pc_enso.query("enso_state == 'La Nina'")
    filt_la = filt_la.drop('enso_state', axis=1).groupby('month').mean()
    filt_la['month'] = filt_la.index
    for col in filt_la.columns:
        filt_la[col] = np.roll(filt_la[col], 6)
    # Filter for years we may want
    data = []
    if years:
        for year in years:
            year_of = pc_enso.query(f'year == {year} and month >= 5')
            year_af = pc_enso.query(f'year == {year+1} and month <= 4')
            data.append(pd.concat([year_of, year_af]))
    
    # Plot our ENSO space
    plt.figure()
    plt.plot(filt_el.E, filt_el.C, label='Mean El Nino', color='darkred')
    plt.scatter(filt_el.E.iloc[-1], filt_el.C.iloc[-1], color='darkred',
                marker='x')
    plt.scatter(filt_el.E.iloc[0], filt_el.C.iloc[0], color='darkred',
                marker='o')
    # La Nina
    plt.plot(filt_la.E, filt_la.C, label='Mean La Nina', color='darkblue')
    plt.scatter(filt_la.E.iloc[-1], filt_la.C.iloc[-1], color='darkblue',
                marker='x')
    plt.scatter(filt_la.E.iloc[0], filt_la.C.iloc[0], color='darkblue',
                marker='o')
    # Specific trajectories
    if years:
        colors = list(mcolors.BASE_COLORS)[1:]
        for n, (year, df) in enumerate(zip(years, data)):
            plt.plot(df.E, df.C, label=year, color=colors[n], 
                     linestyle='dashed')
            plt.scatter(df.E.iloc[-1], df.C.iloc[-1], marker='x',
                        color=colors[n])
            plt.scatter(df.E.iloc[0], df.C.iloc[0], marker='o',
                        color=colors[n])
            
    plt.title('ENSO Phase Space Trajectories')
    plt.xlabel('E Mode')
    plt.ylabel('C Mode')
    plt.legend()
    plt.grid()
    plt.show()    
             

def main():
    global pc_enso, era5_anom, pc_theta
    nino_idx = enso.load_nino_idx('misc_data/nino_all.csv').query('2000 <= year <= 2023').reset_index()
    oni_idx = enso.load_oni_idx('misc_data/oni_index.txt')
    era5_anom = xr.open_dataset('era5_reanal/timeseries/era5_anom.nc')
    
    eof, pc_enso = corr.calc_eof(era5_anom, 'sst', n_pc=2)
    eof, pc_theta = corr.calc_eof(era5_anom, 'theta_700', n_pc=1)
    # Categorize EOFs and indices
    pc_enso = corr.rotate_enso_eof(pc_enso)
    pc_enso = add_enso_state(pc_enso, oni_idx)
    pc_theta = add_enso_state(pc_theta, oni_idx)
    nino_idx = add_enso_state(nino_idx, oni_idx)
    # Actual heating anomaly?
    trop_anom = corr.domain_anom(era5_anom, 'theta_700')
    trop_anom = add_enso_state(trop_anom, oni_idx)
    
    plot_enso_progression(pc_enso, pc_theta, nino_idx, trop_anom,
                          state='El Nino')
    plot_enso_progression(pc_enso, pc_theta, nino_idx, trop_anom,
                          state='La Nina')
    
    plot_enso_space(pc_enso, [2023, 2015, 2020, 2000])
    
    # Comparing the lead-lad relationships
    corr.plot_combined(nino_idx['3.4_anom'], pc_enso['PC1'],
                      era5_anom.time, '3.4 Anom', 'SST PC1',
                      'Months', '', 0, 0, sig=0.99)
    
    corr.plot_combined(nino_idx['3.4_anom'], pc_enso['E'],
                      era5_anom.time, '3.4 Anom', 'E',
                      'Months', '', 0, 0, sig=0.99)
    
    corr.plot_combined(nino_idx['3.4_anom'], pc_enso['C'],
                      era5_anom.time, '3.4 Anom', 'C',
                      'Months', '', 0, 0, sig=0.99)
    
    corr.plot_combined(pc_enso['E'], pc_theta['PC1'],
                      era5_anom.time, 'E Mode', 'Θ₇₀₀ PC1',
                      'Months', '', 0, 0, sig=0.99)
    
    corr.plot_combined(pc_enso['C'], pc_theta['PC1'],
                      era5_anom.time, 'C Mode', 'Θ₇₀₀ PC1',
                      'Months', '', 0, 0, sig=0.99)
    
    # Does mean tropospheric heating respond differently?
    theta_anom = corr.domain_anom(era5_anom, 'theta_700')
    
    corr.plot_combined(pc_enso['E'], theta_anom['theta_700_anom'],
                      era5_anom.time, 'E Mode', 'Mean Θ₇₀₀',
                      'Months', '', 0, 0, sig=0.99)
    
    corr.plot_combined(pc_enso['C'], theta_anom['theta_700_anom'],
                      era5_anom.time, 'C Mode', 'Mean Θ₇₀₀',
                      'Months', '', 0, 0, sig=0.99)
    
    corr.plot_combined(pc_theta['PC1'], theta_anom['theta_700_anom'],
                      era5_anom.time, 'Θ₇₀₀ PC1', 'Mean Θ₇₀₀',
                      'Months', '', 0, 0, sig=0.99)
    
    
if __name__ == '__main__':
    main()

