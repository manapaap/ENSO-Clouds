# -*- coding: utf-8 -*-
"""
Tri-region analysis

Separrate analysis of lcc/eis anoms in NEQP, SEQP, SEP

Particularly curious about PC1/PC2 and Theta 700/SST two-var
multiple regression
"""


import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
import obs_scripts.shared_funcs as share


def isolate_regions(data, var, regions, dataset='ERA5', lowpass=False, f=1/6):
    """
    Returns a pandas dataframe with three time series,
    representing average value of var within the three areas defined
    in regions
    
    Can apply a lowpass fifilter
    """
    arrays = {key: 'temp' for key, value in regions.items()}
    
    for key, value in regions.items():
        if dataset=='ERA5':
            arrays[key] = share.isolate_ep_era5(data, var, value)
        elif dataset=='ISCCP':
            arrays[key] = share.isolate_ep_isccp(data, var, value)
    if lowpass==True:
        for key, value in arrays.items():
            arrays[key] = share.butter_lowpass_filter(value, cutoff=f, fs=1)
    return pd.DataFrame(arrays)


def pred_pc(data_df, preds):
    """
    Constructs linear regression of each of the three regions in data_df
    as a function of arr1, arr2. Returns a pandas df containing slope1, 
    slope2, intercept, r**2
    
    preds is a pandas df containing predictors, e.g. as preds[['PC1', 'PC2']]
    """
    info = pd.DataFrame()
    for loc in data_df.columns:
        Y = data_df[loc]
        X = preds
        X = sm.add_constant(X)
        fit = sm.OLS(Y, X).fit()
        params = dict(fit.params)
        params['rsq_adj'] = fit.rsquared_adj
        # Format into pandas df
        params = pd.DataFrame(params, index=[loc])
        info = pd.concat([info, params])
    return info
        
    
def pred_var(data_df, pred_dfs=[], names=[], normalize=False):
    """
    Constructs linear regression of region-dependent predictors, marking coefficients 
    with * if not significant at 99% confidence (p >= 0.01).
    """
    info = pd.DataFrame()
    for loc in data_df.columns:
        Y = data_df[loc]
        X = pd.DataFrame({name: df[loc] for name, df in zip(names, pred_dfs)})
        if normalize:
            X /= X.std()  # Normalize predictors
        X = sm.add_constant(X)  # Add intercept term
        fit = sm.OLS(Y, X).fit()
        pvalues = fit.pvalues  # Get p-values
        params = {}
        for name in fit.params.index:
            value = fit.params[name]
            # Format intercept without significance marker
            if pvalues[name] >= 0.01:
                params[name] = f"{value:.4f}*"
            else:
                params[name] = f"{value:.4f}"
        # Add adjusted R² (no formatting needed)
        params['rsq_adj'] = fit.rsquared_adj
        # Append to results
        params_df = pd.DataFrame(params, index=[loc])
        info = pd.concat([info, params_df])

    return info
    

def pearson(data_df, pred_dfs, names):
    """
    Returns univariate pearson correlation between variables and data dfs
    """
    info = pd.DataFrame(columns=names)
    for loc in data_df.columns:
        Y = data_df[loc]
        X = pd.DataFrame({name: df[loc] for name, df in zip(names, pred_dfs)})
        denom = X.std() * Y.std()
        x_means = X.mean()
        y_mean = Y.mean()
        pearson = dict()
        for col in X.columns:
            # Corrected line: Added parentheses around the numerator
            pearson[col] = (np.mean(Y * X[col]) - x_means[col] * y_mean) /\
                denom[col]
        info.loc[loc] = pearson
    return info
        

def main():
    # Files of intrest- ISCCP and ERA5
    global era5_isc, isccp_anom, era5_data, pc_enso, pc_1983
    isccp_file = 'era5_reanal/timeseries/isccp_anom.nc'
    file_era5 = 'era5_all/timeseries/era5_anom_all.nc'
       
    if os.path.exists(isccp_file) and os.path.exists(file_era5):
        isccp_anom = xr.load_dataset(isccp_file)
        era5_data = xr.open_dataset(file_era5)
    else:
        print('Files missing; please run vis_clouds and all_cloud_corr')
    # Overlap period with ISCCP
    era5_isc = era5_data.sel({'time': isccp_anom.time})
    # regions of intrest; too much to include in shared funcs for now
    regions = {'NEQP': [0, 10, 240, 280],
               'SEQP': [-10, 0, 240, 280],
               'LSEP': [-20, -10, 240, 280],
               'ALL': [-30, 10, 240, 280]}
    # PCs for different time periods; get sign convention we want
    _, pc_enso = share.calc_eof(era5_data, 'sst', n_pc=2,
                             plot=False, region='equator', detrend=True)
    pc_enso['PC1'] *= -1
    pc_enso = share.rotate_enso_eof(pc_enso)
    _, pc_1983 = share.calc_eof(era5_isc, 'sst', n_pc=2,
                             plot=False, region='equator', detrend=True)
    pc_1983['PC1'] *= -1
    pc_1983 = share.rotate_enso_eof(pc_1983)
    # Get anomalies we care about
    lcc_anoms = isolate_regions(isccp_anom, 'sc_adj', regions, 'ISCCP',
                                lowpass=True)
    eis_anoms = isolate_regions(era5_isc, 'eis', regions, 'ERA5')
    
    # Predictors
    sst_anoms = isolate_regions(era5_data, 'sst', regions, 'ERA5')
    sst_1983 = isolate_regions(era5_isc, 'sst', regions, 'ERA5')
    # theta_anoms = isolate_regions(era5_data, 'theta_700', regions, 'ERA5')
    theta_1983 = isolate_regions(era5_isc, 'theta_700', regions, 'ERA5')
    # Extended list of predictors
    speed_anoms = isolate_regions(era5_isc, 'speed', regions, 'ERA5')
    rh_700_anoms = isolate_regions(era5_isc, 'rh_700', regions, 'ERA5')
    rh_1000_anoms = isolate_regions(era5_isc, 'rh_1000', regions, 'ERA5')
    w_700_anoms = isolate_regions(era5_isc, 'w_700', regions, 'ERA5')
    
    cold_adv_anoms = isolate_regions(era5_isc, 'cold_adv', regions, 'ERA5')
    # Variables we care for
    lcc_pred = pred_pc(lcc_anoms, pc_1983[['PC1', 'PC2']])
    # eis_pred = pred_pc(eis_anoms, pc_enso[['PC1', 'PC2']])
    # Cause of cloud anomalies
    sst_pred = pred_pc(sst_anoms, pc_enso[['PC1', 'PC2']])
    # theta_pred = pred_pc(theta_anoms, pc_enso[['PC1', 'PC2']])
    # Validation check
    #eis_cause = pred_var(eis_anoms, pred_dfs=[sst_anoms, theta_anoms],
    #                     names=['SST', 'Theta 700'])
    #lcc_cause = pred_var(lcc_anoms, pred_dfs=[sst_1983, theta_1983],
    #                     names=['SST', 'Theta 700'])
    # extended lcc check
    
    cirr_anoms = isolate_regions(isccp_anom, 'high', regions, 'ISCCP',
                                lowpass=True)
    
    lcc_causes = pred_var(lcc_anoms, pred_dfs=[sst_1983, eis_anoms,
                                               rh_1000_anoms, cirr_anoms,
                                               cold_adv_anoms],
                         names=['SST', 'EIS',
                                'RH 1000', 'Cirrus Frac', 'Cold Adv.'])
    
    lcc_shapes = pearson(lcc_anoms, pred_dfs=[sst_1983, eis_anoms,
                                               rh_1000_anoms, cirr_anoms,
                                               cold_adv_anoms],
                         names=['SST', 'EIS',
                                'RH 1000', 'Cirrus Frac', 'Cold Adv.'])
    # EIS not good in SEP???
if __name__ == '__main__':
    main()
    