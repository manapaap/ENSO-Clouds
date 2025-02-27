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


def isolate_regions(data, var, regions, dataset='ERA5'):
    """
    Returns a pandas dataframe with three time series,
    representing average value of var within the three areas defined
    in regions
    """
    arrays = {key: 'temp' for key, value in regions.items()}
    
    for key, value in regions.items():
        if dataset=='ERA5':
            arrays[key] = share.isolate_ep_era5(data, var, value)
        elif dataset=='ISCCP':
            arrays[key] = share.isolate_ep_isccp(data, var, value)
    
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
        
    
def pred_var(data_df, pred_df1, pred_df2, name1='sst', name2='theta_700'):
    """
    Constructs linear regression of region-dependent predictors pred_1, pred_2
    onto data_df. All should be pandas dataframes with the same columns
    
    returns a pandas df containing slope1, slope2, intercept, r**2
    """
    info = pd.DataFrame()
    for loc in data_df.columns:
        Y = data_df[loc]
        X = pd.DataFrame({name1: pred_df1[loc], name2: pred_df2[loc]})
        X = sm.add_constant(X)
        fit = sm.OLS(Y, X).fit()
        params = dict(fit.params)
        params['rsq_adj'] = fit.rsquared_adj
        # Format into pandas df
        params = pd.DataFrame(params, index=[loc])
        info = pd.concat([info, params])
    return info
    

def main():
    # Files of intrest- ISCCP and ERA5
    global sst_pred, theta_pred, lcc_pred, eis_pred, lcc_cause, eis_cause
    isccp_file = 'era5_reanal/timeseries/isccp_anom.nc'
    file_era5 = 'era5_all/timeseries/era5_anom_all.nc'
       
    if os.path.exists(isccp_file) and os.path.exists(file_era5):
        isccp_anom = xr.load_dataset(isccp_file)
        era5_data = xr.open_dataset(file_era5)
    else:
        print('Files missing; please run vis_clouds and all_cloud_corr')
    
    # Overlap period with ISCCP
    era5_isc = era5_data = era5_data.sel({'time': isccp_anom.time})
    # regions of intrest; too much to include in shared funcs for now
    regions = {'NEQP': [0, 10, 240, 280],
               'SEQP': [-10, 0, 240, 280],
               'SEP': [-20, -10, 240, 280],
               'ALL': [-20, 10, 240, 280]}
    # PCs for different time periods; get sign convention we want
    _, pc_enso = share.calc_eof(era5_data, 'sst', n_pc=2,
                             plot=False, region='equator', detrend=True)
    pc_enso['PC1'] *= -1
    _, pc_1983 = share.calc_eof(era5_isc, 'sst', n_pc=2,
                             plot=False, region='equator', detrend=True)
    pc_1983['PC1'] *= -1
    # Get anomalies we care about
    lcc_anoms = isolate_regions(isccp_anom, 'stratus', regions, 'ISCCP')
    eis_anoms = isolate_regions(era5_data, 'eis', regions, 'ERA5')
    # Predictors
    sst_anoms = isolate_regions(era5_data, 'sst', regions, 'ERA5')
    sst_1983 = isolate_regions(era5_isc, 'sst', regions, 'ERA5')
    theta_anoms = isolate_regions(era5_data, 'theta_700', regions, 'ERA5')
    theta_1983 = isolate_regions(era5_isc, 'theta_700', regions, 'ERA5')
    # Variables we care for
    lcc_pred = pred_pc(lcc_anoms, pc_1983[['PC1', 'PC2']])
    eis_pred = pred_pc(eis_anoms, pc_enso[['PC1', 'PC2']])
    # Cause of cloud anomalies
    sst_pred = pred_pc(sst_anoms, pc_enso[['PC1', 'PC2']])
    theta_pred = pred_pc(theta_anoms, pc_enso[['PC1', 'PC2']])
    # Validation check
    eis_cause = pred_var(eis_anoms, sst_anoms, theta_anoms, 'SST', 'Theta 700')
    lcc_cause = pred_var(lcc_anoms, sst_1983, theta_1983, 'SST', 'Theta 700')
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    