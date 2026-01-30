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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import linregress, t
from scipy.signal import detrend


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
        
    
def pred_var(data_df, pred_dfs=[], names=[], normalize=True):
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
            Y /= Y.std() # Normalize cloud cover
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


def vif_calc(data_df, pred_dfs=[], names=[], normalize=True):
    """
    Constructs linear regression of region-dependent predictors, marking coefficients 
    with * if not significant at 99% confidence (p >= 0.01).
    """
    info = pd.DataFrame()
    for loc in data_df.columns:
        X = pd.DataFrame({name: df[loc] for name, df in zip(names, pred_dfs)})
        if normalize:
            X /= X.std()  # Normalize predictors
        X = sm.add_constant(X)  # Add intercept term
        vifs = {var: variance_inflation_factor(X.values, i) 
               for i, var in zip(range(X.shape[1]), X.columns)}
        # Append to results
        params_df = pd.DataFrame(vifs, index=[loc])
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


def pred_predictors(pc_enso, pred_vars):
    """
    Returns a pandas df containing the slope and intercept of the 
    predictor variables vs. the C-index over the whole SEP
    """
    data = pd.DataFrame(columns=pred_vars.columns,
                        index=['slope', 'intercept', 
                               'rvalue', 'pvalue'])
    # Normalize
    pred_vars /= pred_vars.std()
    for var in pred_vars.columns:
        reg = linregress(pc_enso['C'], pred_vars[var])
        data.loc['slope', var] = reg.slope
        data.loc['intercept', var] = reg.intercept
        data.loc['rvalue', var] = reg.rvalue
        # data.loc['pvalue', var] = reg.pvalue
        # adjust degrees of freedom 
        df = len(pc_enso) - 2
        cleaned_idx = pc_enso['C'] - np.mean(pc_enso['C'])
        idx_autocorr = (cleaned_idx[1:] * cleaned_idx[:-1]).mean() /\
            np.var(cleaned_idx)
        cleaned_data = pred_vars[var] - np.mean(pred_vars[var])
        data_autocorr = (cleaned_data[1:] * cleaned_data[:-1]).mean() /\
            np.var(cleaned_data)
        df_adjusted = df * ((1 - idx_autocorr * data_autocorr) /\
                            (1 + idx_autocorr * data_autocorr))
        df_adjusted = df
        rvalues = reg.rvalue.astype(float)
        t_stat = rvalues * np.sqrt(df_adjusted / (1 - rvalues**2))
        pvalues = 2 * t.sf(abs(t_stat), df_adjusted)
        data.loc['pvalue', var] = pvalues
    return data


def fit_cloud_data(lcc_anoms, predictors):
    """
    Normalizes data and fits to the cloud cover variable, only for one region 
    (SEP)
    
    Adjusts degrees of freedom via the VIF criteria in Wilks 2011
    """
    X = (predictors - predictors.mean()) / predictors.std()
    X = sm.add_constant(X)
    Y = (lcc_anoms - lcc_anoms.mean()) / lcc_anoms.std()
    fit = sm.OLS(Y, X).fit()
    # adjust pvalues for degrees of freedom
    residuals = Y - fit1.predict()
    autocorr = (residuals[1:] * residuals[:-1]) / np.var(residuals)
    return fit


def isolate_enso(xr_ds, oni_idx, out='El Nino'):
    """
    Isolates El Nino or La Nina months from xr_ds based on ONI criteria
    """
    vect = oni_idx.copy()
    # Convert the vector data to a time-indexed DataArray
    vect['time'] = pd.to_datetime(dict(year=vect['year'], 
                                       month=vect['month'], day=1))
    vect = vect.set_index('time')
    
    if out == 'El Nino':
        vect['month'] = pd.Series(vect.month, dtype=int)
        vect = vect.query('month <= 2 or month >= 11')
        vect = vect.query('oni >= 0.5')
    elif out == 'La Nina':
        vect['month'] = pd.Series(vect.month, dtype=int)
        vect = vect.query('month <= 2 or month >= 11')
        vect = vect.query('oni <= -0.5')
    elif out == 'Neutral':
        vect['month'] = pd.Series(vect.month, dtype=int)
        vect = vect.query('month <= 2 or month >= 11')
        vect = vect.query('-0.5 <= oni <= 0.5')
    elif out =='NDJF':
        vect['month'] = pd.Series(vect.month, dtype=int)
        vect = vect.query('month <= 2 or month >= 11')
    else:
        # do nothing
        vect = vect
    vect_da = xr.DataArray(vect['oni'], coords=[vect.index], dims=['time'])
    # Find the intersection of time periods
    common_times = xr_ds.time.to_index().intersection(vect_da.time.to_index())
    # Align datasets to this common time range
    xr_ds = xr_ds.sel(time=common_times)
    vect_da = vect_da.sel(time=common_times)
    return xr_ds


def isolate_enso_idx(pc_enso, oni_idx, out='El Nino'):
    """
    Isolates El Nino or La Nina months from xr_ds based on ONI criteria
    """
    vect = oni_idx.copy()
    idx = pc_enso.copy()
    # Convert the vector data to a time-indexed DataArray
    vect['time'] = pd.to_datetime(dict(year=vect['year'], 
                                       month=vect['month'], day=1))
    vect = vect.set_index('time')
    idx['time'] = pd.to_datetime(dict(year=idx['year'], 
                                       month=idx['month'], day=1))
    idx = idx.set_index('time')
    if out == 'El Nino':
        vect['month'] = pd.Series(vect.month, dtype=int)
        vect = vect.query('month <= 2 or month >= 11')
        vect = vect.query('oni >= 0.5')
    elif out == 'La Nina':
        vect['month'] = pd.Series(vect.month, dtype=int)
        vect = vect.query('month <= 2 or month >= 11')
        vect = vect.query('oni <= -0.5')
    elif out == 'Neutral':
        vect['month'] = pd.Series(vect.month, dtype=int)
        vect = vect.query('month <= 2 or month >= 11')
        vect = vect.query('-0.5 <= oni <= 0.5')
    elif out =='NDJF':
        vect['month'] = pd.Series(vect.month, dtype=int)
        vect = vect.query('month <= 2 or month >= 11')
    else:
        # do nothing
        vect = vect
    vect_da = xr.DataArray(vect['oni'], coords=[vect.index], dims=['time'])
    idx_da = xr.Dataset(idx)
    for col in idx.columns:
        idx_da[col] = idx[col]
    # Find the intersection of time periods
    common_times = idx_da.time.to_index().intersection(vect_da.time.to_index())
    # Align datasets to this common time range
    idx_da = idx_da.sel(time=common_times)
    return idx_da.to_pandas()


def main():
    # Files of intrest- ISCCP and ERA5
    global pc_1983, pred_vars, ccf_corr, sep_projected, fit1, fit2, lcc_anoms
    global sep_predictors, oni_idx, isccp_anom, era5_data, best, ccf_pred
    isccp_file = 'era5_reanal/timeseries/isccp_anom.nc'
    file_era5 = 'era5_all/timeseries/era5_anom_all.nc'
       
    if os.path.exists(isccp_file) and os.path.exists(file_era5):
        isccp_anom = xr.load_dataset(isccp_file)
        era5_data = xr.open_dataset(file_era5)
    else:
        print('Files missing; please run vis_clouds and all_cloud_corr')
    # Overlap period with ISCCP
    era5_isc = era5_data.sel({'time': isccp_anom.time})
    # PCs for different time periods; get sign convention
    _, pc_1983 = share.calc_eof(era5_isc, 'sst', n_pc=2,
                             plot=False, region='equator', detrend=True)
    pc_1983['PC1'] *= -1
    pc_1983 = share.rotate_enso_eof(pc_1983)
    # load ONI
    oni_idx = share.load_oni_idx(fpath='misc_data/oni_index.txt')
    oni_rel = oni_idx.query('"1983-07" <= time <= "2017-06"').reset_index(drop=True)
    # limit ourselves to warm or cool events
    state = 'None'
    era5_isc = isolate_enso(era5_isc, oni_idx, state)
    isccp_anom = isolate_enso(isccp_anom, oni_idx, state)
    pc_1983 = isolate_enso_idx(pc_1983, oni_idx, state)
    # regions of intrest; too much to include in shared funcs for now
    regions = {'NEQP': [0, 10, 240, 280],
               'SEQP': [-10, 0, 240, 280],
               'LSEP': [-20, -10, 240, 280],
               'ALL': [-20, 10, 240, 280],
               'SEP': [-20, 0, 240, 280]}
    
    # Get anomalies we care about
    lp = False
    lcc_anoms = isolate_regions(isccp_anom, 'sc_adj', regions, 'ISCCP',
                                lowpass=lp)
    eis_anoms = isolate_regions(era5_isc, 'eis', regions, 'ERA5',
                                lowpass=lp)
    
    sst_1983 = isolate_regions(era5_isc, 'sst', regions, 'ERA5',
                               lowpass=lp)
    speed_anoms = isolate_regions(era5_isc, 'speed', regions, 'ERA5',
                                  lowpass=lp)
    rh_700_anoms = isolate_regions(era5_isc, 'rh_700', regions, 'ERA5',
                                   lowpass=lp)
    w_700_anoms = isolate_regions(era5_isc, 'w_700', regions, 'ERA5',
                                  lowpass=lp)
    
    cold_adv_anoms = isolate_regions(era5_isc, 'cold_adv', regions, 'ERA5',
                                     lowpass=lp)

    cirr_anoms = isolate_regions(isccp_anom, 'high', regions, 'ISCCP',
                                lowpass=lp)
    
    pred_vars = [sst_1983, eis_anoms, rh_700_anoms, cirr_anoms,
                 cold_adv_anoms, speed_anoms, w_700_anoms]
    names = ['SST', 'EIS', '700 hPa Relative Humidity', 'Cirrus Fraction',
           'Cold Advection', '10m Windspeed', '700 hPa Subsidence']
    sep_predictors = pd.DataFrame({name: var['SEP'] for name, var in
                                   zip(names, pred_vars)}) 
    # detrend these
    # for predictor in sep_predictors.columns:
    #     sep_predictors[predictor] = detrend(sep_predictors[predictor])
    # include for completeness
    vifs = {var: variance_inflation_factor(sep_predictors.values, i) 
           for i, var in zip(range(sep_predictors.shape[1]), sep_predictors.columns)}
    
    lcc_causes = pred_var(lcc_anoms, pred_dfs= pred_vars,
                         names=names, normalize=True)
    vifs = vif_calc(lcc_anoms, pred_dfs=pred_vars,
                         names=names, normalize=True).T['SEP']
    
    ccf_corr = pred_predictors(pc_1983, sep_predictors).T
    # these are the variables that the C index drives
    ccf_pred = ccf_corr['slope'][ccf_corr['pvalue'] <= 0.01]
    
    # We can now compare models for predicting the lcc anomaly
    # this is the relationship betwen the CCFs and LCC
    fit1 = fit_cloud_data(lcc_anoms['SEP'], sep_predictors)
    # EIS not good in SEP???
    good_fit = fit1.params[fit1.pvalues <= 0.01]
    
    best = ccf_pred * good_fit
if __name__ == '__main__':
    main()
    