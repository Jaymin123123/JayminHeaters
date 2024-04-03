################################################################################
# A command line version of interpreting.ipynb to create the predicted profiles
################################################################################

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

import os

import helpers as h

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--start',
                        '-s',
                        default=datetime(2023,3,1),
                        type=lambda d: datetime.strptime(d, '%Y-%m-%d'),
                        help='Specify the start in the format YYYY-MM-DD')
    
    parser.add_argument('--end',
                        '-e',
                        default=datetime(2023,5,21),
                        type=lambda d: datetime.strptime(d, '%Y-%m-%d'),
                        help='Specify the end in the format YYYY-MM-DD')
    
    parser.add_argument(
        "--inst",
        "-i",
        default = 'OBS',
        choices = ['OBS', 'IBS'],
        type=str,
        help="Which instrument to use, IBS or OBS")
    
    parser.add_argument(
        "--period",
        "-p",
        default=-1,
        type=int,
        help="Period to downsample the profiles",
    )
    
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=15*60,
        help="Duration to downsample the profiles",
    )
    
    parser.add_argument(
        "--tune",
        '-t',
        type=str,
        default='untuned',
        choices = ['untuned', 'bayes', 'tuned'],
        help="How to tune the model",
    )
    
    parser.add_argument(
        "--model",
        '-m',
        type=str,
        default='RF',
        choices = ['RF', 'LGB'],
        help="Which model type",
    )
    
    # was RFE used? if so then need to load the extra _features.joblib file
    parser.add_argument(
        "--rfe",
        default=True,
        action='store_true',
        help="was RFE used or not",
    )
    
    # just use a favourite model
    parser.add_argument(
        "--fav",
        action='store_true',
        help="Just use favourite model specified in the code line 120",
    )
    
    
    args = parser.parse_args()

    ################################################################################
    # Load the models
    ################################################################################


    print("load models")

    # load the models
    from joblib import load
    models = {}

    # these parameters are used to find the model you want to load

    ds_duration = args.duration
    ds_period = args.period
    model_type = args.model
    IBSorOBS = args.inst
    opt = args.tune

    # or just use a favourite model that has been saved separately
    fav = False


    # was RFE used? if so then need to load the extra _features.joblib file
    rfe = args.rfe

    if fav:
        # set the favourite model here
        folder = f'Models/{model_type}/Favs/untuned 2/'
        HK_folder = folder 
    else:
        folder = f'Models/{model_type}/'
        HK_folder = 'HK data/'

    if ds_period < 0:
        HP_TIME_BINS = h.get_var_time_bins(ds_duration)
    else:
        HP_TIME_BINS = np.arange(0, ds_duration, ds_period)

    # define the coordinate system to use
    #spherical
    # components = ['|B|', 'phi', 'theta'] 

    # cartesian
    components = ['R', 'T', 'N'] 


    if model_type == 'RF' or model_type == 'MLP':
        
        models['all'] = load(folder + f"{IBSorOBS}_{opt}_{ds_period}_{ds_duration}.joblib")
        mapper = load(folder + f"{IBSorOBS}_{opt}_{ds_period}_{ds_duration}_scaler.joblib")
        if rfe:
            selected_features = load(folder + f"{IBSorOBS}_{opt}_{ds_period}_{ds_duration}_features.joblib")
    else:
        

        for component in tqdm(components):
            models[component] = load(folder + f"{IBSorOBS}_{component}_{opt}_{ds_period}_{ds_duration}.joblib")
        mapper = load(folder + f"{IBSorOBS}_{opt}_{ds_period}_{ds_duration}_scaler.joblib")
        if rfe:
            selected_features = load(folder + f"{IBSorOBS}_{component}_{opt}_{ds_period}_{ds_duration}_features.joblib")

    ################################################################################
    # Load features
    ################################################################################
    print("load features")
    features = pd.read_csv(HK_folder + "features.csv", parse_dates = ['Date'])
    
    ranges = pd.read_csv("bad_dates.csv", parse_dates=['start', 'end'])

    bad_dates = h.get_forbidden_dates(ranges)

    wb_features = features[~features['Date'].isin(bad_dates)]
    bad_features = features[features['Date'].isin(bad_dates)]
    
    # just use features for now, I want to see on everything
    all_data = h.in_sklearn_format(features, ds_duration, ds_period, IBSorOBS, test = True)
    B, phi, theta = h.cart2sph(
        all_data['R'].values,
        all_data['T'].values,
        all_data['N'].values,
    )
    all_data['|B|'] = B
    all_data['phi'] = phi
    all_data['theta'] = theta
    
    all_data_scaled = mapper.transform(all_data)
    
    # make predictions
    to_drop = ["hp_id", 'R', 'T', 'N', '|B|', 'phi', 'theta']
    for key in all_data_scaled.keys():
        if "_orig" in key or "pred" in key:
            to_drop.append(key)
        all_x = all_data_scaled.drop(to_drop,axis = 1)


    ################################################################################
    # make predictions
    ################################################################################
    
    print("Make predictions")
    
    
    if model_type == 'RF' or model_type == 'MLP':
        if rfe:
            # if RFE then only select those columns
            # doing this here becaues the mapper includes all the columns
            all_x = all_x.loc[:, selected_features]
        
        y_pred = models['all'].predict(all_x)
        
        for i in range(len(components)):
            index = mapper.transformed_names_.index(components[i])
            scaler = mapper.built_features[index][1]
            
            all_data_scaled[f'{components[i]}_pred'] = y_pred[:,i]
            all_data_scaled[f'{components[i]}_pred_orig'] = scaler.inverse_transform(y_pred[:,i].reshape(-1,1)).flatten()
            
    else:
        # set the component you want to look at with LightGBM
        component = 'N'
        if rfe:
            all_x = all_x.loc[:, selected_features]

        y_pred = models[component].predict(all_x)

        index = mapper.transformed_names_.index(component)
        scaler = mapper.built_features[index][1]

        all_data_scaled['pred'] = y_pred 
        all_data_scaled['pred_orig'] = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()

    # unscale the original components
    for i in range(len(components)):
        index = mapper.transformed_names_.index(components[i])
        scaler = mapper.built_features[index][1]
        all_data_scaled[components[i]+'_orig'] = scaler.inverse_transform(all_data_scaled[components[i]].values.reshape(-1,1)).flatten()
        
    # try to see if it got |B| right
    all_data_scaled['|B|_orig'] = np.sqrt(all_data_scaled['R_orig']**2 + all_data_scaled['T_orig']**2 + all_data_scaled['N_orig']**2)
    all_data_scaled['|B|_pred_orig'] = np.sqrt(all_data_scaled['R_pred_orig']**2 + all_data_scaled['T_pred_orig']**2 + all_data_scaled['N_pred_orig']**2)

    ################################################################################
    # Interpolate and save
    ################################################################################

    print("interpolate and save")


    # Interpolate the predictions onto the original time stamps
    # This just makes it easier to use in the pipeline to clean the data

    from scipy.interpolate import interp1d

    fname = h.get_heater_fname(features.loc[0, 'Date'], h.HP_FOLDER)
    og_time_stamps, _ = h.load_profile(fname, IBSorOBS)


    if ds_duration < 900:
        raise Exception("Heater profile needs to be the full 15 mins for use in production")

    dates_to_export = h.get_dates(args.start, args.end)



    for date in dates_to_export:
        print(date)
        selected_day = all_data.loc[all_data['hp_id'] == features.loc[features['Date'] == date].index[0]]
        
        # if the original heater profile was shorter than 900, then reflect that here
        time_stamps_to_interp = og_time_stamps[og_time_stamps < selected_day['Time'].max()]

        
        data_to_export = pd.DataFrame({'Time': time_stamps_to_interp})
        
        for component in ['R', 'T', 'N']:
            # need to add 0 and a time of -1 so interpolation goes right to the start
            pred_times = np.concatenate((np.array([-1]), selected_day['Time'].values))
            pred_comp = np.concatenate((np.array([0]), selected_day[component].values))
            interp = interp1d(pred_times, pred_comp)
            data_to_export[component] = interp(time_stamps_to_interp)

        date_str = date.strftime('%Y-%m-%d')
        
        # save the prediction
        data_to_export.to_csv(h.EXPORT_FOLDER + 'HeaterProfile_' + date_str + '_V999.csv', index=False)