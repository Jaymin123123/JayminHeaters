from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, RobustScaler
import helpers as h
import glob
import h5py as h5


###########################################################################################
# Make sure these point to the right place on your system
###########################################################################################
from settings import LOCAL_HK_FOLDER, RDS_FOLDER,HP_FOLDER,EXPORT_FOLDER

def cart2sph(r,t,n):
    # if all (0,0,0) then return 0
    
    
    mag = np.sqrt(r*r + t*t + n*n)
    theta = np.arcsin(n / mag) * 180/np.pi
    
    phi = np.arctan2(t,r) * 180/np.pi
    
    phi = np.where(r==0, 0, phi)
    theta = np.where(r==0, 0, theta)
    
    return mag, phi, theta

def load_profile(fname, IBSorOBS = 'OBS'):
    with h5.File(fname, 'r') as data:
        # IBS_profile = data['IBS_profile'][:]
        # IBS_time = data['IBS_time'][:]
        profile = data[f'{IBSorOBS}_profile'][:]
        time = data[f'{IBSorOBS}_time'][:]
    return time, profile

def get_forbidden_dates(ranges):
    # make a list of dates with bad profiles that should not be included in training or testing
    if type(ranges) == list:
        dates = []
        for i in np.arange(0, len(ranges)):
            start = ranges[i][0]
            end = ranges[i][1]
            dates.extend(get_dates(start,end))
    elif type(ranges) == pd.DataFrame:
        dates = []
        for i, row in ranges.iterrows():
            start = row['start']
            end = row['end']
            dates.extend(get_dates(start,end))
    return dates

def get_dates(start, end):
    # get all the days between a start and end date
    date = start
    dates = []
    while date <= end:
        dates.append(date)
        date += timedelta(days = 1)
    return dates


def get_heater_fname(date, folder = HP_FOLDER):
    if type(date) == str:
        date_str = date
    else:
        date_str = date.strftime("%Y-%m-%d")
        
    mylist = os.listdir(folder)
    # r = re.compile(r'\*2020-03-01\*.h5')
    r = re.compile(r'\w*'+ date_str + r'\w*')
    fnames = list(filter(r.match, mylist)) # Read Note below
    # fnames = glob.glob(f"{folder}/*{date_str}*.h5")
    # print(fnames)
    if len(fnames) > 0:
        return folder + '/' + sorted(fnames, key=lambda x:x[-7:-4])[-1]
    else:
        return ''

def get_var_time_bins(ds_duration):
    var_hp_time_bins = np.concatenate(
            (
                np.arange(-1, 20, 0.5),
                np.arange(20, 50, 1),
                np.arange(50, 90, 0.5),
                np.arange(90, ds_duration, 5),
            )
        )
    return var_hp_time_bins

IBSorOBS = 'OBS'
ds_duration = 900  
ds_period = -1


def in_sklearn_format_interval(features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER, test = False):
    hp_time_bins = get_var_time_bins(ds_duration)


    dfs = []
    for i, row in features.iterrows():
        date = row['Date']
        fname = get_heater_fname(date, hp_folder)

        if fname != '':
            time, profile = load_profile(fname, IBSorOBS)
            t = len(profile) % 500
        for j in range(0, len(profile)-t, 500):  # Iterate over profile data in chunks of 500
            chunk_time = time[j:j+500]
            chunk_data_R = profile[j:j+500, 0]
            chunk_data_T = profile[j:j+500, 1]
            chunk_data_N = profile[j:j+500, 2]

            j+=500

            #print(chunk_time)
            #print(chunk_data_R)
            #print(chunk_data_T)
            #print(chunk_data_N)
            #print(len(chunk_time))
            #print(len(chunk_data_R))
            #print(len(chunk_data_T))
            #print(len(chunk_data_N))
#            # Check if all chunks have the same length
            if len(chunk_time) == len(chunk_data_R) == len(chunk_data_T) == len(chunk_data_N) == 500:
                hp_df = pd.DataFrame({
                    'R': chunk_data_R,
                    'T': chunk_data_T,
                    'N': chunk_data_N,
                    'Time': chunk_time
                })

            
            hp_df = hp_df.loc[(hp_df['Time'] < ds_duration) & (hp_df['Time'] > 0)]
            hp_df['time_bin'] = np.digitize(hp_df['Time'], bins = hp_time_bins) 
            hp_ds = hp_df.groupby('time_bin').mean()
            hp_ds.reset_index(inplace=True)

            j+=500
            bin_centers = (hp_time_bins[1:] + hp_time_bins[:-1])/2
            
            # because the downsampling goes one too far I think
            N_profile = hp_ds.shape[0]-1
            if bin_centers.shape[0] != hp_ds.shape[0]-1:
                times = bin_centers[:hp_ds.shape[0]-1]
            else:
                times = bin_centers
            
            profile_ds = pd.DataFrame(
                {
                    'R': hp_ds['R'].values[:N_profile],
                    'T': hp_ds['T'].values[:N_profile],
                    'N': hp_ds['N'].values[:N_profile],
                }
           )
            
            profile_ds['hp_id'] = i
            profile_ds['Time'] = times
            
            # add in all the features for that day
            for key in features.keys():
                if key != "Date":
                    profile_ds[key] = row[key]
            
            dfs.append(profile_ds)
            
            
        else:
            if test == True:
                # print(f'No profile on {date}')
                profile_ds = pd.DataFrame({
                    'R': np.nan,
                    'T': np.nan,
                    'N': np.nan,
                    'Time': (hp_time_bins[1:] + hp_time_bins[:-1])/2
                    })
                
                profile_ds['hp_id'] = i
            
                # add in all the features for that day
                for key in features.keys():
                    if key != "Date":
                        profile_ds[key] = row[key]

                dfs.append(profile_ds)
        
    data = pd.concat(dfs).reset_index(drop=True)
    return data



#features_ = pd.read_csv(h.LOCAL_HK_FOLDER + "features.csv", parse_dates = ['Date'])
#
## get rid of the bad times
#ranges = pd.read_csv("bad_dates.csv", parse_dates=['start', 'end'])
##! need to also deal with duplicate days
#bad_dates = h.get_forbidden_dates(ranges)
#wb_features = features_[~features_['Date'].isin(bad_dates)]
#bad_features = features_[features_['Date'].isin(bad_dates)]
#
#
#################################################################
## Validation set ###############################################
#################################################################
#
#
#val_date = datetime(2023,12,31)
#val_features = wb_features.loc[(wb_features['Date'] > val_date)]
#features = wb_features.loc[wb_features['Date'] < val_date]
#
#
#    
#def features2train(features, val_features, ds_duration, ds_period, IBSorOBS, split = 'profiles', weights = False, coord = 'rtn'):
#    if split == 'days':
#        if val_features is None:
#            val_features = features.sample(n=features.shape[0]//5, random_state = 0)
#            # Create a new DataFrame with the remaining rows
#            features = features[~features.index.isin(val_features.index)].copy(deep=True)
#        
#        # take some random points out for testing
#        test_features = features.sample(n=features.shape[0]//5, random_state = 0)
#        # Create a new DataFrame with the remaining rows
#        train_features = features[~features.index.isin(test_features.index)].copy(deep=True)
#        
#        train = in_sklearn_format_interval(train_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
#        val = in_sklearn_format_interval(val_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
#        test = in_sklearn_format_interval(test_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
#    
#    elif split == 'profiles':
#        if val_features is not None:
#            val = in_sklearn_format_interval(val_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
#        all_data = in_sklearn_format_interval(features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
#
#        if val_features is None:
#            val = all_data.sample(n=all_data.shape[0]//5, random_state = 0)
#            all_data = all_data[~all_data.index.isin(val.index)].copy(deep=True)
#        
#        train = all_data.sample(n=all_data.shape[0]//5, random_state = 0)
#        test = all_data[~all_data.index.isin(train.index)].copy(deep=True)
#    else:
#        # I used weight for a bit, but took out. 
#        # Ideally, want the weight to be equal to "how reliable is this heater profile I am learning from?"
#        raise 'BadBoy'
#    
#    # get the weights before scaling
#    if weights:
#        print('make weights represent the amount of variability in B on that day (i.e. how much can we trust the averaging)')
#        raise 'BadBoy 1'
#        # train_weights = get_weights(train['Time'].values, ds_duration)
#        # test_weights = get_weights(test['Time'].values, ds_duration)
#        # val_weights = get_weights(val['Time'].values, ds_duration)
#    
#    
#    B, phi, theta = h.cart2sph(
#        train['R'].values,
#        train['T'].values,
#        train['N'].values,
#    )
#    train['|B|'] = B
#    train['phi'] = phi
#    train['theta'] = theta
#    
#    B, phi, theta = h.cart2sph(
#        val['R'].values,
#        val['T'].values,
#        val['N'].values,
#    )
#    val['|B|'] = B
#    val['phi'] = phi
#    val['theta'] = theta
#    
#    B, phi, theta = h.cart2sph(
#        test['R'].values,
#        test['T'].values,
#        test['N'].values,
#    )
#    test['|B|'] = B
#    test['phi'] = phi
#    test['theta'] = theta
#    
#    # scaling
#    no_scaling_names = [
#        'hp_id',
#        'Heater',
#        'SA change',
#        'HGA azimuth change',
#        'HGA evelvation change',
#        'No time A off'
#    ]
#
#    mapper_list = []
#
#    for key in train.keys():
#        if any(substring in key for substring in no_scaling_names):
#            mapper_list.append((key, None))
#        else:
#            mapper_list.append(([key], RobustScaler()))
#
#    mapper = DataFrameMapper(mapper_list, df_out = True)
#
#
#
#    scaler = mapper.fit(train)
#    train_scaled = mapper.transform(train)
#    test_scaled = mapper.transform(test)
#    val_scaled = mapper.transform(val)
#    
#    x_train = train_scaled.drop(['hp_id', 'R', 'T', 'N', '|B|', 'phi', 'theta'],axis = 1)
#    x_val = val_scaled.drop(['hp_id', 'R', 'T', 'N', '|B|', 'phi', 'theta'], axis = 1)
#    x_test = test_scaled.drop(['hp_id', 'R', 'T', 'N', '|B|', 'phi', 'theta'], axis = 1)
#    
#    if coord == 'sph':
#        y_train = train_scaled[['|B|', 'phi', 'theta']].copy()
#        y_val = val_scaled[['|B|', 'phi', 'theta']].copy()
#        y_test = test_scaled[['|B|', 'phi', 'theta']].copy()
#    else:
#        y_train = train_scaled[['R', 'T', 'N']].copy()
#        y_val = val_scaled[['R', 'T', 'N']].copy()
#        y_test = test_scaled[['R', 'T', 'N']].copy()
#    
#    if weights:
#        return x_train, x_val, x_test, y_train, y_val, y_test, mapper, train_weights, test_weights, val_weights
#    else:
#        return x_train, x_val, x_test, y_train, y_val, y_test, mapper
#
#
#x_train, x_val, x_test, y_train, y_val, y_test, mapper = h.features2train(
#    interval_data,
#    interval_val_features, 
#    ds_duration,
#    ds_period,
#    IBSorOBS,
#    'days',
#    coord = 'rtn'
#    
#
#search.fit(x_train, y_train
#score = search.score(x_test, y_test)
#print("Test Score: ", score)
#score = search.score(x_val, y_val)
#print("Val Score: ", score
#date_str = current_date.strftime('%Y-%m-%d')
#feature_importances = search.feature_importances_
#importance_percentages = 100 * (feature_importances / feature_importances.sum())
#feature_names = interval_train_features.columns
#importance_df = pd.DataFrame({'Feature': feature_names, 'Importance (%)': importance_percentages}).sort_values(by='Importance (%)', ascending=False)
#importance_df.to_csv(f'Models/RF_MFI/{IBSorOBS}_untuned_{ds_period}_{ds_duration}_importances_{date_str}.csv'
## save the mapper
#dump(mapper, f"Models/RF_MFI/{IBSorOBS}_untuned_{ds_period}_{ds_duration}_scaler_{date_str}.joblib"
## save the model
#dump(search, f"Models/RF_MFI/{IBSorOBS}_untuned_{ds_period}_{ds_duration}_{current_date}.joblib")#