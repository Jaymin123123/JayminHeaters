from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, RobustScaler

import glob
import h5py as h5


###########################################################################################
# Make sure these point to the right place on your system
###########################################################################################
from settings import LOCAL_HK_FOLDER, RDS_FOLDER,HP_FOLDER,EXPORT_FOLDER


###########################################################################################
# Loading the House keeping data
###########################################################################################



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


def get_HK_fname(date, folder):
    folder = folder + "/" + str(date.year) + "/"
    date_str = date.strftime("%Y%m%d")
    #! this bit is really slow!
    fnames = glob.glob(f"{folder}/*{date_str}*.txt")
    if len(fnames) > 0:
        # get the last version
        return sorted(fnames, key=lambda x:x[-6:-4])[-1]
    else:
        return ''

def local_HK_fname(hk_name, date):
    date_str = date.strftime("%Y%m%d")
    return LOCAL_HK_FOLDER + '/' + hk_name + '/' + str(date.year) + '/' + hk_name + "_" + date_str + ".csv"

def load_HK(hk_name, start, end, extension = '.txt'):
    
    fname_dates = get_dates(start, end)
    
    # load the rds file list anyway, as I might need to use it
    year = str(fname_dates[0].year)
    year_folder = RDS_FOLDER + '/' + hk_name + '/' + year
    rds_list = os.listdir(year_folder)
    
    dfs = []

    for date in fname_dates:
        date_str = date.strftime("%Y%m%d")
        
        # if this exists as a .csv then load that
        local_name = local_HK_fname(hk_name, date)
        
        if os.path.exists(local_name):
            #load the local data
            df = pd.read_csv(local_name, parse_dates = ['Time'])
            df.set_index('Time', inplace = True)
            dfs.append(df)
        else:
            # check you are in the right rds folder
            if str(date.year) != year:
                year = str(date.year)
                year_folder = RDS_FOLDER + '/' + hk_name + '/' + year
                rds_list = os.listdir(year_folder)
            
            # load rds data
            pattern = r'.*'+ date_str + r'\w*' + extension
            r = re.compile(pattern)
            # fnames = list(filter(r.match, mylist)) # Read Note below
            matches = [string for string in rds_list if re.search(pattern, string)]
            
            if len(matches) > 0:
                # get the last version
                fname = sorted(matches, key=lambda x:x[-6:-4])[-1]
                
                rds_path = RDS_FOLDER + '/' + hk_name + '/' + str(date.year) + '/' + fname
                try:
                    if hk_name == 'solo_ANC_sc-thermal-mag':
                        dtype = {
                            "IFA_HTR1_GR5_ST": str,
                            "IFA_HTR1_GR5_TSW5_ST": str,
                            "IFA_HTR2_GR5_ST": str,
                            "IFA_HTR2_GR5_TSW5_ST": str,
                            "IFA_HTR1_LCL5_TM": float,
                            "IFA_HTR2_LCL5_TM": float,
                            "IFB_HTR1_GR5_ST": str,
                            "IFB_HTR1_GR5_TSW5_ST": str,
                            "IFB_HTR2_GR5_ST": str,
                            "IFB_HTR2_GR5_TSW5_ST": str,
                            "IFB_HTR1_LCL5_TM": float,
                            "IFB_HTR2_LCL5_TM": float,
                            "ANP_1_2_2 MAG OBS": float,
                            "ANP_1_2_3 MAG IBS": float,
                            "ANP_2_1_6 MAG OBS": float,
                            "ANP_2_1_9 MAG IBS": float,
                            "ANP_3_1_13 MAG-OBS": float,
                            "ANP_3_1_14 MAG-IBS": float,
                            "ANP_1_2_2 MAG OBS RIUB": float,
                            "ANP_1_2_3 MAG IBS RIUB": float,
                            "ANP_2_1_6 MAG OBS RIUB": float,
                            "ANP_2_1_9 MAG IBS RIUB": float,
                            "ANP_3_1_13 MAG-OBS RIUB": float,
                            "ANP_3_1_14 MAG-IBS RIUB": float,
                        }
                        
                        # # now take out the two columns with continuous data and save 
                        # before June 22 2020 the datafiles are massive
                        # continuous_data = df[["IFB_HTR1_LCL5_TM", "IFB_HTR2_LCL5_TM"]].copy()
                        # on_off_data = df.drop(columns=["IFB_HTR1_LCL5_TM", "IFB_HTR2_LCL5_TM"]).dropna(how='all', axis = 0)
                    elif hk_name == 'solo_ANC_sc-lcl-mag':
                        dtype = {
                            "A_LCL2_16 MAG-B PWR ST": str,
                            "A_LCL4_17 MAG-A ST": str,
                            "A_LCL2_16 MAG-B PWR TM": float,
                            "A_LCL4_17 MAG-A TM": float,
                            "B_LCL2_16 MAG-B PWR ST": str,
                            "B_LCL4_17 MAG-A ST": str,
                            "B_LCL2_16 MAG-B PWR TM": float,
                            "B_LCL4_17 MAG-A T": float,
                        }
                    else:
                        dtype=False
                    rds_data = load_rds_file(rds_path, dtype = dtype, extension = extension)

                    
                    #save for next time
                    local_folder = LOCAL_HK_FOLDER + '/' + hk_name + '/' + str(date.year)
                    if not os.path.exists(local_folder):
                        os.makedirs(local_folder)

                    # if hk_name == 'solo_ANC_sc-thermal-mag':
                    #     continuous_data.to_csv(local_name[:-4]+ '_continous.txt')
                    #     on_off_data.to_csv(local_name[:-4]+ '_on_off.txt')
                    # else:
                        
                    rds_data.to_csv(local_name)
                    
                    dfs.append(rds_data)
                    
                except IndexError:
                    print(f'Skipped {date}')
                    pass
    
    df = pd.concat(dfs)
    
    return df

def load_rds_file(fname, dtype = False, extension = '.txt'):
    if extension == '.txt':
        names = ['Time']

        with open(fname,"r") as file_handler:
            # get the line with para descriptions
            # last [:-1] gets rid of the \n
            param_desc = file_handler.readlines()[9][:-1]
            names += param_desc.split('\t')[1:]
        if dtype is not False:
            df = pd.read_csv(fname, delimiter = '	', skiprows = 23, names=names, parse_dates=['Time'],infer_datetime_format=True, dtype = dtype)
        else:
            df = pd.read_csv(fname, delimiter = '	', skiprows = 23, names=names, parse_dates=['Time'],infer_datetime_format=True)
        # date_format = "%Y-%m-%dT%H:%M:%S.%f")
        df.set_index('Time', inplace=True)
    # elif extension == '.cdf':
        
    
    #drop all empty columns
    df.dropna(how='all',inplace=True, axis = 1)
    return df

###########################################################################################
# Helpers for generating dates
###########################################################################################

def get_dates(start, end):
    # get all the days between a start and end date
    date = start
    dates = []
    while date <= end:
        dates.append(date)
        date += timedelta(days = 1)
    return dates

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

def random_dates(N, start, end):
    # for the training and test data I want to pick a set of random profiles
    # not sure I used this in the end...
    
    N_to_choose_from = (end-start).days
    
    if N >= N_to_choose_from:
        raise Exception("Can't choose more days then the range given")
    
    dates = get_dates(start, end)
        
    return np.random.choice(dates, N, replace=False)

###########################################################################################
# Dealing with input heater profiles
###########################################################################################

def load_profile(fname, IBSorOBS = 'OBS'):
    with h5.File(fname, 'r') as data:
        # IBS_profile = data['IBS_profile'][:]
        # IBS_time = data['IBS_time'][:]
        profile = data[f'{IBSorOBS}_profile'][:1000]
        time = data[f'{IBSorOBS}_time'][:1000]
    return time, profile

def in_sklearn_format(features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER, test = False):
    # Turn the heater profiles into a DataFrame that SKLearn can understand
    
    # The heater profile is first downsampled to the same time stamps, e.g. every second for 15 minutes
    # That means that we can use "Time" as a feature, since all the profiles will share the same set of times
    
    # Each day will have a value for house keeping data, e.g. Temperature of instrument
    # this is just repeated for each value of Time.
    # i.e. For the 5th second with the house keeping of such a value, what is the prediction?
    
    # if test = True then I will still build the time data and features in, but obviously I don't have a real profile so just leave that blank
    
    if ds_period < 0:
        hp_time_bins = get_var_time_bins(ds_duration)
    else:
        hp_time_bins = np.arange(0, ds_duration, ds_period)

    dfs = []
    for i, row in features.iterrows():
        date = row['Date']
        fname = get_heater_fname(date, hp_folder)
        
        if fname != '':
            time, profile = load_profile(fname, IBSorOBS)
            
        
            R = profile[:,0]
            T = profile[:,1]
            N = profile[:,2]

            hp_df = pd.DataFrame({
                'R': R,
                'T': T,
                'N': N,
                'Time': time
            })
            
            hp_df = hp_df.loc[(hp_df['Time'] < ds_duration) & (hp_df['Time'] > 0)]
            hp_df['time_bin'] = np.digitize(hp_df['Time'], bins = hp_time_bins) 
            hp_ds = hp_df.groupby('time_bin').mean()
            hp_ds.reset_index(inplace=True)
            
            #! I have a choice to make here
            #! turns out that it was only going up to the time of the real profile
            #! do I keep this, or force every profile to last to ds_duration?
            #! but I can't have a np.nan in the training data
            # going for just times of real profile
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
            profile_ds = profile_ds.iloc[:1000]
            
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


def features2train(features, val_features, ds_duration, ds_period, IBSorOBS, split = 'profiles', weights = False, coord = 'rtn'):
    if split == 'days':
        if val_features is None:
            val_features = features.sample(n=features.shape[0]//5, random_state = 0)
            # Create a new DataFrame with the remaining rows
            features = features[~features.index.isin(val_features.index)].copy(deep=True)
        
        # take some random points out for testing
        test_features = features.sample(n=features.shape[0]//5, random_state = 0)
        # Create a new DataFrame with the remaining rows
        train_features = features[~features.index.isin(test_features.index)].copy(deep=True)
        
        train = in_sklearn_format(train_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
        val = in_sklearn_format(val_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
        test = in_sklearn_format(test_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
    
    elif split == 'profiles':
        if val_features is not None:
            val = in_sklearn_format(val_features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)
        all_data = in_sklearn_format(features, ds_duration, ds_period, IBSorOBS, hp_folder = HP_FOLDER)

        if val_features is None:
            val = all_data.sample(n=all_data.shape[0]//5, random_state = 0)
            all_data = all_data[~all_data.index.isin(val.index)].copy(deep=True)
        
        train = all_data.sample(n=all_data.shape[0]//5, random_state = 0)
        test = all_data[~all_data.index.isin(train.index)].copy(deep=True)
    else:
        # I used weight for a bit, but took out. 
        # Ideally, want the weight to be equal to "how reliable is this heater profile I am learning from?"
        raise 'BadBoy'
    
    # get the weights before scaling
    if weights:
        print('make weights represent the amount of variability in B on that day (i.e. how much can we trust the averaging)')
        raise 'BadBoy 1'
        # train_weights = get_weights(train['Time'].values, ds_duration)
        # test_weights = get_weights(test['Time'].values, ds_duration)
        # val_weights = get_weights(val['Time'].values, ds_duration)
    
    
    B, phi, theta = cart2sph(
        train['R'].values,
        train['T'].values,
        train['N'].values,
    )
    train['|B|'] = B
    train['phi'] = phi
    train['theta'] = theta
    
    B, phi, theta = cart2sph(
        val['R'].values,
        val['T'].values,
        val['N'].values,
    )
    val['|B|'] = B
    val['phi'] = phi
    val['theta'] = theta
    
    B, phi, theta = cart2sph(
        test['R'].values,
        test['T'].values,
        test['N'].values,
    )
    test['|B|'] = B
    test['phi'] = phi
    test['theta'] = theta
    
    # scaling
    no_scaling_names = [
        'hp_id',
        'Heater',
        'SA change',
        'HGA azimuth change',
        'HGA evelvation change',
        'No time A off'
    ]

    mapper_list = []

    for key in train.keys():
        if any(substring in key for substring in no_scaling_names):
            mapper_list.append((key, None))
        else:
            mapper_list.append(([key], RobustScaler()))

    mapper = DataFrameMapper(mapper_list, df_out = True)



    scaler = mapper.fit(train)
    train_scaled = mapper.transform(train)
    test_scaled = mapper.transform(test)
    val_scaled = mapper.transform(val)
    
    x_train = train_scaled.drop(['hp_id', 'R', 'T', 'N', '|B|', 'phi', 'theta'],axis = 1)
    x_val = val_scaled.drop(['hp_id', 'R', 'T', 'N', '|B|', 'phi', 'theta'], axis = 1)
    x_test = test_scaled.drop(['hp_id', 'R', 'T', 'N', '|B|', 'phi', 'theta'], axis = 1)
    
    if coord == 'sph':
        y_train = train_scaled[['|B|', 'phi', 'theta']].copy()
        y_val = val_scaled[['|B|', 'phi', 'theta']].copy()
        y_test = test_scaled[['|B|', 'phi', 'theta']].copy()
    else:
        y_train = train_scaled[['R', 'T', 'N']].copy()
        y_val = val_scaled[['R', 'T', 'N']].copy()
        y_test = test_scaled[['R', 'T', 'N']].copy()
    
    if weights:
        return x_train, x_val, x_test, y_train, y_val, y_test, mapper, train_weights, test_weights, val_weights
    else:
        return x_train, x_val, x_test, y_train, y_val, y_test, mapper

###########################################################################################
# Feature manipulation
###########################################################################################

def reduce_cardinality(features, feature_name, N, bins = None):
    if bins is None:
        bins = np.linspace(features[feature_name].min(), features[feature_name].max(), N)
    features['bin'] = np.digitize(features[feature_name], bins = bins) 
    # so just radius_bin as the feature, the model doesn't need to know its true value
    features.drop([feature_name], axis = 1, inplace = True)
    features.rename(columns = {'bin': feature_name}, inplace= True)
    return features

def mode(arr, bins = 10):
    arr = arr[~np.isnan(arr)] #~ means not

    if len(arr) > 0:
        hist, bin_edges = np.histogram(arr, bins=bins)
        centers = 0.5*(bin_edges[1:]+ bin_edges[:-1])
        max_idx = np.argmax(hist)
        mode = centers[max_idx]
        return mode
    else:
        #print('Just nans')
        return np.nan

def get_weights(time, ds_duration):
    # return np.cos((np.pi/2) * time/ds_duration)
    # return np.where(time < 70, 1, 0.1)
    A = 0.5
    B = 100
    C = 10
    return (1-A) + A/(1+ np.exp((time-B)/C))

def cart2sph(r,t,n):
    # if all (0,0,0) then return 0
    
    
    mag = np.sqrt(r*r + t*t + n*n)
    theta = np.arcsin(n / mag) * 180/np.pi
    
    phi = np.arctan2(t,r) * 180/np.pi
    
    phi = np.where(r==0, 0, phi)
    theta = np.where(r==0, 0, theta)
    
    return mag, phi, theta

def add_history(features, col_name, n_days_back, bfill = True):
    for days_back in np.arange(1, n_days_back+1):
        new_col_name = col_name + f' -{days_back}'
        features[new_col_name] = features[col_name].shift(days_back)
        if bfill:
            features[new_col_name].bfill(inplace=True)
    
    # do 1 in the future too
    new_col_name = col_name + f' 1'
    features[new_col_name] = features[col_name].shift(-1)
    if bfill:
        features[new_col_name].ffill(inplace=True)
    return features

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


    # imports
import pandas as pd
from datetime import datetime
import numpy as np
import helpers as h

import argparse
import logging

from sklearn.model_selection import train_test_split, KFold
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, RobustScaler
import skopt
from skopt.callbacks import EarlyStopper
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE

import lightgbm as lgb
from joblib import dump
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
        "--split",
        type=str,
        default='days',
        choices = ['days', 'profiles'],
        help="How to split training and test data. By selecting whole days, or just parts of all profiles",
    )
    
    # Define an argument for the boolean variable
    parser.add_argument('--cv', action='store_true', help='To perform outer CV or not')
    
    # Define an argument for the boolean variable
    parser.add_argument('--weights', '-w', action='store_true', help='To use weights or not')
    
    parser.add_argument(
        "--val",
        "-v",
        type=str,
        default='latest',
        choices = ['latest', 'random'],
        help="How to get the validation set",
    )
    
    # whether to use 
    parser.add_argument('--rfe', action='store_true', help='To use Recursive Feature Elimination (RFE) or not')
    
    parser.add_argument(
        "--coord",
        type=str,
        default='rtn',
        choices = ['rtn', 'sph'],
        help="Coordinate system",
    )

    args = parser.parse_args()
    print('Training with these settings:')
    print(args)
    

    # load data and features
    features_ = pd.read_csv(LOCAL_HK_FOLDER + "features.csv", parse_dates = ['Date'])

    

    # get rid of the bad times
    ranges = pd.read_csv("bad_dates.csv", parse_dates=['start', 'end'])
    #! need to also deal with duplicate days

    bad_dates = get_forbidden_dates(ranges)
    wb_features = features_[~features_['Date'].isin(bad_dates)]
    bad_features = features_[features_['Date'].isin(bad_dates)]
    
    
    ################################################################
    # Validation set ###############################################
    ################################################################
    
    if args.val == 'latest':
        # either take some profiles near the end
        val_date = datetime(2023,12,31)
        val_features = wb_features.loc[(wb_features['Date'] > val_date)]
        features = wb_features.loc[wb_features['Date'] < val_date]
    else:
        features = wb_features.loc[wb_features['Date'] < datetime(2022,2,23)]
        val_features = None
    
    

    # take some random days for validation
    # from test_cv I have changed features_after_val to features
    # val_features = wb_features.sample(n=30, random_state = 0)
    # features = wb_features[~wb_features.index.isin(val_features.index)]
    
    
    
    #! could make these input variables to the script
    ds_period = args.period
    ds_duration = args.duration
    IBSorOBS = args.inst

    HP_TIME_BINS = np.arange(0, ds_duration, ds_period)

    if args.coord == 'rtn':
        components = ['R', 'T', 'N']
    elif args.coord == 'sph':
        components = ['|B|', 'phi', 'theta']
        

    # Define outer CV for model selection and evaluation
    # maybe I don't want to shuffle because then there wont be a day nearby to learn from
    # would learn the relationships better?
    
    if args.cv:
        # take random parts of the training to validate the outer CV
        outer_cv = KFold(n_splits=10, shuffle=False)
    
    # the inner is used to train the BayesSearchCV method
    # therefore, I don't want to shuffle the data as it will just learn which days are good
    inner_cv = KFold(n_splits=3, shuffle=False)
    
    # looking at the last fitted model, there were 250-300+ estimators (boosting rounds)
    stopping_rounds = 20 #50
    early_stopping_callback = lgb.early_stopping(stopping_rounds=stopping_rounds)
    logging_callback = lgb.log_evaluation(period=stopping_rounds)
    
    # SPACE = {
    #     'learning_rate': skopt.space.Real(0.01, 0.3, prior='log-uniform'),
    #     'max_depth': skopt.space.Integer(2, 5),
    #     'n_estimators': skopt.space.Integer(stopping_rounds, 1000, prior='log-uniform'),
    #     'reg_lambda': skopt.space.Real(0, 100),
    #     'reg_alpha': skopt.space.Real(0, 100),
    #     'num_leaves': skopt.space.Integer(2, 1000),
    #     'min_data_in_leaf': skopt.space.Integer(10, 1000),
    #     'feature_fraction': skopt.space.Real(0.1, 1.0, prior='uniform'),
    #     'subsample': skopt.space.Real(0.1, 1.0, prior='uniform'),
    # }
    SPACE = {
        # if early stopping is on, then n_estimators doesn't really matter
        'n_estimators': skopt.space.Integer(2, 1000, prior='log-uniform'),
        'max_depth': skopt.space.Integer(2, 20),
        'min_samples_leaf': skopt.space.Integer(1, 10),
        'max_features': skopt.space.Integer(1, features.shape[0]),
        'min_samples_split': skopt.space.Integer(2, 10),
    }
    
    
    
    
    
    # based on this article https://medium.com/@gabrieltseng/gradient-boosting-and-xgboost-c306c1bcfaf5
    # seems like early stopping is good way to do it
    # so only need to worry about these:
    # n_estimators = (keep it low, but early stopping should help here) (try high once)
    # learning rate = small
    # max_depth = 5 at max
    # reg_alpha and reg_lambda

    # since these jobs are independent, I can run on different cores!
    
    # could use priors from previous fit?
    
    #! for each training, I need:
    # train
    # validation: to use for early stopping
    # test (to do the scoring) 
    
    #! so take a few measurements out of training and use them for validation
    #! Again, I want the validation to act as unseen data (-ish) so that it can't
    #! just figure out we are near this day. DONT USE RANDOM SHUFFLE
    
    if args.cv:
        print("This cole is very old and probably doesn't work. It uses outer validation to prove that the model can adapt and not overfit. Turns out you shouldn't use cross validation to train the model, so I focussed on something else. https://stackoverflow.com/questions/46456381/cross-validation-in-lightgbm/50316411#50316411")
        raise NotImplementedError

    else:

        x_train, x_val, x_test, y_train, y_val, y_test, mapper = features2train(
            features,
            val_features, 
            ds_duration,
            ds_period,
            IBSorOBS,
            args.split,
            coord = args.coord
        )
        
        

        
        if args.rfe:
            
            rfe = RFE(estimator = RandomForestRegressor(), n_features_to_select=20, step = 1, verbose=1)
            
            fittings1 = rfe.fit(x_train, y_train)
            
            for i in range(x_train.shape[1]):
                print(f'Column: {x_train.keys()[i]}, Selected {rfe.support_[i]}, Rank: {rfe.ranking_[i]:.3f}')
            
            x_train = rfe.transform(x_train)
            x_val = rfe.transform(x_val)
            x_test = rfe.transform(x_test)
            
            # save the columns to keep
            selected_features = rfe.support_
            
            dump(selected_features, f'Models/RF_interval/{IBSorOBS}_{args.tune}_{ds_period}_{ds_duration}_features.joblib')
            

        if args.tune == 'bayes':
            search = skopt.BayesSearchCV(RandomForestRegressor(), 
                                        SPACE, 
                                        cv=inner_cv, 
                                        n_jobs = 6, 
                                        pre_dispatch='n_jobs',
                                        refit = True)
            
            
        elif args.tune == 'untuned':
            search = RandomForestRegressor()
        
        elif args.tune == 'tuned':
            raise Exception("Find some good values")
            search = RandomForestRegressor()

        search.fit(x_train, y_train)
        
        # calculate the final scores
        if args.tune == 'bayes':
            score = search.best_estimator_.score(x_test, y_test)
            print("Test Score: ", score)
            score = search.best_estimator_.score(x_val, y_val)
            print("Val Score: ", score)
        else:
            score = search.score(x_test, y_test)
            print("Test Score: ", score)
            score = search.score(x_val, y_val)
            print("Val Score: ", score)
        
        # save the mapper
        dump(mapper, f"Models/RF_interval/{IBSorOBS}_{args.tune}_{ds_period}_{ds_duration}_scaler.joblib")
            
        # save the model
        dump(search, f"Models/RF_interval/{IBSorOBS}_{args.tune}_{ds_period}_{ds_duration}.joblib")