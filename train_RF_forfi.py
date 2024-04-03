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
from sklearn.model_selection import train_test_split, KFold
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, RobustScaler
import lightgbm as lgb
from joblib import dump
import os

if __name__ == "__main__":

    features_ = pd.read_csv(h.LOCAL_HK_FOLDER + "features.csv", parse_dates = ['Date'])
    
    ranges = pd.read_csv("bad_dates.csv", parse_dates=['start', 'end'])
    #! need to also deal with duplicate days

    bad_dates = h.get_forbidden_dates(ranges)
    wb_features = features_[~features_['Date'].isin(bad_dates)]
    bad_features = features_[features_['Date'].isin(bad_dates)]

    # define the features and target
    ds_period = -1
    ds_duration = 900
    IBSorOBS = 'OBS'
    components = ['R', 'T', 'N']
    features = wb_features.loc[wb_features['Date'] < datetime(2023,10,10)]

    SPACE = {
    # if early stopping is on, then n_estimators doesn't really matter
    'n_estimators': skopt.space.Integer(2, 1000, prior='log-uniform'),
    'max_depth': skopt.space.Integer(2, 20),
    'min_samples_leaf': skopt.space.Integer(1, 10),
    'max_features': skopt.space.Integer(1, features.shape[0]),
    'min_samples_split': skopt.space.Integer(2, 10),
    }
    start_date = features_['Date'].min()
    end_date = features_['Date'].max()
    current_date = start_date
    search = RandomForestRegressor()
    while current_date < end_date:
        next_date = current_date + pd.Timedelta(days=100)
        interval_data = features[(features['Date'] >= current_date) & (features['Date'] < next_date)]
        interval_val_features = interval_data.sample(n=20, random_state=0)  # Ensure this samples correctly
        interval_train_features = interval_data[~interval_data.index.isin(interval_val_features.index)]

        x_train, x_val, x_test, y_train, y_val, y_test, mapper = h.features2train(
            interval_data,
            interval_val_features, 
            ds_duration,
            ds_period,
            IBSorOBS,
            'days',
            coord = 'rtn'
            )

        
        search.fit(x_train, y_train)

        score = search.score(x_test, y_test)
        print("Test Score: ", score)
        score = search.score(x_val, y_val)
        print("Val Score: ", score)

        date_str = current_date.strftime('%Y-%m-%d')
        feature_importances = search.feature_importances_
        importance_percentages = 100 * (feature_importances / feature_importances.sum())
        feature_names = interval_train_features.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance (%)': importance_percentages}).sort_values(by='Importance (%)', ascending=False)
        importance_df.to_csv(f'Models/RF_MFI/{IBSorOBS}_untuned_{ds_period}_{ds_duration}_importances_{date_str}.csv')

        # save the mapper
        dump(mapper, f"Models/RF_MFI/{IBSorOBS}_untuned_{ds_period}_{ds_duration}_scaler_{date_str}.joblib")

        # save the model
        dump(search, f"Models/RF_MFI/{IBSorOBS}_untuned_{ds_period}_{ds_duration}_{current_date}.joblib")

        current_date = next_date

        #del search
        #del selector
        #del estimator
        #del x_train
        #del x_val
        #del x_test
        #del y_train
        #del y_val
        #del y_test
        #del interval_data
        #del interval_val_features
        #del interval_train_features
        #del feature_importances
        #del importance_percentages
        #del feature_names
        #del importance_df
        #del mapper



