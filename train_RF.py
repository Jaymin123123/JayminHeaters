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
    features_ = pd.read_csv(h.LOCAL_HK_FOLDER + "features.csv", parse_dates = ['Date'])

    

    # get rid of the bad times
    ranges = pd.read_csv("bad_dates.csv", parse_dates=['start', 'end'])
    #! need to also deal with duplicate days

    bad_dates = h.get_forbidden_dates(ranges)
    wb_features = features_[~features_['Date'].isin(bad_dates)]
    bad_features = features_[features_['Date'].isin(bad_dates)]
    
    
    ################################################################
    # Validation set ###############################################
    ################################################################
    
    if args.val == 'latest':
        # either take some profiles near the end
        val_date = datetime(2022,12,31)
        val_features = wb_features.loc[(wb_features['Date'] > val_date)]
        features = wb_features.loc[wb_features['Date'] < val_date]
    else:
        features = wb_features.loc[wb_features['Date'] < datetime(2023,2,23)]
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

        x_train, x_val, x_test, y_train, y_val, y_test, mapper = h.features2train_interval(
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