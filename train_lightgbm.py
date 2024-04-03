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
        

    array_idx = int(os.environ['PBS_ARRAY_INDEX'])
    counter = 0


    
    

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
    if args.coord == 'sph':
        stopping_rounds = 50
    else:
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
        'learning_rate': skopt.space.Real(0.01, 0.3, prior='log-uniform'),
        'reg_lambda': skopt.space.Real(0, 100),
        'reg_alpha': skopt.space.Real(0, 100),
        'max_depth': skopt.space.Integer(2, 10),
        'num_leaves': skopt.space.Integer(2, 100),
        'min_data_in_leaf': skopt.space.Integer(10, 1000),
        'feature_fraction': skopt.space.Real(0.1, 1.0, prior='uniform'),
    }
    
    
    
    
    
    # based on this article https://medium.com/@gabrieltseng/gradient-boosting-and-xgboost-c306c1bcfaf5
    # seems like early stopping is good way to do it
    # so only need to worry about these:
    # n_estimators = (keep it low, but early stopping should help here) (try high once)
    # learning rate = small
    # max_depth = 5 at max
    # reg_alpha and reg_lambda

    # since these jobs are independent, I can run on different cores!
    for component in components:
        counter += 1
        if counter == array_idx:
            print(component)
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
                #! this won't work because I haven't made a test set yet
                #! once you figure out just inner CV then return to this
                
                models = []
                parameters = []
                scores = []
                #! this is doing args.split == days
                #! haven't tried the profiles one
                for train_index, val_index in outer_cv.split(wb_features):
                    print('')
                    train_features = wb_features.iloc[train_index]
                    test_features = wb_features.iloc[val_index]
                    print('Validation start: ', val_features['Date'].values[0])
                    print('Validation end: ', val_features['Date'].values[-1])
                    
                    train = h.in_sklearn_format(train_features, ds_duration, ds_period, IBSorOBS, hp_folder = h.HP_FOLDER)
                    val = h.in_sklearn_format(val_features, ds_duration, ds_period, IBSorOBS, hp_folder = h.HP_FOLDER)
                    
                    x_train = train.drop(["hp_id", 'R', 'T', 'N'],axis = 1)
                    x_val = val.drop(["hp_id", 'R', 'T', 'N'], axis = 1)

                    y_train = train[['R', 'T', 'N']].copy()
                    y_val = val[['R', 'T', 'N']].copy()
                
                    
                    

                    if args.tune == 'bayes':

                        # inner cv parameter estimation
                        # Define inner CV for parameter search
                        # could just swap this out for RandomSearchCV or GridSearchCV
                        lgb_base = lgb.LGBMRegressor()
                        # Define the early stopping callback
                        

                        #this scoring is negative (lower is better) MSE
                        # this is what the BayesSearchCV method is optimising for
                        search = skopt.BayesSearchCV(lgb_base, 
                                                        SPACE, 
                                                        cv=inner_cv, 
                                                        n_jobs = 6, 
                                                        pre_dispatch='n_jobs')
                        # Fit the bayes search model
                        
                        

                    elif args.tune == 'untuned':
                        search = lgb.LGBMRegressor()
                    
                    # following this article https://www.kaggle.com/questions-and-answers/217679
                    # just put the fit parameters here, they should be passed on in BayesSearchCV 
                    result = search.fit(x_train, y_train[component], 
                                        eval_metric='mae', 
                                        eval_set=[(x_val, y_val[component])], 
                                        callbacks = [early_stopping_callback, logging_callback])
                    
                    
                    # Evaluate the model on the val data and store the accuracy score
                    score = search.score(x_val, y_val[component])
                    scores.append(score)
                    
                    print("Score on this validation: ", score)
                    
                    # important to print the best parameters and score, to see if results are stable
                print('')
                print(scores)

                # Compute the mean and standard deviation of the accuracy scores
                mean_score = sum(scores) / len(scores)
                std_score = np.std(scores)
                
                #this reports how good the fitting method was on generalised data
                print('')
                print('Mean accuracy: %.3f (%.3f)' % (mean_score, std_score))


                # train on all data now. (could keep some data it never ever sees)
                # think I just re-defined test and validation data myself here...
                train = h.in_sklearn_format(train_features, ds_duration, ds_period, IBSorOBS, hp_folder = h.HP_FOLDER)
                test = h.in_sklearn_format(test_features, ds_duration, ds_period, IBSorOBS, hp_folder = h.HP_FOLDER)

                x_train = train.drop(["hp_id", 'R', 'T', 'N'],axis = 1)
                y_train = train[['R', 'T', 'N']].copy()
                
                x_test = test.drop(["hp_id", 'R', 'T', 'N'],axis = 1)
                y_test = test[['R', 'T', 'N']].copy()
                
                #! now what to do?
                print('')
                print('Train final model')
                # refit on all data with the best parameters?
                if args.tune == 'bayes':
                    lgb_base = lgb.LGBMRegressor()
                    final_model = skopt.BayesSearchCV(lgb_base, 
                                                SPACE, 
                                                cv=inner_cv, 
                                                n_jobs = 6, 
                                                pre_dispatch='n_jobs')

                elif args.tune == 'untuned':
                    final_model = lgb.LGBMRegressor()
                
                # the non-shuffling makes the models better, but you have taken random test days 
                # this final training should include a better validation set that isn't the test set
                
                final_model.fit(x_train, y_train[component], 
                                eval_metric='mae', 
                                eval_set=[(x_test, y_test[component])], 
                                callbacks = [early_stopping_callback, logging_callback])

                
                #! This doesn't really matter as a score I don't think, 
                #! cross validation is more important
                print(f'Final score: {final_model.score(x_test, y_test[component]):.3f}')
                # or just use the best model?
                # you might just pick the luckiest model here so be careful
                
                dump(final_model, f"Models/LGB/{IBSorOBS}_{component}_{args.tune}_{ds_period}_{ds_duration}.joblib")
            else:
                
                
                x_train, x_val, x_test, y_train, y_val, y_test, mapper = h.features2train(
                    features,
                    val_features, 
                    ds_duration,
                    ds_period,
                    IBSorOBS,
                    args.split,
                    coord = args.coord
                )

                
                if args.weights:
                    # weights proportional to |B|
                    max_B = max([y_train['|B|'].max(), y_test['|B|'].max(), y_val['|B|'].max()])
                    train_weights = y_train['|B|']/max_B
                    test_weights = y_test['|B|']/max_B
                    valweights = y_val['|B|']/max_B
                
                if args.rfe:
                    
                    rfe = RFE(estimator = lgb.LGBMRegressor(), n_features_to_select=20, step = 1, verbose=1)
                    
                    fittings1 = rfe.fit(x_train, y_train[component])
                    
                    for i in range(x_train.shape[1]):
                        print(f'Column: {x_train.keys()[i]}, Selected {rfe.support_[i]}, Rank: {rfe.ranking_[i]:.3f}')
                    
                    x_train = rfe.transform(x_train)
                    x_val = rfe.transform(x_val)
                    x_test = rfe.transform(x_test)
                    
                    # save the columns to keep
                    selected_features = rfe.support_
                    
                    dump(selected_features, f'Models/LGB/{IBSorOBS}_{component}_{args.tune}_{ds_period}_{ds_duration}_features.joblib')
                    

                if args.tune == 'bayes':
                    # inner cv parameter estimation
                    # Define inner CV for parameter search
                    # could just swap this out for RandomSearchCV or GridSearchCV
                    lgb_base = lgb.LGBMRegressor()
                    
                    #this scoring is negative (lower is better) MSE
                    # this is what the BayesSearchCV method is optimising for
                    search = skopt.BayesSearchCV(lgb_base, 
                                                SPACE, 
                                                cv=inner_cv, 
                                                n_jobs = 6, 
                                                pre_dispatch='n_jobs',
                                                refit = True)
                    
                   
                elif args.tune == 'untuned':
                    search = lgb.LGBMRegressor(first_metric_only = True,
                                )
                
                elif args.tune == 'tuned':
                    search = lgb.LGBMRegressor(
                        n_estimators=1000,
                        max_depth = 20,
                        learning_rate = 0.05,
                        reg_lambda = 100,
                        reg_alpha = 10
                    )
                
                
                fit_args = {
                    'eval_metric': ['l2','l1'],
                    'eval_set': [(x_val, y_val[component])],
                }
                if args.weights:
                    fit_args['sample_weight'] = train_weights
                    fit_args['eval_sample_weight'] = [val_weights]
                
                
                if args.tune != 'bayes':
                    fit_args['callbacks'] = [early_stopping_callback, logging_callback]#[early_stopping_callback, logging_callback]
                else:
                    fit_args['callbacks'] = [logging_callback]
                
                # fit the model
                search.fit(x_train, y_train[component], **fit_args)
                
                
                # set up score arguments
                score_args_test = {}
                score_args_val = {}
                
                if args.weights:
                    score_args_test['sample_weight'] = test_weights
                    score_args_val['sample_weight'] = val_weights
                
                # calculate the final scores
                if args.tune == 'bayes':
                    score = search.best_estimator_.score(x_test, y_test[component], **score_args_test)
                    print("Test Score: ", score)
                    score = search.best_estimator_.score(x_val, y_val[component], **score_args_val)
                    print("Val Score: ", score)
                else:
                    score = search.score(x_test, y_test[component], **score_args_test)
                    print("Test Score: ", score)
                    score = search.score(x_val, y_val[component], **score_args_val)
                    print("Val Score: ", score)
                
                if component == 'N' or component == '|B|':
                    # save the scaler too
                    dump(mapper, f"Models/LGB/{IBSorOBS}_{args.tune}_{ds_period}_{ds_duration}_scaler.joblib")
                    
                # save the model
                if component == '|B|':
                    dump(search, f"Models/LGB/{IBSorOBS}_B_{args.tune}_{ds_period}_{ds_duration}.joblib")
                else:
                    dump(search, f"Models/LGB/{IBSorOBS}_{component}_{args.tune}_{ds_period}_{ds_duration}.joblib")