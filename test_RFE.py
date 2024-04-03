#%%
# imports
import sys
sys.path.append("../")
import pandas as pd
from datetime import datetime
import numpy as np
import helpers as h

import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
%load_ext autoreload
%autoreload 2

# load data and features
features_ = pd.read_csv(h.LOCAL_HK_FOLDER + "features.csv", parse_dates = ['Date'])

val_date = datetime(2023,1,1)
val_features = features_.loc[features_['Date'] >= val_date]
features = features_.loc[features_['Date'] < val_date]

#%%

ds_duration = 300
ds_period = 1
IBSorOBS = 'OBS'

x_train, x_val, x_test, y_train, y_val, y_test, mapper = h.features2train(
                    features,
                    val_features, 
                    ds_duration,
                    ds_period,
                    IBSorOBS,
                    'profiles',
                    coord = 'sph'
)

#%%

min_features_to_select = 1
rfc = lgb.LGBMRegressor() #RandomForestRegressor()
rfe = RFE(estimator=rfc, n_features_to_select= 10, step=1)
fittings1 = rfe.fit(x_train, y_train['N'])

for i in range(x_train.shape[1]):
    print(f'Column: {x_train.keys()[i]}, Selected {rfe.support_[i]}, Rank: {rfe.ranking_[i]:.3f}')