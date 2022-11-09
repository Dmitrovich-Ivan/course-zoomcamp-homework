import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import bentoml


df = pd.read_csv('medicalpremium.csv')

#Converting column names to lowercase
df.columns = df.columns.str.lower()

numerical = [
    'age', 
    'height', 
    'weight'
]

categorical = [
    'diabetes', 
    'bloodpressureproblems', 
    'anytransplants', 
    'anychronicdiseases', 
    'knownallergies', 
    'historyofcancerinfamily', 
    'numberofmajorsurgeries'
]

features = numerical + categorical

#Splitting the data into train - val - test samples 
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)

df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

y_train = np.log1p(df_train['premiumprice'].values)
y_val = np.log1p(df_val['premiumprice'].values)
y_test = np.log1p(df_test['premiumprice'].values)

del df_train['premiumprice']
del df_val['premiumprice']
del df_test['premiumprice']


#Preparing X_train, X_val 
dv = DictVectorizer(sparse = False)

train_dict = df_train[features].to_dict(orient = 'records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[features].to_dict(orient = 'records')
X_val = dv.transform(val_dict)

feature_names_out = dv.get_feature_names_out()


#Random forest model 
rf = RandomForestRegressor(n_estimators = 30, 
                           max_depth = 6, 
                           min_samples_leaf = 5, 
                           random_state = 1, 
                           n_jobs=-1)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

print(mean_squared_error(y_val, y_pred, squared=False))


#XGBoost model 
dtrain = xgb.DMatrix(X_train, label = y_train)
dval = xgb.DMatrix(X_val, label = y_val)

xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,}

watchlist = [(dtrain, 'train'), (dval, 'val')]

xgb_model = xgb.train(xgb_params, dtrain, num_boost_round = 30, evals = watchlist)
y_pred = xgb_model.predict(dval)

print(mean_squared_error(y_val, y_pred, squared = False))

#Saving the models using BentoML
saved_model_rf = bentoml.sklearn.save_model(
    'ins_price_rf_model',
    rf,
    custom_objects={
        'dictVectorizer': dv
    })
print("Model saved: ",saved_model_rf)

saved_model_xgb = bentoml.xgboost.save_model(
    'ins_price_xgb_model',
    xgb_model,
    custom_objects={
        'dictVectorizer': dv
    })
print("Model saved: ",saved_model_xgb)
