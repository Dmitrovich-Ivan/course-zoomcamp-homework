import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import bentoml

#LOAD THE DATA
df = pd.read_csv('oil_spill.csv')
data = df.copy()
data.drop(['f_1', 'f_23'], axis=1, inplace=True)

#PREPARE TRAINING SETS
df_full_train, df_test = train_test_split(data, test_size = 0.2, random_state = 1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)

df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

y_train = (df_train['target'].values)
y_val = (df_val['target'].values)
y_test = (df_test['target'].values)

del df_train['target']
del df_val['target']
del df_test['target']

train_dict = df_train.to_dict(orient = 'records')
val_dict = df_val.to_dict(orient = 'records')
dv = DictVectorizer(sparse = False)

X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)

feature_names_out = dv.get_feature_names_out()

dtrain = xgb.DMatrix(X_train, label = y_train, feature_names = feature_names_out)
dval = xgb.DMatrix(X_val, label = y_val, feature_names = feature_names_out)

xgb_params = {
    'eta': 0.3, 
    'max_depth': 10,
    'min_child_weight': 3,
    'objective': 'binary:logistic',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
    'eval_metric': 'auc'}

watchlist = [(dtrain, 'train'), (dval, 'val')]

#TRAINING THE MODEL
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round = 20, evals = watchlist)

y_pred = xgb_model.predict(dval)

print(roc_auc_score(y_val, y_pred))

#EXPORTING THE MODEL
saved_model_xgb = bentoml.xgboost.save_model(
    'oil_spill_clf_xgb_model',
    xgb_model,
    custom_objects={
        'dictVectorizer': dv
    })

print("Model saved: ",saved_model_xgb)