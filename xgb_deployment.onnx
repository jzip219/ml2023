import numpy as np
import time
import pandas as pd 
import random
import xgboost as xgb

from mytimer import mytimer
from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostRegressor,\
                             ExtraTreesRegressor,\
                             GradientBoostingRegressor,\
                             RandomForestClassifier,\
                             RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,\
                            confusion_matrix,\
                            classification_report,\
                            mean_squared_error,\
                            r2_score
from sklearn.tree import DecisionTreeRegressor

import json
from datetime import datetime
dts = datetime.now().strftime('_%m_%d_%Y_%H_%M_%S_')


#create timer instance
t=mytimer()
# ## Read Data
wine_raw = pd.read_csv('data/winequality-white.csv', delimiter=';')
headers = wine_raw.columns.values.tolist()
# ## Process Data
npwine_raw = wine_raw.to_numpy()
X = npwine_raw[:, :-1]
y = npwine_raw[:, -1] #-1 : (4989,)
###############################################
iters = 13
forest = {}
mindepth = 2
maxdepth = 6
reg=""
for dx in range(mindepth, maxdepth+1):
    times = np.zeros(iters,dtype=float)
    r2scores = np.zeros(iters,dtype=float)
    for iidx in range(iters):
        nestimators = 25*(iidx+1)
        t.start()
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=np.random.randint(0, 429496729))
        reg = xgb.XGBRegressor(n_estimators=nestimators,
                                reg_lambda=1,
                                gamma=0,
                                max_depth=dx).fit( X_train, y_train)
        y_pred = reg.predict(X_test)
        times[iidx]=t.stop()
        r2scores[iidx]=r2_score(y_test, y_pred)
        print(reg)
        forest[str(reg)[:str(reg).index('(')]]={'estimators' : str(nestimators), 'max depth' : str(dx), 'r2scores' : str(r2scores), 'times' : str(times), 'max' : str(np.max(r2scores)), 'r2avg' : str(np.sum(r2scores) / len(r2scores))}


        with open("testruns/jsonforest_mxd"+str(dx)+"_est"+str(nestimators)+dts+".json", 'a') as fout:
            json_dump = json.dumps(forest, indent=4)
            print(json_dump, file=fout)
model = xgb.XGBRegressor(n_estimators=325,
                        reg_lambda=1,
                        gamma=0,
                        max_depth=5).fit( X_train, y_train)
#deploy model to github
import torch
model.eval()
test=torch.randn(1,11)
input_names=['input1']
output_names =['output1']

torch.onnx.export(
    model,
    test,
    'XGB_Winedata.onnx',
    verbose=False,
    input_names =input_names,
    output_names=output_names
)
