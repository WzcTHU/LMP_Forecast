from mlxtend.regressor import StackingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from func import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy
import time
from sklearn.multioutput import MultiOutputRegressor


# import the data and make preprocessing
scaler1 = StandardScaler()
scaler2 = StandardScaler()

x_train, y_train, x_test, y_test, spike_train = divide_data('0702_1001_lmp.csv', '0702_1001_load.csv')

scaler1.fit(x_train)
scaler2.fit(y_train)
x_train = scaler1.transform(x_train)
y_train = scaler2.transform(y_train)
x_test = scaler1.transform(x_test)
y_test = scaler2.transform(y_test)

# generate basic models
xgb = XGBRegressor(reg_lambda=7, gamma=0.01, max_depth=2, colsample_bytree=0.4, subsample=0.6, \
                   n_estimators=150, learning_rate=0.1)
rf = RandomForestRegressor(n_estimators=60, max_depth=5, max_features=2)
lgbm = LGBMRegressor(reg_lambda=7, num_leaves=4, max_depth=3)
svr = svm.SVR(C=0.1, gamma=0.01, max_iter=-1, kernel='rbf')
mlp = MLPRegressor(solver='adam', activation='relu', max_iter=50000, hidden_layer_sizes=(100, 80))

start_time = time.time()
stack = StackingRegressor(regressors=(svr, lgbm, rf, xgb, mlp), meta_regressor=xgb)
stack.fit(x_train, y_train.ravel())
end_time = time.time()

for clf, label in zip([xgb, rf, lgbm, svr, mlp, stack], ['XGB', 'RF','LGBM','SVR', 'MLP', 'STACK']):
    scores = cross_val_score(clf, x_train, y_train.ravel(), cv=5, scoring='neg_mean_squared_error')
    print('MSE Score: %0.4f (+/- %0.4f) [%s]' % (scores.mean(), scores.std(), label))

y_fore_train = stack.predict(x_train)
y_fore_train = scaler2.inverse_transform(y_fore_train)
y_train = scaler2.inverse_transform(y_train)

y_fore_test = stack.predict(x_test)
y_fore_test = scaler2.inverse_transform(y_fore_test)
y_test = scaler2.inverse_transform(y_test)

get_results(y_train, y_fore_train, y_test, y_fore_test, end_time, start_time)