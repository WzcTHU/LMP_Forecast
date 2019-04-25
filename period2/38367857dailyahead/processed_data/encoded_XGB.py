from xgboost import XGBRegressor
from func import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy
import time


def xgb_reg(x_train, y_train, x_test):
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    scaler1.fit(x_train)
    scaler2.fit(y_train)
    x_train = scaler1.transform(x_train)
    y_train = scaler2.transform(y_train)
    x_test = scaler1.transform(x_test)

    # param_test1 ={'subsample':[0.2, 0.4, 0.6, 0.8, 1]}
    # gsearch1 = GridSearchCV(XGBRegressor(reg_lambda=1, reg_alpha=0, max_depth=1, gamma=0.1, \
    #                                      colsample_bytree=0.8, subsample=1, n_estimators=70, \
    #                                      learning_rate=0.1, min_child_weight=1), param_grid=param_test1, \
    #                                      scoring='r2', cv=5)
    # gsearch1.fit(x_train, y_train.ravel())
    # print(gsearch1.best_score_, gsearch1.best_params_)
    x_train_encoded, x_test_encoded = get_encoded(x_train, x_test)

    regressor = XGBRegressor(reg_lambda=8, gamma=0.01, max_depth=3, colsample_bytree=0.4, \
                             subsample=0.6, n_estimators=400, learning_rate=0.1)
    # regressor = XGBRegressor(reg_lambda=1, reg_alpha=0, max_depth=1, gamma=0.1, \
    #                          colsample_bytree=0.8, subsample=1, n_estimators=70, \
    #                          learning_rate=0.1, min_child_weight=1)
    regressor.fit(x_train_encoded, y_train.ravel())

    y_fore_train = regressor.predict(x_train_encoded)
    y_fore_train = scaler2.inverse_transform(y_fore_train)
    y_train = scaler2.inverse_transform(y_train)

    y_fore_test = regressor.predict(x_test_encoded)
    y_fore_test = scaler2.inverse_transform(y_fore_test)
    return y_fore_test, y_fore_train


if __name__ == '__main__':
    start_time = time.time()
    x_train, y_train, x_test, y_test = divide_data('x.csv', 'y.csv')
    y_fore_test, y_fore_train = xgb_reg(x_train, y_train, x_test)

    end_time = time.time()

    get_results(y_train, y_fore_train, y_test, y_fore_test, end_time, start_time)