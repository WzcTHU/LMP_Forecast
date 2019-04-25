from func import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

scaler1 = StandardScaler()
scaler2 = StandardScaler()

x_train, y_train, x_test, y_test, spike_train = divide_data('0702_1001_lmp.csv', '0702_1001_load.csv')

scaler1.fit(x_train)
scaler2.fit(y_train)
x_train = scaler1.transform(x_train)
y_train = scaler2.transform(y_train)
x_test = scaler1.transform(x_test)
y_test = scaler2.transform(y_test)

# param_test1 ={'n_estimators':range(50,201,10), 'max_features':[1, 2, 3, 4, 5]}
# gsearch1 = GridSearchCV(RandomForestRegressor(max_depth=3), param_grid=param_test1, \
#                         scoring='neg_mean_squared_error', cv=5)
# gsearch1.fit(x_train, y_train.ravel())
# print(gsearch1.best_score_, gsearch1.best_params_)

start_time = time.time()
regressor = RandomForestRegressor(n_estimators=60, max_depth=5, max_features=2, oob_score=True)
regressor.fit(x_train, y_train.ravel())
end_time = time.time()

y_fore_train = regressor.predict(x_train)
y_fore_train = scaler2.inverse_transform(y_fore_train)
y_train = scaler2.inverse_transform(y_train)

y_fore_test = regressor.predict(x_test)
y_fore_test = scaler2.inverse_transform(y_fore_test)
y_test = scaler2.inverse_transform(y_test)

get_results(y_train, y_fore_train, y_test, y_fore_test, end_time, start_time)
