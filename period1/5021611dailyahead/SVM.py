from sklearn import svm
from func import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy
import time


scaler1 = StandardScaler()
scaler2 = StandardScaler()

x_train, y_train, x_test, y_test, spike_train = divide_data('0702_1001_lmp.csv', '0702_1001_load.csv')

scaler1.fit(x_train)
scaler2.fit(y_train)
x_train = scaler1.transform(x_train)
y_train = scaler2.transform(y_train)
x_test = scaler1.transform(x_test)
y_test = scaler2.transform(y_test)

start_time = time.time()
regressor = svm.SVR(C=0.1, gamma=0.01, max_iter=10000, kernel='rbf')
regressor.fit(x_train, y_train.ravel())
end_time = time.time()

y_fore_train = regressor.predict(x_train)
y_fore_train = scaler2.inverse_transform(y_fore_train)
y_train = scaler2.inverse_transform(y_train)

y_fore_test = regressor.predict(x_test)
y_fore_test = scaler2.inverse_transform(y_fore_test)
y_test = scaler2.inverse_transform(y_test)

print('time_cost in training is:', end_time - start_time)

get_results(y_train, y_fore_train, y_test, y_fore_test, end_time, start_time)
