from sklearn.neural_network import MLPRegressor
from func import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

scaler1 = StandardScaler()
scaler2 = StandardScaler()

x_train, y_train, x_test, y_test = divide_data('x_rr.csv', 'y.csv')

scaler1.fit(x_train)
scaler2.fit(y_train)
x_train = scaler1.transform(x_train)
y_train = scaler2.transform(y_train)
x_test = scaler1.transform(x_test)
y_test = scaler2.transform(y_test)

x_train_encoded, x_test_encoded = get_encoded(x_train, x_test)

start_time = time.time()
regressor = MLPRegressor(solver='adam', 
    activation='relu', max_iter=70000, 
    hidden_layer_sizes=(400, 300, 150))
regressor.fit(x_train_encoded, y_train.ravel())
end_time = time.time()

y_fore_train = regressor.predict(x_train_encoded)
y_fore_train = scaler2.inverse_transform(y_fore_train)
y_train = scaler2.inverse_transform(y_train)

y_fore_test = regressor.predict(x_test_encoded)
y_fore_test = scaler2.inverse_transform(y_fore_test)
y_test = scaler2.inverse_transform(y_test)

get_results(y_train, y_fore_train, y_test, y_fore_test, end_time, start_time)
