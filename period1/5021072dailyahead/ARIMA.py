import numpy
from func import *
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA 

lmp, *_ = read_data('0702_1001_lmp.csv', '0702_1001_load.csv')
lmp_train = lmp[30*24: 60*24]
lmp_test = lmp[60*24: 61*24]

lmp_train = pd.Series(lmp_train)
# plot_acf(lmp_train)
# plot_pacf(lmp_train)
# plt.show()
# print(adfuller(lmp_train))

# diff1 = lmp_train.diff(1).dropna()
# plot_acf(diff1)
# plot_pacf(diff1)
# plt.show()
# print(adfuller(diff1))
# diff2 = diff1.diff(1).dropna()

# print(adfuller(diff2))
# print(acorr_ljungbox(diff2, lags=1))
# plot_acf(diff2)
# plot_pacf(diff2, lags=1000)
# plt.show()

model = ARIMA(lmp_train, (6, 0, 4)).fit()
result = model.forecast(24)
plt.plot(result[0], marker='o')
plt.plot(lmp_test, marker='*')
plt.show()


