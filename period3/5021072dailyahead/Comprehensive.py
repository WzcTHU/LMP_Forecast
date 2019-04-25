import numpy as np
from sklearn.externals import joblib
from DataStandardScaler import *
from DataCut import *
from SummaryResults import *
import scipy.io as sio

# LGBM, MLP, RF, SVM, XGB, RNN
W = sio.loadmat('res/models_W.mat').get('W_value')
print('loading results from each model...')
y_LGBM = sio.loadmat('ForecastResult/Validation/LGBM.mat').get('LGBMfore')[0]
y_MLP = sio.loadmat('ForecastResult/Validation/MLP.mat').get('MLPfore')[0]
y_RF = sio.loadmat('ForecastResult/Validation/RF.mat').get('RFfore')[0]
y_SVM = sio.loadmat('ForecastResult/Validation/SVM.mat').get('SVMfore')[0]
y_XGB = sio.loadmat('ForecastResult/Validation/XGB.mat').get('XGBfore')[0]
y_RNN = sio.loadmat('ForecastResult/Validation/RNN.mat').get('RNNfore')[0]

print('Cutting dataset...')
data = DataCut('data/x.csv', 'data/y.csv')
data.cut()

y_combine = []

for i in range(0, len(y_LGBM)):
    y_combine.append(W[0] * y_LGBM[i] + W[1] * y_MLP[i]+ W[2] * y_RF[i] + \
        W[3] * y_SVM[i] + W[4] * y_XGB[i] + W[5] * y_RNN[i])

print('Getting results...')
sum_res_validation = SummaryResults(data.validation_yset, y_combine)
print(sum_res_validation.cal_variance())
sum_res_validation.get()