from DataStandardScaler import *
from DataCut import *
from SummaryResults import *
from xgboost import XGBRegressor
from sklearn.externals import joblib
import scipy.io as sio


print('Cutting dataset...')
data = DataCut('data/x.csv', 'data/y.csv')
data.cut()

print('Data standardizating...')
data_scaler = DataStandardScaler(data.train_xset, data.train_yset, 
    data.validation_xset, data.validation_yset)

print('XGB training...')
# regressor = XGBRegressor(n_estimators=100)
regressor = XGBRegressor()
regressor.fit(data_scaler.x_train_standard, data_scaler.y_train_standard.ravel())
joblib.dump(regressor, 'models/xgb_model.m')

y_fore_train = regressor.predict(data_scaler.x_train_standard)
y_fore_validation = regressor.predict(data_scaler.x_validation_standard)
data_scaler.reverse_trans(y_fore_train, y_fore_validation)

print('Getting results...')
sum_res_train = SummaryResults(data.train_yset, data_scaler.rev_y_train)
sum_res_validation = SummaryResults(data.validation_yset, data_scaler.rev_y_validation)
sio.savemat('ForecastResult/Validation/XGB.mat', {'XGBfore': data_scaler.rev_y_validation})
sum_res_train.get()
sum_res_validation.get()
res_list =  sum_res_validation.cal_residual()
sio.savemat('res/XGBres.mat', {'XGB_res': res_list})
print(sum_res_validation.cal_variance())
