import numpy as np
import random
import pandas as pd
import torch
import matplotlib.pyplot as plt

mask = [0, 1, 2, 3, 4, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 
    25, 27, 28, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]

TOTAL_DAYS = 397
TEST_DAYS = 44
SKIP_DAYS = 6
SKIP_HOURS = 24 * SKIP_DAYS
FILE_LIST = ['gen_by_fuel_type_20171001_20181101_processed.csv', 
            'gen_outage_20171001_20181101_processed.csv',
            'hrl_load_prelim_20171001_20181101_processed.csv',
            'lmp_data_20171001_20181101_processed.csv',
            'load_frcstd_hist_20171001_20181101_processed.csv',
            'total_lmp_data_20171001_20181101_processed.csv']

def make_batch(x_train, y_train, batch_size):
    i = random.randint(0, len(x_train) - batch_size)
    if ((i + batch_size) > len(x_train)):
        batch_xs = x_train[i: -1]
        batch_ys = y_train[i: -1]
    else:
        batch_xs = x_train[i: i + batch_size]
        batch_ys = y_train[i: i + batch_size]
    return [batch_xs, batch_ys]

def read_data(file_list):
    df = pd.DataFrame()
    df_fuel_type = pd.read_csv(file_list[0])
    df_outage = pd.read_csv(file_list[1])
    df_load_pre = pd.read_csv(file_list[2])
    df_comp_lmp = pd.read_csv(file_list[3])
    df_load_fore = pd.read_csv(file_list[4])
    df_total_lmp = pd.read_csv(file_list[5])

    y = df_total_lmp.iloc[SKIP_HOURS:, 1:].reset_index()
    y.to_csv('y_t.csv')

    temp_df = df_fuel_type.iloc[SKIP_HOURS:, 1:].reset_index()
    # df = df.append(df_fuel_type.iloc[SKIP_HOURS:, 1:])
    df = pd.concat([df, temp_df.iloc[:, 1:]], axis=1)
    
    df = df.reset_index()
    for i in range(0, 7):
        if(i == 0):
            temp_df = df_load_pre.iloc[SKIP_HOURS:, 1:].reset_index()
            df = pd.concat([df, temp_df.iloc[:, 1:]], axis=1)
        else:
            temp_df = df_load_pre.iloc[SKIP_HOURS-24*i:-24*i, 1:].reset_index()
            df = pd.concat([df, temp_df.iloc[:, 1:]], axis=1)
    # df['forecast_load_mw'] = df_load_fore[SKIP_HOURS:, 1]             #测试集用

    for i in range(1, 7):
        temp_df = df_comp_lmp.iloc[SKIP_HOURS-24*i:-24*i, 2:5].reset_index()
        df = pd.concat([df, temp_df.iloc[:, 1:4]], axis=1)

    for i in range(1, 7):
        temp_df = df_total_lmp.iloc[SKIP_HOURS-24*i:-24*i, 1:].reset_index()
        df = pd.concat([df, temp_df.iloc[:, 1:4]], axis=1)
    
    temp_outage = pd.DataFrame()
    for i in range(SKIP_DAYS, TOTAL_DAYS):
        for j in range(0, 24):
            temp_outage = pd.concat([temp_outage, df_outage.iloc[i, 1:]], axis=0, ignore_index=True)

    temp_df = temp_outage.reset_index()

    df = pd.concat([df, temp_df.iloc[:, 1:]], axis=1)
    print(df)
    df.to_csv('test.csv')


def divide_data(file_namex, file_namey):
    df_x = pd.read_csv(file_namex)
    df_y = pd.read_csv(file_namey)
    train_x = []    
    train_y = []
    test_x = []
    test_y = []
    for i in range(0, len(df_x) - 24 * TEST_DAYS):
        train_x.append(list(df_x.ix[i]))
        train_y.append(list(df_y.ix[i]))

    for i in range(len(df_x) - 24 * TEST_DAYS, len(df_x)):
        test_x.append(list(df_x.ix[i]))
        test_y.append(list(df_y.ix[i]))

    return train_x, train_y, test_x, test_y

def cal_mape(y_test, y_fore):
    '''the y_test is a 2D array, the y_fore is a 1D array'''
    n = len(y_test)
    ape_sum = 0
    for i in range(n):
        ape_sum += abs(y_test[i][0] - y_fore[i]) / abs(y_test[i][0])
    return 100 * ape_sum / n

def get_results(y_train, y_fore_train, y_test, y_fore_test, end_time=0, start_time=0):
    print('time_cost in training is:', end_time - start_time)
    print('the error on train data is:', cal_mape(y_train, y_fore_train))
    print('the error on test data is:', cal_mape(y_test, y_fore_test))

    fig1 = plt.figure()
    plt.title('2018.10.18---2018.11.1 PJM NODE:5021072 DAY AHEAD LMP FORECASTING')
    l1 = plt.plot(y_test, marker='*', label='actual')
    l2 = plt.plot(y_fore_test, marker='o', label='forecast')
    plt.legend()
    plt.show()

def get_encoded(x_train, x_test):
    encoder_net = torch.load('net_rr.pkl') # 提取训练好的encoder
    x_train_encoded = []
    x_test_encoded = []
    for each in x_train:
        x_train_encoded.append(encoder_net(torch.from_numpy(np.array(each)).float()).detach().numpy())
    for each in x_test:
        x_test_encoded.append(encoder_net(torch.from_numpy(np.array(each)).float()).detach().numpy())
    return x_train_encoded, x_test_encoded


def cal_mape_cnn(y_test, y_fore):
    '''the y_test is a 2D array, the y_fore is a 1D array'''
    n = len(y_test)
    ape_sum = 0
    for i in range(n):
        ape_sum += abs(y_test[i][0] - y_fore[i]) / abs(y_test[i][0])
    return 100 * ape_sum / n


def get_results_cnn(y_train, y_fore_train, y_test, y_fore_test):
    print('the error on train data is:', cal_mape_cnn(y_train, y_fore_train))
    print('the error on test data is:', cal_mape_cnn(y_test, y_fore_test))
    fig1 = plt.figure()
    plt.title('2018.10.18---2018.11.1 PJM NODE:5021072 DAY AHEAD LMP FORECASTING')
    l1 = plt.plot(y_test, marker='*', label='actual')
    l2 = plt.plot(y_fore_test, marker='o', label='forecast')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    # read_data(FILE_LIST)
    divide_data('x.csv', 'y.csv')