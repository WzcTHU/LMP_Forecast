import numpy
import random
import pandas as pd
import matplotlib.pyplot as plt
from SpikeForecast import spike_forecast

TEST_DAYS = 14

def make_batch(x_train, y_train, batch_size):
    i = random.randint(0, len(x_train) - batch_size)
    if ((i + batch_size) > len(x_train)):
        batch_xs = x_train[i: -1]
        batch_ys = y_train[i: -1]
    else:
        batch_xs = x_train[i: i + batch_size]
        batch_ys = y_train[i: i + batch_size]
    return [batch_xs, batch_ys]

def read_data(lmp_file, load_file):
    df_lmp = pd.read_csv(lmp_file)
    df_load = pd.read_csv(load_file)
    lmp = df_lmp['total_lmp_da']
    cop = df_lmp['congestion_price_da']
    mlp = df_lmp['marginal_loss_price_da']
    load_each = df_load['mw']
    load_total = []
    temp_load = 0

    for i in range(len(load_each)):
        if((i + 1) % 4 == 0):
            temp_load += load_each[i]
            load_total.append(temp_load)
            temp_load = 0
        else:
            temp_load += load_each[i]

    return list(lmp), list(load_total), list(cop), list(mlp)

def process_data(lmp_file, load_file):
    lmp, load_total, cop, mlp = read_data(lmp_file, load_file)
    data_num = len(lmp)
    data_day = int(data_num / 24)
    list_24 = [i for i in range(1, 25)]
    time_list = []
    for i in range(data_day):
        time_list.extend(list_24)

    # mark the spike price
    spike_mark = cal_spikes(lmp[: -TEST_DAYS*24])

    y = lmp[24 * 7: ]         # 除掉前六天的数据，从第七天开始取日前电价，第八天开始作为标签
    load_total = load_total[24 * 7: ]
    spike_mark = spike_mark[24 * 7: ]

    x_current = lmp[24 * 6: -24]
    x_1_ahead = lmp[24 * 5: -2 * 24]
    x_6_ahead = lmp[0: -7 * 24]

    cop_current = cop[24 * 6: -24] 
    cop_1_ahead = cop[24 * 5: -2 * 24]
    cop_6_ahead = cop[0: -7 * 24]

    mlp_current = mlp[24 * 6: -24]
    mlp_1_ahead = mlp[24 * 5: -2 * 24]
    mlp_6_ahead = mlp[0: -7 * 24]

    y_data = [[each] for each in y]
    x_data = [[load_total[i], x_current[i], x_1_ahead[i], x_6_ahead[i], cop_1_ahead[i], \
               cop_6_ahead[i], mlp_1_ahead[i], mlp_6_ahead[i] , \
              time_list[i]] for i in range(len(y_data))]
    # x_data = [[load_total[i], x_current[i], x_1_ahead[i], x_6_ahead[i], time_list[i]] for i in range(len(y_data))]
    return x_data, y_data, spike_mark

def divide_data(lmp_file, load_file):
    x_data, y_data, spike_train = process_data(lmp_file, load_file)
    x_train = x_data[: -TEST_DAYS*24]
    y_train = y_data[: -TEST_DAYS*24]
    x_test = x_data[-TEST_DAYS*24:]
    y_test = y_data[-TEST_DAYS*24:]
    return x_train, y_train, x_test, y_test, spike_train

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
    plt.title('2018.9.18---2018.10.1 PJM NODE:5021072 DAY AHEAD LMP FORECASTING')
    l1 = plt.plot(y_test, marker='*', label='actual')
    l2 = plt.plot(y_fore_test, marker='o', label='forecast')
    plt.legend()
    plt.show()

def cal_spikes(price_fore):
    # spike_price1 = numpy.mean(price_fore) + 0.5 * numpy.std(price_fore)
    spike_price1 = numpy.mean(price_fore) + 1 * numpy.std(price_fore)
    spike_price2 = numpy.mean(price_fore) + 1.5 * numpy.std(price_fore)
    spike_price3 = numpy.mean(price_fore) + 2.5 * numpy.std(price_fore)
    spike_price_list = [spike_price1, spike_price2, spike_price3]
    spike_mark = []
    for each in price_fore:
        if each < spike_price_list[0]:
            spike_mark.append(0)
        elif spike_price_list[0] <= each < spike_price_list[1]:
            spike_mark.append(1)
        elif spike_price_list[1] <= each < spike_price_list[2]:
            spike_mark.append(2)
        else:
            spike_mark.append(3)
    return spike_mark


def joint_spike(x_train, x_test, spike_train, spike_test):
    for i in range(len(x_train)):
        x_train[i].append(spike_train[i])
    for j in range(len(x_test)):
        x_test[j].append(spike_test[j])
    return x_train, x_test

if __name__ == '__main__':
    [x_train, y_train] = process_data('0702_1001_lmp.csv', '0702_1001_load.csv')