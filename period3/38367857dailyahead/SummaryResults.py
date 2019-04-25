import torch
import matplotlib.pyplot as plt
import numpy as np
import math

class SummaryResults():
    def __init__(self, label_y=[1], fore_y=[1]):
        self.mape = 0
        self.label_y = label_y
        self.fore_y = fore_y
        self.residual_list = []
        self.variance = 0

    def cal_mape(self):
        n = len(self.label_y)
        ape_sum = 0
        for i in range(n):
            ape_sum += abs(self.label_y[i][0] - self.fore_y[i]) / abs(self.label_y[i][0])
        self.mape = 100 * ape_sum / n

    def get(self):
        self.cal_mape()
        print('The MAPE is:', self.mape)
        fig1 = plt.figure()
        # plt.title('2018.10.18---2018.11.1 PJM NODE:5021072 DAY AHEAD LMP FORECASTING')
        l1 = plt.plot(self.label_y, marker='*', label='actual', lw=1, ms=3)
        l2 = plt.plot(self.fore_y, marker='o', label='forecast', lw=1, ms=3)
        plt.legend()
        plt.show()
    
    def cal_residual(self):
        for i in range(0, len(self.label_y)):
            self.residual_list.append(self.fore_y[i] - self.label_y[i])
        return self.residual_list

    def cal_variance(self):
        self.cal_residual()
        for i in range(0, len(self.residual_list)):
            self.variance += pow(self.residual_list[i], 2)
        return self.variance[0]

