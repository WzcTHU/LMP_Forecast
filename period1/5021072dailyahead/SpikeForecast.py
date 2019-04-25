from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy


def spike_forecast(load_train, spike_train, load_test):
    scaler1 = StandardScaler()

    load_train = numpy.reshape(load_train, [-1, 1])
    load_test = numpy.reshape(load_test, [-1, 1])
    spike_train = numpy.reshape(spike_train, [-1, 1])

    scaler1.fit(load_train)
    load_train = scaler1.transform(load_train)
    load_test = scaler1.transform(load_test)

    print(load_train[0:24])
    print(spike_train[0:24])

    clf = XGBClassifier()
    clf.fit(load_train, spike_train.ravel())
    spike_fore = clf.predict(load_test)

    return spike_fore





