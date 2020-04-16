import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

import re
import csv

# this class contain an alternative for the polynomial regression
class KNNRegressor(object):
    def __init__(self, path='output_1_sim_conveyor_V_P_T_2.csv', n_neighbors=5):

        self.n_neighbors = n_neighbors

        with open(path, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)

        data = np.array(data[:])
        data = data.astype(np.float)
        speed = data[:,1]
        power = data[:, 2]
        torque = data[:, 3]
        data = list(zip(speed, torque, power))
        data = [list(elem) for elem in data]
        data = np.array(data)

        self.train_X = data[:,0:2]
        self.train_y = data[:,2]

        knn = neighbors.KNeighborsRegressor(self.n_neighbors, weights='distance')
        self.clf = knn.fit(self.train_X, self.train_y)


    # for now input must be only one sample
    def predict(self, input_array):
        # input array must be numpy array
        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)
        power = self.clf.predict(input_array)

        return power