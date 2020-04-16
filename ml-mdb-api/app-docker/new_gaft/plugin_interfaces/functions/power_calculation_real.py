from math import *
import numpy as np

class PowerCalculationReal(object):

    def __init__(self):
        pass

    def calculation(self, speed, torque = 0.0):
        P_calc = 0
        self.speed = speed
        self.torque = torque
            
        w = self.speed
        T = self.torque

        # the calculation formula for power from regression
        x_degree = 2
        y_degree = 0        
        coef = np.zeros((x_degree+1, y_degree+1))

        # for real motor
        coef[0,0] = 0.11322144
        coef[1,0] = 0.02060969
        coef[2,0] = 0.00072539

        # calculate power
        i = 0
        j = 0
        for i in range(x_degree+1):
            for j in range(y_degree+1):
                if i+j > max(x_degree, y_degree):
                    continue
                else:
                    P_calc += coef[i,j]*pow(w,i)
        
        return P_calc
    

