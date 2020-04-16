from math import *
import numpy as np

class PowerCalculationSimNoise(object):

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
        # y_degree = 2        
        y_degree = 0  
        coef = np.zeros((x_degree+1, y_degree+1))
        # coef[0,0] = 0.08891
        # coef[1,0] = 0.01565
        # coef[0,1] = 3.523
        # coef[2,0] = 0.000442
        # coef[1,1] = 0.2523
        # coef[0,2] = 30.07

        # for simulation with noise
        coef[0,0] = 0.07601603
        coef[1,0] = 0.02706753
        coef[2,0] = 0.0005427        

        # calculate power
        i = 0
        j = 0
        for i in range(x_degree+1):
            for j in range(y_degree+1):
                if i+j > max(x_degree, y_degree):
                    continue
                else:
                    P_calc += coef[i,j]*pow(w,i)
                    # P_calc += coef[i,j]*pow(w,i)*pow(T,j)
        
        return P_calc
    

