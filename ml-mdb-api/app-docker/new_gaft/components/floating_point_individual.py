#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Definition of individual class with decimal encoding.
'''
from numpy.random import normal, uniform
# import json

from .individual import IndividualBase
from ..plugin_interfaces.functions.power_calculation_sim_no_noise import PowerCalculationSimNoNoise
from ..plugin_interfaces.functions.power_calculation_sim_noise import PowerCalculationSimNoise
from ..plugin_interfaces.functions.power_calculation_real import PowerCalculationReal

# this class is modified specifically to meet our requirements of the chromosome space
class FloatingPointIndividual(IndividualBase):
    ''' Individual with decimal encoding.
    '''
    def __init__(self, power_type, ranges, eps=0.001, pi=2, torque=0):
        super(FloatingPointIndividual, self).__init__(power_type, ranges, eps, pi, torque)

        if power_type == 0:
            power_calc = PowerCalculationReal()
        elif power_type == 1:
            power_calc = PowerCalculationSimNoise()
        elif power_type == 2:
            power_calc = PowerCalculationSimNoNoise()

        # historical training data
        min_speed = ranges[0][0]
        max_speed = ranges[0][1]
        vmin_0 = ((uniform(min_speed, max_speed) - min_speed)//pi)*pi + min_speed
        vmax_0 = vmin_0 + pi
        pmin_0 = power_calc.calculation(vmin_0,torque)
        pmax_0 = power_calc.calculation(vmax_0,torque)

        # initialization
        self.init(solution = [vmin_0,vmax_0,pmin_0,pmax_0])

    def encode(self):
        return self.solution

    def decode(self):
        return self.chromsome



