#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Flip Bit mutation implementation. '''

from random import random
from numpy.random import normal, uniform
import json

from ...mpiutil import mpi
from ...plugin_interfaces.operators.mutation import Mutation
from ...components.floating_point_individual import FloatingPointIndividual

from ...plugin_interfaces.functions.power_calculation_sim_no_noise import PowerCalculationSimNoNoise
from ...plugin_interfaces.functions.power_calculation_sim_noise import PowerCalculationSimNoise
from ...plugin_interfaces.functions.power_calculation_real import PowerCalculationReal

# this class is used for adding a random Gaussian noise (mutation) to the individual
class SmallRandomMutation(Mutation):
    def __init__(self, power_type, pm, pe, torque, min_speed = 10.0, max_speed=30.0, int_speed=2):
        '''
        Mutation operator with Flip Bit mutation implementation.

        :param pm: The probability of mutation (usually between 0.001 ~ 0.1)
        :type pm: float in (0.0, 1.0]
        '''
        # super(GeneticAlgorithmControl, self).__init__(torque, max_speed, int_speed)

        if pm <= 0.0 or pm > 1.0:
            raise ValueError('Invalid mutation probability')

        if pe <= 0.0 or pe > 1.0:
            raise ValueError('Invalid mutation parameter')

        self.pm = pm
        self.pe = pe
        self.torque = torque
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.int_speed = int_speed
        if power_type == 0:
            self.power_calc = PowerCalculationReal()
        elif power_type == 1:
            self.power_calc = PowerCalculationSimNoise()
        elif power_type == 2:
            self.power_calc = PowerCalculationSimNoNoise()
            
    def mutate(self, individual, engine):
        '''
        Mutate the individual.
        '''
        if type(individual) is FloatingPointIndividual:

            min_speed = self.min_speed
            max_speed = self.max_speed
            int_speed = self.int_speed

            do_mutation = True if random() <= self.pm else False

            for i, genome in enumerate(individual.chromsome):
                # gaussian distribution, mu: mean = 0, sigma: standard deviation = 0.1
                if i==0 and do_mutation == True:
                    mu = 0
                    sigma = 1
                    # TODO: need to test!
                    alpha = self.pe
                    small_gain = normal(mu, sigma)
                    individual.chromsome[i] = individual.chromsome[i] + alpha*small_gain

                # for historical lower bound speed
                if i==0:
                    # speed range: [min_speed, max_speed]
                    individual.chromsome[i] = max(min(max_speed-int_speed*0.5, individual.chromsome[i]), min_speed)
                    # put into pre-defined interval and set the lower bound
                    individual.chromsome[i] = ((individual.chromsome[i] - min_speed) // int_speed) * int_speed + min_speed
                # for historical upper bound speed
                elif i==1:
                    individual.chromsome[i] = individual.chromsome[i-1] + int_speed
                # for historical power
                elif i<4:
                    individual.chromsome[i] = self.power_calc.calculation(individual.chromsome[i-2], self.torque)

        else:
            raise TypeError('Wrong individual type: {}'.format(type(individual)))

        # Update solution.
        individual.solution = individual.decode()

        return individual