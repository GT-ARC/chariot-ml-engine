from math import *
import numpy as np
import scipy
from scipy import optimize
import json

import sys
import os
from os import path

from new_gaft import GAEngine
from new_gaft.components import Population, FloatingPointIndividual
from new_gaft.operators import TournamentSelection, IntermediateCrossover, SmallRandomMutation

# Analysis plugin base class
from new_gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
# Analysis class for fitness function
from new_gaft.analysis.fitness_store import FitnessStore

# Power calculation classes
from new_gaft.plugin_interfaces.functions.power_calculation_sim_no_noise import PowerCalculationSimNoNoise
from new_gaft.plugin_interfaces.functions.power_calculation_sim_noise import PowerCalculationSimNoise
from new_gaft.plugin_interfaces.functions.power_calculation_real import PowerCalculationReal

# The class for genetic algorithm
class GeneticAlgorithmControl():
    # initialize the population and genetic algorithm operators
    def __init__(self, power_type, error, power, speed, torque=0.0, min_speed = 10.0, max_speed=30.0, int_speed=2, \
            alpha=1, beta=1, ng=50):

        # initialize parameters
        self.torque = torque
        self.speed = speed
        self.power = power
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.int_speed = int_speed
        self.ng = ng
        self.alpha = alpha
        self.beta = beta
        if power_type == 0:
            self.power_calc = PowerCalculationReal()
        elif power_type == 1:
            self.power_calc = PowerCalculationSimNoise()
        elif power_type == 2:
            self.power_calc = PowerCalculationSimNoNoise()
        self.error = error

        a = self.alpha
        b = self.beta * (1+abs(self.error/10.0))
        c = 0.0

        a_norm = a/(a+b+c)
        b_norm = b/(a+b+c)
        c_norm = c/(a+b+c)

        self.alpha = a_norm
        self.beta = b_norm
        self.theta = c_norm

        # init the population, should cover the whole chromosome space
        indv_template = FloatingPointIndividual(power_type=power_type, ranges=[(min_speed, max_speed),(min_speed, max_speed)], \
                            eps=0.001, pi=self.int_speed, torque = self.torque )
        population = Population(indv_template=indv_template, size=60)
        # initialize the whole population with the specified individual
        population.init()

        # create genetic algorithm operators
        selection = TournamentSelection(tournament_size=4)
        
        crossover = IntermediateCrossover(pc=0.5, pe=0.25, \
                        int_speed=self.int_speed, min_speed = self.min_speed, max_speed=self.max_speed)
        
        mutation = SmallRandomMutation(power_type=power_type, pm=0.02, pe=0.2, torque = self.torque, \
                        min_speed = self.min_speed, max_speed=self.max_speed, int_speed = self.int_speed)

        # create genetic alrotithm engine
        self.engine = GAEngine(population=population, selection=selection,
                          crossover=crossover, mutation=mutation,
                          analysis=[FitnessStore])

    # a traditional minimum calculator to verify the correctness of the genetic algorithm
    def f_calc(self):

        a=self.alpha
        b=self.beta  
        a_norm = a/(a+b)
        b_norm = b/(a+b)  
        r = 0.0181
        min_array = optimize.fmin(lambda rpm: -(a_norm*2.0*np.pi/60.0*r*rpm + b_norm*(self.power - self.power_calc.calculation(rpm))), 25, full_output = True)

        return min_array[0], min_array[1]
    
    # the fitness calculation function
    def run(self):
        # register fitness function
        @self.engine.fitness_register
        def fitness(indv):

            a_norm = self.alpha
            b_norm = self.beta
            c_norm = self.theta

            # get the lower and upper bounds of the calculated speed and power from genetic algorithm
            rpm_min, rpm_max, P_min, P_max = indv.solution
            # the calculated speed and power is the average of the upper and lower bounds
            rpm = 0.5*(rpm_min+rpm_max)
            P = 0.5*(P_min+P_max)
            P_err = self.power - P

            # # conveyor belt length
            # s = 0.66
            # # conveyor belt radius
            # r = 0.0181
            # # linear velocity
            # v_const = 2.0*np.pi/60.0*r
            # v = v_const *rpm
            # # total time on the conveyor belt
            # t = s/v
            
            # normalize speed and power by the upper and lower bounds of the chromosome space
            min_v = self.min_speed
            max_v = self.max_speed
            min_P = self.power_calc.calculation(min_v)
            max_P = self.power_calc.calculation(max_v)           
            P_err_min = self.power - max_P
            P_err_max = self.power - min_P

            v_norm = (rpm - min_v) / (max_v - min_v)
            v_real_norm = (self.speed - min_v) / (max_v - min_v)
            v_err_norm = (v_real_norm - v_norm)
            P_norm = (P - min_P) / (max_P - min_P)
            P_err_norm = (P_err - P_err_min) / (P_err_max - P_err_min) 

            self.v_norm = v_norm
            self.P_err_norm = P_err_norm
            self.v_err_norm = v_err_norm

            # fitness function
            # it should achieve the following functionalities:
            # 1. the speed doesn't exceed the given range
            # 2. the power is as close as possible to the regression formula's calculation result
            # 3. the fitness is the least under the given weights
            F = a_norm*v_norm + b_norm*P_err_norm

            # restrain the ranges of power and speed
            # speed cannot change too rapidly
            # speed cannot be smaller than the lower bounds of the give speed range
            if P_err_norm < 0 or abs(self.speed - rpm) > 2:
                F = 0
            self.F = F

            return F


        n_gen = self.ng
        self.engine.run(ng=n_gen)
        best_indv = self.engine.population.best_indv(self.engine.fitness)
        print (best_indv.solution)
        
        # output dict
        out_dict = {}
        text = ["speed_min", "speed_max", "power_min", "power_max"]

        i=0
        for item in text:
            out_dict[item] = best_indv.solution[i]
            i += 1

        out = json.dumps(out_dict)

        # print some result indicators
        print ("beta = %f"%self.beta)
        print ("pid_err_ = %f"%self.error)
        print ("F = %f"%self.F)
            
        return out