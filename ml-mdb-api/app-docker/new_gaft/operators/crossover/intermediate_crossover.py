#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Uniform Crossover operator implementation. '''

from random import random
from numpy.random import normal, uniform
from copy import deepcopy
from math import *

from ...plugin_interfaces.operators.crossover import Crossover

# this class implements the itermediate crossover
class IntermediateCrossover(Crossover):
    def __init__(self, pc, pe, int_speed = 2, min_speed = 10.0, max_speed=30.0):
        '''
        Crossover operator with intermediate crossover algorithm,

        :param pc: The probability of crossover (usaully between 0.25 ~ 1.0)
        :type pc: float in (0.0, 1.0]

        :param pe: Gene exchange probability parameter for intermediate crossover.
        :type pe: float in (0.0, 1.0]
        '''
        # super(GeneticAlgorithmControl, self).__init__(max_speed)

        if pc <= 0.0 or pc > 1.0:
            raise ValueError('Invalid crossover probability')
        self.pc = pc

        if pe <= 0.0 or pe > 1.0:
            raise ValueError('Invalid genome exchange probability')
        self.pe = pe

        self.int_speed = int_speed
        self.max_speed = max_speed
        self.min_speed = min_speed

    def cross(self, father, mother):
        '''
        Cross chromsomes of parent using intermediate crossover method.
        '''
        do_cross = True if random() <= self.pc else False

        if not do_cross:
            return father.clone(), mother.clone()

        # Chromsomes for two children.
        chrom1 = deepcopy(father.chromsome)
        chrom2 = deepcopy(mother.chromsome)

        # interval parameter
        interval = self.pe
        int_speed = self.int_speed
        max_speed = self.max_speed
        min_speed = self.min_speed
        # crossover parameter
        for i in range(len(chrom1)):

            # do nothing with the power
            if i>0:
                continue

            # crossover: speed range: [min_speed, max_speed];
            # only perform crossover for the mean of the speed
            if i == 0:
                alpha = uniform(-interval, 1+interval)
                chrom1[i] = max(min(max_speed, chrom1[i] + alpha * (chrom2[i] - chrom1[i])), min_speed)
                alpha = uniform(-interval, 1+interval)
                chrom2[i] = max(min(max_speed, chrom2[i] + alpha * (chrom1[i] - chrom2[i])), min_speed)
            

        child1, child2 = father.clone(), father.clone()
        child1.init(chromsome=chrom1)
        child2.init(chromsome=chrom2)

        return child1, child2

