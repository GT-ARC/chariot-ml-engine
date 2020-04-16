#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .individual import IndividualBase
import numpy as np


class Memoized(object):
    '''
    Descriptor for population statistical varibles caching.
    '''
    def __init__(self, func):
        self.func = func
        self.result = None
        self.fitness = None

    def __get__(self, instance, cls):
        self.instance = instance
        return self

    def __call__(self, fitness):
        if ((not self.instance.updated)          # population not changed
                and (self.result is not None)    # result already cached
                and (fitness == self.fitness)):  # fitness not changed
            # Return cached result directly.
            return self.result
        else:
            # Update fitness function.
            self.fitness = fitness
            # Update and memoize result.
            self.result = self.func(self.instance, fitness)
            # Recover flag.
            self.instance._updated = False
            return self.result


class Individuals(object):
    '''
    Descriptor for all individuals in population.
    '''
    def __init__(self, name):
        self.name = '_{}'.format(name)

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value
        # Update flag.
        instance.update_flag()


class Population(object):

    # All individuals.
    individuals = Individuals('individuals')

    def __init__(self, indv_template, size=100):
        '''
        Class for representing population in genetic algorithm.

        :param indv_template: A template individual to clone all the other
                              individuals in current population.

        :param size: The size of population, number of individuals in population.
        :type size: int

        '''
        # Population size.
        if size % 2 != 0:
            raise ValueError('Population size must be an even number')
        self.size = size

        # Template individual.
        self.indv_template = indv_template

        # Flag for monitoring changes of population.
        self._updated = False

        # Container for all individuals.
        class IndvList(list):
            '''
            A proxy class inherited from built-in list to contain all
            individuals which can update the population._updated flag
            automatically when its content is changed.
            '''
            # {{{
            # NOTE: Use 'this' here to avoid name conflict.
            def __init__(this, *args):
                super(this.__class__, this).__init__(*args)

            def __setitem__(this, key, value):
                '''
                Override __setitem__ in built-in list type.
                '''
                old_value = this[key]
                if old_value == value:
                    return
                super(this.__class__, self).__setitem__(key, value)
                # Update population flag.
                self.update_flag()

            def append(this, item):
                '''
                Override append method of built-in list type.
                '''
                super(this.__class__, this).append(item)
                # Update population flag.
                self.update_flag()

            def extend(this, iterable_item):
                if not iterable_item:
                    return
                super(this.__class__, this).extend(iterable_item)
                # Update population flag.
                self.update_flag()
            # }}}

        self._individuals = IndvList()

    def init(self, indvs=None):
        '''
        Initialize current population with individuals.

        :param indvs: Initial individuals in population, randomly initialized
                      individuals are created if not provided.
        :type indvs: list of Individual object
        '''
        IndvType = self.indv_template.__class__

        if indvs is None:
            for _ in range(self.size):
                indv = IndvType(power_type=self.indv_template.power_type,
                                ranges=self.indv_template.ranges,
                                eps=self.indv_template.eps, 
                                pi=self.indv_template.pi, 
                                torque=self.indv_template.torque)
                self.individuals.append(indv)
        else:
            # Check individuals.
            if len(indvs) != self.size:
                raise ValueError('Invalid individuals number')
            for indv in indvs:
                if not isinstance(indv, IndividualBase):
                    raise ValueError('individual class must be subclass of IndividualBase')
            self.individuals = indvs

        self._updated = True

        return self

    def update_flag(self):
        '''
        Interface for updating individual update flag to True.
        '''
        self._updated = True

    @property
    def updated(self):
        '''
        Query function for population updating flag.
        '''
        return self._updated

    def new(self):
        '''
        Create a new emtpy population.
        '''
        return self.__class__(indv_template=self.indv_template,
                              size=self.size)

    def __getitem__(self, key):
        '''
        Get individual by index.
        '''
        if key < 0 or key >= self.size:
            raise IndexError('Individual index({}) out of range'.format(key))
        return self.individuals[key]

    def __len__(self):
        '''
        Get length of population.
        '''
        return len(self.individuals)

    def best_indv(self, fitness):
        '''
        The individual with the best fitness.

        '''
        all_fits = self.all_fits(fitness)
        return max(self.individuals,
                   key=lambda indv: all_fits[self.individuals.index(indv)])

    def worst_indv(self, fitness):
        '''
        The individual with the worst fitness.
        '''
        all_fits = self.all_fits(fitness)
        return min(self.individuals,
                   key=lambda indv: all_fits[self.individuals.index(indv)])

    def max(self, fitness):
        '''
        Get the maximum fitness value in population.
        '''
        return max(self.all_fits(fitness))

    def min(self, fitness):
        '''
        Get the minimum value of fitness in population.
        '''
        return min(self.all_fits(fitness))

    def mean(self, fitness):
        '''
        Get the average fitness value in population.
        '''
        all_fits = self.all_fits(fitness)
        return sum(all_fits)/len(all_fits)

    def min_t(self):
        # speed
        rpm = 0.5*(self.individuals[0].solution[0]+self.individuals[0].solution[1])
        # conveyor belt length
        s = 0.66
        # conveyor belt radius
        r = 0.0181
        # linear velocity
        v = 2.0*np.pi/60.0*r*rpm
        # the operation time
        t_min = s/v

        for indv in self.individuals:
            rpm = 0.5*(indv.solution[0]+indv.solution[1])
            t_temp = s/(2.0*np.pi/60.0*r*rpm)
            if t_temp < t_min:
                t_min = t_temp

        return t_min

    def max_t(self):
        # speed
        rpm = 0.5*(self.individuals[0].solution[0]+self.individuals[0].solution[1])
        # conveyor belt length
        s = 0.66
        # conveyor belt radius
        r = 0.0181
        # linear velocity
        v = 2.0*np.pi/60.0*r*rpm
        # the operation time
        t_max = s/v

        for indv in self.individuals:
            rpm = 0.5*(indv.solution[0]+indv.solution[1])
            t_temp = s/(2.0*np.pi/60.0*r*rpm)
            if t_temp > t_max:
                t_max = t_temp

        return t_max

    def min_v(self):
        # speed
        rpm = 0.5*(self.individuals[0].solution[0]+self.individuals[0].solution[1])
        # conveyor belt length
        s = 0.66
        # conveyor belt radius
        r = 0.0181
        # linear velocity
        v_min = 2.0*np.pi/60.0*r*rpm

        for indv in self.individuals:
            rpm = 0.5*(indv.solution[0]+indv.solution[1])
            v_temp = 2.0*np.pi/60.0*r*rpm
            if v_temp < v_min:
                v_min = v_temp

        return v_min

    def max_v(self):
        # speed
        rpm = 0.5*(self.individuals[0].solution[0]+self.individuals[0].solution[1])
        # conveyor belt length
        s = 0.66
        # conveyor belt radius
        r = 0.0181
        # linear velocity
        v_max = 2.0*np.pi/60.0*r*rpm

        for indv in self.individuals:
            rpm = 0.5*(indv.solution[0]+indv.solution[1])
            v_temp = 2.0*np.pi/60.0*r*rpm
            if v_temp > v_max:
                v_max = v_temp

        return v_max

    def min_P(self):
        # power
        P_min = 0.5*(self.individuals[0].solution[2]+self.individuals[0].solution[3])

        for indv in self.individuals:
            P_temp = 0.5*(indv.solution[2]+indv.solution[3])
            if P_temp < P_min:
                P_min = P_temp

        return P_min

    def max_P(self):
        # power
        P_max = 0.5*(self.individuals[0].solution[2]+self.individuals[0].solution[3])

        for indv in self.individuals:
            P_temp = 0.5*(indv.solution[2]+indv.solution[3])
            if P_temp > P_max:
                P_max = P_temp

        return P_max

    @Memoized
    def all_fits(self, fitness):
        '''
        Get all fitness values in population.
        '''
        return [fitness(indv) for indv in self.individuals]

