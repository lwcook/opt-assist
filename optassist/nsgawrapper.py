import numpy as np
import os
import sys
import math
import pdb
import random
from inspect import getargspec

import NSGA
from loggedopt import LoggedOpt

def make_iter(val):
    try:
        iter(val)
    except:
        val = [val]
    return val

class NSGALoggedOpt(LoggedOpt):


    def __init__(self, log_name=None, log_location=None, log=True):
        LoggedOpt.__init__(self, log_name, log_location)
        self.objective = None
        self.bLog = log
        self.lb = None
        self.ub = None


    def _defaultGenerator(self, xdim):

        def gen_func(random, args):
            return [random.uniform(0., 10.) for ii in range(xdim)]

        return gen_func


    def evaluator(self, candidates, args):
        '''Compatible with NSGA module, must take a list of candidates (each of
        which is an iterable and containes design variables scaled to be in
        the interval [0,10]) and return a list of evaluated NSGA Pareto 
        objects.
        Signature must be evaluator(candidates, args) to comply with NSGA'''

        fitness = []
        lb, ub = self.lb, self.ub
        for c in candidates:
            dvs = self.scaleToDV(c)
            objs = self.objective(dvs)
            fitness.append(NSGA.Pareto([ob for ob in make_iter(objs)]))

            self.x_log.append(dvs)
            self.q_log.append(objs)
            if self.bLog:
                self.writeToLog({'x':dvs, 'q':objs})

        return fitness


    def runOptimization(self, seeds=None, **kwargs):
        return self.optimize(seeds, **kwargs)


    def optimize(self, seeds=None, **kwargs):
        '''Performs a multi-objective optimization using the NSGA module. Note
        that it scales the variables that the algorithm operates on to be
        in [0,10] so they have an appropriate design space. However all logs
        and output is in terms of the original design variables, it is scaled
        internally.
        NOTE: objective is set with set_objective and bounds are set with
        set_bounds before optimizing

        Optional Inputs:
            seeds: seeds for the optimizer'''

        if self.objective is None or (self.lb is None or self.ub is None):
            raise Exception('''Objective or bounds not set, please use
                set_objective(fobj) and set_bounds(lb, ub))''')

        lb, ub = self.lb, self.ub

        optlb, optub = [0. for _ in lb], [10. for _ in ub]
        bounder = NSGA.Bounder(optlb, optub)

        generator = self._defaultGenerator(len(lb))
        if seeds:
            for seed in seeds:
                if len(seed) != len(ub):
                    raise ValueError('Length of seeds must be same as bounds')
                for ii, (si, (lbi, ubi)) in enumerate(zip(seed, zip(lb, ub))):
                    if si < lbi or si > ubi:
                        raise ValueError('Seed'+str(i)+' is outside bounds')
            cseeds = [self.scaleToCand(dvs) for dvs in seeds]
        else:
            cseeds = []

        pop_size = kwargs.setdefault('pop', 10)
        max_gen = kwargs.setdefault('gen', 10)

        if kwargs.setdefault('check', True):
            self.overwrite_log_file()

        theNSGA = NSGA.NSGA2()
        final_pop = theNSGA.evolve(seeds=cseeds,
                                pop_size=pop_size,
                                max_generations=max_gen,
                                generator=generator,
                                bounder=bounder,
                                maximize=False,
                                evaluator=self.evaluator)

        flag = 0
        return flag
