import numpy as np
import pdb
import sys
import os
import json

from loggedopt import LoggedOpt

import nlopt

import scipy.optimize as opt


def make_iter(val):
    try:
        iter(val)
    except:
        val = [val]
    return val


class NLOptLoggedOpt(LoggedOpt):

    '''
    IMPORTANT, for NLopt:
        fobj should be in the form:
        def fobj(x, grad):
            if grad.size > 0:
                ...set value of grad
            return fobj(x)

        NLopt expects constraints in the form g <= 0
    '''

    def __init__(self, log_name=None, clog_name=None, log_location=None):

        if log_name is None:
            log_name = 'optlog_q'
        if clog_name is None:
            clog_name = 'optlog_c'

        LoggedOpt.__init__(self, log_name, log_location)
        ## Addditonally need constraint logs
        if log_location:
            self.clog_file = log_location.rstrip('/') + '/' + clog_name + '.txt'
        else:
            self.clog_file = os.getcwd().rstrip('/') + '/output/'+clog_name + '.txt'

        self.objective = None
        self.observer = None
        self.fg = []
        self.fgeq = []
        self.constr_log = []
        self.verbose = False

    def _createLoggedObjective(self, fobj, jac=False):

        def func(x, grad):
            self.evals += 1
            self.eval_log.append(self.evals)

            if jac:
                q, qg = fobj(x)
                self.g_log.append([gi for gi in qg])
                self.writeToLog({'x': x, 'q': q, 'grad': qg, 'evals': self.evals})

                if grad is not None and grad.size > 0:
                    for i, _ in enumerate(x):
                        grad[i] = qg[i]
            else:
                q = fobj(x)
                self.writeToLog({'x': x, 'q': q, 'evals': self.evals})

            self.q_log.append(q)
            self.x_log.append([xi for xi in x])

            if self.verbose:
                print '-----------------------------------------------'
                print 'DVs: ', x
                print 'q: ', q
                if jac:
                    print 'grad: ', qg
                print '-----------------------------------------------'

            if self.observer is not None:
                self.observer()

            return q

        return func

    def _createLoggedConstraint(self, fc, jac=False, iconstraint=0):
        '''NLopt expects constraints in the form g <= 0'''

        def func(x, grad):
            if jac:
                c, cg = fc(x)[0:2]
                if grad.size > 0:
                    for i, _ in enumerate(x):
                        grad[i] = cg[i]
                self.writeToLog({'x': x, 'constraint': c, 'cgrad': cg},
                        log_file=self.clog_file)
            else:
                c = fc(x)
                self.writeToLog({'x': x, 'constraint': c}, log_file=self.clog_file)
            self.constr_log[iconstraint].append(c)

            if self.verbose:
                print 'Constraint: ', c

            return c

        return func

    def optimize(self, **kwargs):
        return self.runOptimization(**kwargs)

    def runOptimization(self, **kwargs):

        if self.objective is None or (self.lb is None or self.ub is None):
            raise Exception('''Objective or bounds not set, please use
                set_objective(fobj) and set_bounds(lb, ub))''')

        x0 = kwargs.setdefault('x0', None)
        if x0 is None:
            x0 = [(lbi + ubi)/2. for (lbi, ubi) in zip(self.lb, self.ub)]

        if kwargs.setdefault('check', True):
            self.overwriteLogFile(self.log_file)
            if self.fg:
                self.overwriteLogFile(self.clog_file)
        else:
            with open(self.log_file, 'w'): pass
            if self.fg:
                with open(self.clog_file, 'w'): pass

        self.verbose = kwargs.setdefault('verbose', False)

        kwargs['ineq_constraints'] = []
        for ig, fg in enumerate(self.fg):
            fun = fg[0]
            jac = fg[1]
            kwargs['ineq_constraints'].append(
                    self._createLoggedConstraint(fun, jac, ig))
        N_ineq = len(self.fg)

        kwargs['eq_constraints'] = []
        for ig, fgeq in enumerate(self.fgeq):
            fun = fgeq[0]
            jac = fgeq[1]
            kwargs['eq_constraints'].append(
                    self._createLoggedConstraint(fun, jac, ig+N_ineq))

        obj_fun = self.objective[0]
        obj_jac = self.objective[1]
        fobj = self._createLoggedObjective(obj_fun, obj_jac)

        algorithm = kwargs.setdefault('alg', 'SLSQP')
        if algorithm.lower() == 'bfgs' or algorithm.lower() == 'lbfgs':
            alg = nlopt.LD_LBFGS
        elif algorithm.lower() == 'slsqp' or algorithm.lower() == 'sqp':
            alg = nlopt.LD_SLSQP
        elif algorithm.lower() == 'mma':
            alg = nlopt.LD_MMA
        elif algorithm.lower() == 'cobyla':
            alg = nlopt.LN_COBYLA
        elif algorithm.lower() == 'bobyqa':
            alg = nlopt.LN_BOBYQA
        else:
            raise Exception('Unsupported algorithm')

        theNLopt = nlopt.opt(alg, len(x0))

        theNLopt.set_min_objective(fobj)
        theNLopt.set_lower_bounds(self.lb)
        theNLopt.set_upper_bounds(self.ub)
        theNLopt.set_xtol_rel(kwargs.setdefault('xtol_rel', 1e-4)) # (xi - xi+1)/x
        theNLopt.set_xtol_abs(kwargs.setdefault('xtol_abs', 1e-4)) # (xi - xi+1)
        theNLopt.set_ftol_rel(kwargs.setdefault('ftol_rel', 1e-4)) # (fi - fi+1)/f
        theNLopt.set_ftol_abs(kwargs.setdefault('ftol_abs', 1e-4)) # (fi - fi+1)
        theNLopt.set_maxeval(kwargs.setdefault('maxeval', 100))

        if kwargs.setdefault('ineq_constraints', []):
            for g in kwargs.setdefault('ineq_constraints', []):
                theNLopt.add_inequality_constraint(g, 1e-3)

        if kwargs.setdefault('eq_constraints', []):
            for g in kwargs.setdefault('eq_constraints', []):
                theNLopt.add_equality_constraint(g, 1e-3)

        #### RUN THE OPTIMIZATION ###
        theNLopt.optimize(x0)
        theNLopt.last_optimum_value()

        flag = 1
        return flag


def main():
    pass

if __name__ == "__main__":
    main()
