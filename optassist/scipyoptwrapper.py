import numpy as np
import pdb
import sys
import os
import json

from loggedopt import LoggedOpt

import scipy.optimize as opt

def make_iter(val):
    try:
        iter(val)
    except:
        val = [val]
    return val

class ScipyLoggedOpt(LoggedOpt):

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
            self.clog_file = log_location.rstrip('/') + '/' + clog_name +'.txt'
        else:
            self.clog_file = os.getcwd().rstrip('/') + '/output/'+clog_name+'.txt'

        self.objective = None
        self.observer = None
        self.fg = []
        self.fgeq = []
        self.constr_log = []
        self.verbose = False

    def setObjective(self, fobj):
        self.objective = fobj

    def setBounds(self, lb, ub):
        if len(lb) != len(ub):
            raise ValueError('Length of bounds must be the same')
        else:
            self.lb = make_iter(lb)
            self.ub = make_iter(ub)
        self.bounds = [(l, u) for l, u in zip(self.lb, self.ub)]

    def addConstraint(self, func):
        self.fg.append(func)
        self.constr_log.append([])

    def addEqConstraint(self, func):
        self.fgeq.append(func)
        self.constr_log.append([])

    def addObserver(self, func):
        self.observer = func

    def _createLoggedObjective(self, fobj):

        def func(x):

            out = make_iter(fobj(x))
            if len(out) == 1:
                raise Exception('''Objective funciton should return the objective
                and a gradient (which can be empty)''')
            if len(out) >= 2:
                q, g = out[0], out[1]
                self.q_log.append(q)
                self.g_log.append(g)
            if len(out) >= 3:
                mdict = out[2]
                self.dict_log.append(mdict)
            if len(out) >= 4:
                raise Exception('Objective funciton returning too many values')

            self.g_log.append([gi for gi in g])

            if self.verbose:
                print '-----------------------------------------------'
                print 'DVs: ', x
                print 'q: ', q
                print 'grad: ', g
                print '-----------------------------------------------'

            self.x_log.append([xi for xi in x])
            self.evals += 1
            self.eval_log.append(self.evals)
            self.writeToLog({'x': x, 'q':q, 'grad':g, 'evals':self.evals})

            if self.observer is not None:
                self.observer()

            return q, np.array(g)

        return func

    def _createLoggedConstraint(self, fg, iconstraint=0):
        '''NLopt expects constraints in the form g <= 0'''

        def func(x, grad):
            g, ggrad = fg(x)[0:2]

            if self.verbose:
                print 'Constraint: ', g

            self.constr_log[iconstraint].append(g)
            self.writeToLog({'x':x, 'constraint':g, 'cgrad':ggrad},
                    log_file=self.clog_file)

            return g, np.array(ggrad)

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
            with open(self.log_file, 'w') as f: pass
            if self.fg:
                with open(self.clog_file, 'w') as f: pass

        self.verbose = kwargs.setdefault('verbose', False)

        kwargs['ineq_constraints'] = \
            [self._createLoggedConstraint(fg, ig)
                for ig, fg in enumerate(self.fg)]
        N_ineq = len(self.fg)

        kwargs['eq_constraints'] = \
            [self._createLoggedConstraint(fg, ig+N_ineq)
                for ig, fg in enumerate(self.fgeq)]

        fobj = self._createLoggedObjective(self.objective)

        algorithm = kwargs.setdefault('alg', 'SLSQP')
#        if algorithm.lower() == 'bfgs':
#            method = 'bfgs'
#        elif algorithm.lower() == 'l_bfgs_b':
#            method = 'l_bfgs_b'
#        elif algorithm.lower() == 'slsqp' or algorithm.lower() == 'sqp':
#            method = 'slsqp'
#        else:
#            raise Exception('Unsupported algorithm')
#
#        theNLopt = nlopt.opt(alg, len(x0))
#        theNLopt.set_min_objective(fobj)
#        theNLopt.set_lower_bounds(self.lb)
#        theNLopt.set_upper_bounds(self.ub)
#        theNLopt.set_xtol_rel(kwargs.setdefault('xtol_rel', 1e-4)) # (xi - xi+1)/x
#        theNLopt.set_xtol_abs(kwargs.setdefault('xtol_abs', 1e-4)) # (xi - xi+1)
#        theNLopt.set_ftol_rel(kwargs.setdefault('ftol_rel', 1e-4)) # (fi - fi+1)/f
#        theNLopt.set_ftol_abs(kwargs.setdefault('ftol_abs', 1e-4)) # (fi - fi+1)
#        theNLopt.set_maxeval(kwargs.setdefault('maxeval', 100))
#
#        if kwargs.setdefault('ineq_constraints', []):
#            for g in kwargs.setdefault('ineq_constraints', []):
#                theNLopt.add_inequality_constraint(g, 1e-3)
#
#        if kwargs.setdefault('eq_constraints', []):
#            for g in kwargs.setdefault('eq_constraints', []):
#                theNLopt.add_equality_constraint(g, 1e-3)
        tolA = kwargs.setdefault('ftol_abs', 1e-4)
        tol = kwargs.setdefault('tol', tolA)
        opts = kwargs.setdefault('options', {})

        self.result = opt.minimize(fobj, x0, method=algorithm.upper(),
                jac=True, bounds=self.bounds, tol=tol, options=opts)


        #### RUN THE OPTIMIZATION ###
#        xmin = theNLopt.optimize(x0)
#        fmin = theNLopt.last_optimum_value()

        flag = 1
        return flag


def main():
    pass

if __name__ == "__main__":
    main()
