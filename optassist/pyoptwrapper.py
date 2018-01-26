import numpy as np
import pdb
import sys
import os
import json

from loggedopt import LoggedOpt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import pyOpt

def make_iter(val):
    try:
        iter(val)
    except:
        val = [val]
    return val


class PyOptLoggedOpt(LoggedOpt):

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
        self.verbose = False

    def setBounds(self, lb, ub):
        if len(lb) != len(ub):
            raise ValueError('Length of bounds must be the same')
        else:
            self.lb = make_iter(lb)
            self.ub = make_iter(ub)
        self.bounds = [(l, u) for l, u in zip(self.lb, self.ub)]

    def _createLoggedObjective(self, fobj, jac):

        def func(x):
            self.evals += 1
            self.eval_log.append(self.evals)
            if jac:
                q, g = fobj(x)
                self.g_log.append([gi for gi in g])
                self.writeToLog({'x': x, 'q': q, 'grad': g, 'evals': self.evals})
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
                    print 'grad: ', g
                print '-----------------------------------------------'

            if self.observer is not None:
                self.observer()

            if jac:
                return q, np.array(g)
            else:
                return q

        return func

    def _createLoggedConstraint(self, fg, jac):
        '''Optassist expects constraints in the form g <= 0'''

        def func(x):
            if jac:
                c, cgrad = fg(x)
                self.writeToLog({'x': x, 'constraint': c, 'cgrad': cgrad},
                        log_file=self.clog_file)
            else:
                c = fg(x)
                self.writeToLog({'x': x, 'constraint': c},
                        log_file=self.clog_file)

            if self.verbose:
                print 'Constraint: ', c

            ### Scipy needs constraints >= 0, but the interface is <= 0
            if jac:
                return -1*c, np.array([-1*g for g in cgrad])
            else:
                return -1*c

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

        ######################################################################
        ## Constraints and objectives
        ######################################################################

        scipy_constrs = []

        cjac = kwargs.setdefault('jac', False)
        for fg in self.fg:
            ctype = 'ineq'
            cfun = self._createLoggedConstraint(fg[0], jac=False)
            cjac = self._createLoggedConstraint(fg[1], jac=False)
            scipy_constrs.append({'type': ctype, 'fun': cfun, 'jac': cjac})

#        for fgeq in self.fgeq:
#            ctype = 'eq'
#            cfun = self._createLoggedConstraint(fgeq[0], jac=False)
#            cjac = self._createLoggedConstraint(fgeq[1], jac=False)
#            scipy_constrs.append({'type': ctype, 'fun': cfun, 'jac': cjac})
        if len(scipy_constrs) == 2:
            scipy_constrs = scipy_constrs[0]

        obj_fun = self.objective[0]
        obj_jac = self.objective[1]
        fobj = self._createLoggedObjective(obj_fun, obj_jac)

        algorithm = kwargs.setdefault('alg', 'SLSQP')
        tol = kwargs.setdefault('tol', 5e-4)
        maxeval = kwargs.setdefault('maxeval', 100)
        opts = kwargs.setdefault('options', {'maxiter': maxeval})

#        self.result = opt.minimize(fobj, x0, jac=obj_jac, method=algorithm.upper(),
#                bounds=self.bounds, constraints=scipy_constrs, tol=tol, options=opts)
        theProb = pyOpt.Optimization('Logged Optimization', obj_fun)
        theProb.addObj('f')
        for i, b in enumerate(self.bounds):
            theProb.addVar('x'+str(i), 'c', lower=b[0], upper=b[1], value=x0[i])

        theOpt = pyOpt.SLSQP()
        theOpt(theProb, sens_type=obj_jac)

        ### INCOMPLETE

        print theProb

        flag = 1
        return flag


def main():
    pass

if __name__ == "__main__":
    main()
