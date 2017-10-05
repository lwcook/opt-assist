import numpy as np
import pdb
import sys
import os
import json

import scipy.optimize as opt

def make_iter(val):
    try:
        iter(val)
    except:
        val = [val]
    return val

class Log():

    def __init__(self, log_name=None, log_location=None, date=None, bDate=True):

        if log_name is None:
            log_name = 'log'
        self.log_name = log_name

        if date is not None:
            self.date = '_' + str(date)
        else:
            self.date = ''

        if log_location:
            self.log_file = log_location.rstrip('/') + '/' + log_name +\
                self.date + '.txt'
        else:
            self.log_file = os.getcwd().rstrip('/') + '/output/' + log_name +\
                self.date + '.txt'

    def overwrite_log_file(self, log_file=None, bCheck=True):
        return self.overwriteLogFile(log_file)

    def overwriteLogFile(self, log_file=None, bCheck=True):

        if log_file is None:
            log_file = self.log_file

        if os.path.isfile(log_file):
            if bCheck:
                ow = raw_input('Overwrite '+str(log_file)+'? y/N \r\n')
            else:
                ow = 'y'

            if not ow or ow[0].lower() != 'y':
                raise Exception('Aborting')
            else:
                if os.path.isfile(self.log_file):
                    os.remove(self.log_file)
                with open(self.log_file, 'w') as f:
                    pass

    def write_to_log(self, data, log_file=None):
        return self.writeToLog(data, log_file)

    def writeToLog(self, data, log_file=None):

        if not isinstance(data, dict):
            raise TypeError('''Data to write to file must be a dictionary to save
                in json format''')

        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()

        if log_file is None:
            log_file = self.log_file

        with open(log_file, 'a') as f:
            json.dump(data, f)
            f.write('\r\n')


    def readFromLog(self, log_file=None):

        if log_file is None:
            log_file = self.log_file

        with open(log_file, 'r') as f:
            data = [json.loads(line) for line in f]

        return data


class LoggedOpt(Log):


    def __init__(self, log_name=None, log_location=None):

        Log.__init__(self, log_name, log_location)

        self.objective = None
        self.lb = None
        self.ub = None

        self._resetLogs()


    def _resetLogs(self):
        self.x_log = []
        self.q_log = []
        self.g_log = []
        self.dict_log = []
        self.eval_log = []
        self.dv_log = []
        self.gen_log = []
        self.constr_log = []
        self.evals = 0


    def setObjective(self, fobj):
#        argspec = getargspec(fobj)
#        if len(argspec[0]) > 1:
#            raise TypeError('''Input function must take one argument''')
#        else:
#            self.objective = fobj
        self.objective = fobj


    def setBounds(self, lb, ub):
        if len(lb) != len(ub):
            raise ValueError('Length of bounds must be the same')
        else:
            self.lb = make_iter(lb)
            self.ub = make_iter(ub)


    def scaleToCand(self, dvs):
        lb, ub = self.lb, self.ub
        return [10.*(dvs[i]-lb[i])/(ub[i]-lb[i]) for i in range(len(dvs))]


    def scaleToDV(self, cand):
        lb, ub = self.lb, self.ub
        return [lb[i] + (ub[i]-lb[i])*(cand[i]/10.) for i in range(len(cand))]


    def runOptimization(self):
        raise Exception('Overwrite optimize function')


    def _tryOptimziation(self, f, x0, lb, ub, **kwargs):

        algorithm = kwargs.setdefault('alg', 'SLSQP')
        if algorithm.lower() == 'slsqp' or algorithm.lower() == 'sqp':
            alg = nlopt.LD_SLSQP
        if algorithm.lower() == 'coblya':
            alg = nlopt.LN_COBYLA

        opt = nlopt.opt(alg, len(x0))

        opt.set_min_objective(f)
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)

        maxeval = kwargs.setdefault('maxeval', 50)
        xtol_rel = kwargs.setdefault('xtol_rel', 1e-4)  # (xi - xi+1)/x
        xtol_abs = kwargs.setdefault('xtol_abs', 1e-4)  # (xi - xi+1)
        ftol_rel = kwargs.setdefault('ftol_rel', 1e-4)  # (fi - fi+1)/f
        ftol_abs = kwargs.setdefault('ftol_abs', 1e-4)  # (fi - fi+1)
        gtol = kwargs.setdefault('gtol', 1e-3)
        # Create the optimzitaion object by specifying algorithm and dimension
        # opt = nlopt.opt(nlopt.LD_MMA, len(xk))

        opt.set_xtol_rel(xtol_rel)
        opt.set_xtol_abs(xtol_abs)
        opt.set_ftol_rel(ftol_rel)
        opt.set_ftol_abs(ftol_abs)
        opt.set_maxeval(maxeval)

        if kwargs.setdefault('ineq_constraints', []):
            for g in kwargs.setdefault('ineq_constraints', []):
                opt.add_inequality_constraint(g, gtol)

        if kwargs.setdefault('eq_constraints', []):
            for g in kwargs.setdefault('eq_constraints', []):
                opt.add_equality_constraint(g, gtol)

        xmin = opt.optimize(x0)
        fmin = opt.last_optimum_value()

        return xmin, fmin


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

            return q, g

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

            return g, ggrad

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

#        algorithm = kwargs.setdefault('alg', 'SLSQP')
#        if algorithm.lower() == 'bfgs' or algorithm.lower() == 'lbfgs':
#            alg = nlopt.LD_LBFGS
#        elif algorithm.lower() == 'slsqp' or algorithm.lower() == 'sqp':
#            alg = nlopt.LD_SLSQP
#        elif algorithm.lower() == 'coblya':
#            alg = nlopt.LN_COBYLA
#        elif algorithm.lower() == 'bobyqa':
#            alg = nlopt.LN_BOBYQA
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

        opt.minimize(fobj, x0, method='BFGS', jac=True, bounds=self.bounds,
                tol=1e-5)

        #### RUN THE OPTIMIZATION ###
#        xmin = theNLopt.optimize(x0)
#        fmin = theNLopt.last_optimum_value()

        flag = 1
        return flag


def main():
    pass

if __name__ == "__main__":
    main()
