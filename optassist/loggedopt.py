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

    def __init__(self, log_name=None, log_location=None,
            date=None, bDate=True):

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
                with open(self.log_file, 'w'):
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
            f.write(json.dumps(data))
            f.write('\r\n')

    def readFromLog(self, log_file=None):

        if log_file is None:
            log_file = self.log_file

        with open(log_file, 'r') as f:
            data = [json.loads(line) for line in f]

        return data


class LoggedOpt(Log):

    def __init__(self, log_name=None, log_location=None, verbose=False):

        self.verbose = verbose

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

    def setObjective(self, fobj, jac=False):
        self.objective = (fobj, jac)

    def setBounds(self, lb, ub):
        if len(lb) != len(ub):
            raise ValueError('Length of bounds must be the same')
        else:
            self.lb = make_iter(lb)
            self.ub = make_iter(ub)

    def addConstraint(self, func, jac=False):
        self.fg.append((func, jac))
        self.constr_log.append([])

    def addEqConstraint(self, func, jac=False):
        self.fgeq.append((func, jac))
        self.constr_log.append([])

    def addObserver(self, func):
        self.observer = func

    def scaleToCand(self, dvs):
        lb, ub = self.lb, self.ub
        return [10.*(dvs[i]-lb[i])/(ub[i]-lb[i]) for i in range(len(dvs))]

    def scaleToDV(self, cand):
        lb, ub = self.lb, self.ub
        return [lb[i] + (ub[i]-lb[i])*(cand[i]/10.) for i in range(len(cand))]

    def runOptimization(self):
        raise Exception('Overwrite optimize function')


def main():
    pass

if __name__ == "__main__":
    main()
