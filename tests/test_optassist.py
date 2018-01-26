import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
        '../optassist/')))

import scipy.optimize as sco
from scipyoptwrapper import ScipyLoggedOpt

class TestNLOpt(unittest.TestCase):

    def testOptimization(self):
        pass

    def testSLSQP(self):

        def fun(x):
            q = sco.rosen(x)
            grad = sco.rosen_der(x)
            return q, grad

        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        bounds = [(-20, 20) for _ in x0]
        lb = [-20 for _ in x0]
        ub = [20 for _ in x0]

        opt = ScipyLoggedOpt(log_location='./', verbose=True)
        opt.setObjective(fun)
        opt.setBounds(lb, ub)
        opt.optimize(x0=x0, alg='Nelder-Mead')


if __name__ == "__main__":
    unittest.main()
