import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
        '../optassist/')))

from nloptwrapper import NLOptLoggedOpt

class TestNLOpt(object):

    def testOptimization(unittest.testcase):

        theOpt = NLOptLoggedOpt()


if __name__ == "__main__":
    unittest.main()
