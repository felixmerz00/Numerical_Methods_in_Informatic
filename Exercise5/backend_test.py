import numpy as np
import sys
import traceback
import backend

class Tester:
    def __init__(self):
        self.module = None
    #############################################
    # Task a
    #############################################

    def testA(self, l: list):
        task = "5.1a)"
        comments = ""

        def evaluate(X, Y):
            nonlocal comments
            try:
                m, b = self.module.linearLSQ(np.copy(X), np.copy(Y))
                A = np.ones([X.shape[0], 2])
                A[:, 0] = np.copy(X)
                x = np.linalg.lstsq(A, Y, rcond=None)[0]
                if (np.abs(m - x[0]) < 1e-6 and np.abs(b - x[1]) < 1e-6):
                    comments += "passed. "
                else:
                    comments += "failed. "
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "

        # Y = 0 case
        comments += "Y = 0 case "

        X = np.linspace(5, 15, 100)
        Y = (np.random.random(100) - 0.5) * 3.
        evaluate(X, Y)

        # Y = X - 5 case
        comments += "Y = X - 5 case "

        X = np.linspace(5, 15, 100)
        Y = np.copy(X) + (np.random.random(100) - 0.5) * 3.
        evaluate(X, Y)

        # Y = 100X - 10 case
        comments += "Y = 100X - 10 case "

        X = np.linspace(5, 15, 100)
        Y = np.copy(X) * 100 - 10 + (np.random.random(100) - 0.5) * 3.
        evaluate(X, Y)

        l.extend([task, comments])

    #############################################
    # Task b
    #############################################

    def testB(self, l: list):
        task = "5.1b)"
        comments = ""

        def evaluate(inputBase):
            nonlocal comments

            def testBases(base):
                passed = True
                additionalComments = ""
                if (len(inputBase) != len(base)):
                    additionalComments += "Number of basevectors has changed. "
                    passed = False
                for i in range(len(base)):
                    if (np.abs(np.linalg.norm(base[i]) - 1) > 1e-6):
                        additionalComments += "Base is not normal. "
                        passed = False
                    for j in range(i + 1, len(base)):
                        dot = base[i].dot(base[j])
                        norms = np.linalg.norm(base[i]) * np.linalg.norm(base[j])
                        if (norms < 1e-6):
                            additionalComments += "Base has 0 vector. "
                            passed = False
                            norms = 1
                        if (np.abs(dot / norms) > 1e-6):
                            additionalComments += "Base is not orthorgonal. "
                            passed = False

                    M = np.zeros((len(inputBase[0]), len(inputBase)))
                    t = np.zeros((len(base[0]), len(base)))
                    for i, source in enumerate(inputBase):
                        M[:, i] = source
                    for i, b in enumerate(base):
                        t[:, i] = b
                    res = np.linalg.lstsq(M, t, rcond=None)[1]
                    if ((np.abs(res) > 1e-16).any()):
                        additionalComments += "Base spans different Space. "
                        passed = False
                return additionalComments, passed

            try:
                base = self.module.orthonormalize(inputBase.copy())
                additionalComments, passed = testBases(base)
                comments += additionalComments
                if (passed):
                    comments += "passed. "
                else:
                    comments += "failed. "
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "

        # Identity case
        comments += "Identity case "

        inputBase = list(np.identity(5))
        evaluate(inputBase)

        # Nonnormal case
        comments += "NonNormal case "

        inputBase = list(np.identity(5) * 2)
        evaluate(inputBase)

        # Nonorthorgonal case
        comments += "NonOrthorgonal case "

        inputBase = [np.array([1. / np.sqrt(3), 1. / np.sqrt(3), 1. / np.sqrt(3)]), np.array([-1., 1. / np.sqrt(3), 0]),
                     np.array([0, -1. / np.sqrt(3), 1. / np.sqrt(3)])]
        evaluate(inputBase)

        # Skew 3D in 4D case
        comments += "Skew 3D in 4D case "

        inputBase = [np.array([1., 1., 1., 0.]), np.array([1., 1., 0., 0.]), np.array([1., 0., 1., 0.])]
        evaluate(inputBase)

        l.extend([task, comments])

    def performTest(self, func):
        l = []
        try:
            func(l)
            return l
        except Exception as e:
            return []

    def runTests(self, module, l):
        self.module = module

        def evaluateResult(task, result):
            if (len(result) == 0):
                l.append([task, 0, "Interrupt."])
            else:
                l.append(result)

        result = self.performTest(self.testA)
        evaluateResult("5.1a)", result)

        result = self.performTest(self.testB)
        evaluateResult("5.1b)", result)

        return l
tester = Tester()
comments = []
tester.runTests(backend, comments)
print(comments)
