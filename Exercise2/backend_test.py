import numpy as np
import sys
import traceback
import backend


class Tester:
    def __init__(self):
        self.module = None
        self.runtime = 300

    def testP(self, P):
        # Square
        if (not P.shape[0] == P.shape[1]):
            return False
        # Only 0 and 1
        if (not (np.unique(P) == [0, 1]).all()):
            return False
        # Amount of 1s has to be equal to rows/columns
        if (not np.count_nonzero(P) == P.shape[0]):
            return False
        # Only one 1 per row/column
        for i in range(P.shape[0]):
            if (np.count_nonzero(P[i]) != 1 or
                    np.count_nonzero(P[:, i]) != 1):
                return False
        # Determinant has to be 1
        if (np.abs(np.linalg.det(P)) - 1 > 1e-16):
            return False
        return True

    def testL(self, L):
        # Lower Triangle
        if (not (np.abs(L - np.tril(L)) < 1e-6).all()):
            return False
        return True

    def testU(self, U):
        # Upper Triangle
        if (not (np.abs(U - np.triu(U)) < 1e-6).all()):
            return False
        return True

    def testPLU(self, P, L, U, A):
        passed = True
        additionalComments = ""
        if (not self.testP(P)):
            additionalComments += "P failed. "
            passed = False
        if (not self.testL(L)):
            additionalComments += "L failed. "
            passed = False
        if (not self.testU(U)):
            additionalComments += "U failed. "
            passed = False
        if (not (np.abs(P.dot(A) - L.dot(U)) < 1e-6).all()):
            additionalComments += "result imprecise. "
            passed = False
        else:
            additionalComments += "passed. "
        return passed, additionalComments

    #############################################
    # Task a
    #############################################

    def testA(self):
        task = "2.2a)"
        points = 0
        comments = ""

        def evaluate(A, numPoints):
            nonlocal comments, points
            try:
                P, L, U = self.module.lu(np.copy(A))
                passed, additionalComments = self.testPLU(P, L, U, A)
                comments += additionalComments + "\n "
                if (passed):
                    points += numPoints
            except Exception as e:
                comments += "crashed. \n " + str(e) + " \n "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " \n "
            finally:
                print(comments)

        # 10x10 upper triangular
        comments = task + " 10x10 upper triangle case "
        A = np.triu(np.ones((10, 10)))
        evaluate(A, 1)

        # 10x10 floats
        comments = task + " 10x10 case "

        A = np.triu(np.ones((10, 10)))
        for i in range(10):
            A[i] *= (i + 1) / 10.
            A[:, i] *= (10 - i + 1) / 10.
        A = A.transpose().dot(A).dot(A) * np.pi / np.e * 50.

        evaluate(A, 2)

        # Pivoting case
        comments = task + " 10x10 Pivoting case "

        A = np.triu(np.ones((10, 10)))
        A = np.roll(A, 1, axis=0)
        evaluate(A, 2)


    #############################################
    # Task b
    #############################################

    def testB(self):
        task = "2.2b) "
        points = 0
        comments = ""

        def evaluate(A, numPoints):
            nonlocal comments, points
            try:
                reference = np.linalg.det(A)
                det = self.module.determinant(np.copy(A))
                if (np.abs(reference - det) < 1e-6):
                    points += numPoints
                    comments += "passed. \n "
                else:
                    comments += "failed. \n "
            except Exception as e:
                comments += "crashed. \n " + str(e) + " \n "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " \n "
            finally:
                print(comments)

        # 10x10 Identity
        comments = task + "Identity case "
        A = np.identity(10)
        evaluate(A, 1)

        # 10x10 Zeros
        comments = task + "Zero case "
        A = np.zeros((10, 10))
        evaluate(A, 1)

        # 10x10 Ones
        comments = task + "Ones case "
        A = np.ones((10, 10))
        evaluate(A, 1)

        # 10x10 floats
        comments = task + "10x10 case "
        A = np.triu(np.ones((10, 10)))
        for i in range(10):
            A[i] *= (i + 1) / 10.
            A[:, i] *= (10 - i + 1) / 10.
        A = A.transpose().dot(A).dot(A) * np.pi / np.e * 50.
        evaluate(A, 2)


    def runTests(self, module):
        self.module = module
        self.testA()
        self.testB()

tester = Tester()
tester.runTests(backend)