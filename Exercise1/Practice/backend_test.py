import numpy as np
from backend import rowrank, solveLinearSystem, setUpLinearSystem
from ctscanner import CTScanner


def testA():
    def evaluate(A, reference, comment):
        try:
            if (rowrank(np.copy(A)) == reference):
                comment += "passed."
            else:
                comment += "failed."
        except Exception as e:
            comment += "crashed. \n " + str(e)
        finally:
            print(comment)

    # 10x10 upper triangular
    comment = "a) 10x10 upper triangle case "

    A = np.triu(np.ones((10, 10)))
    reference = 10
    evaluate(A, reference, comment)

    # 10x10 floats
    comment = "a) 10x10 case with floating numbers "

    A = np.triu(np.ones((10, 10)))

    for i in range(10):
        A[i] *= (i + 1) / 10.
        A[:, i] *= (10 - i + 1) / 10.
    A = A.transpose().dot(A).dot(A) * np.pi / np.e * 50.
    A[-1] = A[0] + A[-2]
    reference = np.linalg.matrix_rank(A)
    evaluate(A, reference, comment)

#############################################
# Task b
#############################################

def testB():
    def evaluate(scanner, comment):
        try:
            A, b = setUpLinearSystem(scanner)
            x = np.linalg.lstsq(A, b.reshape(A.shape[0]), rcond=None)[0]
            if ((np.abs(x.reshape(scanner.resolution, scanner.resolution) - scanner.image) < 1e-3).all()):
                comment += "passed."
            else:
                comment += "failed."
        except Exception as e:
            comment += "crashed. \n " + str(e)
        finally:
            print(comment)

    # Default 20x20
    comment = "b) Default case "
    scanner = CTScanner()
    evaluate(scanner, comment)

    # 30x30
    comment = "b) 30x30 case "
    scanner = CTScanner(30)
    evaluate(scanner, comment)
#############################################
# Task c
#############################################

def testC():
    def evaluate(A, b, reference, comment, epsilon=1e-16):
        try:
            x = solveLinearSystem(A, b)
            if ((np.abs(x - reference) < epsilon).all()):
                comment += "passed."
            else:
                comment += "failed."
        except Exception as e:
            comment += "crashed. \n " + str(e)
        finally:
            print(comment)

    # Identity
    comment = "c) Identity case "

    A = np.identity(30)
    b = np.ones(30)
    for i in range(30):
        b[i] = (i + 1)
    reference = np.copy(b)
    evaluate(A, b, reference, comment)

    # 30x30 floats
    comment = "c) 30x30 case "

    A = np.tril(np.ones((30, 30)))
    tmp = np.triu(np.ones((30, 30)))
    for i in range(30):
        A[i] *= i + 1
        A[:, i] /= i + 1
        tmp[i] *= (i + 1) ** 2
        tmp[:, i] /= (i + 1) ** 2
    A = A.dot(tmp)
    A *= np.pi / np.e
    b = np.ones(30)
    reference = np.linalg.solve(A, b)
    evaluate(A, b, reference, comment, 1e-10)

    # 30x30 instable
    comment = "c) 30x30 unstable case "

    A = np.tril(np.ones((30, 30)))
    tmp = np.triu(np.ones((30, 30)))
    for i in range(30):
        A[i] *= i + 1
        A[:, i] /= i + 1
        tmp[i] *= (i + 1) ** 2
        tmp[:, i] /= (i + 1) ** 2
    A = A.dot(tmp)
    A *= np.pi / np.e
    for i in range(30):
        A[i] *= np.exp(i)
        A[:, i] /= np.exp(i)
    b = np.ones(30)
    reference = np.linalg.solve(A, b)
    evaluate(A, b, reference, comment, 1e-2)

    # 10x10 Pivoting
    comment = "c) 10x10 Pivoting case "

    A = np.triu(np.ones((10, 10)))
    A = np.roll(A, 1, axis=0)
    b = np.ones(10)
    reference = np.linalg.solve(A, b)
    evaluate(A, b, reference, comment)

def runTests():
    testA()
    testB()
    testC()

runTests()

