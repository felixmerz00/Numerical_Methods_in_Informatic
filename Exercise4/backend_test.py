import numpy as np
import sys
import backend
import traceback

class Tester:
  def __init__(self):
    self.module = None

  #############################################
  # Task a
  #############################################

  def testA(self):
    task = "4.1a)"
    comments = ""
    
    def evaluate(A, b):
      nonlocal comments
      try:
        eigenVector = self.module.powerMethod(np.copy(A), np.copy(b))
        eigenVector /= np.linalg.norm(eigenVector)
        vals, reference = np.linalg.eig(A)
        maxReference = reference[:,np.argmax(np.abs(vals))]
        dot = eigenVector.dot(maxReference)
        if(1 - np.abs(dot) < 1e-6):
          comments += "passed. "
        else:
          comments += "failed. "
      except Exception as e:
        comments += "crashed. " + str(e) + " "
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        fname = str(tb.filename.split("/")[-1])
        lineno = str(tb.lineno)
        comments += "Here: " + str(fname) + ":" + str(lineno) + " "

    # Positive case
    comments += "Positive case "

    A = np.triu(np.ones((10, 10)))
    for i in range(10):
      A[i] *= (i + 1) / 10.
      A[:, i] *= (10 - i + 1) / 10.
    A = A.transpose().dot(A).dot(A) * np.pi / np.e
    b = np.ones(10)
    evaluate(A, b)
    
    # Negative case
    comments += "Negative case "
    
    A = np.triu(np.ones((10, 10)))
    for i in range(10):
      A[i] *= (i + 1) / 10.
      A[:, i] *= (10 - i + 1) / 10.
    A = A.transpose().dot(A).dot(A) * -np.pi / np.e
    b = np.ones(10)
    evaluate(A, b)

    print(f'{task} {comments}')

  #############################################
  # Task b
  #############################################

  def testB(self):
    task = "4.1b)"
    comments = ""
    
    def evaluate(A, b):
      nonlocal comments
      try:
        eigenVector = self.module.inversePowerMethod(np.copy(A), np.copy(b))
        eigenVector /= np.linalg.norm(eigenVector)
        vals, reference = np.linalg.eig(A)
        minReference = reference[:,np.argmin(np.abs(vals))]
        dot = eigenVector.dot(minReference)
        dot /= np.linalg.norm(eigenVector)
        if(1 - np.abs(dot) < 1e-6):
          comments += "passed. "
        else:
          comments += "failed. "
      except Exception as e:
        comments += "crashed. " + str(e) + " "
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        fname = str(tb.filename.split("/")[-1])
        lineno = str(tb.lineno)
        comments += "Here: " + str(fname) + ":" + str(lineno) + " "

    # Positive case
    comments += "Positive case "

    A = np.triu(np.ones((10, 10)))
    for i in range(10):
      A[i] *= (i + 1) / 10.
      A[:, i] *= (10 - i + 1) / 10.
    A = A.transpose().dot(A).dot(A) * np.pi / np.e
    evaluate(A, np.ones(10))
    
    # Negative case
    comments += "Negative case "
    
    A = np.triu(np.ones((10, 10)))
    for i in range(10):
      A[i] *= (i + 1) / 10.
      A[:, i] *= (10 - i + 1) / 10.
    A = A.transpose().dot(A).dot(A) * -np.pi / np.e
    evaluate(A, np.ones(10))

    print(f'{task} {comments}')

  #############################################
  # Task c
  #############################################

  def testC(self):
    task = "4.1c)"
    comments = ""
    
    def evaluate(X, Y):
      nonlocal comments
      try:
        eigenVector = self.module.linearPCA(np.copy(X), np.copy(Y))
        eigenVector /= np.linalg.norm(eigenVector)
        data = np.array([X, Y]).T
        data -= np.mean(data, axis = 0)
        vals, reference = np.linalg.eig(data.T.dot(data))
        referenceVector = reference[:,np.argmax(np.abs(vals))]
        dot = eigenVector.dot(referenceVector)
        dot /= np.linalg.norm(eigenVector)
        if(1 - np.abs(dot) < 1e-6):
          comments += "passed. "
        else:
          comments += "failed. "
      except Exception as e:
        comments += "crashed. " + str(e) + " "
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        fname = str(tb.filename.split("/")[-1])
        lineno = str(tb.lineno)
        comments += "Here: " + str(fname) + ":" + str(lineno) + " "

    # X case
    comments += "X case "

    X = np.linspace(5, 15, 100)
    Y = (np.random.random(100) - 0.5) * 3. + 100
    evaluate(X, Y)

    # Y case
    comments += "Y case "

    X = (np.random.random(100) - 0.5) * 3. + 100.
    Y = np.linspace(5, 15, 100)
    evaluate(X, Y)

    # XY case
    comments += "XY case "

    X = np.linspace(5, 15, 100)
    Y = np.copy(X) + (np.random.random(100) - 0.5) * 3. + 100
    evaluate(X, Y)

    print(f'{task} {comments}')

  def performTest(self, func):
    try:
      func()
    except Exception as e:
      return print(e)

  def runTests(self, module):
    self.module = module
    self.performTest(self.testA)
    self.performTest(self.testB)
    self.performTest(self.testC)

tester = Tester()
tester.runTests(backend)