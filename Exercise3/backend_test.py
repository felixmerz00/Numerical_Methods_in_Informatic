import numpy as np
import sys
import traceback
import backend

class Tester:
  def __init__(self):
    self.module = None
    self.runtime = 300

  #############################################
  # Task a
  #############################################

  def testA(self, l: list):
    task = "3.2a)"
    points = 0
    comments = ""
    
    def evaluate(sourceBase, targetBase, reference, numPoints):
      nonlocal comments, points
      try:
        A = self.module.changeBase(sourceBase, targetBase)
        if((np.abs(A - reference) < 1e-6).all()):
          comments += "passed. \n "
          points += numPoints
        else:
          if((np.abs(np.linalg.inv(A) - reference) < 1e-6).all()):
            comments += "inverse. \n "
            points += numPoints / 2.
          else:
            comments += "failed. \n "
      except Exception as e:
        comments += "crashed. \n " + str(e) + " \n "
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        fname = str(tb.filename.split("/")[-1])
        lineno = str(tb.lineno)
        comments += "Here: " + str(fname) + ":" + str(lineno) + " \n "

    # 45 degree case
    comments += "45 degree case "
    base = np.identity(3).T
    sourceBase = [np.array([1. / np.sqrt(3), 1. / np.sqrt(3), -1. / np.sqrt(3)]),
                  np.array([-1. / np.sqrt(3), 1. / np.sqrt(3), 1. / np.sqrt(3)]),
                  np.array([1. / np.sqrt(3), -1. / np.sqrt(3), 1. / np.sqrt(3)])]
    targetBase = list(base)
    reference = np.array(sourceBase).T
    
    evaluate(sourceBase, targetBase, reference, 1)
    
    # 45 degree to -45 degree case
    comments += "45 degree to -45 degree case "
    sourceBase = [np.array([1. / np.sqrt(3), 1. / np.sqrt(3), -1. / np.sqrt(3)]),
                  np.array([-1. / np.sqrt(3), 1. / np.sqrt(3), 1. / np.sqrt(3)]),
                  np.array([1. / np.sqrt(3), -1. / np.sqrt(3), 1. / np.sqrt(3)])]
    targetBase = [np.array([1. / np.sqrt(3), -1. / np.sqrt(3), 1. / np.sqrt(3)]),
                  np.array([1. / np.sqrt(3), 1. / np.sqrt(3), -1. / np.sqrt(3)]),
                  np.array([-1. / np.sqrt(3), 1. / np.sqrt(3), 1. / np.sqrt(3)])]
    reference = np.linalg.inv(np.array(targetBase).T).dot(np.array(sourceBase).T)
    
    evaluate(sourceBase, targetBase, reference, 2)
    
    # Roll case
    comments += "Roll case "

    base = np.identity(10) + np.roll(np.identity(10), 2, axis = 0)
    base[-1, 0] = 0
    sourceBase = list(base)
    base = np.identity(10) + np.roll(np.identity(10), 1, axis = 0) + np.roll(np.identity(10), -1, axis = 0)
    base[0, -1] = 0
    base[-1, 0] = 0
    targetBase = list(base)
    reference = np.linalg.inv(np.array(targetBase).T).dot(np.array(sourceBase).T)
    evaluate(sourceBase, targetBase, reference, 2)
    
    l.extend([task, points, comments])

    
  #############################################
  # Task b
  #############################################
  
  def testB(self, l: list):
    task = "3.2b)"
    points = 0
    comments = ""
    
    def evaluate(base, subBase, reference, numPoints):
      nonlocal comments, points
      try:
        isSubSpace = self.module.spansSubSpace(base, subBase)
        if(isSubSpace == reference):
          comments += "passed. \n "
          points += numPoints
        else:
          comments += "failed. \n "
      except Exception as e:
        comments += "crashed. \n " + str(e) + " \n "
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        fname = str(tb.filename.split("/")[-1])
        lineno = str(tb.lineno)
        comments += "Here: " + str(fname) + ":" + str(lineno) + " \n "

    # R2-R2 case
    comments += "R2-R2 case "

    base = list(np.identity(2))
    subBase = list(np.identity(2))
    evaluate(base, subBase, True, 1)

    # R3-R2 case°
    comments += "R3-R2 case "
    
    base = list(np.identity(3))
    subBase = list(np.identity(3))[:-1]
    evaluate(base, subBase, True, 1)
    
    # XY-XZ case
    comments += "XY-XZ case "
    
    base = list(np.identity(3))[:-1]
    subBase = list(np.identity(3))[0::2]
    evaluate(base, subBase, False, 1)
    
    # R2-R3 case
    comments += "R2-R3 case "

    base = list(np.identity(3))[:-1]
    subBase = list(np.identity(3))
    evaluate(base, subBase, False, 1)
    
    # R1-Zero case
    comments += "R1-Zero "
    
    base = [np.array([1.])]
    subBase = [np.array([0.])]
    evaluate(base, subBase, True, 1)
    
    l.extend([task, points, comments])

  def performTest(self, func):
    # manager = multiprocessing.Manager()
    # localList = manager.list()
    # p = multiprocessing.Process(target = func, args = (localList,))
    # p.start()
    # p.join(self.runtime)
    # p.kill()
    # if(p.exitcode != 0):
    #   return []
    # else:
    #   return list(localList)
    l = []
    try:
      func(l)
      return l
    except Exception as e:
      return []

  def runTests(self, module, l):
    self.module = module

    def evaluateResult(task, result):
      if(len(result) == 0):
        l.append([task, 0, "Interrupt."])
      else:
        l.append(result)

    result = self.performTest(self.testA)
    evaluateResult("3.2a)", result)

    result = self.performTest(self.testB)
    evaluateResult("3.2b)", result)

    print(l)

tester = Tester()
tester.runTests(backend, [])