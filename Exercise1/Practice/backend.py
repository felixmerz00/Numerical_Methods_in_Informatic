import numpy as np
from ctscanner import CTScanner

# Task a)
# Implement a method, calculating the rowrank of a given matrix and return it.
# Obviously, you're not allowed to use any method solving that problem for you.
def rowrank(matrix):
  return 0

# Task b)
# Implement a method setting up the linear system, as described in the exercise.
# Make use of the scanner.shootRays(angle) function.
def setUpLinearSystem(scanner):
  A = np.ones((scanner.resolution ** 2, scanner.resolution ** 2))
  b = np.zeros(scanner.resolution ** 2)
  return A, b

# Task c)
# Implement the gaussian elimination method, to solve the given system of linear equations
# Add full pivoting to increase accuracy and stability of the solution
def solveLinearSystem(A, b):
  return np.ones(b.shape)

