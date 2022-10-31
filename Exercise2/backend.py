import numpy as np

# Task a)
# Implement a method, calculating the LU factorization of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: Matrices P, L and U - same shape as A each.
def lu(A):
  return np.identity(A.shape[0]), np.zeros(A.shape), np.zeros(A.shape)

# Task b)
# Implement a method, calculating the determinant of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: The determinant - a floating number
def determinant(A):
  return 0

