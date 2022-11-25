import numpy as np

# Task a)
# Implement a method, calculating the largest eigenvector of A with b as an initial guess.
# Input: Matrix A, Vector b. A - 2D numpy array, b - 1D numpy array
# Output: The Eigenvector of A with the largest (absolute) Eigenvalue, given as 1D np.array.
def powerMethod(A: np.array, b: np.array) -> np.array:
  return np.ones(b.shape)

# Task b)
# Implement a method, calculating the smallest eigenvector of A with b as an initial guess.
# Input: Matrix A, Vector b. A - 2D numpy array, b - 1D numpy array
# Output: The Eigenvector of A with the smallest (absolute) Eigenvalue, given as 1D np.array.
def inversePowerMethod(A: np.array, b: np.array) -> np.array:
  return np.ones(b.shape)

# Task c)
# Implement a method performing a PCA on given data.
# Input: Vectors x,y. Both 1D np.array of same size.
# Output: The Principal direction of the given data, represented as 1D np.array
def linearPCA(x: np.array, y: np.array) -> np.array:
  return np.ones(2)
