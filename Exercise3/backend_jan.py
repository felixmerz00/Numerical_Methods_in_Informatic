import numpy as np

# Task a)
# Implement a method, calculating a base change matrix.
# Input: lists sourceBase and targetBase - lists of vectors (e.g. [np.array([1, 2, 3]), np.array([2, 0, 1]), ...])
# Output: Matrix A - a 2D np.array, with len(sourceBase) x len(targetBase) entries
def changeBase(sourceBase: list, targetBase: list) -> np.array:
  inv = np.linalg.inv(targetBase)
  return np.matmul(inv, np.transpose(sourceBase))

# Task b)
# Implement a method, checking if a subBase spans a Subvectorspace of the space spanned by the given base
# Input: lists sourceBase and subSpace - lists of vectors (e.g. [np.array([1, 2, 3]), np.array([2, 0, 1]), ...])
# Output: bool
def spansSubSpace(sourceBase: list, subBase: list) -> bool:
  #if (len(sourceBase[0]) != len(subBase[0])): return False
  u = sourceBase + subBase
  return np.linalg.matrix_rank(u) <= np.linalg.matrix_rank(sourceBase)
