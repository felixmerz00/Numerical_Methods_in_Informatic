import numpy as np

# Task a)
# Implement a method, calculating a base change matrix.
# Input: lists sourceBase and targetBase - lists of vectors (e.g. [np.array([1, 2, 3]), np.array([2, 0, 1]), ...])
# Output: Matrix A - a 2D np.array, with len(sourceBase) x len(targetBase) entries
def changeBase(sourceBase: list, targetBase: list) -> np.array:
  sourceMatrix = np.array(sourceBase).T
  targetMatrix = np.array(targetBase).T
  return np.matmul(np.linalg.inv(targetMatrix), sourceMatrix)

# Task b)
# Implement a method, checking if a subBase spans a Subvectorspace of the space spanned by the given base
# Input: lists sourceBase and subSpace - lists of vectors (e.g. [np.array([1, 2, 3]), np.array([2, 0, 1]), ...])
# Output: bool
def spansSubSpace(sourceBase: list, subBase: list) -> bool:
  sourceBaseMatrix = np.array(sourceBase)
  subBaseMatrix = np.array(subBase)
  union = np.append(sourceBaseMatrix, subBaseMatrix, axis=0)
  return np.linalg.matrix_rank(sourceBaseMatrix) == np.linalg.matrix_rank(union)