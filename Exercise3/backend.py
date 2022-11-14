import numpy as np

# Task a)
# Implement a method, calculating a base change matrix.
# Input: lists sourceBase and targetBase - lists of vectors (e.g. [np.array([1, 2, 3]), np.array([2, 0, 1]), ...])
# Output: Matrix A - a 2D np.array, with len(sourceBase) x len(targetBase) entries
def changeBase(sourceBase: list, targetBase: list) -> np.array:
  # target base = source base setzten
  # Gleichungssystem in reduced row echelon form bringen
  # Die Lösung von diesem Gleichungssytem ist die 
  return np.identity(len(targetBase))

# Task b)
# Implement a method, checking if a subBase spans a Subvectorspace of the space spanned by the given base
# Input: lists sourceBase and subSpace - lists of vectors (e.g. [np.array([1, 2, 3]), np.array([2, 0, 1]), ...])
# Output: bool
def spansSubSpace(sourceBase: list, subBase: list) -> bool:
  h = sourceBase + subBase
  out = np.linalg.matrix_rank(h) <= np.linalg.matrix_rank(sourceBase)
  return out
