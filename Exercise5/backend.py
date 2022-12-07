import numpy as np

# Task a)
# Implement a method performing least squares approximation of a linear courve.
# Input: Vectors x,y. Both 1D np.array of same size.
# Output: list of factors [m, b] representing the linear courve f(x) = mx + b.
def linearLSQ(x: np.array, y: np.array) -> list:
  return [1, 0]

# Task b)
# Implement a method, orthogornalizing the given basis.
# Input: sourceBase - list of vectors, as in a)
# Output: orthoronalizedBase - list of vectors, with same size and shape as sourceBase
def orthonormalize(sourceBase: list) -> list:
  return [np.ones(sourceBase[0].shape[0])] * len(sourceBase)

