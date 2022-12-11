import numpy as np

# Task a)
# Implement a method performing least squares approximation of a linear courve.
# Input: Vectors x,y. Both 1D np.array of same size.
# Output: list of factors [m, b] representing the linear courve f(x) = mx + b.
def linearLSQ(x: np.array, y: np.array) -> list:
  # Create matrix X with v_1 = (1, 1, ...)^T and v_2 = x
  X = np.c_[np.ones(len(x)), x]
  # Calculate X^T*X --> left side of equation
  left_side = np.dot(X.T, X)
  # Calcuelate X^T*y  --> right side of equation
  right_side = np.dot(X.T, y)
  # Solve(right_side = left_side) --> vecor beta
  beta = np.dot(np.linalg.inv(left_side), right_side)
  # Return [beta_2, beta_1]
  return [beta[1], beta[0]]

# Task b)
# Implement a method, orthogornalizing the given basis.
# Input: sourceBase - list of vectors, as in a)
# Output: orthoronalizedBase - list of vectors, with same size and shape as sourceBase
def orthonormalize(sourceBase: list) -> list:
  # in sourcebase, a row is a column
  orthonormalBasis = np.zeros((len(sourceBase), len(sourceBase[0])))
  orthonormalBasis[0] = sourceBase[0]
  for column_number in range(1, len(sourceBase)):
    xp = np.array(sourceBase[column_number]).T
    subtrahend = np.zeros(len(sourceBase[column_number]))
    for i in range(0, column_number):
      vpLast = np.array(orthonormalBasis[i]).T
      subtrahend += (xp.dot(vpLast)/np.dot(vpLast, vpLast))*vpLast
    orthonormalBasis[column_number] = (xp - subtrahend).T

  for column_number in range(len(orthonormalBasis)):
    vp = orthonormalBasis[column_number]
    orthonormalBasis[column_number] = vp/np.sqrt(np.dot(vp, vp))
  
  return orthonormalBasis

