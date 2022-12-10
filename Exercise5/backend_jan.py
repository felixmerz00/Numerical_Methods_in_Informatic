import numpy as np

# Task a)
# Implement a method performing least squares approximation of a linear courve.
# Input: Vectors x,y. Both 1D np.array of same size.
# Output: list of factors [m, b] representing the linear courve f(x) = mx + b.
def linearLSQ(x: np.array, y: np.array) -> list:
  A = np.c_[x, np.ones(len(x))]
  ATA = np.dot(A.T, A)
  ATb = np.dot(A.T, y)
  result = np.dot(np.linalg.inv(ATA), ATb)
  return result

def orthogonalProjection(x, v):
  scalar = x.dot(v)/np.dot(v,v)
  return scalar*v

# Task b)
# Implement a method, orthogornalizing the given basis.
# Input: sourceBase - list of vectors, as in a)
# Output: orthoronalizedBase - list of vectors, with same size and shape as sourceBase
def orthonormalize(sourceBase: list) -> list:
  # in sourcebase, a row is a column
  orthonormalBase = np.zeros((len(sourceBase), len(sourceBase[0])))
  orthonormalBase[0] = sourceBase[0]
  for colNum in range(1, len(sourceBase)):
    xp = np.array(sourceBase[colNum]).T
    subtraction = np.zeros(len(sourceBase[colNum]))
    for i in range(0, colNum):
      vpLast = np.array(orthonormalBase[i]).T
      subtraction += orthogonalProjection(xp, vpLast)
    orthonormalBase[colNum] = (xp - subtraction).T

  for colNum in range(len(orthonormalBase)):
    vp = orthonormalBase[colNum]
    orthonormalBase[colNum] = vp/np.sqrt(np.dot(vp, vp))
  
  return orthonormalBase

