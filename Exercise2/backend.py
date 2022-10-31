import numpy as np

# Task a)
# Implement a method, calculating the LU factorization of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: Matrices P, L and U - same shape as A each.
def lu(A):
  U, r, L, P = toEchelonForm(A)
  return P, L, U

def swapRows(M, r1, r2):
  t = []
  for e in M[r2]:
    t.append(e)
  M[r2] = M[r1]
  M[r1] = t

# Test if value1 and value2 are approximately equal.
def isEqualFloat(value1, value2, epsilon):
  return (value1 >= value2 - epsilon and value1 <= value2 + epsilon)

def toEchelonForm(A):
  m = len(A)
  n = len(A[0])
  pivot_row = 0
  pivot_column = 0
  r = 0
  L = np.identity(m)
  P = np.identity(m)
  A1 = []
  for i in range(m):
    line = []
    for j in range(n):
      line.append(A[i][j])
    A1.append(line)

  # Reduce whole Array A1 to row echelon form
  while(pivot_column < n):
    # Does the first pivot element equal zero?
    if (isEqualFloat(A1[pivot_row][pivot_column], 0.0, 1e-16)):
      hasNonZero = False
      for i in range(pivot_row+1, m):
        # If there is a position below the pivot which is non zero, exchange the rows.
        if (not isEqualFloat(A1[i][pivot_column], 0.0, 1e-16)):
          swapRows(A1, pivot_row, i)
          swapRows(P, pivot_row, i)
          r += 1
          hasNonZero = True
          break
      # If there are only zero elements in the pivot column, 
      # row by row search for non zero elements in the columns 
      # to the right of the pivot_column and save the position
      # of the leading entry.
      if (not hasNonZero):
        leadingEntryColumn = n + 1
        leadingEntryRow = -1
        for i in range(pivot_row, m):
          for j in range(pivot_column+1, n):
            if not isEqualFloat(A1[i][j], 0.0, 1e-16):
              if j < leadingEntryColumn:
                leadingEntryColumn = j
                leadingEntryRow = i
        if leadingEntryColumn < n + 1:
          pivot_column = leadingEntryColumn
          if pivot_row != leadingEntryRow:
            swapRows(A1, pivot_row, leadingEntryRow)
            swapRows(P, pivot_row, leadingEntryRow)
            r+=1
        else:
          # all rows below this are 0
          break

    # Multiplication of rows below the current row with matrix[i_1][j]/pivot_element
    for i in range(pivot_row + 1, m):
      if not isEqualFloat(A1[i][pivot_column], 0.0, 1e-16):
        factor = A1[i][pivot_column]/A1[pivot_row][pivot_column]
        L[i][pivot_column] = factor
        A1[i][pivot_column] = 0
        for j in range(pivot_column+1, n):
          A1[i][j] = A1[i][j] - (factor*A1[pivot_row][j])
    pivot_row += 1
    pivot_column += 1

  # Make sure all entries in the zero rows at the bottom are exactly zero.
  if pivot_row < m:
    for i in range(pivot_row,m):
      A1[i] = np.zeros(n)
  return (A1, r, L, P)
        

# Task b)
# Implement a method, calculating the determinant of A.
# Input: Matrix A - 2D numpy array (e.g. np.array([[1,2],[3,4]]))
# Output: The determinant - a floating number
def determinant(A):
  E, r, L, P = toEchelonForm(A)
  return (-1)**r * np.prod(np.diagonal(E))
