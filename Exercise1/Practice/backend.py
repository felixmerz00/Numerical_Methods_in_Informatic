import numpy as np
from ctscanner import CTScanner

# Task a)
# Implement a method, calculating the rowrank of a given matrix and return it.
# Obviously, you're not allowed to use any method solving that problem for you.
def rowrank(matrix):
  A = matrix

  # Put matrix into Echelon form
  m = len(A)  # Number of rows
  n = len(A[0]) # Number of columns
  # Find the pivot elements. A has at most m pivot elements.
  j = -1 # Column of current pivot element
  for i in range(m-1):
    # Step 1: Begin in the leftmost nonzero column
    non_zero_value = False
    while(non_zero_value == False and j < n-1):
      j += 1
      # Check the whole column for non zero values
      for i1 in range(i, m):
        if(A[i1][j] != 0):
          non_zero_value = True

    # Step 2: Select the biggest absolute element in the column as pivot
    # If necessary, interchange rows to move this entry into the pivot position
    i_max = i # Index of biggest element
    for i1 in range(i+1, m):
      if(abs(A[i1][j]) > abs(A[i_max][j])):
        i_max = i1
    if(i != i_max):
      A[[i, i_max]] = A[[i_max, i]]

    # The pivot element is now at A[i][j]
    
    # Step 3: Create zeros in all positions below the pivot by unsing row replacement operations
    for i1 in range(i+1, m):
      if(A[i1][j] != 0):
        x = (-1)*A[i1][j]/A[i][j]  # Factor for the row operation
        # Execute the row operation onto the whole row
        for j1 in range(0, n):
          A[i1][j1] += x * A[i][j1]
      
    # Step 4: Cover the row containing the pivot position and all rows above
    # Then Apply steps 1-3 to the submatrix that remains
    # This step is done by increasing the value of i in the for loop.
  
  # Count pivot elements
  rank = 0
  for i in range(m):
    for j in range(n):
      if(A[i][j] != 0):
        rank += 1
        break
  return rank



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

def forwardSubstitution(A, b):
  m = len(A)  # Number of rows
  n = len(A[0]) # Number of columns
  # Find the pivot elements. A has at most m pivot elements.
  j = -1 # Column of current pivot element
  for i in range(m-1):
    # Step 1: Begin in the leftmost nonzero column
    non_zero_value = False
    while(non_zero_value == False and j < n-1):
      j += 1
      # Check the whole column for non zero values
      for i1 in range(i, m):
        if(A[i1][j] != 0):
          non_zero_value = True

    # Step 2: Select the biggest absolute element in the column as pivot
    # If necessary, interchange rows to move this entry into the pivot position
    i_max = i # Index of biggest element
    for i1 in range(i+1, m):
      if(abs(A[i1][j]) > abs(A[i_max][j])):
        i_max = i1
    if(i != i_max):
      A[[i, i_max]] = A[[i_max, i]]
      (b[i], b[i_max]) = (b[i_max], b[i])

    # The pivot element is now at A[i][j]
    
    # Step 3: Create zeros in all positions below the pivot by unsing row replacement operations
    for i1 in range(i+1, m):
      if(A[i1][j] != 0):
        x = (-1)*A[i1][j]/A[i][j]  # Factor for the row operation
        # Execute the row operation onto the whole row
        for j1 in range(0, n):
          A[i1][j1] += x * A[i][j1]
        b[i1] += x * b[i]
      
    # Step 4: Cover the row containing the pivot position and all rows above
    # Then Apply steps 1-3 to the submatrix that remains
    # This step is done by increasing the value of i in the for loop.
  return A, b

  
def backwardSubstitution(A, b):
  print(A)
  print("\n")
  # STEP 5: 
  # Find the rightmost pivot and working upward and to the left
  m = len(A)
  n = len(A[0])
  for i in range(m-1, -1, -1):
    j = 0
    while(A[i][j] == 0):
      j += 1
      if(j == n):
        j = 0
        i -= 1
        
    # i, j now point to the pivot element
    # Make the pivot to one by dividing the whole row by the value in the pivot
    if(A[i][j] != 1):
      x = A[i][j]
      for j1 in range(j, n):
        A[i][j1] /= x
      b[i] /= x

    # Create zeros above each pivot.
    for i1 in range(i-1, -1, -1):
      x = A[i1][j]
      for j1 in range(j, n):
        A[i1][j1] -= x*A[i][j1]
      b[i1] -= x*b[i]

  return A, b

def solveLinearSystem(A, b):
  A, b = forwardSubstitution(A, b)
  A, b = backwardSubstitution(A, b)

  return np.ones(b.shape)

