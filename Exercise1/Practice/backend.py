import numpy as np
from ctscanner import CTScanner

def getReducedEchelonForm(matrix):
    pivot_column = 0
    pivot_row = 0
    is_in_echelon_form = False

    (row_count, column_count) = matrix.shape

    while is_in_echelon_form == False:

        # 1. Get non zero pivot element from matrix (only lower rows)
        for i in range(pivot_row, row_count - 1):
            # If the row is all zero: exchange row with last non zero row
            if not np.any(matrix[i, :]):
                LastNonZeroRow = getLastNonZeroRow(matrix, i)
                matrix[[i, LastNonZeroRow]] = matrix[[LastNonZeroRow, i]]

        free_var_col = True
        for i in range(pivot_row, row_count):
            # If the pivot element is non zero
            if not -0.00001 < matrix[i, pivot_column] < 0.00001:
                free_var_col = False
                matrix[[i, pivot_row]] = matrix[[pivot_row, i]]
                break

        if free_var_col:
            pivot_column += 1
            if pivot_row == row_count or pivot_column == column_count:
                is_in_echelon_form = True

            elif not np.any(matrix[[pivot_row, row_count - 1], [pivot_column, column_count - 1]]):
                is_in_echelon_form = True

            continue

        # 2. Divide the row by its pivot element
        scalar = matrix[pivot_row, pivot_column]
        matrix[pivot_row, :] = matrix[pivot_row, :] / scalar

        # 3. Subtract x times the pivot row from the rows below and above
        for i in range(0, pivot_row):
            scalar = matrix[i, pivot_column]
            if scalar != 0:
                matrix[i, :] = matrix[i, :] - scalar * matrix[pivot_row, :]

        for i in range(pivot_row + 1, row_count):
            scalar = matrix[i, pivot_column]
            if scalar != 0:
                matrix[i, :] = matrix[i, :] - scalar * matrix[pivot_row, :]

        # 4. Move one row down and one column to the left
        pivot_column += 1
        pivot_row += 1
        if pivot_row == row_count or pivot_column == column_count:
            is_in_echelon_form = True

        elif not np.any(matrix[[pivot_row, row_count - 1], [pivot_column, column_count - 1]]):
            is_in_echelon_form = True

    return matrix

def getLastNonZeroRow(matrix, other_Row):
    (row_count, column_count) = matrix.shape
    for i in range(row_count - 1, other_Row, -1):
        if np.any(matrix[i, :]):
            return i


# Task a)
# Implement a method, calculating the rowrank of a given M and return it.
# Obviously, you're not allowed to use any method solving that problem for you.
def rowrank(matrix):
    matrix = getReducedEchelonForm(matrix)
    matrix = matrix.round(3)
    row_rank = getLastNonZeroRow(matrix, 0) + 1
    return row_rank


# Task b)
# Implement a method setting up the linear system, as described in the exercise.
# Make use of the scanner.shootRays(angle) function.
def setUpLinearSystem(scanner):

    scanner_res = scanner.resolution ** 2
    A = np.zeros((scanner_res, scanner_res))
    b = np.zeros(scanner_res)

    current_row = 0
    zero_row = np.zeros(scanner_res)

    # Repeat until full rank
    while np.linalg.matrix_rank(A) < scanner_res:
        # Shoot rays from random angle
        ray_idxs, ray_int, ray_len = scanner.shootRays(np.pi * np.random.rand())
        for i, cell_idxs in enumerate(ray_idxs):

            # Until the zero matrix is fully filled
            if len(A) > current_row:
                b[current_row] = ray_int[i]
                np.put(A[current_row], cell_idxs, ray_len[i])

            # Append row
            else:
                b = np.append(b, ray_int[i])
                A = np.vstack([A, zero_row])

            np.put(A[current_row], cell_idxs, ray_len[i])
            current_row = current_row+1

    return A, b


# Task c)
# Implement the gaussian elimination method, to solve the given system of linear equations
# Add full pivoting to increase accuracy and stability of the solution
def solveLinearSystem(A, b):
    matrix = np.column_stack((A, b))
    matrix = getReducedEchelonForm(matrix)
    return matrix[:, -1]