import numpy as np
from ctscanner import CTScanner

def getReducedEchelonForm(M):
    pivot_spalte = 0
    pivot_zeile = 0
    echelon_form = False

    (row_count, column_count) = M.shape

    while echelon_form == False:

        # 1. Get Non-Zero Pivot-element from M (only lower rows)
        for i in range(pivot_zeile, row_count - 1):
            # If row is all zero: exchange row with last non zero row
            if not np.any(M[i, :]):
                LastNonZeroRow = getLastNonZeroRow(M, i)
                M[[i, LastNonZeroRow]] = M[[LastNonZeroRow, i]]

        free_var_col = True
        for i in range(pivot_zeile, row_count):
            # If pivot element is non zero
            if not -0.00001 < M[i, pivot_spalte] < 0.00001:
                free_var_col = False
                M[[i, pivot_zeile]] = M[[pivot_zeile, i]]
                break

        if free_var_col:
            pivot_spalte += 1
            if pivot_zeile == row_count or pivot_spalte == column_count:
                echelon_form = True

            elif not np.any(M[[pivot_zeile, row_count - 1], [pivot_spalte, column_count - 1]]):
                echelon_form = True

            continue

        # 2. Divide row by pivot element
        scalar = M[pivot_zeile, pivot_spalte]
        M[pivot_zeile, :] = M[pivot_zeile, :] / scalar

        # 3. Subtract x times pivoting row from lower & upper rows
        for i in range(0, pivot_zeile):
            scalar = M[i, pivot_spalte]
            if scalar != 0:
                M[i, :] = M[i, :] - scalar * M[pivot_zeile, :]

        for i in range(pivot_zeile + 1, row_count):
            scalar = M[i, pivot_spalte]
            if scalar != 0:
                M[i, :] = M[i, :] - scalar * M[pivot_zeile, :]

        # 4. Move one down and left
        pivot_spalte += 1
        pivot_zeile += 1
        if pivot_zeile == row_count or pivot_spalte == column_count:
            echelon_form = True

        elif not np.any(M[[pivot_zeile, row_count - 1], [pivot_spalte, column_count - 1]]):
            echelon_form = True

    return M

def getLastNonZeroRow(matrix, otherRow):
    (row_count, column_count) = matrix.shape
    for i in range(row_count - 1, otherRow, -1):
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

            # Until 0-matrix fully filled
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