import backend
import frontend
import numpy as np

# Maybe change this to a propper matrix, which you can calculate by hand as well.
A = np.random.random((10, 10))
A += np.identity(10)

# These are the methods you are supposed to implement in backend.py
P, L, U = backend.lu(A)
det = backend.determinant(A)

frontend.displayLU(A, P, L, U)


print("Your determinant:", det)
print("Reference: ", np.linalg.det(A))
