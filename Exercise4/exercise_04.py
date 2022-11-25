import backend
import frontend
import numpy as np

x = np.linspace(0,10,100)
b = (np.random.random() - 0.5) * 4
noise = np.random.normal(size=100)
angle = np.random.random()
y = angle * x + noise + b

eigenMatrix = np.random.random((2,2))
eigenMatrix += eigenMatrix.T

# These are the methods you are supposed to implement in backend.py
largestEigenVector = backend.powerMethod(eigenMatrix, np.random.random(2))
smallestEigenVector = backend.inversePowerMethod(eigenMatrix, np.random.random(2))
principalDirection = backend.linearPCA(x, y)

# Displaying stuff
frontend.displayResults(largestEigenVector, smallestEigenVector, principalDirection, eigenMatrix, x, y, angle, b)
