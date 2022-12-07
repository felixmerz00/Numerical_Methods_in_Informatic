import backend
import frontend
import numpy as np

x = np.linspace(0,10,100)
originalB = (np.random.random() - 0.5) * 4
noise = np.random.normal(size=100)
angle = np.random.random()
y = angle * x + noise + originalB
nonorthonormalBase = [np.array([1. / np.sqrt(3), 1. / np.sqrt(3), 1. / np.sqrt(3)]), np.array([-1., 1. / np.sqrt(3), 0]), np.array([0, -1. / np.sqrt(3), 1. / np.sqrt(3)])]

# These are the methods you are supposed to implement in backend.py
m, b = backend.linearLSQ(x, y)
orthonormalizedBase = backend.orthonormalize(nonorthonormalBase)

# Displaying stuff
frontend.displayResults(m, b, x, y, angle, originalB, nonorthonormalBase, orthonormalizedBase)

