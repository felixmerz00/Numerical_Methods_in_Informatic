import backend
import frontend
import numpy as np

# Maybe change this to a propper matrix, which you can calculate by hand as well.
A = np.random.random((10, 10))
A += np.identity(10)

sourceBase = [np.array([1. / np.sqrt(2), 1. / np.sqrt(2)]), np.array([-1. / np.sqrt(2), 1. / np.sqrt(2)])]
subBase = [np.array([1., 1.]), np.array([1., 0.])]
targetBase = [np.array([1., 0.]), np.array([0., 1.])]

# These are the methods you are supposed to implement in backend.py
transformMatrix = backend.changeBase(sourceBase, targetBase)

isSubSpace = backend.spansSubSpace(sourceBase, subBase)

# Displaying stuff
frontend.displayBaseChange(sourceBase, targetBase, transformMatrix)

# Sorry, again no good visualization technique for bools 
print(isSubSpace)
