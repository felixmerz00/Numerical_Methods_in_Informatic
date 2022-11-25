import numpy as np
import matplotlib.pyplot as plt

# This file only exists to keep the main code clean of windowing stuff.
# Please ignore this file.

def setupFigure():
  fig, ax = plt.subplots(1, 3)
  fig.set_figwidth(18)
  fig.set_figheight(5)
  ax[0].set_title("Powermethod")
  ax[1].set_title("Inverse Powermethod")
  ax[2].set_title("PCA")
  
  for i in range(2):
    ax[i].set_xlim(-1.05, 1.05)
    ax[i].set_ylim(-1.05, 1.05)
  return ax

def displayResults(largestEigenVector, smallestEigenVector, principalDirection, eigenMatrix, x, y, angle, b):
  ax = setupFigure()
  val, vec = np.linalg.eig(eigenMatrix)
  index = np.argmax(np.abs(val))

  # Task a
  ax[0].quiver(0, 0, vec[0, index], vec[1, index], angles='xy', scale=1, scale_units='xy', label='Reference', color = 'r')
  ax[0].quiver(0, 0, largestEigenVector[0], largestEigenVector[1], angles='xy', scale=1, scale_units='xy', label='Your Solution', color = 'k')

  # Task b
  ax[1].quiver(0, 0, vec[0, 1 - index], vec[1, 1 - index], angles='xy', scale=1, scale_units='xy', label='Reference', color = 'r')
  ax[1].quiver(0, 0, smallestEigenVector[0], smallestEigenVector[1], angles='xy', scale=1, scale_units='xy', label='Your Solution', color = 'k')
  
  # Task c
  
  ax[2].plot(x, y, "b.", label = "Data")
  ax[2].plot(x, angle * x + b, "r--", label = "Reference")
  mean = [x.mean(), y.mean()]
  ySolution = mean[1] + (x - mean[0]) * principalDirection[1] / principalDirection[0]
  ax[2].plot(x, ySolution, "k", label = "Your solution")
  
  for a in ax:
    a.legend()

  plt.show()
