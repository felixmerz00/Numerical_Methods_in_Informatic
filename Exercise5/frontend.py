import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# This file only exists to keep the main code clean of windowing stuff.
# Please ignore this file.

def setupFigure():
  fig = plt.figure()
  ax = [fig.add_subplot(131), fig.add_subplot(132, projection='3d'), fig.add_subplot(133, projection='3d')]
  fig.set_figwidth(18)
  fig.set_figheight(5)
  ax[0].set_title("Least Squares")
  ax[1].set_title("Non-Orthonormal")
  ax[2].set_title("Orthonormal")
  
  for i in range(1,3):
    ax[i].set_xlim(-1.05, 1.05)
    ax[i].set_ylim(-1.05, 1.05)
    ax[i].set_zlim(-1.05, 1.05)
  
  return ax

def displayResults(m, b, x, y, angle, originalB, nonorthonormalBase, orthonormalizedBase):
  ax = setupFigure()
  
  # Task a
  ax[0].plot(x, y, "b.", label = "Data")
  ax[0].plot(x, angle * x + originalB, "r--", label = "Reference")
  ySolution = m * x + b
  ax[0].plot(x, ySolution, "k", label = "Your solution")
  ax[0].legend()
  
  for vec in nonorthonormalBase:
    ax[1].quiver(0, 0, 0, vec[0], vec[1], vec[2])
  for vec in orthonormalizedBase:
    ax[2].quiver(0, 0, 0, vec[0], vec[1], vec[2])
  
  plt.show()
