import numpy as np
import matplotlib.pyplot as plt

# This file only exists to keep the main code clean of windowing stuff.
# Please ignore this file.

def removeTicks(ax):
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  ax.set_xticks([])
  ax.set_yticks([])

def setupFigure():
  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.set_figwidth(12)
  fig.set_figheight(5)
  ax1.set_title("Your result")
  ax2.set_title("Reference")
  removeTicks(ax1)
  removeTicks(ax2)
  return ax1, ax2

def displayResults(solution, scanner):
  ax1, ax2 = setupFigure()
  ax1.imshow(solution.reshape(scanner.resolution, scanner.resolution), cmap = 'gray', vmin = 0, vmax = 1)
  ax2.imshow(scanner.image, cmap = 'gray', vmin = 0, vmax = 1)
  plt.show()
