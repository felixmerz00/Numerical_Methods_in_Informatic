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
  fig, ax = plt.subplots(1, 4)
  fig.set_figwidth(24)
  fig.set_figheight(5)
  ax[0].set_title("L")
  ax[1].set_title("U")
  ax[2].set_title("LU")
  ax[3].set_title("PA")
  for a in ax:
    removeTicks(a)
  return ax

def displayLU(A, P, L, U):
  ax1, ax2, ax3, ax4 = setupFigure()
  ax1.imshow(L, cmap = 'gray')
  ax2.imshow(U, cmap = 'gray')
  ax3.imshow(L.dot(U), cmap = 'gray')
  ax4.imshow(P.dot(A), cmap = 'gray')
  plt.show()
