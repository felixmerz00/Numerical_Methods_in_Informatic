import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
from mpl_toolkits.axisartist import Subplot
from mpl_toolkits.mplot3d import Axes3D

# This file only exists to keep the main code clean of windowing stuff.
# Please ignore this file.

class BaseDisplayer:
  def __init__(self, sourceBase, targetBase, transformMatrix):
    self.sourceBase = np.zeros((len(sourceBase), len(sourceBase)))
    self.targetBase = np.zeros((len(sourceBase), len(sourceBase)))
    for i in range(len(sourceBase)):
      self.sourceBase[:,i] = sourceBase[i]
    for i in range(len(targetBase)):
      self.targetBase[:,i] = targetBase[i]
    self.invSource = np.linalg.inv(self.sourceBase)
    self.invTarget = np.linalg.inv(self.targetBase)
    self.transformMatrix = transformMatrix
    self.setupBaseFigure()

  def trSource(self, x, y):
    x, y = np.asarray(x), np.asarray(y)
    return self.sourceBase[0,0] * x + self.sourceBase[1,0] * y, self.sourceBase[0,1] * x + self.sourceBase[1,1] * y

  def inv_trSource(self, x, y):
    x, y = np.asarray(x), np.asarray(y)
    return self.invSource[0,0] * x + self.invSource[1,0] * y, self.invSource[0,1] * x + self.invSource[1,1] * y
  
  def trTarget(self, x, y):
    return self.targetBase[0,0] * x + self.targetBase[0,1] * y, self.targetBase[1,0] * x + self.targetBase[1,1] * y
  
  def inv_trTarget(self, x, y):
    return self.invTarget[0,0] * x + self.invTarget[0,1] * y, self.invTarget[1,0] * x + self.invTarget[1,1] * y
    
  def setupBaseFigure(self):
    fig = plt.figure()
    source_grid_helper = GridHelperCurveLinear((self.trSource, self.inv_trSource))
    target_grid_helper = GridHelperCurveLinear((self.trTarget, self.inv_trTarget))
    self.ax = [Subplot(fig, 1, 2, 1, grid_helper=source_grid_helper), Subplot(fig, 1, 2, 2, grid_helper=target_grid_helper)]
    self.ax[0].set_title("Source Base")
    self.ax[1].set_title("Target Base")

    for a in self.ax:
      a.grid(True)
      a.set_xlim(-1.05, 1.05)
      a.set_ylim(-1.05, 1.05)
      a.set_aspect(aspect='equal')
      fig.add_subplot(a)

  def draw(self):
    sourceBase = np.identity(2)
    transformedBase = self.transformMatrix.dot(sourceBase)
    x = transformedBase[:,0]
    y = transformedBase[:,1]
    xx, yy = self.trTarget(x, y)
    transformedBase = np.array([[xx[0],yy[0]],[xx[1],yy[1]]])

    self.ax[0].quiver(0, 0, xx[0], yy[0], angles = 'xy', scale = 1, scale_units = 'xy', color = 'r')
    self.ax[0].quiver(0, 0, xx[1], yy[1], angles = 'xy', scale = 1, scale_units = 'xy', color = 'b')

    self.ax[1].quiver(0, 0, transformedBase[0,0], transformedBase[1,0], angles = 'xy', scale = 1, scale_units = 'xy', color = 'r')
    self.ax[1].quiver(0, 0, transformedBase[0,1], transformedBase[1,1], angles = 'xy', scale = 1, scale_units = 'xy', color = 'b')
    plt.show()

def displayBaseChange(sourceBase, targetBase, transformMatrix):
  baseDisplayer = BaseDisplayer(sourceBase, targetBase, transformMatrix)
  baseDisplayer.draw()

def displayOrthonormalization(sourceBase, orthonormalizedBase):
  fig = plt.figure()
  ax = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
  ax[0].set_title("Unnormalized")
  ax[1].set_title("Normalized")
  for i in range(2):
    ax[i].grid(True)
    ax[i].set_xlim(-1.05, 1.05)
    ax[i].set_ylim(-1.05, 1.05)
    ax[i].set_zlim(-1.05, 1.05)
  for vec in sourceBase:
    ax[0].quiver(0, 0, 0, vec[0], vec[1], vec[2])
  for vec in orthonormalizedBase:
    ax[1].quiver(0, 0, 0, vec[0], vec[1], vec[2])
  plt.show()
