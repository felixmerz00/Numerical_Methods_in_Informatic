import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize

def generateRays(angle, resolution):
  numRays = resolution
  numIntersections = resolution + 1

  rayDirection = np.array([np.cos(angle), np.sin(angle)])
  # We want to shoot from left to right
  if(rayDirection[0] < 0):
    rayDirection *= -1

  projectedImageWidth = (np.abs(np.sin(angle)) + np.abs(np.cos(angle))) * resolution
  rayDistance = projectedImageWidth / (numRays)

  rayOrigins = np.zeros((numRays, 2))
  rayOrigins[:,1] = rayDistance * np.linspace(numRays / 2. - 0.5, -numRays / 2. + .5, numRays)

  rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
  rayOrigins = rayOrigins.dot(rotation)
  rayOrigins += resolution / 2.

  # Either set the Origins to (0,y), (x,0) or (x,resolution), depending on the rayDirection
  if(abs(rayDirection[0]) > abs(rayDirection[1])):
    # More horizontal shots
    a = rayOrigins[:,0] / rayDirection[0]
    rayOrigins[:,0] = -1e-6
    rayOrigins[:,1] -= a * rayDirection[1]
  else:
    # More vertical shots
    a = rayOrigins[:,1] / rayDirection[1]
    if(rayDirection[1] > 0):
      # Shots are going upwards
      rayOrigins[:,1] = -1e-6
      rayOrigins[:,0] -= a * rayDirection[0]
    else:
      # Shots are going downwards
      rayOrigins[:,1] = resolution + 1e-6
      rayOrigins[:,0] += a * rayDirection[0]
  return rayOrigins, rayDirection


def intersectGrid(rayOrigins, rayDirection, resolution):
  numRays = resolution
  grid = np.array([np.mgrid[0.:numRays, 0.:resolution + 1][1].transpose(), np.mgrid[0.:numRays, 0.:resolution + 1][1].transpose()])
  for i in range(2):
    grid[i] -= rayOrigins[:,i]
    if(np.abs(rayDirection[i]) < 1e-6):
      grid[i] = np.inf
    else:
      grid[i] /= rayDirection[i]
    grid[i] *= rayDirection[(i + 1) % 2]
    grid[i] += rayOrigins[:,(i + 1) % 2]
  intersections = np.column_stack([np.dstack((np.mgrid[0.:numRays, 0.:resolution + 1][1], grid[0].transpose())), np.dstack((grid[1].transpose(), np.mgrid[0.:numRays, 0.:resolution + 1][1]))])
  indices = [[] for j in range(numRays)]
  lengths = [[] for j in range(numRays)]
  for i in range(intersections.shape[0]):
    validIntersections = filter(lambda x: x[0] > -1e-6 and x[1] > -1e-6 and x[0] < resolution + 1e-6 and x[1] < resolution + 1e-6, intersections[i])
    sortedIntersections = sorted(validIntersections, key = lambda x: np.linalg.norm(x - rayOrigins[i]))
    indices[i] = [0] * (len(sortedIntersections) - 1)
    lengths[i] = [0] * (len(sortedIntersections) - 1)    
    for j in range(len(sortedIntersections) - 1):
      length = np.linalg.norm(sortedIntersections[j + 1] - sortedIntersections[j])
      if(length < 1e-5):
        continue
      midPoint = 0.5 * (sortedIntersections[j + 1] + sortedIntersections[j])
      indices[i][j] = int(midPoint[0]) * resolution + int(midPoint[1])
      lengths[i][j] = length
  for index in indices:
    if(len(index) == 0):
      print("?")
  return indices, lengths

def calculateIntensities(lengths, indices, image):
  intensities = [0] * len(lengths)
  for i in range(len(lengths)):
    intensities[i] = (image[indices[i]] * lengths[i]).sum()
  return intensities

class CTScanner:
  image = None
  resolution = None
  
  def __init__(self, resolution = 20):
    self.resolution = resolution
    self.readImage()
  
  def readImage(self):
    self.image = resize(rgb2gray(plt.imread("Images/Raccoon-small-squared.jpg")), (self.resolution, self.resolution))
    self.flattenedImage = self.image.reshape(self.resolution * self.resolution)
  
  # Input: An angle. Please note, that opposing angles will result in the same measurement.
  # Output:
  #   indices: list of index lists, with (NumRays, NumIntersectionsPerRay) entries.
  #   intensities: Vector of measured intensities with (NumRays) entries.
  #   lengths: list of intersection lengths, with (NumRays, NumIntersectionsPerRay) entries.
  def shootRays(self, angle):
    numRays = self.resolution
    rayOrigins, rayDirection = generateRays(angle, self.resolution)
    indices, lengths = intersectGrid(rayOrigins,rayDirection, self.resolution)
    intensities = calculateIntensities(lengths, indices, self.flattenedImage)
    return indices, intensities, lengths
