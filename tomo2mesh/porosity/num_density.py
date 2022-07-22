import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


def num_density(list_centers, radius):
    dist = squareform(pdist(list_centers, 'euclidean'))
    #count = np.array([np.count_nonzero(arr<=radius) for arr in dist])-1
    count = [np.count_nonzero(arr<=radius)-1 for arr in dist]
    #rho = count/((4/3)*np.pi*radius**3) 
    return np.asarray(count)
    #return rho

