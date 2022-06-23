import numpy as np
from scipy.spatial.distance import pdist
import itertools
import matplotlib.pyplot as plt

from tomo2mesh.misc.voxel_processing import edge_map
from scipy.spatial import ConvexHull


def max_feret_dm(x_void, unit_vect = np.array([1,0,0]), normalize = True):
    #Calculate maximum feret diameter
    verts = np.asarray(np.where(edge_map(x_void))).T
    verts = verts[ConvexHull(verts).vertices]
    dist = pdist(verts, 'euclidean')
    idx = np.argmax(dist)
    feret_dm = dist[idx]
    eq_sphere = np.cbrt(6.0/np.pi*np.sum(x_void))
    normalized_feret_dm = feret_dm/eq_sphere

    #Calculate orientation of feret diameter
    coord = list(itertools.combinations(verts,2))[idx]
    vect = coord[1]-coord[0]
    #phi = np.arccos(np.dot(vect,unit_vect)/(np.sqrt(vect.dot(vect))*np.sqrt(unit_vect.dot(unit_vect)))) 
    phi = np.arccos(vect[0]/np.sqrt(vect[0]**2+vect[1]**2+vect[2]**2))*(180/np.pi)
    theta = np.arctan(vect[1]/max(vect[2],1E-12))*(180/np.pi)
    return (feret_dm, theta, phi, eq_sphere, normalized_feret_dm)

if __name__ == "__main__":
    print("")