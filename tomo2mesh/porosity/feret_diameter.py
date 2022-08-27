import numpy as np
from scipy.spatial.distance import pdist
import itertools
import matplotlib.pyplot as plt

from tomo2mesh.misc.voxel_processing import edge_map
from scipy.spatial import ConvexHull

def calc_dia_cross_mean(p, q, rs):
    '''
    https://stackoverflow.com/questions/50727961/shortest-distance-between-a-point-and-a-line-in-3-d-space
    '''
    x = p-q
    dist = np.linalg.norm(np.outer(np.dot(rs-q, x)/np.dot(x, x), x)+q-rs,axis=1)
    mean_dist = np.sqrt(np.mean(dist**2))
    return mean_dist*2

def max_feret_dm(x_void, unit_vect = np.array([1,0,0]), normalize = True):
    #Calculate maximum feret diameter
    verts_e = np.asarray(np.where(edge_map(x_void))).T
    verts = verts_e[ConvexHull(verts_e).vertices]

    # max feret
    dist = pdist(verts, 'euclidean')
    idx = np.argmax(dist)
    feret_dm = dist[idx]

    # mean diameter of cross-section along feret axis
    coord = list(itertools.combinations(verts,2))[idx]
    cross_dia = calc_dia_cross_mean(coord[0], coord[1], np.asarray(np.where(x_void)).T)

    #Calculate orientation of feret diameter
    vect = coord[1]-coord[0]
    phi = np.arccos(vect[0]/np.sqrt(vect[0]**2+vect[1]**2+vect[2]**2))*(180/np.pi)
    theta = np.arctan(vect[1]/max(vect[2],1E-12))*(180/np.pi)
    
    normalized_feret_dm = feret_dm/cross_dia #feret_dm/eq_sphere
    eq_sphere = np.cbrt(6.0/np.pi*np.sum(x_void))
    return (feret_dm, theta, phi, eq_sphere, normalized_feret_dm, cross_dia)

if __name__ == "__main__":
    print("")