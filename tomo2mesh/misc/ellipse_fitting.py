#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Yashas Satapathy
Email: yashsatapathy[at]yahoo[dot]com

"""


import numpy as np
import sys
import time
from numpy import linalg
from multiprocessing import Pool


def max_ellip_rad(x_void):
    size = np.sum(x_void)

    #Edge map
    msk = np.zeros_like(x_void)
    tmp = x_void[:-1]!=x_void[1:]
    msk[:-1][tmp] = 1
    msk[1:][tmp] = 1
    tmp = x_void[:,:-1]!=x_void[:,1:]
    msk[:,:-1][tmp] = 1
    msk[:,1:][tmp] = 1
    tmp = x_void[:,:,:-1]!=x_void[:,:,1:]
    msk[:,:,:-1][tmp] = 1
    msk[:,:,1:][tmp] = 1
    x_void_adj = msk>0

    #Find all the points (x,y,z) where the void exists
    if size<2000:
        P = np.asarray(np.where(x_void_adj)).T 
        P = np.array(P)
    
    if size>=2000 and size<1e6: #Trim data for larger voids to reduce computation time
        # P = np.asarray(np.where(x_void_adj)).T 
        # P = np.array(P[::2])
        P = np.asarray(np.where(x_void_adj[::2,::2,::2])).T 
        P = np.array(P)*2

    if size>=1e6:
        P = np.asarray(np.where(x_void_adj[::4,::4,::4])).T 
        P = np.array(P)*4


    #Calculate the center location, radii, and orientation matrix of the fitted ellipsoid
    tolerance = 0.01
    (N, d) = np.shape(P)
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)]) 
    QT = Q.T
    
    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse 
    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = linalg.inv(
                    np.dot(P.T, np.dot(np.diag(u), P)) - 
                    np.array([[a * b for b in center] for a in center])
                    ) / d
                    
    # Get the values we'd like to return
    U, s, rotation = linalg.svd(A)
    radii = 1.0/np.sqrt(s)

    rad = radii

    #Calculate the ellipticity of the ellipsoid
    a,b,c = sorted(rad, reverse=True)
    elp_E = np.sqrt((a**2-b**2)/a**2) #Equatorial ellipticity of tri-axial ellipsoid
    elp_P = np.sqrt((a**2-c**2)/a**2) #Polar ellipticity of tri-axial ellipsoid
    ellipticity_E = elp_E
    ellipticity_P = elp_P

    return ellipticity_E, ellipticity_P #max(rad)/max(min(rad), 1.0)


def get_all_max_ellip_rad(x_voids):
    p = Pool(processes = 36)
    OutList = p.map(max_ellip_rad, x_voids)
    p.close()
    p.join()
    return np.asarray(OutList)



if __name__ == "__main__":
    
    sys.path.append('/data01/AMPolyCalc/code')
    from rw_utils import read_raw_data_1X
    from params import pixel_size_1X as pixel_size
    from void_mapping import void_map_gpu
    
    projs, theta, center, dark, flat = read_raw_data_1X("1", "1")
    b = 4
    voids_4 = void_map_gpu(projs, theta, center, dark, flat, b, pixel_size)
    start = time.time()
    L = get_all_max_ellip_rad(voids_4["x_voids"])
    end = time.time()
    print(end-start)

    


