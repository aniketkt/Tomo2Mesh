#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implementation of the patches data structure  


"""

import pandas as pd
import os
import glob
import numpy as np


import h5py
import cupy as cp

from multiprocessing import Pool, cpu_count
import functools
import time

from numpy.random import default_rng
import abc
import tensorflow as tf
from tensorflow.keras.layers import UpSampling3D


class Grid(dict):
    
    def __init__(self, vol_shape, initialize_by = "grid-params", xp = np, **kwargs):
        '''
        A patch is the set of all pixels in a rectangle / cuboid sampled from a (big) image / volume. The Patches data structure allows the following. Think of this as a pandas DataFrame. Each row stores coordinates and features corresponding to a new patch constrained within a big volume of shape vol_shape.  
        
        1. stores coordinates and widths of the patches as arrays of shape (n_pts, z, y, x,) and (n_pts, pz, py, px) respectively.
        2. extracts patches from a big volume and reconstructs a big volume from patches
        3. filters, sorts and selects patches based on a feature

        A Grid class is similar to Patches with the main difference that the width of all patches in a grid is equal, hence width is not stored as an array but a single integer value.
        '''
        self.vol_shape = vol_shape
        self.xp = xp
        initializers = {"data" : self._set_from_data, \
                        "grid-params" : self._set_regular_grid, \
                        "file": self._load_from_disk}

        self["source"] = initialize_by
        self.points, self.wd = initializers[initialize_by](**kwargs)
        self['points'] = self.points
        self['width'] = self.wd
        return

    def __len__(self):
        return len(self.points)
    
    def dump(self, fpath):
        """create df from points"""
        
        with h5py.File(fpath, 'w') as hf:
            hf.create_dataset("vol_shape", data = self.vol_shape)
            hf.create_dataset("points", data = self.points)
            hf.create_dataset("width", data = self.wd)
        return

    def _set_regular_grid(self, width = None, n_points = None):

        '''
        Initialize (n,3) points on the corner of volume patches placed on a grid. No overlap is used. Instead, the volume is cropped such that it is divisible by the patch_size in that dimension.  
        
        Parameters  
        ----------  
        width : int  
            width of a unit patch volume  
        
        '''
        
        _ndim = len(self.vol_shape)
        assert not any([self.vol_shape[i]%width > 0 for i in range(_ndim)]), "vol_shape must be multiple of patch width"

        # Find optimum number of patches to cover full image
        m = list(self.vol_shape)
        p = [width]*len(self.vol_shape)
        
        nsteps = [int(m[i]//p[i]) for i in range(len(m))]
        stepsize = tuple(p)
        
        points = []
        if len(m) == 3:
            for ii in range(nsteps[0]):
                for jj in range(nsteps[1]):
                    for kk in range(nsteps[2]):
                        points.append([ii*stepsize[0], jj*stepsize[1], kk*stepsize[2]])
        elif len(m) == 2:
            for ii in range(nsteps[0]):
                for jj in range(nsteps[1]):
                    points.append([ii*stepsize[0], jj*stepsize[1]])
        
        if n_points is not None:
            n_points = min(n_points, len(points))
            points = self.xp.asarray(points)
            # sample randomly
            rng = default_rng()
            idxs = self.xp.sort(rng.choice(points.shape[0], n_points, replace = False))
            points = points[idxs,...].copy()
        
        return self.xp.asarray(points).astype(np.uint32), int(width)


    # use this when initialize_by = "file"
    def _load_from_disk(self, fpath = None):
        
        with h5py.File(fpath, 'r') as hf:
            self.vol_shape = tuple(self.xp.asarray(hf["vol_shape"]))
            return self.xp.asarray(hf["points"]).astype(np.uint32), \
                   int(self.xp.asarray(hf["width"]))

    def _set_from_data(self, points = None, width = None):

        return points.astype(self.xp.uint32), int(width)
    
    def append(self, more_patches):
        
        '''
        Append the input patches to self in place.  
        
        Parameters  
        ----------  
        more_patches : Patches
            additional rows of patches to be appended.  
            
        Returns
        -------
        None
            Append in place so nothing is returned.  
        '''
        
        if self.vol_shape != more_patches.vol_shape:
            raise ValueError("patches data is not compatible. Ensure that big volume shapes match")
        assert self.wd == more_patches.wd, "this is a Grid data structure. All widths must be equal"    
        self.points = self.xp.concatenate([self.points, more_patches.points], axis = 0)
        
        return

    
    def slices(self):
        '''  
        Get python slice objects from the list of coordinates  
        
        Returns  
        -------  
        self.xp.ndarray (n_pts, 3)    
            each element of the array is a slice object  
        
        '''  
        _ndim = len(self.vol_shape)

        s = [[slice(self.points[ii,jj], self.points[ii,jj] + self.wd) for jj in range(_ndim)] for ii in range(len(self))]
        
        return self.xp.asarray(s)
    
    def centers(self):
        '''  
        Get centers of the patch volumes.    
        
        Returns  
        -------  
        self.xp.ndarray (n_pts, 3)    
            each element of the array is the z, y, x coordinate of the center of the patch volume.    
        
        '''  
        _ndim = len(self.vol_shape)
        s = [[int(self.points[ii,jj] + self.wd//2) for jj in range(_ndim)] for ii in range(len(self.points))]
        return self.xp.asarray(s)
    
    def _is_within_cylindrical_crop(self, mask_ratio, height_ratio):
        
        '''
        returns a boolean array
        '''
        assert self.vol_shape[1] == self.vol_shape[2], "must be tomographic CT volume (ny = nx = n)"
        nz, n = self.vol_shape[:2]
        centers = self.centers()
        radii = self.xp.sqrt(self.xp.power(centers[:,1] - n/2.0, 2) + self.xp.power(centers[:,2] - n/2.0, 2))
        clist1 = radii < mask_ratio*n/2.0
        
        heights = self.xp.abs(centers[:,0] - nz/2.0)
        clist2 = heights < height_ratio*nz/2.0

        cond_list = clist1&clist2
#         print("CALL TO: %s"%self._is_within_cylindrical_crop.__name__)
        return cond_list
        
    def filter_by_cylindrical_mask(self, mask_ratio = 0.9, height_ratio = 1.0):
        '''
        Selects patches whose centers lie inside a cylindrical volume of radius = mask_ratio*nx/2. This assumes that the volume shape is a tomogram where ny = nx. The patches are filtered along the vertical (or z) axis if height_ratio < 1.0.  
        '''
        
        cond_list = self._is_within_cylindrical_crop(mask_ratio, height_ratio)
        return self.filter_by_condition(cond_list)
    
    def filter_by_condition(self, cond_list):
        '''  
        Select coordinates based on condition list. Here we use numpy.compress. The input cond_list can be from a number of classifiers.  
        
        Parameters  
        ----------  
        cond_list : self.xp.ndarray  
            array with shape (n_pts, n_conditions). Selection will be done based on ALL conditions being met for the given patch.  
        '''  
        
        if cond_list.shape[0] != len(self.points):
            raise ValueError("length of condition list must same as the current number of stored points")
        
        if cond_list.ndim == 2:
            cond_list = self.xp.prod(cond_list, axis = 1) # AND operator on all conditions
        elif cond_list.ndim > 2:
            raise ValueError("condition list must have 1 or 2 dimensions like so (n_pts,) or (n_pts, n_conditions)")
            
        return Grid(self.vol_shape, initialize_by = "data", \
                       points = self.xp.compress(cond_list, self.points, axis = 0),\
                       width = self.wd)
    
    def copy(self):
        return Grid(self.vol_shape, initialize_by = "data", \
                       points = self.points.copy(),\
                       width = int(self.wd))

    def select_by_range(self, s_sel):

        '''
        Parameters
        ----------
        s_sel : tuple
            range (start, stop)
        '''
        s_sel = slice(s_sel[0], s_sel[1], None)
        return Grid(self.vol_shape, initialize_by = "data", \
                       points = self.points.copy()[s_sel,...],\
                       width = self.wd)
        
    def pop(self, n_pop):
        
        '''
        Parameters
        ----------
        n_pop : int
            If n_pop is negative, pop from end else pop from beginning
        
        '''
        if n_pop > 0:
            spop = slice(n_pop, None, None)
        elif n_pop < 0:
            spop = slice(None, n_pop, None)
        else:
            return self.copy()
            
        return Grid(self.vol_shape, initialize_by = "data", \
                       points = self.points.copy()[spop,...],\
                       width = self.wd)
        
    def rescale(self, fac):
        '''
        '''
        
        fac = int(fac)
        new_vol_shape = tuple([int(self.vol_shape[i]*fac) for i in range(len(self.vol_shape))])        
        return Grid(new_vol_shape, initialize_by = "data", \
                        points = self.points.copy()*fac,\
                        width = int(self.wd*fac))
    
    def select_by_indices(self, idxs):

        '''
        Select patches corresponding to the input list of indices.  
        Parameters
        ----------
        idxs : list  
            list of integers as indices.  
        '''
        
        return Grid(self.vol_shape, initialize_by = "data", \
                       points = self.points[idxs].copy(),\
                       width = self.wd)
        
    def select_random_sample(self, n_points):
        
        '''
        Select a given number of patches randomly without replacement.  
        
        Parameters
        ----------
        n_points : list  
            list of integers as indices.  
        '''
        rng = default_rng()
        idxs = self.xp.sort(rng.choice(self.points.shape[0], n_points, replace = False))
        return self.select_by_indices(idxs)
    
    def sort_by(self, feature):
        '''  
        Sort patches list in ascending order of the value of a feature.    
        
        Parameters  
        ----------  
        feature : self.xp.ndarray  
            array with shape (n_pts,). If provided separately, ife will be ignored.  
        
        '''  
        assert feature.ndim == 1, "feature must be 1D array"
        assert len(feature) == len(self.points), "length mismatch"
        idxs = self.xp.argsort(feature)
        return self.select_by_indices(idxs)
    
    def extract(self, vol):

        '''  
        Returns a list of volume patches at the active list of coordinates by drawing from the given big volume 'vol'  
        
        Returns
        -------
        self.xp.ndarray  
            shape is (n_pts, wd, wd, wd)  
        
        '''  
        xp = cp.get_array_module(vol)
        assert vol.shape == self.vol_shape, "Shape of big volume does not match vol_shape attribute of patches data"
        
        # make a list of patches
        x = []
        for ii in range(len(self)):
            s = (slice(self.points[ii,0], self.points[ii,0] + self.wd),\
                slice(self.points[ii,1], self.points[ii,1] + self.wd),\
                slice(self.points[ii,2], self.points[ii,2] + self.wd))
            # x[ii,...] = vol[s]
            x.append(vol[s])
        x = xp.array(x)
        return x
    
    def fill_patches_in_volume(self, sub_vols, vol_out):
        
        '''
        fill patches in volume or image (3d or 2d patches)
        '''
        
        s = self.slices()
        for idx in range(len(self)):
                vol_out[tuple(s[idx,...])] = sub_vols[idx]
            
        return
    


        
        
if __name__ == "__main__":
    
    print('just a bunch of functions')
    
