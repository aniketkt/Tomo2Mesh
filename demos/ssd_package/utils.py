#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

from tomo2mesh.structures.voids import Voids
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage
from scipy.ndimage import find_objects
from tomo2mesh.structures.grid import Grid

class Bumps(Voids):
    def __init__(self):
        super().__init__()

    def import_from_grid(self, voids_b, x_grid, p_grid, kernel_size = 3):

        '''import voids data from grid data
        Parameters
        ----------
        voids_b : Voids  
            voids data structure. may be binned with corresponding binning value stored as voids_b.b
        x_grid : np.asarray  
            list of cubic sub_volumes, all of same width  
        p_grid : Grid  
            instance of grid data structure  

        '''

        Vp = np.zeros(p_grid.vol_shape, dtype = np.uint8)
        p_grid.fill_patches_in_volume(x_grid, Vp)

        self["sizes"] = []
        self["cents"] = []
        self["cpts"] = []
        self["s_voids"] = []
        self["x_voids"] = []
        self.vol_shape = p_grid.vol_shape
        self.b = 1
        b = voids_b.b

        from tqdm import tqdm
        pbar = tqdm(total=len(voids_b))
        for iv, s_b in enumerate(voids_b["s_voids"]):
            s = tuple([slice(s_b[i].start*b, s_b[i].stop*b) for i in range(3)])
            void = Vp[s]

            cent = [int((s[i].start + s[i].stop)//2) for i in range(3)]
            self["s_voids"].append(s)
            self["cents"].append(cent)
            self["cpts"].append([int((s[i].start)) for i in range(3)])

            # make sure no other voids fall inside the bounding box
            
            k = kernel_size
            void = cp.array(void, dtype = cp.uint8)           
            # void = ndimage.median_filter(void,3)
            void = ndimage.binary_opening(void,structure = cp.ones((k,k,k), dtype = cp.uint8))
            void, n_objs = ndimage.label(void, structure = cp.ones((3,3,3), dtype = cp.uint8))
            void = void.get()
            
            objs = find_objects(void)
            
            counts = [np.sum(void[objs[i]] == i+1) for i in range(n_objs)]
            if len(counts) > 1:
                i_main = np.argmax(counts)    
                void = (void == (i_main+1)).astype(np.uint8)            
            else:
                void = (void > 0).astype(np.uint8)

            self["sizes"].append(np.sum(void))
            self["x_voids"].append(void)
            pbar.update(1)
        pbar.close()

        self["sizes"] = np.asarray(self["sizes"])
        self["cents"] = np.asarray(self["cents"])
        self["cpts"] = np.asarray(self["cpts"])

        # copy over any other keys
        for key in voids_b.keys():
            if key in self.keys():
                continue
            else:
                self[key] = voids_b[key]
                
        return self

    
    def export_grid(self, wd):
        
        wd = wd//self.b
        V_bin = np.zeros(self.vol_shape, dtype = np.uint8)
        for ii, s_void in enumerate(self["s_voids"]):
            V_bin[s_void] = 1
            # V_bin[s_void] += self["x_voids"][ii]
        
        # find patches on surface
        p3d = Grid(V_bin.shape, width = wd)
        x = p3d.extract(V_bin)
        is_sel = (np.sum(x, axis = (1,2,3)) > 0)

        p3d_sel = p3d.filter_by_condition(is_sel)
        if self.b > 1:
            p3d_sel = p3d_sel.rescale(self.b)
        r_fac = len(p3d_sel)*(p3d_sel.wd**3)/np.prod(p3d_sel.vol_shape)
        print(f"\tSTAT: 1/r value: {1/r_fac:.4g}")
        
        return p3d_sel, r_fac


if __name__ == "__main__":


    print("nothing here")
