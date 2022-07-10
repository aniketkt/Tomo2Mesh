#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
import time
import seaborn as sns
import pandas as pd

import cupy as cp

read_path = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic_raw'
raw_fname = os.path.join(read_path, 'all_layers.hdf5')
raw_fname_16bit = os.path.join(read_path, 'all_layers_16bit.hdf5')

id_start = np.asarray([130,219,222,225,223,225])
id_end = np.asarray([1023,1026,1027,1025,1027,1027])
# id_start = np.asarray([130,219,221,225,223,225])
# id_end = np.asarray([878,879,879,878,877,1027])
hts = id_end - id_start
ntheta = 3000

if __name__ == "__main__":

    
    tot_ht = int(np.sum(hts))
    FULL_SHAPE = (ntheta, tot_ht, 4200)
    print(f'shape of full projection array {FULL_SHAPE}')
    center = 2100.0
    theta = np.linspace(0,np.pi,ntheta,dtype='float32')

    hf = h5py.File(raw_fname, 'w')
    hf.create_dataset("data",FULL_SHAPE)
    hf.create_dataset("theta", data = theta)
    hf.create_dataset("center", data = center)
    hf.close()


    
    for ii in range(0,6):
        
        if ii == 0:
            sw = slice(0, hts[0])        
        else:
            sw = slice(sw.stop, sw.stop + hts[ii])
        
        hf = h5py.File(os.path.join(read_path,f'layer{ii+1}.hdf5'), 'r')
        projs = np.asarray(hf["data"][:,id_start[ii]:id_end[ii],:])
        hf.close() 
        print(f'shape of projection data from layer {ii+1}: {projs.shape}')    
        hf_out = h5py.File(raw_fname, 'a')
        print(f'position in stitched projection array {sw.start}, {sw.stop}')
        hf_out["data"][:,sw,:] = projs
        hf_out.close()


    # normalize to 16-bit
    print("normalization to 16bit: estimating data range (min, max)")
    hf = h5py.File(raw_fname, 'r')
    binned_projs = np.asarray(hf["data"][::4,::4,::4])
    [ntheta, nz, n] = hf["data"].shape
    theta = np.asarray(hf["theta"][:])
    min_val = binned_projs.min()
    max_val = binned_projs.max()
    print("\tdone")
    
    
    hf_16bit = h5py.File(raw_fname_16bit,'w')
    hf_16bit.create_dataset("data",(ntheta,nz,n), dtype = np.uint16)
    hf_16bit.create_dataset("theta", data = theta)
    hf_16bit.create_dataset("center", data = center)


    from tqdm import trange
    for ii in trange(ntheta):
        proj = np.asarray(hf["data"][ii,...]).astype(np.float32)
        proj = (proj - min_val)/(max_val - min_val)
        hf_16bit["data"][ii,...] = ((2**16-1)*proj).astype(np.uint16)

    hf.close()
    hf_16bit.close()    
        

    
    
    
    

    
    
    