#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import h5py
import numpy as np
import sys
import os


# coarse mapping parameters
b = 4
b_K = 4
wd = 32

base_fpath = '/data02/MyArchive/tomo_datasets/data_porosity_mapping_paper'

# detailed mapping parameters
pixel_res = 1.17
size_um = -1 # um
void_rank = 1
radius_around_void_um = 1000.0 # um


# U-net parameters
model_tag = "M_a07"
model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}


## file names
raw_fname = os.path.join(base_fpath, 'mosaic_raw/all_layers_16bit.hdf5')
raw_fname_b2 = os.path.join(base_fpath, 'mosaic_raw/all_layers_b2_16bit.hdf5')
voids_dir = os.path.join(base_fpath,'voids_data')
data_output = os.path.join(base_fpath, 'output_vis')
time_logs = os.path.join(base_fpath,'time_logs')
ply_dir = os.path.join(base_fpath, 'ply')
rec_dir = os.path.join(base_fpath, 'full_rec')
model_path = os.path.join(base_fpath, 'models_unet')

if not os.path.exists(time_logs):
    os.makedirs(time_logs)
if not os.path.exists(ply_dir):
    os.makedirs(ply_dir)
if not os.path.exists(voids_dir):
    os.makedirs(voids_dir)


# def read_raw_data(fname, multiplier):
#     hf = h5py.File(fname, 'r')
#     crop_wdb = int(hf["data"].shape[-1]%(multiplier))
#     if crop_wdb:
#         sw = slice(None,-crop_wdb)
#     else:
#         sw = slice(None,None)
#     crop_z_wdb = int(hf["data"].shape[1]%(multiplier))
#     # if crop_z_wdb:
#     #     sz = slice(None,-crop_z_wdb)
#     # else:
#     #     sz = slice(None,None)
#     sz = slice(-4096,None)
#     projs = np.asarray(hf["data"][:, sz, sw])    
#     theta = np.asarray(hf["theta"][:])
#     center = float(np.asarray(hf["center"]))
#     hf.close()

#     return projs, theta, center


def read_raw_data_b1(dtype = np.uint16):
    hf = h5py.File(raw_fname, 'r')
    sz = slice(-4096,None)
    sw = slice(0,4096)
    projs = np.asarray(hf["data"][:, sz, sw]).astype(dtype)
    # projs = np.ascontiguousarray(projs.swapaxes(0,1)) 
    projs = projs.swapaxes(0,1)
    theta = np.asarray(hf["theta"][:])
    center = float(np.asarray(hf["center"]))
    hf.close()
    return projs, theta, center


def read_raw_data_b2():
    hf = h5py.File(raw_fname_b2, 'r')
    projs = np.asarray(hf["data"][:]).astype(np.float32)   
    projs = np.ascontiguousarray(projs.swapaxes(0,1)) 
    print(projs.shape)
    theta = np.asarray(hf["theta"][:])
    center = float(np.asarray(hf["center"]))
    hf.close()

    return projs, theta, center



if __name__ == "__main__":

    pass    
    
