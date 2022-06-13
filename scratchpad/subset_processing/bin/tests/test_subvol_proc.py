#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
"""


import numpy as np
from tomo2mesh.misc.digital_zoom import process_subset
from tomo2mesh.misc.voxel_processing import TimerGPU
from tomo2mesh import Grid
import pandas as pd
import sys
import os
from tomo2mesh.unet3d.surface_segmenter import SurfaceSegmenter
######## START GPU SETTINGS ############
########## SET MEMORY GROWTH to True ############
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass        
######### END GPU SETTINGS ############

from utils import model_path, get_model_params
model_tag = "M_a07"
model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}
model_params = get_model_params(model_tag)


wd = 32
n = 2048
ntheta = 1500
nz = 32



from utils import time_logs as output_path


def run(projs, theta, center, r_fac, fe):
    
    ntheta, nz, n = projs.shape

    n_sel = int(nz*n*n*r_fac/(32**3))
    timer = TimerGPU("secs")
    # arguments to recon_chunk2: data, theta, center, p3d
    
    times = []
    for ii in range(25):
        p_sel = Grid((nz, n, n), width = wd, n_points = n_sel)
        timer.tic()
        x, p_sel, = process_subset(projs, theta, center, fe, p_sel, (-1.0,1.0), seg_batch = True)
        t_tot = timer.toc("subset processing")
        print(f'r = N(P)/N(V): {len(p_sel)*32**3/(nz*n*n):.2f}')    
        print(f'returned sub_vols shape: {x.shape}')
        columns = ["r_fac", "ntheta", "nz", "n", "t_tot"]
        times.append([r_fac, ntheta, nz, n, t_tot])
    df = pd.DataFrame(data = np.median(times, axis=0).reshape(1,-1), columns = columns)
    return df


if __name__ == "__main__":

    # initialize segmenter fCNN
    fe = SurfaceSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    fe.test_speeds(128,n_reps = 15, input_size = (wd,wd,wd))    

    sparsity = np.logspace(0,2,10)
    r_fac_list = np.sort(1.0/sparsity)
    
    projs = np.random.normal(0,1,(ntheta, nz, n)).astype(np.float32)
    theta = np.linspace(0, np.pi, ntheta, dtype = np.float32)
    center = n/2.0

    dfs = []
    for r_fac in r_fac_list:
        print(f'experiment: n={n}, nz={nz}, ntheta={ntheta}, 1/r={(1.0/r_fac):.2f}')
        dfs.append(run(projs, theta, center, r_fac, fe))
    df_fin = pd.concat(dfs, ignore_index = True)
    df_fin.to_csv(os.path.join(output_path, f'runtime_{model_tag}_n{n}_nz{nz}_ntheta{ntheta}.csv'), index=False)
