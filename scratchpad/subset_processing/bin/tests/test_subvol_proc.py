#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
"""


import numpy as np
from tomo_encoders.tasks.digital_zoom import process_patches
from tomo_encoders import Grid
import pandas as pd
import sys
import os
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
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
sys.path.append('/home/atekawade/TomoEncoders/scratchpad/voids_paper/configs')
from params import model_path, get_model_params
model_tag = "M_a07"
model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}
model_params = get_model_params(model_tag)
wd = 32

output_path = '/data02/MyArchive/aisteer_3Dencoders/voids_paper_data'


def run(ntheta, nc, n, r_fac, fe):
    n_sel = int(nc*n*n*r_fac/(32**3))
    # arguments to recon_chunk2: data, theta, center, p3d
    projs = np.random.normal(0,1,(ntheta, nc, n)).astype(np.float32)
    theta = np.linspace(0, np.pi, ntheta, dtype = np.float32)
    center = n/2.0
    
    times = []
    for ii in range(3):
        p_sel = Grid((nc, n, n), width = wd, n_points = n_sel)
        x, p_sel, t_rec, t_seg = process_patches(projs, theta, center, fe, p_sel, (-1.0,1.0), TIMEIT = True)
        print(f'r = N(P)/N(V): {len(p_sel)*32**3/(nc*n*n):.2f}')    
        print(f'returned sub_vols shape: {x.shape}')
        columns = ["r_fac", "ntheta", "nz", "n", "t_rec", "t_seg"]
        times.append([r_fac, ntheta, nc, n, t_rec, t_seg])
    df = pd.DataFrame(data = np.median(times, axis=0).reshape(1,-1), columns = columns)
    return df


if __name__ == "__main__":

    # initialize segmenter fCNN
    fe = SurfaceSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    fe.test_speeds(128,n_reps = 5, input_size = (wd,wd,wd))    

    n = 4096
    ntheta = 1500
    nc = 1024
    sparsity = np.logspace(0,2,10)
    r_fac_list = np.sort(1.0/sparsity)
    
    dfs = []
    for r_fac in r_fac_list:
        print(f'experiment: n={n}, nz={nc}, ntheta={ntheta}, 1/r={(1.0/r_fac):.2f}')
        dfs.append(run(ntheta, nc, n, r_fac, fe))
    pd.concat(dfs, ignore_index = True).to_csv(os.path.join(output_path, 'output_recseg_%s_%i_%i.csv'%(model_tag,n,nc)), index=False)

    


    pass