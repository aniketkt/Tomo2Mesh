#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import numpy as np 
import cupy as cp 
from tomo2mesh.unet3d.surface_segmenter import SurfaceSegmenter
import os 

from tomo2mesh.projects.subset_processing.params import *
from tomo2mesh.misc.voxel_processing import TimerGPU
#### THIS EXPERIMENT ####
wd = 32
nb = 64
model_tag = sys.argv[1] #if len(sys.argv) > 4 else "M_a01"
######### DEFINE EXPERIMENT ON 'nb'
def infer(segmenter):

#     Possible slowdown of first iteration due to tensorflow Dataset creation?
#     https://github.com/tensorflow/tensorflow/issues/46950

    print("#"*55,"\n")
    timer = TimerGPU("ms")
    counter = []
    for jj in range(10):
        x = np.random.uniform(0, 1, (nb,wd,wd,wd,1)).astype(np.float32)
        x = cp.array(x, dtype = cp.float32)
        timer.tic()
        x[:] = cp.clip(x, 0.0, 1.0)
        x[:] = (x - 0.0) / (1.0)
        cap = x.toDlpack()
        x_in = tf.experimental.dlpack.from_dlpack(cap)
        yp_cpu = np.round(segmenter.models["segmenter"](x_in)).astype(np.uint8)
        
        t_tot = timer.toc()
        t_vox = t_tot/(nb*wd**3)*1.0e6
        print(f"inf. time per voxel {t_vox:.2f} ns")
        print("\n")
        counter.append(t_vox)

    print(f"MEASURED INFERENCE TIME PER VOXEL: {np.median(counter):.2f} ns")

        
    
if __name__ == "__main__":

    print("EXPERIMENT WITH MODEL %s"%model_tag)
    model_params = get_model_params(model_tag)
    fe = SurfaceSegmenter(model_initialization = 'define-new', \
                         descriptor_tag = model_tag,\
                         **model_params) 
    print(f"EXPERIMENT WITH INPUT_SIZE = {wd}, BATCH_SIZE = {nb}")

    infer(fe)
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
