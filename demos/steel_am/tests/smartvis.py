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
import tensorflow

from tomo2mesh.projects.steel_am.coarse2fine import coarse_map, process_subset
from tomo2mesh.misc.voxel_processing import TimerGPU, TimerCPU
from tomo2mesh.structures.voids import Voids
from tomo2mesh.projects.steel_am.rw_utils import *
from tomo2mesh.porosity.params_3dunet import *
from tomo2mesh.unet3d.surface_segmenter import SurfaceSegmenter
from copy import deepcopy
import gc


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

# SELECTION CRITERIA
criteria_list = ["all_geq_14um", "cylindrical_neighborhood", "spherical_neighborhood"]
columns = ["criteria", "sparsity (${1/r}$)", "${t_{mapping}}$",	"${t_{mesh}}$", "n_voids"]
# n_iters = 1
n_iters = 5

if __name__ == "__main__":
    
    rows = []
    timer = TimerGPU("secs")
    timer_cpu = TimerCPU("secs")
    
    # load the U-net model
    model_params = get_model_params(model_tag)
    segmenter = SurfaceSegmenter(model_initialization = 'load-model', \
                            model_names = model_names, \
                            model_path = model_path)    

    vol_name = str(sys.argv[1])
    # read data and initialize output arrays
    print("BEGIN: Read projection data from disk")

    if vol_name == "2k":
        b, b_K = 2, 2
        pixel_size = 2.34
        projs, theta, center = read_raw_data_b2()
        csv_path = os.path.join(time_logs, "smartvis_times_2k.csv")
    elif vol_name == "4k":
        b, b_K = 4, 4
        pixel_size = 1.17
        projs, theta, center = read_raw_data_b1()
        csv_path = os.path.join(time_logs, "smartvis_times_4k.csv")
    else:
        raise NotImplementedError("unacceptable value")
    
    print(f"EXPERIMENT with b, b_K {(b,b_K)}")
    print(f"SHAPE OF PROJECTIONS DATA: {projs.shape}")
    for criteria in criteria_list:
        
        for i_iter in range(n_iters):
            
            # MAPPING
            timer.tic()
            cp._default_memory_pool.free_all_blocks(); cp.fft.config.get_plan_cache().clear()
            voids_b = coarse_map(projs, theta, center, b, b_K, 0) 
            # voids selection criteria goes here
            void_id = np.argmax(voids_b["sizes"])
            if criteria == "cylindrical_neighborhood":
                print("select cylindrical neighborhood")
                z_min = voids_b["cents"][void_id,0] - 800.0/(b*pixel_size)
                z_max = voids_b["cents"][void_id,0] + 800.0/(b*pixel_size)
                voids_b.select_z_slab(z_min, z_max)
            elif criteria == "spherical_neighborhood":
                print("select spherical neighborhood")
                voids_b.select_around_void(void_id, 800.0, pixel_size_um=pixel_size)
            elif criteria == "all_geq_14um":
                voids_b_coarse_14um = deepcopy(voids_b)
                voids_b.select_by_size(14, pixel_size, sel_type="geq")

            # export subset coordinates here
            p_voids, r_fac = voids_b.export_grid(wd)    
            cp._default_memory_pool.free_all_blocks(); cp.fft.config.get_plan_cache().clear()        
            # process subset reconstruction
            x_voids, p_voids = process_subset(projs, theta, center, segmenter, p_voids, voids_b["rec_min_max"])
            
            # import voids data from subset reconstruction
            voids = Voids().import_from_grid(voids_b, x_voids, p_voids)

            # select voids with size greater than 3 pixels across
            idxs = np.where(voids["sizes"] > 3**3)[0]
            voids.select_by_indices(idxs)
            voids_b.select_by_indices(idxs)


            t_mapping = timer.toc(f"CRITERIA: {criteria}")

            # MESHING        
            timer_cpu.tic()
            nprocs = 36
            surf = voids.export_void_mesh_mproc("sizes", edge_thresh = 1.0, preserve_feature = False, nprocs = nprocs, pool_context="spawn" if vol_name == "4k" else "fork")        
            t_mesh = timer_cpu.toc()

            
            rows.append([criteria, 1/r_fac, t_mapping, t_mesh, len(voids)])
            print("#"*55 + "\n")
            print(pd.DataFrame(columns = columns, data = rows))
            print("#"*55 + "\n")
            _ = gc.collect()
        
        print("saving voids data now...")
        timer.tic()
        voids.write_size_data(os.path.join(voids_dir,f"sizes_{vol_name}_b_{b}_{criteria}"))
        if n_iters > 1:
            surf.write_ply(os.path.join(ply_dir, f"voids_{vol_name}_{b}_subset_{criteria}.ply"))
            print(f"\nCRITERIA: {criteria}; t_mapping: {t_mapping:.2f} secs; t_mesh: {t_mesh:.2f} secs; 1/r value: {1/r_fac:.4g}\n")
            timer.toc("saving data")
            df = pd.DataFrame(columns = columns, data = rows)
            df.to_csv(csv_path, index=False)

    if n_iters > 1:
        surf_b = voids_b_coarse_14um.export_void_mesh_mproc("sizes", edge_thresh = 0.0, nprocs = 6, pool_context="spawn" if vol_name == "4k" else "fork")
        surf_b.write_ply(os.path.join(ply_dir, f"voids_{vol_name}_b_{b}_coarse.ply"))
        print(os.path.join(ply_dir, f"voids_{vol_name}_b_{b}_coarse.ply"))
        for criteria in criteria_list:
            print(os.path.join(ply_dir, f"voids_{vol_name}_{b}_subset_{criteria}.ply"))
        print("Done\n")


    
    

        
    
    
