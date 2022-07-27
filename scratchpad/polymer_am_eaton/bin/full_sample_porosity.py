from cmath import nan
from tomo2mesh.structures.voids import VoidLayers
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys
import os

from tomo2mesh.projects.eaton.rw_utils_ae import read_raw_data_1X, save_path
from tomo2mesh.projects.eaton.void_mapping import void_map_gpu, void_map_all
from tomo2mesh.projects.eaton.params import pixel_size_1X as pixel_size
plots_dir = '/home/yash/eaton_plots/'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
import matplotlib as mpl
mpl.use('Agg')
CUTOFF_CRACKS = 4.0


def merge_void_layers(sample_tag, b, raw_pixel_size, dust_thresh, number_density_radius = 50, local_otsu_psize = 384):

    rdf = pd.read_csv(save_path)
    info = rdf[(rdf["sample_num"] == str(sample_tag))]
    scan_num = list(info["scan_num"])
    
    y_pos_list = list(info['y_pos'])
    y_pos_list = np.asarray(y_pos_list)*1.0e3/(raw_pixel_size*b)
    y_pos_list = (y_pos_list - y_pos_list[0]).astype(np.uint64)
    z_max = np.uint32(np.diff(y_pos_list))

    voids_all = VoidLayers()
    porosity_z_all = []
    porosity_all = []
    for ii, scan_tag in enumerate(range(scan_num[0], scan_num[-1]+1)): 
        projs, theta, center, dark, flat = read_raw_data_1X(sample_tag, scan_tag)
        
        if ii==0:
            z_crop = (1536//2,1536)
        elif scan_tag == scan_num[-1]:
            z_crop = (0,1536//2)
        else:
            z_crop = (0,1536)
        cp._default_memory_pool.free_all_blocks(); cp.fft.config.get_plan_cache().clear()  
        
        voids = void_map_all(projs, theta, center, dark, flat, b, raw_pixel_size, dust_thresh, z_crop, local_otsu_psize = local_otsu_psize)

        if scan_tag != scan_num[-1]:
            voids.select_by_z_coordinate(z_max[ii]) #Find voids in each layer (w/o overlap)
            porosity_z_all.append(voids["porosity_z"][0:z_max[ii]])
        else:
            porosity_z_all.append(voids["porosity_z"])
        porosity_all.append(voids["porosity"])            
        voids_all.add_layer(voids,y_pos_list[ii])
        print("added a layer")


    porosity_z_all = np.concatenate(porosity_z_all, axis = 0)
    voids_all["porosity_z"] = porosity_z_all
    voids_all["porosity"] = porosity_all

    if b != 1:
        voids_all.calc_max_feret_dm()
        voids_all.calc_number_density(number_density_radius)

    return voids_all


if __name__ == "__main__":

    b = 1
    dust_thresh = 2
    number_density_radius = 50 # assumes b = 4?
    sample_name = str(sys.argv[1])
    
    from tomo2mesh.misc.voxel_processing import TimerCPU
    timer_full = TimerCPU("secs")
    timer_full.tic()
    voids_all = merge_void_layers(sample_name, b, pixel_size, dust_thresh)
    timer_full.toc("FULL SAMPLE VOID MAPPING")

    #Save porosity values for all layers in all samples
    fpath = f"/data01/Eaton_Polymer_AM/csv_files/porosity_sample_{sample_name}.csv"
    info = {"layer": np.arange(len(voids_all["porosity"])), "porosity": voids_all["porosity"]}
    df2 = pd.DataFrame(info)
    df2.to_csv(fpath, mode = 'w', index = False)

    fpath = f"/data01/Eaton_Polymer_AM/csv_files/local_porosity_sample_{sample_name}.csv"
    info = {"porosity_z": voids_all["porosity_z"]}
    df2 = pd.DataFrame(info)
    df2.to_csv(fpath)


    # voids_all.select_by_feret_dia_norm(3.0)
    # voids.select_by_size(100.0, pixel_size_um = pixel_size) # remove later
    # voids_all.export_void_mesh_with_texture("number_density").write_ply(f'/data01/Eaton_Polymer_AM/ply_files/sample_{sample_name}.ply')
    # voids_all.export_void_mesh_mproc("max_feret", edge_thresh=1.0).write_ply(f'/data01/Eaton_Polymer_AM/ply_files/sample_{sample_name}_maxferet.ply')
    # voids_all.export_void_mesh_mproc("number_density", edge_thresh=1.0).write_ply(f'/data01/Eaton_Polymer_AM/ply_files/sample_{sample_name}_numdensity.ply')
    # voids_all.write_to_disk(f'/data01/Eaton_Polymer_AM/voids_data/sample_{sample_name}_all_layers_dust{dust_thresh}_b{b}')

    exit()

