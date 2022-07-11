from operator import mod
from tomo2mesh.misc.voxel_processing import TimerGPU, edge_map, modified_autocontrast, get_values_cyl_mask, cylindrical_mask
from tomo2mesh.projects.eaton.recon import recon_binned, recon_all
from tomo2mesh.projects.eaton.params import pixel_size_1X as pixel_size
from tomo2mesh.projects.eaton.rw_utils_ae import read_raw_data_1X, save_path
from tomo2mesh.unet3d.surface_segmenter import SurfaceSegmenter
from tomo2mesh.structures.grid import Grid
from tomo2mesh.structures.patches import Patches
from tomo2mesh.projects.eaton.void_mapping import void_map_gpu, void_map_all
from tomo2mesh.projects.eaton.params import pixel_size_1X as pixel_size

import sys
import os
import pandas as pd
import cc3d

import cupy as cp
import numpy as np
from tomo2mesh.structures.voids import Voids
from skimage.filters import threshold_otsu
from skimage.filters import threshold_local
from cupyx.scipy import ndimage
from scipy import ndimage as ndimage_cpu
from scipy import ndimage as ndimage_cpu
import matplotlib.pyplot as plt

plots_dir = '/home/yash/eaton_plots2/'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
import matplotlib as mpl
mpl.use('Agg')


b = 1
raw_pixel_size = pixel_size
df1 = pd.read_csv(save_path)
sample_tag = df1["sample_num"]
scan_tag =  df1["scan_num"]

porosity = []
sample = []
scan = []
for j in range(len(sample_tag)):
    projs, theta, center, dark, flat = read_raw_data_1X(str(sample_tag[j]), str(scan_tag[j]))
    t_gpu = TimerGPU("secs")
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)

    ntheta, nz, n = projs.shape
    projs = np.mean(projs.reshape(ntheta,nz//b,b,n//b,b), axis = (2,4))
    projs = np.array(projs, dtype = np.float32)
    dark = np.mean(dark.reshape(nz//b, b, n//b, b), axis = (1,3))
    flat = np.mean(flat.reshape(nz//b, b, n//b, b), axis = (1,3))
    dark = np.array(dark.astype(np.float32), dtype = np.float32)
    flat = np.array(flat.astype(np.float32), dtype = np.float32)
    theta = np.array(theta, dtype = np.float32)
    center = np.float32(center/float(b))

    #FBP
    t_gpu.tic()
    V = recon_all(projs, theta, center, 32, dark, flat, pixel_size) 
    V_rec = V
    
    #Segmentation
    p_size = 144
    dust_thresh = 4
    patches = Patches(V_rec.shape, initialize_by = "regular-grid", patch_size = (p_size,p_size,p_size))
    x_vols = patches.extract(V_rec, (p_size,p_size,p_size))

    thresh_list = []
    for i in range(len(x_vols)):
        thresh_list.append((x_vols[i]<threshold_otsu(x_vols[i][::2,::2,::2])).astype(np.uint8))


    V_seg = np.empty(V_rec.shape, dtype = np.uint8)
    patches.fill_patches_in_volume(thresh_list, V_seg)
    cylindrical_mask(V_seg, 1, mask_val = 1)
    V_seg = cc3d.connected_components(V_seg)

    #porosity.append((np.sum(V_seg)-(n**2)*(1-np.pi/4)*nz-count)/(n*n*nz*np.pi/4))

    porosity.append([np.sum(i)/(n*n*nz*np.pi/4) for i in V_seg if np.all(np.clip(i.shape-np.array((dust_thresh,dust_thresh,dust_thresh)), 0, None))==True][0])
    sample.append(sample_tag[j])
    scan.append(scan_tag[j])
    
    info = {"sample_tag": sample, "scan_tag": scan, "porosity": porosity}
    df2 = pd.DataFrame(info)
    fpath = "/data01/Eaton_Polymer_AM/csv_files/porosity_vals_adj.csv"
    df2.to_csv(fpath)
    
    cp._default_memory_pool.free_all_blocks(); cp.fft.config.get_plan_cache().clear()
    

samp1 = np.mean([porosity[i] for i in sample_tag if i==1])
samp2 = np.mean([porosity[i] for i in sample_tag if i==2])
samp3 = np.mean([porosity[i] for i in sample_tag if i==3])
samp4 = np.mean([porosity[i] for i in sample_tag if i==4])
samp5 = np.mean([porosity[i] for i in sample_tag if i==5])
samp6 = np.mean([porosity[i] for i in sample_tag if i==6])
samp7 = np.mean([porosity[i] for i in sample_tag if i==7])
samp8 = np.mean([porosity[i] for i in sample_tag if i==8])
samp9 = np.mean([porosity[i] for i in sample_tag if i==9])
samp10 = np.mean([porosity[i] for i in sample_tag if i==10])
samp11 = np.mean([porosity[i] for i in sample_tag if i==11])
samp12 = np.mean([porosity[i] for i in sample_tag if i==12])

list_porosity = np.array([samp1, samp2, samp3, samp4, samp5, samp6, samp7, samp8, samp9, samp10, samp11, samp12])
samples = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

fig, ax = plt.subplots(1,1, figsize = (16,8))
ax.bar(samples, list_porosity)
ax.set_title("Porosity of Each Sample")
ax.set_ylabel("Porosity")
ax.set_xlabel("Sample Number")
#ax.set(xlim=(20, 1130))
plt.savefig(plots_dir + f'porosity_bar_graph.png', format='png')
plt.close()
