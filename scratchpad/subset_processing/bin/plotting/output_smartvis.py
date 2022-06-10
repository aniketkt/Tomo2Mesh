#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 


import numpy as np
from tomo_encoders import Voids
import matplotlib.pyplot as plt
import open3d as o3d
from tomo_encoders.misc.voxel_processing import _edge_map
from tomo_encoders.misc import viewer
from tomo_encoders import DataFile
from scipy.ndimage import label, find_objects
import os
save_path = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/void_comparison'
from shutil import rmtree
if os.path.exists:
    rmtree(save_path)
os.makedirs(save_path)    
voids_hr_path = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/voids_highres'
voids_lr_path = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/voids_lowres'
import matplotlib as mpl
mpl.use('Agg')


def plot_stuff(voids_lr, voids_hr, Vx, Vx_bin, ax, ii):
    imp = viewer.get_orthoplanes(vol = voids_lr["x_voids"][ii])
    imx = viewer.get_orthoplanes(vol = Vx_bin[voids_lr["s_voids"][ii]])
    for i3 in range(3):
        viewer.edge_plot(imx[i3], imp[i3], ax[0,i3], color = [255,0,0])

    imp = viewer.get_orthoplanes(vol = voids_hr["x_voids"][ii])
    imx = viewer.get_orthoplanes(vol = Vx[voids_hr["s_voids"][ii]])
    for i3 in range(3):
        viewer.edge_plot(imx[i3], imp[i3], ax[1,i3], color = [255,0,0])        
    
    ax[0,0].set_title("lowres voidnum %i"%ii, fontsize = 12)
    ax[1,0].set_title("highres voidnum %i"%ii, fontsize = 12)    
    return

if __name__ == "__main__":

    Vx = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/test_x_rec/', tiff = True).read_full()
    Vx_bin = Vx[::4,::4,::4].copy()    

    voids_hr = Voids().import_from_disk(voids_hr_path)
    voids_lr = Voids().import_from_disk(voids_lr_path)
    assert len(voids_hr) == len(voids_lr)
    idxs = np.argsort(voids_hr["sizes"])[::-1]
    voids_hr.select_by_indices(idxs)
    voids_lr.select_by_indices(idxs)    

    for ii in range(len(voids_lr)):
        fig, ax = plt.subplots(2,3, figsize = (12,6))
        plot_stuff(voids_lr, voids_hr, Vx, Vx_bin, ax, ii)
        plt.savefig(os.path.join(save_path, "voidnum_%03d"%ii))   
        plt.close() 
        print(r'ii = %i'%ii)

