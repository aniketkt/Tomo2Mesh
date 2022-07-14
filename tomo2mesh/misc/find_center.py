#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find centers using cross-correlation. nothing fancy.

"""

import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt



from skimage.feature import match_template
from tomo2mesh.misc.img_stats import Parallelize
from tomo2mesh.projects.eaton.rw_utils_ae import read_raw_data_1X, save_path
from tomo2mesh.projects.eaton.recon import recon_slice, recon_binned
from tomo2mesh.misc import viewer
from tomo2mesh.misc.voxel_processing import cylindrical_mask, modified_autocontrast
import h5py
from cupyx.scipy.ndimage import median_filter
import cupy as cp

plots_dir = '/home/yash/eaton_plots2/'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
import matplotlib as mpl
mpl.use('Agg')

def preprocess(data, dark, flat):

    '''
    Parameters
    ----------
    data : np.ndarray  
        data has shape ntheta, nz, n  
        
    '''
    
    data = (data-dark)/(cp.maximum(flat-dark, 1.0e-6))                
    
    fdata = median_filter(data,[3,3,3])
    ids = cp.where(cp.abs(fdata-data)>0.5*cp.abs(fdata))
    data[ids] = fdata[ids]        
    
    data = -cp.log(cp.maximum(data,1.0e-6))
    
    return data

def read_opposing_pairs(fpath):
    
    # to cupy
    hf = h5py.File(fpath, 'r')
    data = [np.asarray(hf['exchange/data'][0]), np.asarray(hf['exchange/data'][-1])]
    data = cp.array(np.float32(data), dtype = cp.float32)
    flat = cp.array(np.mean(hf['exchange/data_white'][:], axis = 0).astype(np.float32))
    dark = cp.array(np.mean(hf['exchange/data_dark'][:], axis = 0).astype(np.float32))
    hf.close()
    data = preprocess(data, dark, flat)
    return data

def match_opposing(center_guess, projs = None, roi_width = None, metric = 'NCC', roi_height = None):
    
    '''
    translates projection at theta = 0
    '''
    
    center_guess = int(center_guess)
    
    nz, n = projs[0].shape
    sx = slice(center_guess - int(roi_width*n//2), center_guess + int(roi_width*n//2))
    sz = slice(int(roi_height[0]*nz), int(roi_height[1]*nz))
    
    if metric == "NCC":
        match_val = match_template(projs[0, sz, sx], \
                                   np.fliplr(projs[-1, sz, sx]))[0][0]
    elif metric == "MSE":
        match_val = cp.linalg.norm(projs[0, sz , sx] - cp.fliplr(projs[-1, sz, sx]), ord = 2)
    
    return match_val

def estimate_center(projs, search_range, roi_width = 0.8, metric = "NCC", procs = 12, roi_height = (0.0, 1.0)):
    
    center_guesses = np.arange(*search_range).tolist()
    
    if metric == "NCC":
        projs = projs.get()
        match_vals = Parallelize(center_guesses, match_opposing, \
                                 projs = projs, roi_width = roi_width, \
                                 metric = metric, procs = procs, roi_height = roi_height)
        match_vals = np.asarray(match_vals)
        idx_correct = np.argmax(match_vals)
    elif metric == "MSE":
        match_vals = []
        for center_guess in center_guesses:
            match_vals.append(match_opposing(center_guess, projs = projs, roi_width = roi_width, \
                                             metric = metric, roi_height = roi_height))
                                             
        match_vals = cp.array(match_vals).get()
        idx_correct = np.argmin(match_vals)
    
    if idx_correct in [0, len(match_vals)-1]:
        raise ValueError("center finding failed")
    else:
        return center_guesses[idx_correct]
    # print(f"match_vals: {match_vals}")
    

def main(args):

    projs = read_opposing_pairs(args.input_fname)

    if args.center_guess == 0:
        args.center_guess = projs.shape[-1]//2

    print(f"shape of projection images: {projs.shape}")
    # print(args.center_guess)
    # print(args.search_width)
    print(f"Search will be around center values: {args.center_guess-args.search_width}, {args.center_guess+args.search_width}")
    # input("continue?")

    from tomo2mesh.misc.voxel_processing import TimerGPU
    timer = TimerGPU("secs")
    timer.tic()
    search_range = (args.center_guess - args.search_width, args.center_guess + args.search_width)
    center_val = estimate_center(projs, search_range, roi_width = args.roi_width, metric = args.metric, procs = args.procs, \
                                 roi_height = (args.roi_height_start, args.roi_height_end))
    timer.toc("finding center")
    print("%s center = %.2f"%(args.input_fname, center_val))

    #Save center in csv file
    df = pd.read_csv(save_path)
    df.loc[(df["sample_num"] == args.sample_num) & (df["scan_num"] == int(args.scan_num)),"rot_cen"] = center_val
    df.to_csv(save_path,index = False)

    #Save reconstructed slice as a check
    projs, theta, center_guess, dark, flat = read_raw_data_1X(str(args.sample_num), str(args.scan_num))
    V = recon_binned(projs, theta, center_val, dark, flat, 4, 3.13).get()
    h = modified_autocontrast(V)
    V = np.clip(V, *h)
    cylindrical_mask(V, 1.0, mask_val = V.min())
    imgs = viewer.get_orthoplanes(vol = V)
    fig, ax = plt.subplots(1,1, figsize = (12,12))
    ax.imshow(imgs[0], cmap = 'gray')
    plt.savefig(plots_dir + f'check_recon_sample{args.sample_num}_scan{args.scan_num}.png', format='png')
    plt.close()
    
    return


if __name__ == "__main__":

   
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--input-fname", required = True, type = str, help = "Path to tiff folder or hdf5 file")
    parser.add_argument('-i', "--stats_only", required = False, action = "store_true", default = False, help = "show stats only")
    parser.add_argument('-v', "--verbosity", required = False, type = int, default = 0, help = "read / write verbosity; 0 - silent, 1 - important stuff, 2 - print everything")
    parser.add_argument('-g', "--center_guess", required = False, type = int, default = 0, help = "initial guess is horizontal center of projection if not provided")
    parser.add_argument('--search_width', required = False, type = int, default = 50, help = "search width around guessed center, e.g. default width of 50 and guess of image center will search on range 550 - 650")
    parser.add_argument('--roi_width', required = False, type = float, default = 0.8, help = "fraction of horizontal roi to use for cross correlation match")
    parser.add_argument('-roi_height_start', required = False, type = float, default = 0.0, help = "fraction of vertical roi start val to use for cross correlation match")
    parser.add_argument('-roi_height_end', required = False, type = float, default = 1.0, help = "fraction of vertical roi end val to use for cross correlation match")
    parser.add_argument('--metric', required = False, type = str, default = "NCC", help = "metric can be MSE or NCC. NCC is preferrred")
    parser.add_argument('--procs', required = False, type = int, default = 12, help = "number of processes to spawn for multiprocessing")
    parser.add_argument('--sample_num', required = True, type = str, help = "Sample number") #
    parser.add_argument('--scan_num', required = True, type = str, help = "Scan number") #

    args = parser.parse_args()

    if args.stats_only:
        print("not implemented")
    else:
        main(args)
 