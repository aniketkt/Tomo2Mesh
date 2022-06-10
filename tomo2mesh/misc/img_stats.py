#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate image statistics (2D or 3D). Combination of ct_segnet.stats and ct_segnet.measurements.  
    1. signal-to-noise ratio (SNR) for binarizable datasets.  
    2. accuracy metrics for segmentation maps.  

"""

import numpy as np
from multiprocessing import cpu_count
import matplotlib as mpl
import matplotlib.pyplot as plt
import functools
from multiprocessing import Pool, cpu_count
from scipy.ndimage import label, find_objects

def calc_SNR(img, seg_img, labels = (0,1), mask_ratio = None):
    """
    SNR =  1     /  s*sqrt(std0^^2 + std1^^2)  
    where s = 1 / (mu1 - mu0)  
    mu1, std1 and mu0, std0 are the mean / std values for each of the segmented regions respectively (pix value = 1) and (pix value = 0).  
    seg_img is used as mask to determine stats in each region.  

    Parameters
    ----------
    img : np.array  
        raw input image (2D or 3D)  
    
    seg_img : np.array  
        segmentation map (2D or 3D)  
        
    labels : tuple  
        an ordered list of two label values in the image. The high value is interpreted as the signal and low value is the background.  
        
    mask_ratio : float or None
        If not None, a float in (0,1). The data are cropped such that the voxels / pixels outside the circular mask are ignored.  

    Returns
    -------
    float
        SNR of img w.r.t seg_img  

    """
    eps = 1.0e-12
    # handle circular mask  
    if mask_ratio is not None:
        crop_val = int(img.shape[-1]*0.5*(1 - mask_ratio/np.sqrt(2)))
        crop_slice = slice(crop_val, -crop_val)    

        if img.ndim == 2: # 2D image
            img = img[crop_slice, crop_slice]
            seg_img = seg_img[crop_slice, crop_slice]
        elif img.ndim == 3: # 3D image
            vcrop = int(img.shape[0]*(1-mask_ratio))
            vcrop_slice = slice(vcrop, -vcrop)
            img = img[vcrop_slice, crop_slice, crop_slice]
            seg_img = seg_img[vcrop_slice, crop_slice, crop_slice]
            
    pix_1 = img[seg_img == labels[1]]
    pix_0 = img[seg_img == labels[0]]
    
    if np.any(pix_1) and np.any(pix_0):
        mu1 = np.mean(pix_1)
        mu0 = np.mean(pix_0)
        s = abs(1/(mu1 - mu0 + eps))
        std1 = np.std(pix_1)
        std0 = np.std(pix_0)
        std = np.sqrt(0.5*(std1**2 + std0**2))
        std = s*std
        return 1/(std + eps)
    else:
        return 1/(np.std(img) + eps)

def calc_jac_acc(true_img, seg_img):
    """
    Parameters
    ----------
    true_img : np.array
            ground truth segmentation map (ny, nx)
            
    seg_img : np.array
            predicted segmentation map (ny, nx)
     
    Returns
    -------
    float
        Jaccard accuracy or Intersection over Union  
    """
    seg_img = np.round(np.copy(seg_img))
    
    jac_acc = (np.sum(seg_img*(true_img == 1)) + 1) / (np.sum(seg_img) + np.sum((true_img == 1)) - np.sum(seg_img*(true_img == 1)) + 1)
    return jac_acc

def calc_dice_coeff(true_img, seg_img):
    """
    Parameters
    ----------
    true_img : np.array
            ground truth segmentation map (ny, nx)
            
    seg_img : np.array
            predicted segmentation map (ny, nx)
     
    Returns
    -------
    float
        Dice coefficient  

    """
    seg_img = np.round(np.copy(seg_img))
    
    dice = (2*np.sum(seg_img*(true_img == 1)) + 1) / (np.sum(seg_img) + np.sum((true_img == 1)) + 1)
    return dice

def fidelity(true_imgs, seg_imgs, tolerance = 0.95):
    """
    Fidelity is number of images with IoU > tolerance  
    
    Parameters
    ----------
    tolerance : float
                tolerance (default  = 0.95)
                
    true_imgs : numpy.array
                list of ground truth segmentation maps (nimgs, ny, nx)
                
    seg_imgs  : numpy.array
                list of predicted segmentation maps (nimgs, ny, nx)
     
    Returns
    -------
    float
        Fidelity  
    """

    XY = [(true_imgs[ii], seg_imgs[ii]) for ii in range(true_imgs.shape[0])]
    del true_imgs
    del seg_imgs
    
    jac_acc = np.asarray(Parallelize(XY, calc_jac_acc, procs = cpu_count()))
    
    mean_IoU = np.mean(jac_acc)

    jac_fid = np.zeros_like(jac_acc)
    jac_fid[jac_acc > tolerance] = 1
    jac_fid = np.sum(jac_fid).astype(np.float32) / np.size(jac_acc)
    
    
    return jac_fid, mean_IoU, jac_acc


def pore_analysis(seg_img, features = ["fraction", "number", "size"], invert = False):
    
    '''
    Calculates void fraction, number of pores and mean size of pores. Pore size is simply defined by sqrt of sum of pixels in the pore phase.  
    
    Returns  
    -------  
    np.array  
        vector of measurements  
    
    Parameters   
    ----------  
    seg_img : np.array  
        segmented image (output from Segmenter class)  
        
    true_img : np.array  
        true segmentation map (same shape as seg_img)  
        
    features : list  
        list of porosity features to measure from "fraction", "number" and "size"    
        
    invert : bool  
        if the image contains particles instead of pores, invert = True  
        
    '''
    
    if any(f in features for f in ["number", "size"]):
        img_labeled, n_objs = label(seg_img if invert else seg_img^1)
        obj_list = find_objects(img_labeled)
    
    values = []
    for feature in features:
        if feature == "fraction":
            values.append(1 - np.sum(seg_img)/np.size(seg_img))
        elif feature == "number":
            values.append(n_objs)
        elif feature == "size":
            
            p_size = []
            for idx in range(n_objs):

                sub_img = img_labeled[obj_list[idx]]
                p_area = np.sum(sub_img==(idx+1))                
                p_dia_px = np.sqrt(p_area)#*2*3/(4*np.pi)
                p_size.append(p_dia_px)

            values.append(np.mean(p_size))
        
    return np.asarray(values)

def Parallelize(ListIn, f, procs = -1, **kwargs):
    
    """This function packages the "starmap" function in multiprocessing, to allow multiple iterable inputs for the parallelized function.  
    
    Parameters
    ----------
    ListIn: list
        each item in the list is a tuple of non-keyworded arguments for f.  
    f : func
        function to be parallelized. Signature must not contain any other non-keyworded arguments other than those passed as iterables.  
    
    Example:  
    
    .. highlight:: python  
    .. code-block:: python  
    
        def multiply(x, y, factor = 1.0):
            return factor*x*y
    
        X = np.linspace(0,1,1000)  
        Y = np.linspace(1,2,1000)  
        XY = [ (x, Y[i]) for i, x in enumerate(X)] # List of tuples  
        Z = Parallelize_MultiIn(XY, multiply, factor = 3.0, procs = 8)  
    
    Create as many positional arguments as required, but all must be packed into a list of tuples.
    
    """
    if type(ListIn[0]) != tuple:
        ListIn = [(ListIn[i],) for i in range(len(ListIn))]
    
    reduced_argfunc = functools.partial(f, **kwargs)
    
    if procs == -1:
        opt_procs = int(np.interp(len(ListIn), [1,100,500,1000,3000,5000,10000] ,[1,2,4,8,12,36,48]))
        procs = min(opt_procs, cpu_count())

    if procs == 1:
        OutList = [reduced_argfunc(*ListIn[iS]) for iS in range(len(ListIn))]
    else:
        p = Pool(processes = procs)
        OutList = p.starmap(reduced_argfunc, ListIn)
        p.close()
        p.join()
    
    return OutList




