from tomo2mesh.fbp.cuda_kernels import rec_all
from tomo2mesh.fbp.retrieve_phase import paganin_filter
from tomo2mesh.fbp.prep import fbp_filter
from cupyx.scipy import ndimage
import cupy as cp
import numpy as np
from tomo2mesh.projects.eaton.params import *
import numexpr as ne


def preprocess(data, dark, flat, pixel_size_cm, apply_pg = True, outlier_removal = False):
    
    data = (data-dark)/(cp.maximum(flat-dark, 1.0e-6))                
    
    if outlier_removal:
        fdata = ndimage.median_filter(data,[3,3,3])
        ids = cp.where(cp.abs(fdata-data)>0.5*cp.abs(fdata))
        data[ids] = fdata[ids]        
    
    if apply_pg:
        data[:] = paganin_filter(data, alpha = pg_alpha, energy = energy, pixel_size = pixel_size_cm, dist = detector_dist)

    data[:] = -cp.log(cp.maximum(data,1.0e-6))
    
    return data

def recon_slice(projs, theta, center, dark, flat, sino_pos, pixel_size, pg_pad = 10):

    ntheta, nz, n = projs.shape
    
    if type(sino_pos) is float:
        sino_pos = int(sino_pos*nz)
    elif type(sino_pos) is int:
        print(sino_pos)
    else:
        raise ValueError("")
    sino_slice = slice(sino_pos-pg_pad, sino_pos + 1 + pg_pad)
    
    theta = cp.array(theta, dtype = cp.float32)
    center = cp.float32(center)
    dark = cp.array(dark)
    flat = cp.array(flat)
    obj_gpu = cp.empty((1+2*pg_pad,n,n), dtype = cp.float32)

    stream = cp.cuda.Stream()
    
    with stream:
        data = cp.array(projs[:,sino_slice,:].astype(np.float32), dtype = cp.float32)
        dark = cp.array(dark[sino_slice,:].astype(np.float32), dtype = cp.float32)
        flat = cp.array(flat[sino_slice,:].astype(np.float32), dtype = cp.float32)

        print(f"data shape: (ntheta, nz, n) {data.shape}")
        # dark-flat correction, phase retrieval
        data[:] = preprocess(data, dark, flat, pixel_size*1.0e-4, apply_pg=True if pg_pad > 0 else False)
        fbp_filter(data)
        rec_all(obj_gpu, data, theta, center)

    stream.synchronize()
    
    return obj_gpu.get()[pg_pad,...]

import tqdm
def recon_all(projs, theta, center, nc, dark, flat, pixel_size, outlier_removal = False, pg_pad = 8):

    stream = cp.cuda.Stream()
    with stream:
        ntheta, nz, n = projs.shape
        data = cp.empty((ntheta, nc, n), dtype = cp.float32)
        theta = cp.array(theta, dtype = cp.float32)
        center = cp.float32(center)
        dark = cp.array(dark.astype(np.float32), dtype = cp.float32)
        flat = cp.array(flat.astype(np.float32), dtype = cp.float32)

        obj_gpu = cp.empty((nc, n, n), dtype = cp.float32)
        obj_out = np.empty((nz, n, n), dtype = np.float32)
    stream.synchronize()

    for ic in tqdm.trange(int(np.ceil(nz/nc))):
        
        if ic == 0:
            s_in = slice(ic*nc, (ic+1)*nc+pg_pad)
            s_out = slice(None, -pg_pad)
        elif ic == int(np.ceil(nz/nc))-1:
            s_in = slice(ic*nc-pg_pad, (ic+1)*nc)
            s_out = slice(pg_pad, None)
        else:
            s_in = slice(ic*nc-pg_pad, (ic+1)*nc+pg_pad)
            s_out = slice(pg_pad,-pg_pad)
        s_chunk = slice(ic*nc, (ic+1)*nc)

        # dark-flat correction, phase retrieval
        stream = cp.cuda.Stream()
        with stream:
            
            data[:] = preprocess(cp.array(projs[:,s_in,:].astype(np.float32), dtype = cp.float32), \
                                 dark[s_in], flat[s_in], pixel_size*1.0e-4, outlier_removal = outlier_removal)[:,s_out,:]
            fbp_filter(data)
            rec_all(obj_gpu, data, theta, center)
            obj_gpu[:] = ndimage.gaussian_filter(obj_gpu, 0.5)
            # obj_gpu[:] = ndimage.median_filter(obj_gpu,[1,1,1])
                
            obj_out[s_chunk] = obj_gpu.get()
        stream.synchronize()

    del obj_gpu, data, theta, center    
    cp._default_memory_pool.free_all_blocks()    
    
    return obj_out





def bin_projections(data, b):
    """Downsample data"""
    binning = int(np.log2(b))
    for j in range(binning):
        x = data[:, :, ::2]
        y = data[:, :, 1::2]
        data = ne.evaluate('0.5*(x + y)')  # should use multithreading
        
    for k in range(binning):
        x = data[:, ::2]
        y = data[:, 1::2]
        data = ne.evaluate('0.5*(x + y)')
    return data.astype(np.uint16)


def recon_binned(projs, theta, center, dark, flat, b, pixel_size):
    
    # projs = bin_projections(projs, b)
    # nz, n = dark.shape
    # dark = np.mean(dark.reshape(nz//b, b, n//b, b), axis = (1,3))
    # flat = np.mean(flat.reshape(nz//b, b, n//b, b), axis = (1,3))
    # center = center/float(b)
    # ntheta, nz, n = projs.shape
    
    cp.fft.config.clear_plan_cache()
    stream = cp.cuda.Stream()
    
    with stream:

        ntheta, nz, n = projs.shape
        data = cp.array(projs, dtype = cp.uint8) # send to gpu
        data = cp.mean(data.reshape(ntheta,nz//b,b,n//b,b), axis = (2,4), dtype = cp.uint8) # binning
        data = cp.array(data, dtype = cp.float32) # change dtype to float32
        dark = cp.array(dark.astype(np.float32), dtype = cp.float32) # send dark to gpu
        flat = cp.array(flat.astype(np.float32), dtype = cp.float32) # send flat to gpu
        dark = cp.mean(dark.reshape(nz//b, b, n//b, b), axis = (1,3)) # binning dark
        flat = cp.mean(flat.reshape(nz//b, b, n//b, b), axis = (1,3)) # binning flat
        theta = cp.array(theta, dtype = cp.float32)
        center = cp.float32(center/float(b))

        # dark-flat correction, phase retrieval
        
        data[:] = preprocess(data, dark, flat, pixel_size*1.0e-4*b, outlier_removal=False)
        fbp_filter(data)
        ntheta, nz, n = data.shape
        obj_gpu = cp.empty((nz,n,n), dtype = cp.float32)
        rec_all(obj_gpu, data, theta, center)

    # obj_gpu[:] = ndimage.gaussian_filter(obj_gpu,0.3)
    stream.synchronize()
    return obj_gpu

