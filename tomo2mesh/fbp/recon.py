from tomo2mesh.fbp.cuda_kernels import rec_all
from tomo2mesh.fbp.retrieve_phase import paganin_filter
from tomo2mesh.fbp.prep import fbp_filter
from cupyx.scipy import ndimage
import cupy as cp
import numpy as np
from tomo2mesh.projects.eaton.params import *
import numexpr as ne
import tqdm

# typical for 2-BM pink beam
# pg_dict_default = {"alpha" : 0.00005, "energy" : 30.0, "detector_dist_cm" : 15.0, "pixel_size_um" : 3.13, "pg_pad" : 10}

def preprocess(data, dark, flat, outlier_removal = False, pg_dict = None):
    
    data = (data-dark)/(cp.maximum(flat-dark, 1.0e-6))                
    
    if outlier_removal:
        fdata = ndimage.median_filter(data,[3,3,3])
        ids = cp.where(cp.abs(fdata-data)>0.5*cp.abs(fdata))
        data[ids] = fdata[ids]        
    
    if pg_dict is not None:
        data[:] = paganin_filter(data, alpha = pg_dict["alpha"], \
                                 energy = pg_dict["energy"], \
                                 pixel_size = pg_dict["pixel_size_um"]*1.0e-4,\
                                 dist = detector_dist)

    data[:] = -cp.log(cp.maximum(data,1.0e-6))
    return data

def recon_slice(projs, theta, center, dark, flat, sino_pos, pg_dict = None):

    ntheta, nz, n = projs.shape
    pg_pad = 0 if pg_dict is None else pg_dict["pg_pad"]
    
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
        data[:] = preprocess(data, dark, flat, pg_dict = pg_dict)
        fbp_filter(data)
        rec_all(obj_gpu, data, theta, center)

    stream.synchronize()
    
    return obj_gpu.get()[pg_pad,...]


def recon_all(projs, theta, center, nc, dark, flat, outlier_removal = False, pg_dict = None):


    pg_pad = 0 if pg_dict is None else pg_dict["pg_pad"]
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
            s_out = slice(None, -pg_pad or None)
        elif ic == int(np.ceil(nz/nc))-1:
            s_in = slice(ic*nc-pg_pad, (ic+1)*nc)
            s_out = slice(pg_pad or None, None)
        else:
            s_in = slice(ic*nc-pg_pad, (ic+1)*nc+pg_pad)
            s_out = slice(pg_pad or None,-pg_pad or None)
        s_chunk = slice(ic*nc, (ic+1)*nc)

        # dark-flat correction, phase retrieval
        stream = cp.cuda.Stream()
        with stream:
            
            data[:] = preprocess(cp.array(projs[:,s_in,:].astype(np.float32), dtype = cp.float32), \
                                 dark[s_in], flat[s_in], outlier_removal = outlier_removal, pg_dict = pg_dict)[:,s_out,:]
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


def recon_binned(projs, theta, center, dark, flat, b, pg_dict = None):
    

    assert projs.dtype == np.uint8, "projection data must be 8-bit for this to work"
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
        
        data[:] = preprocess(data, dark, flat, outlier_removal=False, pg_dict = pg_dict)
        fbp_filter(data)
        ntheta, nz, n = data.shape
        obj_gpu = cp.empty((nz,n,n), dtype = cp.float32)
        rec_all(obj_gpu, data, theta, center)

    # obj_gpu[:] = ndimage.gaussian_filter(obj_gpu,0.3)
    stream.synchronize()
    return obj_gpu

