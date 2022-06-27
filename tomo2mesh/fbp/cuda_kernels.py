#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""
import numpy as np
import cupy as cp
import time
import tensorflow as tf
# from cupyx.scipy.fft import rfft, irfft, rfftfreq


source = """
extern "C" {
    void __global__ rec_pts(float *f, float *g, float *theta, int *pts, float center, int ntheta, int nz, int n, int npts)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tx >= npts)
            return;
        int s0 = 0;
        int ind = 0;
        float f0 = 0;
        float sp = 0;
        
        for (int k = 0; k < ntheta; k++)
        {
            sp = (pts[3*tx+2] - n / 2) * __cosf(theta[k]) - (pts[3*tx+1] - n / 2) * __sinf(theta[k]) + center; //polar coordinate
            //linear interpolation
            s0 = roundf(sp);
            ind = k * n * nz + pts[3*tx+0] * n + s0;
            if ((s0 >= 0) & (s0 < n - 1))
                f0 += g[ind] + (g[ind+1] - g[ind]) * (sp - s0) / n;
        }
        f[tx] = f0;
    }



    void __global__ rec_pts_xy(float *f, float *g, float *theta, int *pts, float center, int ntheta, int nz, int n, int npts)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        if (tx >= npts || ty >= nz)
            return;
        int s0 = 0;
        int ind = 0;
        float f0 = 0;
        float sp = 0;
        
        for (int k = 0; k < ntheta; k++)
        {
            sp = (pts[2*tx+1] - n / 2) * __cosf(theta[k]) - (pts[2*tx] - n / 2) * __sinf(theta[k]) + center; //polar coordinate
            //linear interpolation
            s0 = roundf(sp);
            ind = k * n * nz + ty * n + s0;
            if ((s0 >= 0) & (s0 < n - 1))
                f0 += g[ind] + (g[ind+1] - g[ind]) * (sp - s0) / n;
        }
        f[ty*npts+tx] = f0;
    }




    void __global__ rec(float *f, float *g, float *theta, float center, int ntheta, int nz, int n, int stx, int px, int sty, int py, int stz, int pz)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= px || ty >= py || tz>=pz)
            return;
        stx += tx;
        sty += ty;
        stz += tz;
        int s0 = 0;
        int ind = 0;
        float f0 = 0;
        float sp = 0;
        
        for (int k = 0; k < ntheta; k++)
        {
            sp = (stx - n / 2) * __cosf(theta[k]) - (sty - n / 2) * __sinf(theta[k]) + center; //polar coordinate
            //linear interpolation
            s0 = roundf(sp);
            ind = k * n * nz + stz * n + s0;
            if ((s0 >= 0) & (s0 < n - 1))
                f0 += g[ind] + (g[ind+1] - g[ind]) * (sp - s0) / n;
        }
        f[tz*px*py+ty*px+tx] = f0;
    }

    void __global__ rec_mask(float *f, float *g, float *theta, float center, int ntheta, int nz, int n)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz>=nz)
            return;
        int s0 = 0;
        int ind = 0;
        float f0 = 0;
        float sp = 0;
        
        if (f[tz*n*n+ty*n+tx] > 0)
            for (int k = 0; k < ntheta; k++)
            {
                sp = (tx - n / 2) * __cosf(theta[k]) - (ty - n / 2) * __sinf(theta[k]) + center; //polar coordinate
                //linear interpolation
                s0 = roundf(sp);
                ind = k * n * nz + tz * n + s0;
                //ind = tz * n * ntheta + k * n + s0;
                if ((s0 >= 0) & (s0 < n - 1))
                    f0 += g[ind] + (g[ind+1] - g[ind]) * (sp - s0) / n;
            }
            f[tz*n*n+ty*n+tx] = f0;
    }

    void __global__ rec_all(float *f, float *g, float *theta, float center, int ntheta, int nz, int n)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz>=nz)
            return;
        int s0 = 0;
        int ind = 0;
        float f0 = 0;
        float sp = 0;
        
        for (int k = 0; k < ntheta; k++)
        {
            sp = (tx - n / 2) * __cosf(theta[k]) - (ty - n / 2) * __sinf(theta[k]) + center; //polar coordinate
            //linear interpolation
            s0 = roundf(sp);
            ind = k * n * nz + tz * n + s0;
            if ((s0 >= 0) & (s0 < n - 1))
                f0 += g[ind] + (g[ind+1] - g[ind]) * (sp - s0) / n;
        }
        f[tz*n*n+ty*n+tx] = f0;
    }


}

"""
    
    
module = cp.RawModule(code=source)
rec_kernel = module.get_function('rec')
rec_pts_kernel = module.get_function('rec_pts')
rec_pts_xy_kernel = module.get_function('rec_pts_xy')
rec_mask_kernel = module.get_function('rec_mask')
rec_all_kernel = module.get_function('rec_all')

def rec_pts(data, theta, center, pts):
    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    stream_rec = cp.cuda.Stream()
    
    with stream_rec:
        [ntheta, nz, n] = data.shape
        obj = cp.zeros(len(pts),dtype='float32', order='C')
        data = cp.ascontiguousarray(data)
        theta = cp.ascontiguousarray(theta)     
        pts = cp.ascontiguousarray(pts)     

        rec_pts_kernel((int(np.ceil(len(pts)/1024)),1), (1024,1), \
                   (obj, data, theta, pts, cp.float32(center), ntheta, nz, n, len(pts)))
        stream_rec.synchronize()

    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    print("TIME rec_pts: %.2f ms"%t_gpu)
        
    return obj
    
def rec_pts_xy(data, theta, center, pts):
    
    '''
    pts is a sorted array of points y,x with shape (npts,2)
    
    '''
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    stream_rec = cp.cuda.Stream()
    
    with stream_rec:
        [ntheta, nz, n] = data.shape
        obj = cp.zeros((len(pts)*nz),dtype='float32', order='C')
        data = cp.ascontiguousarray(data)
        theta = cp.ascontiguousarray(theta)     
        pts = cp.ascontiguousarray(pts)     

        nbkx = 256
        nthx = int(np.ceil(len(pts)/nbkx))
        nbkz = 4
        nthz = int(np.ceil(nz/nbkz))
        rec_pts_xy_kernel((nthx,nthz), (nbkx,nbkz), \
                   (obj, data, theta, pts, cp.float32(center), ntheta, nz, n, len(pts)))
        stream_rec.synchronize()

    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    print("TIME rec_pts: %.2f ms"%t_gpu)
        
    return obj
    
def rec_mask(obj, data, theta, center):
    """Reconstruct mask on GPU"""
    [ntheta, nz, n] = data.shape
    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    stream_rec = cp.cuda.Stream()
    with stream_rec:
        
        data = cp.ascontiguousarray(data)
        theta = cp.ascontiguousarray(theta)
        
        rec_mask_kernel((int(cp.ceil(n/16)), int(cp.ceil(n/16)), \
                    int(cp.ceil(nz/4))), (16, 16, 4), \
                   (obj, data, theta, cp.float32(center),\
                    ntheta, nz, n))
        stream_rec.synchronize()
    
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    # print("TIME rec_mask: %.2f ms"%t_gpu)
    return t_gpu
    

def rec_all(obj, data, theta, center):
    """Reconstruct all data array on GPU"""
    [ntheta, nz, n] = data.shape
    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    stream_rec = cp.cuda.Stream()
    with stream_rec:
        
        data = cp.ascontiguousarray(data)
        theta = cp.ascontiguousarray(theta)
        
        rec_all_kernel((int(cp.ceil(n/16)), int(cp.ceil(n/16)), \
                    int(cp.ceil(nz/4))), (16, 16, 4), \
                   (obj, data, theta, cp.float32(center),\
                    ntheta, nz, n))
    stream_rec.synchronize()
    
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    # print("TIME rec_mask: %.2f ms"%t_gpu)
    return t_gpu




def rec_patch(data, theta, center, stx, px, sty, py, stz, pz, TIMEIT = False):
    """Reconstruct subvolume [stz:stz+pz,sty:sty+py,stx:stx+px] on GPU"""
    [ntheta, nz, n] = data.shape
    
    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    stream_rec = cp.cuda.Stream()
    with stream_rec:
        
        obj = cp.zeros([pz, py, px], dtype='float32', order = 'C')
        data = cp.ascontiguousarray(data)
        theta = cp.ascontiguousarray(theta)
        
        rec_kernel((int(cp.ceil(px/16)), int(cp.ceil(py/16)), \
                    int(cp.ceil(pz/4))), (16, 16, 4), \
                   (obj, data, theta, cp.float32(center),\
                    ntheta, nz, n, stx, px, sty, py, stz, pz))
        stream_rec.synchronize()
    
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    
    if TIMEIT:
#         print("TIME rec_patch: %.2f ms"%t_gpu)
        return obj, t_gpu
    else:
        return obj








if __name__ == "__main__":
    
    print('just a bunch of functions')

    
