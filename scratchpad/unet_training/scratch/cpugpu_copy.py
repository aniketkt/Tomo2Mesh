#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import numpy as np
import cupy as cp


from tomo2mesh.projects.subset_processing.utils import *
from tomo2mesh.misc.voxel_processing import TimerGPU


if __name__ == "__main__":

    timer_transfer = TimerGPU("ms")
    # projs, theta, center = read_raw_data_b1()
    # projs = projs[:1024,:,:].astype(np.float32)

    projs, theta, center = read_raw_data_b2()

    ntheta, nz, n = projs.shape

    a = cp.empty((ntheta, 32, n), dtype = cp.float32)
    stream1 = cp.cuda.Stream(non_blocking=True)
    for ii in range(10):
        timer_transfer.tic()
        with stream1:
            a.set(projs[:,5:32+5,...])
        stream1.synchronize()
        timer_transfer.toc("transfer time along axis 1")

    timer_transfer.tic()
    projs = np.ascontiguousarray(projs.swapaxes(0,1))
    timer_transfer.toc("swapping contiguous array")
    
    
    a = cp.empty((32, ntheta, n), dtype = cp.float32)
    stream1 = cp.cuda.Stream(non_blocking=True)
    for ii in range(10):
        timer_transfer.tic()
        with stream1:
            a.set(projs[5:32+5,...])
        stream1.synchronize()
        timer_transfer.toc("transfer time along axis 0")


    # case 2: approx 39 ms
    # b = np.ascontiguousarray(np.float32(np.random.normal(0, 1, (nz, ntheta, n))))
    # print(b.shape)
    # a = cp.empty((nc, ntheta, n), dtype = cp.float32)
    
    # timer_transfer = TimerGPU()
    # stream1 = cp.cuda.Stream(non_blocking=True)
    # for ii in range(10):
    #     timer_transfer.tic()
    #     with stream1:
    #         a.set(b[5:32+5])
    #     stream1.synchronize()
    #     timer_transfer.toc("transfer time")

    # case 3: approx 117 ms
    # b = np.ascontiguousarray(np.random.normal(0, 1, (nz, ntheta, n)).astype(np.float32))
    # b = b.swapaxes(0,1)
    # print(b.shape)
    # a = cp.ascontiguousarray(cp.empty((ntheta, nc, n), dtype = cp.float32))
    
    # timer_transfer = TimerGPU()
    # stream1 = cp.cuda.Stream(non_blocking=True)
    # for ii in range(10):
    #     timer_transfer.tic()
    #     with stream1:
    #         a.set(b[:,5:32+5])
            
    #     stream1.synchronize()
    #     timer_transfer.toc("transfer time")    
    
    # case 4: approx 117 ms
    # b = np.ascontiguousarray(np.random.normal(0, 1, (ntheta, nz, n)).astype(np.float32))
    # b = b.swapaxes(0,1) # back to nz, ntheta, n
    # print(b.shape)
    # a = cp.ascontiguousarray(cp.empty((nc, ntheta, n), dtype = cp.float32))
    
    # timer_transfer = TimerGPU()
    # stream1 = cp.cuda.Stream(non_blocking=True)
    # for ii in range(10):
    #     timer_transfer.tic()
    #     with stream1:
    #         a.set(b[5:32+5])
            
    #     stream1.synchronize()
    #     timer_transfer.toc("transfer time")    
    

