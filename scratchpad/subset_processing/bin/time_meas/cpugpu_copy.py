#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import numpy as np
import cupy as cp

class TimerGPU():

    def __init__(self):
        pass

    def tic(self):
        self.start = cp.cuda.Event()
        self.end = cp.cuda.Event()
        self.start.record()
        return
    
    def toc(self, msg = "execution"):
        self.end.record()
        self.end.synchronize()
        t_elapsed = cp.cuda.get_elapsed_time(self.start, self.end)

        if msg != "execution":
            print(f"\tTIME: {msg} {t_elapsed:.2f} ms")
        return t_elapsed

n = 2048
ntheta = 1500
nc = 32
nz = 128

def pinned_array(array):
    # first constructing pinned memory
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(
                mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src

if __name__ == "__main__":

    # next step: try this
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryPointer.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryPointer.html#cupy.cuda.MemoryPointer.copy_from_host_async

    # case 1: approx 117 ms
    b = np.float32(np.random.normal(0, 1, (ntheta, nz, n)))
    print(b.shape)
    a = cp.empty((ntheta, nc, n), dtype = cp.float32)
    
    # timer_transfer = TimerGPU()
    # stream1 = cp.cuda.Stream(non_blocking=True)
    # for ii in range(10):
    #     timer_transfer.tic()
    #     with stream1:
    #         a.set(b[:,5:32+5])
    #     stream1.synchronize()
    #     timer_transfer.toc("transfer time")


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
    

