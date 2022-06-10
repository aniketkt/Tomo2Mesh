#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import numpy as np
import cupy as cp


batch_size = 1024
wd = 32
input_shape = (batch_size,wd,wd,wd)
a = np.random.normal(0,1,input_shape).astype(np.float32)
b = cp.random.normal(0,1,input_shape).astype(cp.float32)
print(f"cpu array shape: {a.shape}")
print(f"gpu array shape: {b.shape}")
stream = cp.cuda.Stream()
print("CPU to GPU")
for i in range(5):
  with stream:
    st = cp.cuda.Event(); end = cp.cuda.Event(); st.record()                
    _ = cp.array(a)
    end.record(); end.synchronize()
    print(f'time {cp.cuda.get_elapsed_time(st,end)/(np.prod(input_shape))*1e6:.4f} ns')
    stream.synchronize()

print("GPU to CPU")
for i in range(5):
  with stream:
    st = cp.cuda.Event(); end = cp.cuda.Event(); st.record()                
    _ = b.get()
    end.record(); end.synchronize()
    print(f'time {cp.cuda.get_elapsed_time(st,end)/(np.prod(input_shape))*1e6:.4f} ns')
    stream.synchronize()