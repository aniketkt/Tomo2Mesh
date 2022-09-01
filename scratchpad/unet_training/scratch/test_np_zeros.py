#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import numpy as np
from tomo2mesh.misc.voxel_processing import TimerCPU

if __name__ == "__main__":

  timer = TimerCPU("ms")
  
  timer.tic()
  a = np.ascontiguousarray(np.zeros((4096,4096,4096), dtype = np.uint8))
  # a[12,15,17] = 0
  # print(a.sum())
  timer.toc("numpy zeros")
