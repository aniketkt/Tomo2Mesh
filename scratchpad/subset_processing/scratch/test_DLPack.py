#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import numpy as np
import tensorflow as tf
import cupy as cp

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)


# convert a TF tensor to a cupy array
with tf.device('/GPU:0'):
    a = tf.random.uniform((10,))



print(a)
print(a.device)
cap = tf.experimental.dlpack.to_dlpack(a)
b = cp.from_dlpack(cap)
b *= 3
print(b)
print(a)
# convert a cupy array to a TF tensor
a = cp.arange(10)
cap = a.toDlpack()
b = tf.experimental.dlpack.from_dlpack(cap)
b.device
print(b)
print(a)


