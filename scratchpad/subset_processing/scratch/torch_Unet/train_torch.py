#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862

""" 
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
from tomo_encoders import Patches, DataFile
import time
import torch



if __name__ == "__main__":

    # ds1 = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/train_x', tiff = True)
    # ds2 = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/train_x_rec', tiff = True)
    # Xs = []
    # Xs.append(ds1.read_full())
    # Xs.append(ds2.read_full())
    # Ys = [DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/train_y', tiff = True).read_full()]

    from model import UNet
    model = UNet(in_channels=1,
                out_channels=1,
                n_blocks=3,
                start_filters=8,
                activation='relu',
                normalization='batch',
                conv_mode='same',
                dim=3)

    nb = 256
    wd = 32
    x = torch.randn(size=(nb, 1, wd, wd, wd), dtype=torch.float32)
    
    with torch.no_grad():
        for ii in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            out = model(x)
            end.record()
            torch.cuda.synchronize()
            t_vox = start.elapsed_time(end)/(nb*wd**3)
            print(f'vox time: {t_vox*1.0e6:.2f} ns')


    print(f'Out: {out.shape}')
    from torchinfo import summary
    summary(model, input_size = (256,1,32,32,32))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
