import numpy as np
import h5py 
import matplotlib.pyplot as plt
import pandas as pd
import json
import os



def read_raw_data_any(fpath):
    '''
    
    return projs, theta, center, dark, flat
    '''

    #Create file name
    f = h5py.File(fpath, 'r')
    projs = np.asarray(f['exchange/data'][:])
    theta = np.radians(np.asarray(f['exchange/theta'][:]))%(2*np.pi)
    dark = np.mean(f['exchange/data_dark'][:], axis = 0)
    flat = np.mean(f['exchange/data_white'][:], axis = 0)
    f.close()

    return projs, theta, dark, flat


if __name__ == "__main__":

    print("nothing here")