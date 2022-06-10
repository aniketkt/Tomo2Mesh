#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import os
import sys

data_path = '/data02/MyArchive/tomo_datasets/AM_part_Xuan' #ensure this path matches where your data is located.
dataset_names = ['mli_L206_HT_650_L3', 'AM316_L205_fs_tomo_L5']


########## CROPPING AND BINNING ##########        
#### to-do: apply this as filter on patches ####
# load vols here and quick look

def _get_scrops(test_binning):
    
    dict_scrops = {'mli_L206_HT_650_L3' : (slice(100,-100, test_binning), \
                                        slice(None,None, test_binning), \
                                        slice(None,None, test_binning)), \
                'AM316_L205_fs_tomo_L5' : (slice(50,-50, test_binning), \
                                           slice(None,None, test_binning), \
                                           slice(None,None, test_binning))}
    return dict_scrops

def get_datasets(names, test_binning = 1):
    '''
    create datasets input for train method
    '''
    
    datasets = {}
    dict_scrops = _get_scrops(test_binning)
    
    for filename in names:
        ct_fpath = os.path.join(data_path, 'data', \
                                filename + '_rec_1x1_uint16_tiff')
        seg_fpath = os.path.join(data_path, 'seg_data', \
                                 filename, filename + '_GT')

        datasets.update({filename : {'fpath_X' : ct_fpath, \
                                     'fpath_Y' : seg_fpath, \
                                     'data_tag_X' : 'data', \
                                     'data_tag_Y' : 'SEG', \
                                     's_crops' : dict_scrops[filename]}})
    return datasets

if __name__ == "__main__":

    print("nothing here")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
