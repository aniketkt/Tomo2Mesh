import numpy as np
import h5py 
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

from tomo2mesh.projects.eaton.params import save_path, data_path, voids_path
# rdf = pd.read_csv(save_path)

def get_filename(sample_tag, scan_tag, csv_path = save_path):
    '''
    The function reads individual projection data in the form of h5 files given the sample number, layer number (or field of view),
    and the path where csv data file is. Produces the h5 file name . 
    
    Input: 
    sample_tag: Single string value of the sample number
    layer_tag: Single string value of the layer number
    csv_path = Saved path location of the csv file (contains information on each sample)
    
    Output:
    fname = file name
    
    return fname
    '''

    fname = "PE6_"+str(sample_tag)+"_"+"0"*((len(str(scan_tag))-3)*-1)+str(scan_tag)+".h5"
    
    return fname


def read_raw_data_1X(sample_tag, scan_tag, csv_path = save_path):
    '''
    The function reads individual projection data in the form of h5 files given the sample number, layer number (or field of view),
    and the path where csv data file is. Produces the projection data, dark-field data, flat-field data, rotation center
    value, and theta values. 
    
    Input: 
    sample_tag: Single string value of the sample number
    layer_tag: Single string value of the layer number
    csv_path = Saved path location of the csv file (contains information on each sample)
    
    Output:
    projs = Projection data
    theta = Theta data
    center = Rotational center data
    dark = Dark-field data
    flat = Flat-field data
    
    return projs, theta, center, dark, flat
    '''

    #Read data from csv file
    rdf = pd.read_csv(csv_path)
    info = rdf[(rdf["sample_num"] == str(sample_tag)) & (rdf["scan_num"] == int(scan_tag))].iloc[0]

    rot_cen = info["rot_cen"]

    #Create file name
    fname = "PE6_"+str(sample_tag)+"_"+"0"*((len(str(scan_tag))-3)*-1)+str(scan_tag)+".h5"
    f = h5py.File(os.path.join(data_path, fname), 'r')
    projs = np.asarray(f['exchange/data'][:])
    theta = np.radians(np.asarray(f['exchange/theta'][:]))%(2*np.pi)
    dark = np.mean(f['exchange/data_dark'][:], axis = 0)
    flat = np.mean(f['exchange/data_white'][:], axis = 0)
    f.close()
    center = rot_cen

    return projs, theta, center, dark, flat

def adj_csv_file(sample_tag_list, scan_tag_list, csv_path = save_path):
    '''
    The function reads adjusts the csv data file by reading the projection data and copying it into the csv file. 
    
    Input: 
    sample_tag_list: List of string values of the sample number
    layer_tag_list: List of string value of the layer number
    csv_path = Saved path location of the csv file (contains information on each sample)
    
    Output:
    fname = file name
    
    return fname
    '''

    rdf = pd.read_csv(csv_path)
    sample_pitch = []
    sample_roll = []
    sample_rotary = []
    top_x = []
    top_z = []
    x_pos = []
    y_pos = []
    for i in range(len(sample_tag_list)):
        fname = get_filename(str(sample_tag_list[i]), str(scan_tag_list[i]))
        f = h5py.File(os.path.join(data_path,fname),'r')
        sample_pitch.append(float(f['measurement/instrument/sample_motor_stack/setup']["pitch"][:]))
        sample_roll.append(float(f['measurement/instrument/sample_motor_stack/setup']["roll"][:]))
        sample_rotary.append(float(f['measurement/instrument/sample_motor_stack/setup']["rotary"][:]))
        top_x.append(float(f['measurement/instrument/sample_motor_stack/setup']["top/x"][:]))
        top_z.append(float(f['measurement/instrument/sample_motor_stack/setup']["top/z"][:]))
        x_pos.append(float(f['measurement/instrument/sample_motor_stack/setup']["x"][:]))
        y_pos.append(float(f['measurement/instrument/sample_motor_stack/setup']["y"][:]))
        f.close()
    #import pdb; pdb.set_trace()
    rdf['sample_pitch'] = sample_pitch 
    rdf['sample_roll'] = sample_roll 
    rdf['sample_rotary'] = sample_rotary
    rdf['top_x'] = top_x 
    rdf['top_z'] = top_z 
    rdf['x_pos'] = x_pos 
    rdf['y_pos'] = y_pos 
    rdf.to_csv(csv_path, index=False)
    return


def test():

    #Test Cases
    
    
    #Test case
    rdf = pd.read_csv(save_path)
    sample_tag_list = rdf["sample_num"]
    scan_tag_list = rdf["scan_num"]
    adj_csv_file(sample_tag_list, scan_tag_list)

    #Testing files
    # rdf = pd.read_csv(save_path)
    # sample_tag_list = rdf["sample_num"]
    # scan_tag_list = rdf["scan_num"]

    # for i in range(len(sample_tag_list)):
    #     projs, theta, center, dark, flat = read_raw_data_1X(str(sample_tag_list[i]),str(scan_tag_list[i]))
    #     print("Sample #:", sample_tag_list[i], "Layer #:", scan_tag_list[i], "Shape:", str(projs.shape))
    #     print(f'max theta: {theta.max():.2f}, min theta: {theta.min():.2f}')
    #     print(f'max proj: {projs.max():.2f}, min proj: {projs.min():.2f}')
    #     print("proj data type:", projs.dtype)
    return

if __name__ == "__main__":

    test()    
