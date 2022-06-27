import numpy as np
import h5py 
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

from tomo2mesh.projects.eaton.params import save_path, data_path, voids_path
# rdf = pd.read_csv(save_path)

def get_filename(sample_tag, layer_tag, csv_path = save_path):
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

    rdf = pd.read_csv(csv_path)
    info = rdf[(rdf["sample_num"] == int(sample_tag)) & (rdf["layer"] == int(layer_tag))].iloc[0]

    scan_num = int(info["scan_num"])
    FOV = int(info["layer"])
    samp_num = int(info["sample_num"])
    flag  = int(info["flag"])

    if flag==0:
        fname = "PE6_"+str(samp_num)+"_FOV"+str(FOV)+"_"+"0"*((len(str(scan_num))-3)*-1)+str(scan_num)+".h5"
    else:
        fname = "PE6_"+sample_tag+"_"+str(scan_num)+".h5"
    
    return fname


def read_raw_data_1X(sample_tag, layer_tag, csv_path = save_path):
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
    info = rdf[(rdf["sample_num"] == int(sample_tag)) & (rdf["layer"] == int(layer_tag))].iloc[0]

    scan_num = int(info["scan_num"])
    FOV = int(info["layer"])
    samp_num = int(info["sample_num"])
    rot_cen = info["rot_cen"]
    flag  = int(info["flag"])

    #Create file name
    if flag==0:
        fname = "PE6_"+str(samp_num)+"_FOV"+str(FOV)+"_"+"0"*((len(str(scan_num))-3)*-1)+str(scan_num)+".h5"
        f = h5py.File(os.path.join(data_path, fname), 'r')
        projs = np.asarray(f['exchange/data'][:])
        theta = np.radians(np.asarray(f['exchange/theta'][:]))%(2*np.pi)
        f.close()
        f = h5py.File(os.path.join(data_path, 'dark_fields_'+fname), 'r')
        dark = np.mean(f['exchange/data_dark'][:], axis = 0)
        f.close()
        f = h5py.File(os.path.join(data_path,'flat_fields_'+fname), 'r')
        flat = np.mean(f['exchange/data_white'][:], axis = 0)
        f.close()
        center = rot_cen
    else:
        fname = "PE6_"+sample_tag+"_"+str(scan_num)+".h5"
        f = h5py.File(os.path.join(data_path,fname), 'r')
        projs = np.asarray(f['exchange/data'][:])
        flat = np.mean(f['exchange/data_white'][:], axis = 0)
        dark = np.mean(f['exchange/data_dark'][:], axis = 0)
        theta = np.radians(np.asarray(f['exchange/theta'][:]))
        center = rot_cen
        f.close()

    return projs, theta, center, dark, flat

def adj_csv_file(sample_tag_list, layer_tag_list, csv_path = save_path):
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
    sample_x = []
    sample_x_cent = []
    sample_y = []
    sample_z_cent = []
    for i in range(len(sample_tag_list)):
        fname = get_filename(str(sample_tag_list[i]), str(layer_tag_list[i]))
        f = h5py.File(os.path.join(data_path,fname),'r')
        sample_pitch.append(float(f['measurement/instrument/sample_motor_stack/setup']["sample_pitch"][:]))
        sample_roll.append(float(f['measurement/instrument/sample_motor_stack/setup']["sample_roll"][:]))
        sample_rotary.append(float(f['measurement/instrument/sample_motor_stack/setup']["sample_rotary"][:]))
        sample_x.append(float(f['measurement/instrument/sample_motor_stack/setup']["sample_x"][:]))
        sample_x_cent.append(float(f['measurement/instrument/sample_motor_stack/setup']["sample_x_cent"][:]))
        sample_y.append(float(f['measurement/instrument/sample_motor_stack/setup']["sample_y"][:]))
        sample_z_cent.append(float(f['measurement/instrument/sample_motor_stack/setup']["sample_z_cent"][:]))
        f.close()
    rdf['sample_pitch'] = sample_pitch
    rdf['sample_roll'] = sample_roll
    rdf['sample_rotary'] = sample_rotary
    rdf['sample_x'] = sample_x
    rdf['sample_x_cent'] = sample_x_cent
    rdf['sample_y'] = sample_y
    rdf['sample_z_cent'] = sample_z_cent
    rdf.to_csv(csv_path, index=False)
    return


def test():

    #Test Cases
    
    
    #Test case
    rdf = pd.read_csv(save_path)
    sample_tag_list = rdf["sample_num"]
    layer_tag_list = rdf["layer"]
    adj_csv_file(sample_tag_list, layer_tag_list)

    #Testing files
    rdf = pd.read_csv(save_path)
    sample_tag_list = rdf["sample_num"]
    layer_tag_list = rdf["layer"]

    for i in range(len(sample_tag_list)):
        projs, theta, center, dark, flat = read_raw_data_1X(str(sample_tag_list[i]),str(layer_tag_list[i]))
        print("Sample #:", sample_tag_list[i], "Layer #:", layer_tag_list[i], "Shape:", str(projs.shape))
        print(f'max theta: {theta.max():.2f}, min theta: {theta.min():.2f}')
        print(f'max proj: {projs.max():.2f}, min proj: {projs.min():.2f}')
        print("proj data type:", projs.dtype)
    return

if __name__ == "__main__":

    test()    
