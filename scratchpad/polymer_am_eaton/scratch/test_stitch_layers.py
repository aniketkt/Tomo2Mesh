import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import time
import pandas as pd

import sys
sys.path.append("/data01/Tomo2Mesh/scratchpad/polymer_am_eaton/code")
from rw_utils import read_raw_data_1X
from void_mapping import void_map_gpu
from params import pixel_size_1X as pixel_size
from tomo2mesh.misc.num_density import num_density
from params import save_path
from tomo2mesh.structures.voids import VoidLayers

if __name__ == "__main__":
    
    b = 4
    num_layers = 5
    sample_tag = '3'
    z_max = []
    voids_list = []
    voids_all = VoidLayers()
    rdf = pd.read_csv(save_path)
    
    y_pos_list = []
    for layer in range(1,1+num_layers):
        info = rdf[(rdf["sample_num"] == int(sample_tag)) & (rdf["layer"] == int(layer))].iloc[0]
        y_pos_list.append(info['sample_y'])
    
    y_pos_list = np.asarray(y_pos_list)*1.0e3/(pixel_size*b)
    y_pos_list = y_pos_list - y_pos_list[0]
    z_max = np.uint32(np.diff(y_pos_list))



    for i in range(num_layers):
        layer = i + 1
        projs, theta, center, dark, flat = read_raw_data_1X(sample_tag, layer)
        voids = void_map_gpu(projs, theta, center, dark, flat, b, pixel_size)
        # voids.select_by_size(100.0, pixel_size_um = pixel_size) # remove later

        if i != num_layers-1:
            voids.select_by_z_coordinate(z_max[i]) #Find voids in each layer (w/o overlap)
            
        voids_all.add_layer(voids,y_pos_list[i])
        # surf = voids_all.export_void_mesh_with_texture("sizes")
        # surf.write_ply(f'/data01/Eaton_Polymer_AM/ply_files/sample{sample_tag}_layer{layer}.ply')
        
        print("added a layer")


    voids_all.calc_max_feret_dm()
    voids_all.calc_number_density(10)
    surf = voids_all.export_void_mesh_with_texture("number_density")
    surf.write_ply(f'/data01/Eaton_Polymer_AM/ply_files/sample{sample_tag}_all_layers.ply')
    voids_all.write_to_disk(f'/data01/Eaton_Polymer_AM/voids_data/sample{sample_tag}_all_layers')




