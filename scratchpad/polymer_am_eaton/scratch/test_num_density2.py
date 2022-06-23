import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import time

import sys
sys.path.append("/data01/Tomo2Mesh/scratchpad/polymer_am_eaton/code")
sys.path.append("/data01/Tomo2Mesh/tomo2mesh/misc")
from rw_utils import read_raw_data_1X
from void_mapping import void_map_gpu
from params import pixel_size_1X as pixel_size
from num_density import num_density


if __name__ == "__main__":
    b = 4
    edge_thresh = 0.75 #float(sys.argv[1])
    layer = 1
    sample_tag = 1
    projs, theta, center, dark, flat = read_raw_data_1X(sample_tag, layer)
    voids_4 = void_map_gpu(projs, theta, center, dark, flat, b, pixel_size)
    
    voids_4.calc_max_feret_dm()
    voids_4.calc_number_density(10)
    surf = voids_4.export_void_mesh_with_texture("number_density")
    surf.write_ply(f'/data01/Eaton_Polymer_AM/ply_files/num_density_layer{layer}_sample{sample_tag}.ply')


    
    # zmax = (Ypos[layer2] - Ypos[layer1])/pixel_size_1X  # this needs to be a list of values
    

    # voids_all = VoidsLayers()
    # for i in range(4):
    #     voids_list[ii].select_by_z_coordinate(z_max)
    # for i in range(4):
    #     voids_all.add_layer(voids_list[i])
    




