import numpy as np
import itertools
import matplotlib.pyplot as plt

import sys
sys.path.append("/data01/Tomo2Mesh/scratchpad/polymer_am_eaton/AMPolyCalc/code")
from rw_utils import read_raw_data_1X
from void_mapping import void_map_gpu
from params import pixel_size_1X as pixel_size



if __name__ == "__main__":
    #Create and export ply files
    b = 4
    feret_thresh =3.0
    edge_thresh = float(sys.argv[1])#0.75
    layer = 1
    sample_tag = 1
    projs, theta, center, dark, flat = read_raw_data_1X(sample_tag, layer)
    voids_4 = void_map_gpu(projs, theta, center, dark, flat, b, pixel_size)
    x_voids = voids_4["x_voids"]

    voids_4.calc_max_feret_dm()
    voids_4.select_by_feret_dia_norm(feret_thresh)
    surf = voids_4.export_void_mesh_with_texture("max_feret", edge_thresh = edge_thresh)
    
    print(f"number of vertices: {len(surf['vertices'])}; faces: {len(surf['faces'])}")
    
    
    # surf.write_ply(f'/data01/Eaton_Polymer_AM/ply_files/ply_bin{b}_sample{sample_tag}_layer{layer}_edge{}.ply')
    surf.write_ply(f'/data01/Eaton_Polymer_AM/ply_files/test_edge{edge_thresh}.ply')

    exit()
    #Graph data
    feret_dm = voids_4["max_feret"]["dia"]
    eq_sph_dm = voids_4["max_feret"]["eq_sph"]
    norm_dm = voids_4["max_feret"]["norm_dia"]
    theta = voids_4["max_feret"]["theta"]
    phi = voids_4["max_feret"]["phi"]

    labels = ["$d_{fe}$", "$d_{sp}$", "$d^{*}_{fe}$", "$\Theta$", "$\phi$"]
    fig, ax = plt.subplots(3,1, figsize = (8,8), sharex = False)
    ax[0].hist(feret_dm, bins = 100, density = True)
    ax[1].hist(eq_sph_dm, bins = 100, density = True)
    ax[2].hist(norm_dm, bins = 100, density = True)
    ax[0].set_title("Histograms on the Size of Voids")
    ax[1].set_ylabel("Probability Density")
    ax[0].set_xlabel(labels[0])
    ax[1].set_xlabel(labels[1])
    ax[2].set_xlabel(labels[2])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.scatter(eq_sph_dm,norm_dm, s=30)
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[2])
    ax.set_title("Equivalent Sphere Diameter vs. Normalized Feret Diameter")
    plt.axvline(x = 15, color = "black")
    plt.axhline(y = 3, color = "red")
    plt.show()

    feret_dm_adj = [] 
    eq_sph_dm_adj = []
    norm_dm_adj = []
    theta_adj = []
    phi_adj = []
    thresh = 6
    for i in range(len(norm_dm)):
        if norm_dm[i]>=thresh:
            feret_dm_adj.append(feret_dm[i])
            eq_sph_dm_adj.append(eq_sph_dm[i])
            norm_dm_adj.append(norm_dm[i])
            theta_adj.append(theta[i])
            phi_adj.append(phi[i])

    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.scatter(eq_sph_dm_adj,norm_dm_adj, s=30)
    ax.set_title("Equivalent Sphere Diameter vs. Normalized Feret Diameter")
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[2])
    ax.set(xlim=(0, 50), ylim=(0, 12))
    plt.show()

    fig, ax = plt.subplots(2,1, figsize = (8,8))
    ax[0].hist(theta, bins=100, color = 'blue', density = True)
    ax[0].hist(theta_adj, bins=100, color = 'red', density = True)
    ax[1].hist(phi, bins=100, color = 'blue', density = True)
    ax[1].hist(phi_adj, bins=100, color = 'red', density = True)
    ax[0].set_title("Histograms on the Orientation of Voids")
    ax[0].set_xlabel(labels[3])
    ax[1].set_xlabel(labels[4])
    ax[0].set_ylabel("Probability Density")
    ax[1].set_ylabel("Probability Density")
    plt.tight_layout()
    plt.show()



