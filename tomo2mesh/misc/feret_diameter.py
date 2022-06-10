import numpy as np
from scipy.spatial.distance import pdist
import itertools

# import sys
# sys.path.append("/data01/AMPolyCalc/code/")
from tomo_encoders.misc.voxel_processing import edge_map
# from rw_utils import read_raw_data_1X
# from void_mapping import void_map_gpSu
from scipy.spatial import ConvexHull


def max_feret_dm(x_void, unit_vect = np.array([1,0,0]), normalize = True):
    #Calculate maximum feret diameter
    verts = np.asarray(np.where(edge_map(x_void))).T
    verts = verts[ConvexHull(verts).vertices]
    dist = pdist(verts, 'euclidean')
    idx = np.argmax(dist)
    feret_dm = dist[idx]
    eq_sphere = np.cbrt(6.0/np.pi*np.sum(x_void))
    normalized_feret_dm = feret_dm/eq_sphere

    #Calculate orientation of feret diameter
    coord = list(itertools.combinations(verts,2))[idx]
    vect = coord[1]-coord[0]
    #phi = np.arccos(np.dot(vect,unit_vect)/(np.sqrt(vect.dot(vect))*np.sqrt(unit_vect.dot(unit_vect)))) 
    phi = np.arccos(vect[0]/np.sqrt(vect[0]**2+vect[1]**2+vect[2]**2))*(180/np.pi)
    theta = np.arctan(vect[1]/vect[2])*(180/np.pi)
    return (feret_dm, theta, phi, eq_sphere, normalized_feret_dm)


#%%
if __name__ == "__main__":
    projs, theta, center, dark, flat = read_raw_data_1X("1", "1")
    b = 4
    voids_4 = void_map_gpu(projs, theta, center, dark, flat, b, pixel_size)
    x_voids = voids_4["x_voids"]

    voids_4.calc_max_feret_dm()
    surf = voids_4.export_void_mesh_with_texture("max_feret")
    surf.write_ply('/data01/Eaton_Polymer_AM/ply_files/ply_bin4_sample1_layer1.ply')
    
    
    # import matplotlib.pyplot as plt
    # labels = ["$d_{fe}$", "$d_{sp}$", "$d^{*}_{fe}$"]
    # fig, ax = plt.subplots(3,1, figsize = (8,8), sharex = False)
    # for ii, idx_item in enumerate([0,3,4]):
    #     ax[ii].hist([item[idx_item] for item in info],bins = 100, density = True)
    #     ax[ii].set_title(labels[ii])

    # plt.show()

    # #%%
    # fig, ax = plt.subplots(1,1, figsize = (8,8))
    # ax.scatter([item[3] for item in info],[item[4] for item in info], s=30)
    # ax.set_xlabel(labels[1])
    # ax.set_ylabel(labels[2])

    # # %%
    # info_adj = []
    # for i in range(len(info)):
    #     if info[i][4]>=6:
    #         info_adj.append(info[i])

    # fig, ax = plt.subplots(1,1, figsize = (8,8))
    # ax.scatter([item[3] for item in info_adj],[item[4] for item in info_adj], s=30)
    # ax.set_xlabel(labels[1])
    # ax.set_ylabel(labels[2])
    # ax.set(xlim=(0, 50), ylim=(0, 12))

    # fig, ax = plt.subplots(2,1, figsize = (8,8))
    # ax[0].hist([item[1] for item in info], bins=100, color = 'blue', density = True)
    # ax[0].hist([item[1] for item in info_adj], bins=100, color = 'red', density = True)
    # ax[1].hist([item[2] for item in info], bins=100, color = 'blue', density = True)
    # ax[1].hist([item[2] for item in info_adj], bins=100, color = 'red', density = True)
    # plt.show()

# %%
