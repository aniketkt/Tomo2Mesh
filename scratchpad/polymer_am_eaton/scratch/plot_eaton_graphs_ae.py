from cmath import nan
from tomo2mesh.structures.voids import VoidLayers
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys
import os

#from tomo2mesh.projects.eaton.rw_utils import read_raw_data_1X, save_path
from tomo2mesh.projects.eaton.rw_utils_ae import read_raw_data_1X, save_path
from tomo2mesh.projects.eaton.void_mapping import void_map_gpu, void_map_all
from tomo2mesh.projects.eaton.params import pixel_size_1X as pixel_size
plots_dir = '/home/yash/eaton_plots/'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
import matplotlib as mpl
mpl.use('Agg')

CUTOFF_CRACKS = 4.0

def number_density_z(voids, axis = 0):
    z_cents = voids["cents"][:,axis]*3.13*4*10**(-3)
    slab_id = (z_cents//0.05).astype(np.uint64)
    num_den = np.array(voids_all["number_density"])/eq_sph_vol

    num_den_mean = []
    num_den_std = []
    slab_id_adj = []
    for i in range(np.max(slab_id)):
        idxs = np.where(slab_id==i)
        if np.isnan(np.mean(np.mean(num_den[idxs]))) or np.isnan(np.std(num_den[idxs])/np.sqrt(np.size(idxs))):
            continue
        else:
            slab_id_adj.append(i)
            num_den_mean.append(np.mean(num_den[idxs]))
            num_den_std.append(np.std(num_den[idxs])/np.sqrt(np.size(idxs)))
        
    #x_mm = np.arange(0,np.max(slab_id))*0.05+0.025
    x_mm = np.array(slab_id_adj)*0.05+0.025
    return x_mm, np.asarray(num_den_mean)*10**4, np.asarray(num_den_std)*10**4

def plot_mean_size_z_slabs(voids):
    z_cents = voids["cents"][:,0]*3.13*4*10**(-3)
    x_voids_vol = np.cbrt(voids["sizes"])
    slab_id = (z_cents//0.05).astype(np.uint64)

    mean_size_mean = []
    mean_size_std = []
    slab_id_adj = []
    for i in range(np.max(slab_id)):
        idxs = np.where(slab_id==i)
        if np.isnan(np.mean(x_voids_vol[idxs])) or np.isnan(np.std(x_voids_vol[idxs])/np.sqrt(np.size(idxs))):
            continue
        else:
            slab_id_adj.append(i)
            mean_size_mean.append(np.mean(x_voids_vol[idxs])) #to-do: convert to mm
            mean_size_std.append(np.std(x_voids_vol[idxs])/np.sqrt(np.size(idxs)))

    #x_mm = np.arange(0,np.max(slab_id))*0.05+0.025
    x_mm = np.array(slab_id_adj)*0.05+0.025
    return x_mm, np.asarray(mean_size_mean), np.asarray(mean_size_std) #*10**4


def count_cracks(voids, cutoff = 3.0):
    z_cents = voids["cents"][:,0]*3.13*4*10**(-3)
    slab_id = (z_cents//0.05).astype(np.uint64)
    norm_feret = np.array(voids["max_feret"]["norm_dia"])
    
    count = []
    for i in range(np.max(slab_id)):
        idxs = np.where(slab_id==i)
        count.append(np.sum(norm_feret[idxs] > cutoff))

    x_mm = np.arange(0,np.max(slab_id))*0.05+0.025
    #x_mm = np.array(slab_id)*0.05+0.025
    return x_mm, np.asarray(count)


def crack_orientation(voids, cutoff = 3.0):
    z_cents = voids["cents"][:,0]*3.13*4*10**(-3)
    slab_id = (z_cents//0.05).astype(np.uint64)
    
    theta_mean = []
    theta_std = []
    phi_mean = []
    phi_std = []
    slab_id_adj = []
    for i in range(np.max(slab_id)):
        idxs = np.where((slab_id==i) & (voids["max_feret"]["norm_dia"] > cutoff))
        if np.isnan(np.mean(voids["max_feret"]["theta"][idxs])) or np.isnan(np.std(voids["max_feret"]["theta"][idxs])/np.sqrt(np.size(idxs))) or np.isnan(np.mean(voids["max_feret"]["phi"][idxs])) or np.isnan(np.std(voids["max_feret"]["phi"][idxs])/np.sqrt(np.size(idxs))):
            continue
        else:
            slab_id_adj.append(i)
            theta_mean.append(np.mean(voids["max_feret"]["theta"][idxs]))
            theta_std.append(np.std(voids["max_feret"]["theta"][idxs])/np.sqrt(np.size(idxs)))
            phi_mean.append(np.mean(voids["max_feret"]["phi"][idxs]))
            phi_std.append(np.std(voids["max_feret"]["phi"][idxs])/np.sqrt(np.size(idxs)))

    #x_mm = np.arange(0,np.max(slab_id))*0.05+0.025
    x_mm = np.array(slab_id_adj)*0.05+0.025
    return x_mm, np.asarray(theta_mean), np.asarray(theta_std), np.asarray(phi_mean), np.asarray(phi_std)



def merge_void_layers(sample_tag, b, raw_pixel_size, dust_thresh, number_density_radius = 50):
    z_max = []

    rdf = pd.read_csv(save_path)
    info = rdf[(rdf["sample_num"] == str(sample_tag))]
    scan_num = list(info["scan_num"])
    y_pos_list = list(info['y_pos'])

    y_pos_list = np.asarray(y_pos_list)*1.0e3/(raw_pixel_size*b)
    y_pos_list = (y_pos_list - y_pos_list[0]).astype(np.uint64)
    
    # y_pos_list[0] += 1536//2//b
    z_max = np.uint32(np.diff(y_pos_list))
    

    voids_all = VoidLayers()
    porosity_z_all = []
    for ii, scan_tag in enumerate(range(scan_num[0], scan_num[-1]+1)): 
        projs, theta, center, dark, flat = read_raw_data_1X(sample_tag, scan_tag)
        
        if ii==0:
            z_crop = (1536//2,1536)
        elif scan_tag == scan_num[-1]:
            z_crop = (0,1536//2)
        else:
            z_crop = (0,1536)
        cp._default_memory_pool.free_all_blocks(); cp.fft.config.get_plan_cache().clear()  
        
        voids = void_map_all(projs, theta, center, dark, flat, b, raw_pixel_size, dust_thresh, z_crop)
        # voids = void_map_gpu(projs, theta, center, dark, flat, b, raw_pixel_size)
        # print(f"BOUNDARY SHAPE: {voids['x_boundary'].shape}")
        # voids.select_by_size(100.0, pixel_size_um = pixel_size) # remove later

        if scan_tag != scan_num[-1]:
            # print(f"z_max: {z_max}")
            voids.select_by_z_coordinate(z_max[ii]) #Find voids in each layer (w/o overlap)
        #     porosity_z_all.append(voids["porosity_z"][0:z_max[ii]])
        # else:
        #     porosity_z_all.append(voids["porosity_z"])
            
        voids_all.add_layer(voids,y_pos_list[ii])
        print("added a layer")

        # if ii==0:
        #     break

        #Save porosity values for all layers in all samples
        # if b == 1:
        #     if str(sample_tag)=="1" and str(scan_tag)=="124":
        #         fpath = "/data01/Eaton_Polymer_AM/csv_files/porosity_vals_adj2.csv"
        #         info = {"sample_tag": [sample_tag], "scan_tag": [scan_tag], "porosity": [voids["porosity"]]}
        #         df2 = pd.DataFrame(info)
        #         df2.to_csv(fpath)
        #     else:
        #         fpath = "/data01/Eaton_Polymer_AM/csv_files/porosity_vals_adj2.csv"
        #         info = {"sample_tag": [sample_tag], "scan_tag": [scan_tag], "porosity": [voids["porosity"]]}
        #         df2 = pd.DataFrame(info)
        #         df2.to_csv(fpath, mode = 'a',header=False)


    # porosity_z_all = np.concatenate(porosity_z_all, axis = 0)
    # voids_all["porosity_z"] = porosity_z_all
    #Save local porosity values for all layers in sample 1
    # if b == 1:
    #     fpath = "/data01/Eaton_Polymer_AM/csv_files/local_porosity_vals.csv"
    #     info = {"porosity_z": porosity_z_all}
    #     df2 = pd.DataFrame(info)
    #     df2.to_csv(fpath)

    if b != 1:
        voids_all.calc_max_feret_dm()
        voids_all.calc_number_density(number_density_radius)

    return voids_all


if __name__ == "__main__":

    b = 4
    dust_thresh = 2
    number_density_mesh = False
    feret_dm_mesh = False

    sample_name = str(sys.argv[1])
    
    z_max = []
    voids_list = []
    
    rdf = pd.read_csv(save_path)
    number_density_radius = 50
    
    voids_all = merge_void_layers(sample_name, b, pixel_size, dust_thresh)
    if b != 1:
        #voids_all.export_void_mesh_with_texture("number_density").write_ply(f'/data01/Eaton_Polymer_AM/ply_files/sample_{sample_name}.ply')
        if number_density_mesh:
            voids_all.export_void_mesh_mproc("number_density", edge_thresh=1.0).write_ply(f'/data01/Eaton_Polymer_AM/ply_files/sample_{sample_name}_numdensity.ply')
        if feret_dm_mesh:
            voids_all.select_by_feret_dia_norm(3.0)
            voids_all.export_void_mesh_mproc("max_feret", edge_thresh=1.0).write_ply(f'/data01/Eaton_Polymer_AM/ply_files/sample_{sample_name}_maxferet.ply')
        
    voids_all.write_to_disk(f'/data01/Eaton_Polymer_AM/voids_data/sample_{sample_name}_all_layers_dust{dust_thresh}_b{b}')

    exit()

    #################################################

    # crack_count_list = []
    # mean_size_list = []
    # num_density_list = []
    # sample_name = [1,2,3,4,5,6,7,9,10,11,12]
    # for val in sample_name:
    #     z_max = []
    #     voids_list = []
    #     voids_all = VoidLayers()
    #     rdf = pd.read_csv(save_path)
    #     number_density_radius = 50
        
    #     voids_all, porosity = merge_void_layers(sample_tag, end_layer, b, pixel_size)
    #     #voids_all.export_void_mesh_with_texture("number_density").write_ply(f'/data01/Eaton_Polymer_AM/ply_files/sample_{sample_name}_layers_{start_layer}_{end_layer}.ply')
    #     voids_all.write_to_disk(f'/data01/Eaton_Polymer_AM/voids_data/sample{sample_name_adj}_all_layers')

    #     feret_dm = voids_all["max_feret"]["dia"]
    #     eq_sph_dm = voids_all["max_feret"]["eq_sph"]
    #     norm_dm = voids_all["max_feret"]["norm_dia"]
    #     theta = voids_all["max_feret"]["theta"]
    #     phi = voids_all["max_feret"]["phi"]

    #     eq_sph_vol = (4/3)*np.pi*number_density_radius**3
    #     x_voids = voids_all["x_voids"]
    #     z_cent = voids_all["cents"][:,0]
    #     y_cent = voids_all["cents"][:,1]
    #     x_cent = voids_all["cents"][:,2]
    #     num_den = np.array(voids_all["number_density"])/eq_sph_vol

    #     x_mm, num_den_mean, num_den_std = number_density_z(voids_all)
    #     x_mm, crack_count = count_cracks(voids_all, cutoff = CUTOFF_CRACKS)
    #     x_mm, mean_size, mean_size_std = plot_mean_size_z_slabs(voids_all)

    #     crack_count_list.append(np.sum(crack_count))
    #     mean_size_list.append(np.sum(np.array([np.cbrt(np.sum(i)) for i in x_voids]))/len(x_voids))
    #     num_density_list.append(np.sum(np.array([np.sum(i) for i in num_den]))/len(x_voids))

    # fig, ax = plt.subplots(1,1, figsize = (8,8))
    # ax.bar(sample_name, crack_count_list)
    # ax.set_title("Number of Cracks in each Sample")
    # ax.set_xlabel("Sample Number")
    # ax.set_ylabel("Number of Cracks")
    # #plt.show()
    # plt.savefig(plots_dir+f'barplot_crack_number_all_samples_{CUTOFF_CRACKS}.png', format = 'png')
    # plt.close()

    # fig, ax = plt.subplots(1,1, figsize = (8,8))
    # ax.bar(sample_name, mean_size_list)
    # ax.set_title("Average Void Size in each Sample")
    # ax.set_xlabel("Sample Number")
    # ax.set_ylabel("Average Voids Size")
    # #plt.show()
    # plt.savefig(plots_dir+f'barplot_mean_size_all_samples.png', format = 'png')
    # plt.close()
    
    # fig, ax = plt.subplots(1,1, figsize = (8,8))
    # ax.bar(sample_name, num_density_list)
    # ax.set_title("Average Number Density in each Sample")
    # ax.set_xlabel("Sample Number")
    # ax.set_ylabel("Average Number Density")
    # #plt.show()
    # plt.savefig(plots_dir+f'barplot_mean_number_density_all_samples.png', format = 'png')
    # plt.close()

    # exit()

    ################################

    feret_dm = voids_all["max_feret"]["dia"]
    eq_sph_dm = voids_all["max_feret"]["eq_sph"]
    norm_dm = voids_all["max_feret"]["norm_dia"]
    theta = voids_all["max_feret"]["theta"]
    phi = voids_all["max_feret"]["phi"]

    eq_sph_vol = (4/3)*np.pi*number_density_radius**3
    x_voids = voids_all["x_voids"]
    z_cent = voids_all["cents"][:,0]
    y_cent = voids_all["cents"][:,1]
    x_cent = voids_all["cents"][:,2]
    num_den = np.array(voids_all["number_density"])/eq_sph_vol

    #Number Density plot
    x_mm, num_den_mean, num_den_std = number_density_z(voids_all)

    sns.set(font_scale=1.3)
    sns.set_style(style = "white")
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    ax.errorbar(x_mm, num_den_mean, xerr=0, yerr=num_den_std, fmt='-o', color = 'black', ecolor = 'red')
    ax.set_title("Number Density vs. Location of Voids")
    ax.set_ylabel("Number Density (10^-4)")
    ax.set_xlabel("Sample Width Location of Voids (mm)")
    #ax.set(xlim=(20, 1130))

    top = np.max(num_den_mean)+0.5
    bot = np.min(num_den_mean)-0.5
    lines_2500um = np.arange(0,max(x_mm),0.6)
    ax.vlines(lines_2500um, bot, top,linestyles='--')
    ax.set_ylim([bot,top])
    # plt.show()
    plt.savefig(plots_dir + f'number_density_sample{sample_name}.png', format='png')
    plt.close()

    #Crack count plot
    x_mm, crack_count = count_cracks(voids_all, cutoff = CUTOFF_CRACKS)

    sns.set(font_scale=1.3)
    sns.set_style(style = "white")
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    ax.errorbar(x_mm, crack_count, xerr=0, yerr=0, fmt='-o', color = 'black', ecolor = 'red')
    ax.set_title("Count Cracks vs. Location of Voids")
    ax.set_ylabel("Crack Count")
    ax.set_xlabel("Sample Width Location of Voids (mm)")
    #ax.set(xlim=(20, 1130))

    top = np.max(crack_count)+5
    bot = np.min(crack_count)-5
    lines_2500um = np.arange(0,max(x_mm),0.6)
    ax.vlines(lines_2500um, bot, top,linestyles='--')
    ax.set_ylim([bot,top])
    # plt.show()
    plt.savefig(plots_dir + f'crack_count_sample{sample_name}_cutoff{CUTOFF_CRACKS}.png', format = 'png')
    plt.close()


    # Orientation plot
    x_mm, theta_mean, theta_std, phi_mean, phi_std = crack_orientation(voids_all, cutoff = 3.0)
    
    sns.set(font_scale=1.3)
    sns.set_style(style = "white")
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    ax.scatter(x_mm, theta_mean)
    ax.errorbar(x_mm, theta_mean, xerr=0, yerr=theta_std, fmt='-o', color = 'black', ecolor = 'red')
    ax.set_title("Mean Orientation of Voids vs. Location of Voids")
    ax.set_ylabel("Theta (Degrees)")
    ax.set_xlabel("Sample Width Location of Voids (mm)")
    #ax.set(xlim=(0,8))

    top = 90
    bot = -90
    lines_2500um = np.arange(0,max(x_mm),0.6)
    ax.vlines(lines_2500um, bot, top,linestyles='--')
    ax.set_ylim([bot,top])
    # plt.show()
    plt.savefig(plots_dir+f'theta_orientation_sample{sample_name}.png', format = 'png')
    plt.close()

    sns.set(font_scale=1.3)
    sns.set_style(style = "white")
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    ax.scatter(x_mm, phi_mean)
    ax.errorbar(x_mm, phi_mean, xerr=0, yerr=phi_std, fmt='-o', color = 'black', ecolor = 'red')
    ax.set_title("Orientation of Voids")
    ax.set_ylabel("Phi (Degrees)")
    ax.set_xlabel("Sample Width Location of Voids (mm)")
    #ax.set(xlim=(0,8))

    top = 0
    bot = 90
    lines_2500um = np.arange(0,max(x_mm),0.6)
    ax.vlines(lines_2500um, bot, top,linestyles='--')
    ax.set_ylim([bot,top])
    # plt.show()
    plt.savefig(plots_dir+f'phi_orientation_sample{sample_name}.png', format = 'png')
    plt.close()

    #Mean size plot
    x_mm, mean_size, mean_size_std = plot_mean_size_z_slabs(voids_all)

    sns.set(font_scale=1.3)
    sns.set_style(style = "white")
    fig, ax = plt.subplots(1,1, figsize = (16,8))
    ax.scatter(x_mm, mean_size)
    ax.errorbar(x_mm, mean_size, xerr=0, yerr=mean_size_std, fmt='-o', color = 'black', ecolor = 'red')
    ax.set_title("Mean Size vs. Location of Voids")
    ax.set_ylabel("Mean Size (10^4)")
    ax.set_xlabel("Sample Width Location of Voids (mm)")
    #ax.set(xlim=(0,8))

    top = np.max(mean_size)+0.5
    bot = np.min(mean_size)-0.5
    lines_2500um = np.arange(0,max(x_mm),0.6)
    ax.vlines(lines_2500um, bot, top,linestyles='--')
    ax.set_ylim([bot,top])
    # plt.show()
    plt.savefig(plots_dir+f'mean_size_sample{sample_name}.png', format = 'png')
    plt.close()

    #Histograms of Size
    labels = ["$d_{fe}$", "$d_{sp}$", "$d^{*}_{fe}$", "$\Theta$", "$\phi$"]
    fig, ax = plt.subplots(3,1, figsize = (8,8), sharex = False)
    ax[0].hist(feret_dm, bins = 100, density = True)
    ax[0].axvline(np.mean(feret_dm), color = "black")
    ax[1].hist(eq_sph_dm, bins = 100, density = True)
    ax[1].axvline(np.mean(eq_sph_dm), color = "black")
    ax[2].hist(norm_dm, bins = 100, density = True)
    ax[2].axvline(np.mean(norm_dm), color = "black")
    ax[0].set_title("Histograms on the Size of Voids")
    ax[1].set_ylabel("Probability Density")
    ax[0].set_xlabel(labels[0])
    ax[1].set_xlabel(labels[1])
    ax[2].set_xlabel(labels[2])
    plt.tight_layout()
    plt.savefig(plots_dir+f'histogram_size_sample{sample_name}.png', format = 'png')

    #Histogram of Orientation
    theta_adj = []
    phi_adj = []
    for i in range(len(norm_dm)):
        if norm_dm[i]>=CUTOFF_CRACKS:
            theta_adj.append(theta[i])
            phi_adj.append(phi[i])
            
    fig, ax = plt.subplots(2,1, figsize = (8,8))
    ax[0].hist(theta, bins=100, color = 'blue', density = True)
    ax[0].axvline(np.mean(theta), color = "green")
    ax[0].hist(theta_adj, bins=100, color = 'red', density = True)
    ax[0].axvline(np.mean(theta_adj), color = "orange")
    ax[1].hist(phi, bins=100, color = 'blue', density = True)
    ax[1].axvline(np.mean(phi), color = "green")
    ax[1].hist(phi_adj, bins=100, color = 'red', density = True)
    ax[1].axvline(np.mean(feret_dm), color = "orange")
    ax[0].set_title("Histograms on the Orientation of Voids")
    ax[0].set_xlabel(labels[3])
    ax[1].set_xlabel(labels[4])
    ax[0].set_ylabel("Probability Density")
    ax[1].set_ylabel("Probability Density")
    plt.tight_layout()
    plt.savefig(plots_dir+f'histogram_orientation_sample{sample_name}.png', format = 'png')

    #Porosity value
    print("Porosity:", porosity)
    print("Theta (Std):", np.std(theta))
    print(f"Theta (Std), CUTOFF {CUTOFF_CRACKS}:", np.std(theta_adj))
    print("Phi (Std):", np.std(phi))
    print(f"Phi (Std), CUTOFF {CUTOFF_CRACKS}:", np.std(phi_adj))
    print("Feret Diameter (Std):", np.std(feret_dm))
    print("Eq. Sphere Diameter (Std):", np.std(eq_sph_dm))
    print("Normalized Feret Diameter (Std):", np.std(norm_dm))
    print("Sample", sample_name)
    