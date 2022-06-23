from tomo2mesh.structures.voids import VoidLayers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys
import os
sys.path.append('/data01/Tomo2Mesh/scratchpad/polymer_am_eaton/code/')
from rw_utils import read_raw_data_1X, save_path
from void_mapping import void_map_gpu
from params import pixel_size_1X as pixel_size
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
    for i in range(np.max(slab_id)):
        idxs = np.where(slab_id==i)
        num_den_mean.append(np.mean(num_den[idxs]))
        num_den_std.append(np.std(num_den[idxs])/np.sqrt(np.size(idxs)))
        

    x_mm = np.arange(0,np.max(slab_id))*0.05+0.025
    return x_mm, np.asarray(num_den_mean)*10**4, np.asarray(num_den_std)*10**4

def plot_mean_size_z_slabs(voids):
    z_cents = voids["cents"][:,0]*3.13*4*10**(-3)
    x_voids_vol = np.cbrt(voids["sizes"])
    slab_id = (z_cents//0.05).astype(np.uint64)
    
    mean_size_mean = []
    mean_size_std = []
    for i in range(np.max(slab_id)):
        idxs = np.where(slab_id==i)
        mean_size_mean.append(np.mean(x_voids_vol[idxs])) #to-do: convert to mm
        mean_size_std.append(np.std(x_voids_vol[idxs])/np.sqrt(np.size(idxs)))

    x_mm = np.arange(0,np.max(slab_id))*0.05+0.025
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
    return x_mm, np.asarray(count)


def crack_orientation(voids, cutoff = 3.0):
    z_cents = voids["cents"][:,0]*3.13*4*10**(-3)
    slab_id = (z_cents//0.05).astype(np.uint64)
    
    theta_mean = []
    theta_std = []
    phi_mean = []
    phi_std = []
    for i in range(np.max(slab_id)):
        idxs = np.where((slab_id==i) & (voids["max_feret"]["norm_dia"] > cutoff))
        theta_mean.append(np.mean(voids["max_feret"]["theta"][idxs]))
        theta_std.append(np.std(voids["max_feret"]["theta"][idxs])/np.sqrt(np.size(idxs)))
        phi_mean.append(np.mean(voids["max_feret"]["phi"][idxs]))
        phi_std.append(np.std(voids["max_feret"]["phi"][idxs])/np.sqrt(np.size(idxs)))

    x_mm = np.arange(0,np.max(slab_id))*0.05+0.025
    return x_mm, np.asarray(theta_mean), np.asarray(theta_std), np.asarray(phi_mean), np.asarray(phi_std)



def merge_void_layers(sample_tag, start_layer, end_layer, b, raw_pixel_size, number_density_radius = 50):

    y_pos_list = []
    for layer in range(start_layer, end_layer+1):
        info = rdf[(rdf["sample_num"] == int(sample_tag)) & (rdf["layer"] == int(layer))].iloc[0]
        y_pos_list.append(info['sample_y'])
    
    y_pos_list = np.asarray(y_pos_list)*1.0e3/(raw_pixel_size*b)
    y_pos_list = y_pos_list - y_pos_list[0]
    z_max = np.uint32(np.diff(y_pos_list))

    for ii, layer in enumerate(range(start_layer, end_layer+1)): 
        projs, theta, center, dark, flat = read_raw_data_1X(sample_tag, layer)
        voids = void_map_gpu(projs, theta, center, dark, flat, b, raw_pixel_size)
        print(f"BOUNDARY SHAPE: {voids['x_boundary'].shape}")
        # voids.select_by_size(100.0, pixel_size_um = pixel_size) # remove later

        if layer != end_layer:
            print(f"z_max: {z_max}")
            voids.select_by_z_coordinate(z_max[ii]) #Find voids in each layer (w/o overlap)
            
        voids_all.add_layer(voids,y_pos_list[ii])
        print("added a layer")

    #Porosity Calculations
    count = np.sum([np.sum(k) for k in voids_all['x_voids']])
    ht_pix = y_pos_list[-1] + 288
    print(f"height pix: {ht_pix}")
    vol = ht_pix*612*612*(np.pi/4)
    porosity = count/vol

    voids_all.calc_max_feret_dm()
    voids_all.calc_number_density(number_density_radius)

    return voids_all, porosity




if __name__ == "__main__":

    b = 4
    
    #####

    # sample_name = '1'
    # start_layer = 1
    # end_layer = 4
    # sample_tag = '1'

    # sample_name = str(sys.argv[1])
    # start_layer = int(sys.argv[2])
    # end_layer = int(sys.argv[3])
    # sample_tag = str(sys.argv[4])

    #####
    
    # z_max = []
    # voids_list = []
    # voids_all = VoidLayers()
    # rdf = pd.read_csv(save_path)
    # number_density_radius = 50
    
    # voids_all, porosity = merge_void_layers(sample_tag, start_layer, end_layer, b, pixel_size)
    # #voids_all.export_void_mesh_with_texture("number_density").write_ply(f'/data01/Eaton_Polymer_AM/ply_files/sample_{sample_name}_layers_{start_layer}_{end_layer}.ply')
    # voids_all.write_to_disk(f'/data01/Eaton_Polymer_AM/voids_data/sample{sample_name}_all_layers')


    #################################################

    crack_count_list = []
    mean_size_list = []
    num_density_list = []
    sample_name = [1,2,3]#,4,5,6,7,9,10,11,12]
    for val in sample_name:
        sample_tag = 0
        start_layer = 0
        end_layer = 0
        if val==1:
            start_layer = 1
            end_layer = 4
            sample_tag = "1"
        if val==2 or val==3:
            start_layer = 1
            end_layer = 5
            sample_tag = val
        if val==4 or val==6 or val==8 or val==10:
            start_layer = 1
            end_layer = 4
            if val==4:
                sample_tag = "45"
            if val==6:
                sample_tag = "67"
            if val==8:
                sample_tag = "89"
            if val==10:
                sample_tag = "1011"
        if val==5 or val==7 or val==9 or val==11:
            start_layer = 7
            end_layer = 10
            if val==5:
                sample_tag = "45"
            if val==7:
                sample_tag = "67"
            if val==9:
                sample_tag = "89"
            if val==11:
                sample_tag = "1011"
        if val==12:
            start_layer = 1
            end_layer = 2
            sample_tag = "12"

        sample_name_adj = str(val)

        #print(sample_name, sample_tag, start_layer, end_layer)
        
        z_max = []
        voids_list = []
        voids_all = VoidLayers()
        rdf = pd.read_csv(save_path)
        number_density_radius = 50
        
        voids_all, porosity = merge_void_layers(sample_tag, start_layer, end_layer, b, pixel_size)
        #voids_all.export_void_mesh_with_texture("number_density").write_ply(f'/data01/Eaton_Polymer_AM/ply_files/sample_{sample_name}_layers_{start_layer}_{end_layer}.ply')
        voids_all.write_to_disk(f'/data01/Eaton_Polymer_AM/voids_data/sample{sample_name_adj}_all_layers')

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

        x_mm, num_den_mean, num_den_std = number_density_z(voids_all)
        x_mm, crack_count = count_cracks(voids_all, cutoff = CUTOFF_CRACKS)
        x_mm, mean_size, mean_size_std = plot_mean_size_z_slabs(voids_all)

        crack_count_list.append(np.sum(crack_count))
        mean_size_list.append(np.sum(np.array([np.cbrt(np.sum(i)) for i in x_voids]))/len(x_voids))
        num_density_list.append(np.sum(np.array([np.sum(i) for i in num_den]))/len(x_voids))

    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.bar(sample_name, crack_count_list)
    ax.set_title("Number of Cracks in each Sample")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Number of Cracks")
    #plt.show()
    plt.savefig(plots_dir+f'barplot_crack_number_all_samples_{CUTOFF_CRACKS}.png', format = 'png')
    plt.close()

    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.bar(sample_name, mean_size_list)
    ax.set_title("Average Void Size in each Sample")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Average Voids Size")
    #plt.show()
    plt.savefig(plots_dir+f'barplot_mean_size_all_samples.png', format = 'png')
    plt.close()
    
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    ax.bar(sample_name, num_density_list)
    ax.set_title("Average Number Density in each Sample")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Average Number Density")
    #plt.show()
    plt.savefig(plots_dir+f'barplot_mean_number_density_all_samples.png', format = 'png')
    plt.close()

    exit()
    # start a new script by reading voids data

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
    ax.set_title("Mean Size vs. Location of Voids")
    ax.set_ylabel("Mean Size (10^4)")
    ax.set_xlabel("Sample Width Location of Voids (mm)")
    #ax.set(xlim=(0,8))

    top = np.max(theta_mean)+0.5
    bot = np.min(theta_mean)-0.5
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
    ax.set_title("Mean Size vs. Location of Voids")
    ax.set_ylabel("Mean Size (10^4)")
    ax.set_xlabel("Sample Width Location of Voids (mm)")
    #ax.set(xlim=(0,8))

    top = np.max(phi_mean)+0.5
    bot = np.min(phi_mean)-0.5
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

    #Porosity value
    print("Porosity:", porosity)
    print("Sample", sample_name)
    