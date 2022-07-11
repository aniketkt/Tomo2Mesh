#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import ast

#%%
#Plot important parameters of voids
w_num = 3
s_num = 5

por_all = []
for i in range(w_num):
    for j in range(s_num):
        w = str(i+1)
        s = str(j+1)
        csv_path = "/data01/csv_files/stats_w"+w+"_s"+s+".csv"
        rdf = pd.read_csv(csv_path)

        vol = rdf["Volume"]
        por = rdf["Porosity"][0]
        feret_dm = rdf["Feret_Diameter"]
        ellip_E = rdf["Equatorial_Ellipticity"]
        ellip_P = rdf["Polar_Ellipticity"]
        sph_diam = rdf["Equivalent_Sphere_Volume_Diameter"]

        por_all.append(por)

        ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
        ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
        ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
        ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
        ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

        #Plot histogram of volumes of the voids
        vol_avg = np.mean(vol)
        vol_median = np.median(vol)
        vol_std = np.std(vol)
        ax1.axvline(x = vol_avg, color = 'red')
        ax1.axvline(x = vol_median, color = 'black')
        ax1.hist(vol, bins=1000)
        ax1.set(xlabel='(a)', ylabel='Frequency')
        ax1.set_xlim([0,1900])
        print("Max Volume:",max(vol))

        #Plot histogram of feret diameter
        feret_dm_avg = np.mean(feret_dm)
        feret_dm_median = np.median(feret_dm)
        feret_dm_std = np.std(feret_dm)
        ax2.axvline(x = feret_dm_avg, color = 'red')
        ax2.axvline(x = feret_dm_median, color = 'black')
        ax2.hist(feret_dm, bins=100)
        ax2.set(xlabel='(b)', ylabel='')
        ax2.set_xlim([0,30])
        print("Max Maximum Feret Diameter:",max(feret_dm))

        #Plot histogram of equivalent sphere volume diameter of the voids
        sph_diam_avg = np.mean(sph_diam)
        sph_diam_median = np.median(sph_diam)
        sph_diam_std = np.std(sph_diam)
        ax3.axvline(x = sph_diam_avg, color = 'red')
        ax3.axvline(x = sph_diam_median, color = 'black')
        ax3.hist(sph_diam, bins=100)
        ax3.set(xlabel='(c)', ylabel='')
        ax3.set_xlim([0,20])
        print("Max Equivalent Sphere Diameter:",max(sph_diam))

        #Plot histogram of equatorial ellipticity of the voids
        ellip_E_avg = np.mean(ellip_E)
        ellip_E_median = np.median(ellip_E)
        ellip_E_std = np.std(ellip_E)
        ax4.axvline(x = ellip_E_avg, color = 'red')
        ax4.axvline(x = ellip_E_median, color = 'black')
        ax4.hist(ellip_E, bins=100)
        ax4.set(xlabel='(d)', ylabel='Frequency')
        print("Max Equatorial Ellipticity:",max(ellip_E))

        #Plot histogram of polar ellipticity of the voids
        ellip_P_avg = np.mean(ellip_P)
        ellip_P_median = np.median(ellip_P)
        ellip_P_std = np.std(ellip_P)
        ax5.axvline(x = ellip_P_avg, color = 'red')
        ax5.axvline(x = ellip_P_median, color = 'black')
        ax5.hist(ellip_P, bins=100)
        ax5.set(xlabel='(e)', ylabel='')
        print("Max Polar Ellipticity:",max(ellip_E))
        plt.tight_layout()
        plt.show()


        table = [["w"+w+"_s"+s, 'Volume (um^3)', 'Max. Feret Diameter (um)', 'Eq. Vol. Diameter (um)', 'Ellipticity (Equ.)', 'Ellipticity (Polar)'], 
        ["Mean",round(vol_avg,3), round(feret_dm_avg,3), round(sph_diam_avg,3), round(ellip_E_avg,3), round(ellip_P_avg,3)],
        ["Median",round(vol_median,3), round(feret_dm_median,3), round(sph_diam_median,3), round(ellip_E_median,3), round(ellip_P_median,3)],
        ["Std. Dev.",round(vol_std,3), round(feret_dm_std,3), round(sph_diam_std,3), round(ellip_E_std,3), round(ellip_P_std,3)]]
        print(tabulate(table, tablefmt='fancy_grid'))



#%%
por_avg = round(np.mean(por_all),5)
por_median = round(np.median(por_all),5)
por_std = round(np.std(por_all),6)
print("Porosity Values:", por_all)

table = [['Porosity (Mean)', 'Porosity (Median)', 'Porosity (Std. Dev.)'], [por_avg,por_median,por_std]]
print(tabulate(table, tablefmt='fancy_grid'))

#%%
#Plot stability test
csv_path = "/data01/csv_files/stability_test.csv"
rdf = pd.read_csv(csv_path)
por = np.array(rdf["Porosity"])
s_val = np.array(rdf["s"])*100
plt.scatter(s_val,por)
plt.title("Porosity vs. Clipping %")
plt.xlabel("Clipping %")
plt.ylabel("Porosity")
plt.show()

#%%
#Plot data trimming sensitivity test for ellipsoid fitting
bin_fact = [1,2,4,8,16,32]
vol_lists = []
ellip_E_lists = []
ellip_P_lists = []
maj_ax_lists = []
inter_ax_lists = []
min_ax_lists = []
time_list = []
for i in range(len(bin_fact)):
    csv_path = "/data01/csv_files/trim_sensitivity_f"+str(bin_fact[i])+".csv"
    rdf = pd.read_csv(csv_path)
    ax_len = rdf["Axis_Lengths"]
    ellip_E_lists.append(rdf["Equatorial_Ellipticity"])
    ellip_P_lists.append(rdf["Polar_Ellipticity"])
    time_list.append(rdf["Time"][0])

    empty_max = []
    empty_inter = []
    empty_min = []
    for j in range(len(ax_len)):
        temp = ax_len[j].strip('][').strip(" ").replace(" ",",").replace(",,",",").replace(",,",",").replace(",,",",")
        temp = "["+temp+"]"
        empty_max.append(ast.literal_eval(temp)[0])
        empty_inter.append(ast.literal_eval(temp)[1])
        empty_min.append(ast.literal_eval(temp)[2])

    maj_ax_lists.append(empty_max)
    inter_ax_lists.append(empty_inter)
    min_ax_lists.append(empty_min)
    vol_lists.append(rdf["Volume"])


ellip_E_err1 = (np.abs(np.array(ellip_E_lists[1])-np.array(ellip_E_lists[0]))/np.array(ellip_E_lists[0]))*100
ellip_E_err2 = (np.abs(np.array(ellip_E_lists[2])-np.array(ellip_E_lists[0]))/np.array(ellip_E_lists[0]))*100
ellip_E_err3 = (np.abs(np.array(ellip_E_lists[3])-np.array(ellip_E_lists[0]))/np.array(ellip_E_lists[0]))*100
ellip_E_err4 = (np.abs(np.array(ellip_E_lists[4])-np.array(ellip_E_lists[0]))/np.array(ellip_E_lists[0]))*100
ellip_E_err5 = (np.abs(np.array(ellip_E_lists[5])-np.array(ellip_E_lists[0]))/np.array(ellip_E_lists[0]))*100

ellip_P_err1 = (np.abs(np.array(ellip_P_lists[1])-np.array(ellip_P_lists[0]))/np.array(ellip_P_lists[0]))*100
ellip_P_err2 = (np.abs(np.array(ellip_P_lists[2])-np.array(ellip_P_lists[0]))/np.array(ellip_P_lists[0]))*100
ellip_P_err3 = (np.abs(np.array(ellip_P_lists[3])-np.array(ellip_P_lists[0]))/np.array(ellip_P_lists[0]))*100
ellip_P_err4 = (np.abs(np.array(ellip_P_lists[4])-np.array(ellip_P_lists[0]))/np.array(ellip_P_lists[0]))*100
ellip_P_err5 = (np.abs(np.array(ellip_P_lists[5])-np.array(ellip_P_lists[0]))/np.array(ellip_P_lists[0]))*100

maj_err1 = (np.abs(np.array(maj_ax_lists[1])-np.array(maj_ax_lists[0]))/np.array(maj_ax_lists[0]))*100
maj_err2 = (np.abs(np.array(maj_ax_lists[2])-np.array(maj_ax_lists[0]))/np.array(maj_ax_lists[0]))*100
maj_err3 = (np.abs(np.array(maj_ax_lists[3])-np.array(maj_ax_lists[0]))/np.array(maj_ax_lists[0]))*100
maj_err4 = (np.abs(np.array(maj_ax_lists[4])-np.array(maj_ax_lists[0]))/np.array(maj_ax_lists[0]))*100
maj_err5 = (np.abs(np.array(maj_ax_lists[5])-np.array(maj_ax_lists[0]))/np.array(maj_ax_lists[0]))*100

min_err1 = (np.abs(np.array(min_ax_lists[1])-np.array(min_ax_lists[0]))/np.array(min_ax_lists[0]))*100
min_err2 = (np.abs(np.array(min_ax_lists[2])-np.array(min_ax_lists[0]))/np.array(min_ax_lists[0]))*100
min_err3 = (np.abs(np.array(min_ax_lists[3])-np.array(min_ax_lists[0]))/np.array(min_ax_lists[0]))*100
min_err4 = (np.abs(np.array(min_ax_lists[4])-np.array(min_ax_lists[0]))/np.array(min_ax_lists[0]))*100
min_err5 = (np.abs(np.array(min_ax_lists[5])-np.array(min_ax_lists[0]))/np.array(min_ax_lists[0]))*100

inter_err1 = (np.abs(np.array(inter_ax_lists[1])-np.array(inter_ax_lists[0]))/np.array(inter_ax_lists[0]))*100
inter_err2 = (np.abs(np.array(inter_ax_lists[2])-np.array(inter_ax_lists[0]))/np.array(inter_ax_lists[0]))*100
inter_err3 = (np.abs(np.array(inter_ax_lists[3])-np.array(inter_ax_lists[0]))/np.array(inter_ax_lists[0]))*100
inter_err4 = (np.abs(np.array(inter_ax_lists[4])-np.array(inter_ax_lists[0]))/np.array(inter_ax_lists[0]))*100
inter_err5 = (np.abs(np.array(inter_ax_lists[5])-np.array(inter_ax_lists[0]))/np.array(inter_ax_lists[0]))*100

##############################################################################################

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False)
fig.tight_layout()

ax1.scatter(vol_lists[0],ellip_E_err1)
ax1.axvline(x = 3219.681, color = 'black')
ax1.axvline(x = 16098.404, color = 'black')
ax1.set(xlabel='(a)', ylabel='Percent Error (%)')
#plt.setp(ax1.get_xticklabels(), visible=False)

ax2.scatter(vol_lists[0],ellip_E_err5)
ax2.axvline(x = 3219.681, color = 'black')
ax2.axvline(x = 16098.404, color = 'black')
ax2.set(xlabel='(b)', ylabel='')

ax3.scatter(vol_lists[0],ellip_E_err1)
ax3.scatter(vol_lists[0],ellip_E_err2)
ax3.scatter(vol_lists[0],ellip_E_err3)
ax3.scatter(vol_lists[0],ellip_E_err4)
ax3.scatter(vol_lists[0],ellip_E_err5)
ax3.axvline(x = 3219.681, color = 'black')
ax3.axvline(x = 16098.404, color = 'black')
ax3.set(xlabel='(c)', ylabel='')

##############################################################################################

ax4.scatter(vol_lists[0],ellip_P_err1)
ax4.axvline(x = 3219.681, color = 'black')
ax4.axvline(x = 16098.404, color = 'black')
ax4.set(xlabel='(d)', ylabel='Percent Error (%)')

ax5.scatter(vol_lists[0],ellip_P_err5)
ax5.axvline(x = 3219.681, color = 'black')
ax5.axvline(x = 16098.404, color = 'black')
ax5.set(xlabel='(e)', ylabel='')

ax6.scatter(vol_lists[0],ellip_P_err1)
ax6.scatter(vol_lists[0],ellip_P_err2)
ax6.scatter(vol_lists[0],ellip_P_err3)
ax6.scatter(vol_lists[0],ellip_P_err4)
ax6.scatter(vol_lists[0],ellip_P_err5)
ax6.axvline(x = 3219.681, color = 'black')
ax6.axvline(x = 16098.404, color = 'black')
ax6.set(xlabel='(f)', ylabel='')
plt.show()

##############################################################################################

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False)
fig.tight_layout()

ax1.scatter(vol_lists[0],maj_err1)
ax1.axvline(x = 3219.681, color = 'black')
ax1.axvline(x = 16098.404, color = 'black')
ax1.set(xlabel='(a)', ylabel='')

ax2.scatter(vol_lists[0],maj_err5)
ax2.axvline(x = 3219.681, color = 'black')
ax2.axvline(x = 16098.404, color = 'black')
ax2.set(xlabel='(b)', ylabel='')

ax3.scatter(vol_lists[0],maj_err1)
ax3.scatter(vol_lists[0],maj_err2)
ax3.scatter(vol_lists[0],maj_err3)
ax3.scatter(vol_lists[0],maj_err4)
ax3.scatter(vol_lists[0],maj_err5)
ax3.axvline(x = 3219.681, color = 'black')
ax3.axvline(x = 16098.404, color = 'black')
ax3.set(xlabel='(c)', ylabel='')

ax4.scatter(vol_lists[0],min_err1)
ax4.axvline(x = 3219.681, color = 'black')
ax4.axvline(x = 16098.404, color = 'black')
ax4.set(xlabel='(d)', ylabel='Percent Error (%)')

ax5.scatter(vol_lists[0],min_err5)
ax5.axvline(x = 3219.681, color = 'black')
ax5.axvline(x = 16098.404, color = 'black')
ax5.set(xlabel='(e)', ylabel='')

ax6.scatter(vol_lists[0],min_err1)
ax6.scatter(vol_lists[0],min_err2)
ax6.scatter(vol_lists[0],min_err3)
ax6.scatter(vol_lists[0],min_err4)
ax6.scatter(vol_lists[0],min_err5)
ax6.axvline(x = 3219.681, color = 'black')
ax6.axvline(x = 16098.404, color = 'black')
ax6.set(xlabel='(f)', ylabel='')

ax7.scatter(vol_lists[0],inter_err1)
ax7.axvline(x = 3219.681, color = 'black')
ax7.axvline(x = 16098.404, color = 'black')
ax7.set(xlabel='(g)', ylabel='')

ax8.scatter(vol_lists[0],inter_err5)
ax8.axvline(x = 3219.681, color = 'black')
ax8.axvline(x = 16098.404, color = 'black')
ax8.set(xlabel='(h)', ylabel='')

ax9.scatter(vol_lists[0],inter_err1)
ax9.scatter(vol_lists[0],inter_err2)
ax9.scatter(vol_lists[0],inter_err3)
ax9.scatter(vol_lists[0],inter_err4)
ax9.scatter(vol_lists[0],inter_err5)
ax9.axvline(x = 3219.681, color = 'black')
ax9.axvline(x = 16098.404, color = 'black')
ax9.set(xlabel='(i)', ylabel='')
plt.show()


plt.scatter(bin_fact, time_list)
plt.title("Computation Time vs. Data Trimming Factor")
plt.xlabel("Data Trimming Factor")
plt.ylabel("Time (s)")
plt.show()

# # %%
# #Plot Slice of Segmented Data
# from skimage import io
# num = 1000
# seg_path = '/data01/AM_steel_project/xzhang_feb22_rec/seg_data/wheel1_sam1/segmented/'
# im_arr = io.imread(seg_path+"segmented"+"0001"+".tif")
# plt.imshow(im_arr)
# plt.show()

# %%
