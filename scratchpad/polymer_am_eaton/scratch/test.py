

import numpy as np
from tomo2mesh.misc.voxel_processing import TimerCPU
from tomo2mesh.structures.patches import Patches

V_shape = (1536,2448,2448)
p_size = 144

patches = Patches(V_shape, initialize_by = "bestfit_grid", patch_size = (p_size,p_size,p_size))
thresh_list = [np.ones((144,144,144), dtype = np.uint8) for i in range(len(patches))]


timer = TimerCPU("secs")


timer.tic()
V_seg = np.empty(V_shape, dtype = np.uint8)
patches.fill_patches_in_volume(thresh_list, V_seg)
timer.toc("fill patches")

