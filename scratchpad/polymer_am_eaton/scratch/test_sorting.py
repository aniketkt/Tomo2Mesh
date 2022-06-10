

import numpy as np

nv = 102
nproc = 4
if __name__ == "__main__":

    x_voids = [np.random.randint(0,2,(32,32,32)) for i in range(nv)]
    sizes = np.random.normal(0,1,nv)
    


    
    idx = np.argsort(sizes)
    idx_old = np.arange(len(sizes))
    idx_old = idx_old[idx]

    idxs = np.array_split(idx, nproc)

    x_voids_nested = [[x_voids[ii] for ii in idxs[jj]] for jj in range(nproc)]


    rad_max_nested = [np.random.normal(0,1,26), np.random.normal(0,1,26), np.random.normal(0,1,25), np.random.normal(0,1,25)]
    rad_max = np.concatenate(rad_max_nested, axis = 0)
    rad_max = rad_max[idx_old]
    
    
    



