#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import cupy as cp
import numpy as np
from tomo2mesh import Grid
from cupyx.scipy.ndimage import label
from scipy.ndimage import label as label_np
from scipy.ndimage import find_objects
from tomo2mesh.misc.voxel_processing import TimerGPU, TimerCPU
import os
from tifffile import imsave, imread
import h5py
from skimage.measure import marching_cubes
from tomo2mesh.porosity.feret_diameter import max_feret_dm
import pymesh
from tomo2mesh.porosity.num_density import num_density
import functools
from multiprocessing import Pool, cpu_count


class Surface(dict):
    def __init__(self, vertices, faces, texture = None):
        self["vertices"] = vertices
        self["faces"] = faces
        self["texture"] = texture
        return

    def __len__(self):
        return(len(self["vertices"]))
    
    def write_ply(self, filename):
        '''
        Source: https://github.com/julienr/meshcut/blob/master/examples/ply.py
        
        '''
        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % len(self["vertices"])) #verts.shape[0]
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            if self["texture"] is not None:
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')        
            f.write('element face %d\n' % len(self["faces"]))
            f.write('property list uchar int vertex_indices\n')
            f.write('end_header\n')
            for i in range(len(self["vertices"])): #verts.shape[0]
                if self["texture"] is not None:
                    f.write('%f %f %f %f %f %f\n' % (self["vertices"][i,0], self["vertices"][i,1], self["vertices"][i,2], self["texture"][i,0], self["texture"][i,1], self["texture"][i,2]))
                else:
                    f.write('%f %f %f\n' % (self["vertices"][i,0], self["vertices"][i,1], self["vertices"][i,2]))
            for i in range(len(self["faces"])):
                f.write('3 %d %d %d\n' % (self["faces"][i,0], self["faces"][i,1], self["faces"][i,2])) 
        return

    def write_ply_npy(self, filename):
        if os.path.exists(filename):
            from shutil import rmtree
            rmtree(filename)
        
        os.makedirs(filename)
        np.save(os.path.join(filename, "vertices.npy"), np.asarray(self["vertices"]))
        np.save(os.path.join(filename, "faces.npy"), np.asarray(self["faces"]))
        if self["texture"] is not None:
            np.save(os.path.join(filename, "texture.npy"), np.asarray(self["texture"]))
        return

    def show_vis_o3d(self):
        import open3d as o3d
        pcd = o3d.geometry.TriangleMesh()
        pcd.vertices = o3d.utility.Vector3dVector(self["vertices"])
        pcd.triangles = o3d.utility.Vector3iVector(self["faces"])
        pcd.vertex_colors = o3d.utility.Vector3dVector(self["texture"])
        o3d.visualization.draw_geometries([pcd])        
        return


class Voids(dict):

    def __init__(self, pad_bb = 0):

        self["sizes"] = []
        self["cents"] = []
        self["cpts"] = []
        self["s_voids"] = []
        self["x_voids"] = []
        self["x_boundary"] = []
        self["surfaces"] = []
        self.vol_shape = (None,None,None)
        self.b = 1
        self.pad_bb = pad_bb
        self.marching_cubes_algo = "skimage"#"vedo"

        return
        
    def transform_linear_shift(self, shift):
        shift = np.asarray(shift)
        self["cents"] = np.asarray(self["cents"]) + shift
        self["cpts"] = np.asarray(self["cpts"]) + shift
        s_voids = []
        for s_void in self["s_voids"]:
            s_new = tuple([slice(s_void[i3].start+shift[i3], s_void[i3].stop + shift[i3]) for i3 in range(3)])
            s_voids.append(s_new)
        self["s_voids"] = s_voids
        return


    def __len__(self):
        return len(self["x_voids"])

    def write_to_disk(self, fpath, overwrite = True):

        '''write voids data to disk.'''

        if os.path.exists(fpath):
            assert overwrite, "overwrite not permitted"
            from shutil import rmtree
            rmtree(fpath)
        
        os.makedirs(fpath)
        
        # write void data to tiff
        void_dir = os.path.join(fpath, "voids")
        os.makedirs(void_dir)
        tot_num = len(str(len(self)))
        for iv, void in enumerate(self["x_voids"]):
            void_id_str = str(iv).zfill(tot_num)
            imsave(os.path.join(void_dir, "void_id_" + void_id_str + ".tiff"), void)
        # write void meta data to hdf5
        with h5py.File(os.path.join(fpath, "meta.hdf5"), 'w') as hf:
            hf.create_dataset("vol_shape", data = self.vol_shape)
            hf.create_dataset("b", data = self.b)
            hf.create_dataset("sizes", data = self["sizes"])
            hf.create_dataset("cents", data = self["cents"])
            hf.create_dataset("cpts", data = self["cpts"])
            s = []
            for s_void in self["s_voids"]:
                sz, sy, sx = s_void
                s.append([sz.start, sz.stop, sy.start, sy.stop, sx.start, sx.stop])
            hf.create_dataset("s_voids", data = np.asarray(s))
            if np.any(self["x_boundary"]):
                hf.create_dataset("x_boundary", data = np.asarray(self["x_boundary"]))

            if "max_feret" in self.keys():
                hf.create_dataset("max_feret/dia", data = self["max_feret"]["dia"])
                hf.create_dataset("max_feret/eq_sph", data = self["max_feret"]["eq_sph"])
                hf.create_dataset("max_feret/norm_dia", data = self["max_feret"]["norm_dia"])
                hf.create_dataset("max_feret/theta", data = self["max_feret"]["theta"])
                hf.create_dataset("max_feret/phi", data = self["max_feret"]["phi"])

            if "number_density" in self.keys():
                hf.create_dataset("number_density", data = self["number_density"])
            
        # # write ply mesh of each void into folder (if available)
        # if np.any(self["surfaces"]):
        #     surf_dir = os.path.join(fpath, "mesh")
        #     os.makedirs(surf_dir)
        #     tot_num = len(str(len(self)))
        #     for iv, surface in enumerate(self["surfaces"]):
        #         void_id_str = str(iv).zfill(tot_num)
        #         surface.write_ply(os.path.join(void_dir, \
        #             "void_id_" + void_id_str + ".ply"), void)
        return
    
    def import_from_disk(self, fpath):

        import glob
        assert os.path.exists(fpath), "path not found to import voids data from"
        with h5py.File(os.path.join(fpath, "meta.hdf5"), 'r') as hf:
            self.vol_shape = tuple(np.asarray(hf["vol_shape"]))
            self.b = int(np.asarray(hf["b"]))
            self["sizes"] = np.asarray(hf["sizes"][:])
            self["cents"] = np.asarray(hf["cents"][:])
            self["cpts"] = np.asarray(hf["cpts"][:])

            self["s_voids"] = []
            for s_void in np.asarray(hf["s_voids"]):
                self["s_voids"].append((slice(s_void[0], s_void[1]), slice(s_void[2], s_void[3]), slice(s_void[4], s_void[5])))

            if "x_boundary" in hf.keys():
                self["x_boundary"] = np.asarray(hf["x_boundary"][:])                
            else:
                self["x_boundary"] = []


            if "max_feret" in hf.keys():
                self["max_feret"] = {}
                self["max_feret"]["dia"] = np.asarray(hf["max_feret/dia"])
                self["max_feret"]["eq_sph"] = np.asarray(hf["max_feret/eq_sph"])
                self["max_feret"]["norm_dia"] = np.asarray(hf["max_feret/norm_dia"])
                self["max_feret"]["theta"] = np.asarray(hf["max_feret/theta"])
                self["max_feret"]["phi"] = np.asarray(hf["max_feret/phi"])

            if "number_density" in hf.keys():
                self['number_density'] = np.asarray(hf["number_density"])



        flist = sorted(glob.glob(os.path.join(fpath,"voids", "*.tiff")))
        self["x_voids"] = []
        for f_ in flist:
            self["x_voids"].append(imread(f_))

        return self



    # def copy_from(self, voids):

    #     self.vol_shape = voids.shape
    #     self.b = voids.b
    #     self["sizes"] = 
    #     self["cents"] = 
    #     self["cpts"] = 

    #     self["s_voids"] = voids["s_voids"]

    #         if "x_boundary" in hf.keys():
    #             self["x_boundary"] = np.asarray(hf["x_boundary"][:])                
    #         else:
    #             self["x_boundary"] = []


    #         if "max_feret" in hf.keys():
    #             self["max_feret"] = {}
    #             self["max_feret"]["dia"] = np.asarray(hf["max_feret/dia"])
    #             self["max_feret"]["eq_sph"] = np.asarray(hf["max_feret/eq_sph"])
    #             self["max_feret"]["norm_dia"] = np.asarray(hf["max_feret/norm_dia"])
    #             self["max_feret"]["theta"] = np.asarray(hf["max_feret/theta"])
    #             self["max_feret"]["phi"] = np.asarray(hf["max_feret/phi"])

    #         if "number_density" in hf.keys():
    #             self['number_density'] = np.asarray(hf["number_density"])



    #     flist = sorted(glob.glob(os.path.join(fpath,"voids", "*.tiff")))
    #     self["x_voids"] = []
    #     for f_ in flist:
    #         self["x_voids"].append(imread(f_))

    #     return







    def import_from_grid(self, voids_b, x_grid, p_grid):

        '''import voids data from grid data
        Parameters
        ----------
        voids_b : Voids  
            voids data structure. may be binned with corresponding binning value stored as voids_b.b
        x_grid : np.asarray  
            list of cubic sub_volumes, all of same width  
        p_grid : Grid  
            instance of grid data structure  

        '''

        Vp = np.zeros(p_grid.vol_shape, dtype = np.uint8)
        p_grid.fill_patches_in_volume(x_grid, Vp)

        self["sizes"] = []
        self["cents"] = []
        self["cpts"] = []
        self["s_voids"] = []
        self["x_voids"] = []
        self.vol_shape = p_grid.vol_shape
        self.b = 1
        b = voids_b.b

        for iv, s_b in enumerate(voids_b["s_voids"]):
            s = tuple([slice(s_b[i].start*b, s_b[i].stop*b) for i in range(3)])
            void = Vp[s]

            cent = [int((s[i].start + s[i].stop)//2) for i in range(3)]
            self["s_voids"].append(s)
            self["cents"].append(cent)
            self["cpts"].append([int((s[i].start)) for i in range(3)])

            # make sure no other voids fall inside the bounding box
            void, n_objs = label_np(void,structure = np.ones((3,3,3),dtype=np.uint8)) #8-connectivity
            
            # nv = np.asarray(void.shape).astype(np.uint32)
            # nv_cent = (nv//2).astype(np.uint32)
            # s_cent = tuple([slice(nv_cent[i3]-1, nv_cent[i3]+2) for i3 in range(3)])
            # idx_cent = np.median(void[s_cent])
            # void = (void == idx_cent).astype(np.uint8)
            
            
            # _s = tuple([slice(self.pad_bb*self.b,-self.pad_bb*self.b)]*3)
            # _idx = np.median(np.clip(void[_s],1,None))
            # void = (void == _idx).astype(np.uint8)

            objs = find_objects(void)
            counts = [np.sum(void[objs[i]] == i+1) for i in range(n_objs)]
            if len(counts) > 1:
                i_main = np.argmax(counts)    
                void = (void == (i_main+1)).astype(np.uint8)            
            else:
                void = (void > 0).astype(np.uint8)

            self["sizes"].append(np.sum(void))
            self["x_voids"].append(void)

        self["sizes"] = np.asarray(self["sizes"])
        self["cents"] = np.asarray(self["cents"])
        self["cpts"] = np.asarray(self["cpts"])

        # copy over any other keys
        for key in voids_b.keys():
            if key in self.keys():
                continue
            else:
                self[key] = voids_b[key]
                
        return self


    def count_voids(self, V_lab, b, dust_thresh, boundary_loc = (0,0,0)):
        self.dust_thresh = dust_thresh
        self.vol_shape = V_lab.shape
        boundary_id = V_lab[boundary_loc]
        slices = find_objects(V_lab)

        self["sizes"] = []
        self["cents"] = []
        self["cpts"] = []
        self["x_voids"] = []
        self["s_voids"] = []
        


        for idx, s in enumerate(slices):

            if s is None:
                continue

            cpt = np.asarray([s[i3].start for i3 in range(3)])
            ept = np.asarray([s[i3].stop for i3 in range(3)])
            
            if not np.all(np.clip(ept-cpt-self.dust_thresh,0,None)):
                continue
            cpt = np.maximum(cpt-self.pad_bb, 0)
            ept = np.minimum(ept+self.pad_bb, np.asarray(self.vol_shape))
            s = tuple([slice(cpt[i3], ept[i3]) for i3 in range(3)])
            void = (V_lab[s] == idx+1)
            if idx + 1 == boundary_id:
                self["x_boundary"] = void.copy()
                continue
            self["sizes"].append(np.sum(void))
            self["cents"].append(list((cpt + ept)//2))
            self["cpts"].append(list(cpt))
            self["x_voids"].append(void)
            self["s_voids"].append(s)

        self["sizes"] = np.asarray(self["sizes"])
        self["cents"] = np.asarray(self["cents"])
        self["cpts"] = np.asarray(self["cpts"])
        self.b = b
        self.n_voids = len(self["sizes"])
        print(f"\tSTAT: voids found - {self.n_voids}")
        
        return self

    def select_by_indices(self, idxs):

        self["x_voids"] = [self["x_voids"][ii] for ii in idxs]
        self["sizes"] = self["sizes"][idxs]
        self["cents"] = self["cents"][idxs]
        self["cpts"] = self["cpts"][idxs]
        self["s_voids"] = [self["s_voids"][ii] for ii in idxs]

        if "max_feret" in self.keys():
            self["max_feret"]["dia"] = self["max_feret"]["dia"][idxs]
            self["max_feret"]["eq_sph"] = self["max_feret"]["eq_sph"][idxs]
            self["max_feret"]["norm_dia"] = self["max_feret"]["norm_dia"][idxs]
            self["max_feret"]["theta"] = self["max_feret"]["theta"][idxs]
            self["max_feret"]["phi"] = self["max_feret"]["phi"][idxs]
        
        if "number_density" in self.keys():
            self["number_density"] = self["number_density"][idxs]

        return

    def select_by_z_coordinate(self, z_pt):

        idxs = np.arange(len(self))
        
        cond_list = np.asarray([1 if self["cents"][:,0][idx] < z_pt else 0 for idx in idxs])
        
        idxs = idxs[cond_list == 1]
        self.select_by_indices(idxs)
        
        return


    def select_z_slab(self, z_start, z_end):

        idxs = np.arange(len(self))
        cond_list = np.asarray([1 if ((self["cents"][:,0][idx] >= z_start) & (self["cents"][:,0][idx] <= z_end))  else 0 for idx in idxs])
        
        idxs = idxs[cond_list == 1]
        self.select_by_indices(idxs)
        
        return



    def select_by_size(self, size_thresh_um, pixel_size_um = 1.0, sel_type = "geq"):
        
        '''
        selection type could be geq or leq.  

        '''
        if size_thresh_um <= 0:
            return
        size_thresh = size_thresh_um/(self.b*pixel_size_um)
        idxs = np.arange(len(self["sizes"]))
        cond_list = np.asarray([1 if self["sizes"][idx] >= size_thresh**3 else 0 for idx in idxs])
        if sel_type == "leq":
            cond_list = cond_list^1

        idxs = idxs[cond_list == 1]
        print(f'\tSTAT: size thres: {size_thresh:.2f} pixel length for {size_thresh_um:.2f} um threshold')          
        self.select_by_indices(idxs)
        return

    def select_by_feret_dia_norm(self, thresh, sel_type = "geq"):
        
        '''
        selection type could be geq or leq.  

        '''
        
        idxs = np.arange(len(self))
        cond_list = np.asarray([1 if self["max_feret"]["norm_dia"][idx] >= thresh else 0 for idx in idxs])
        if sel_type == "leq":
            cond_list = cond_list^1

        idxs = idxs[cond_list == 1]
        self.select_by_indices(idxs)
        return

    def sort_by_size(self, reverse = False):
        
        '''
        selection type could be geq or leq.  
        '''
        idxs = np.argsort(self["sizes"])
        if reverse:
            idxs = idxs[::-1]    
        self.select_by_indices(idxs)
        return

    def select_around_void(self, void_id, radius_um, pixel_size_um = 1.0):
        '''select voids around a given void within a spherical region'''
        
        radius_pix = radius_um/(self.b*pixel_size_um)
        idxs = np.arange(len(self["sizes"]))
        dist = np.linalg.norm((self["cents"] - self["cents"][void_id]), ord = 2, axis = 1)
        self["distance_%i"%void_id] = dist
        cond_list = dist < radius_pix
        idxs = idxs[cond_list == 1]
        self.select_by_indices(idxs)
        return

    def export_grid(self, wd):
        
        wd = wd//self.b
        V_bin = np.zeros(self.vol_shape, dtype = np.uint8)
        for ii, s_void in enumerate(self["s_voids"]):
            V_bin[s_void] = 1#self["x_voids"][ii]
        
        # find patches on surface
        p3d = Grid(V_bin.shape, width = wd)
        x = p3d.extract(V_bin)
        is_sel = (np.sum(x, axis = (1,2,3)) > 0)

        p3d_sel = p3d.filter_by_condition(is_sel)
        if self.b > 1:
            p3d_sel = p3d_sel.rescale(self.b)
        r_fac = len(p3d_sel)*(p3d_sel.wd**3)/np.prod(p3d_sel.vol_shape)
        print(f"\tSTAT: 1/r value: {1/r_fac:.4g}")
        
        return p3d_sel, r_fac

    def _void2mesh(self, void_id, tex_vals, edge_thresh):

        void = self["x_voids"][void_id]
        spt = self["cpts"][void_id]
        # make watertight
        void = np.pad(void, tuple([(2,2)]*3), mode = "constant", constant_values = 0)
        
        
        if np.std(void) == 0:
            return Surface(None, None, texture=None)
            
        # try skimage measure
        verts, faces, _, __ = marching_cubes(void, 0.5)

        verts -= 2 # correct for padding
        for i3 in range(3):
            verts[:,i3] += spt[i3] # z, y, x to x, y, z
        
        # if b > 1, scale up the size
        verts *= (self.b)

        # Decimate
        if edge_thresh > 0:
            verts, faces, info = pymesh.collapse_short_edges_raw(verts, faces, edge_thresh, preserve_feature = True)
        
        # set texture
        texture = np.empty((len(verts),3), dtype = np.float32)
        texture[:,0] = float(tex_vals[0])
        texture[:,1] = float(tex_vals[1])
        texture[:,2] = float(tex_vals[2])

        return Surface(verts, faces, texture=texture)


    def _gray_to_rainbow(self, gray):
        '''
        implements short rainbow colormap
        https://www.particleincell.com/2014/colormap/

        '''        
        a = (1-gray)/0.25
        X = int(a)
        Y = int(255*(gray-X))
        if X == 0:
            r = 255; g = Y; b = 0
        elif X == 1:
            r = 255-Y; g = 255; b = 0
        elif X == 2:
            r = 0; g = 255; b = Y
        elif X == 3:
            r = 0; g = 255-Y; b = 255
        elif X == 4:
            r = 0; g = 0; b = 255
        return r, g, b
    


    def export_void_mesh_with_texture(self, texture_key, edge_thresh = 1.0):

        '''export with texture, slower but vis with color coding
        '''
        st_chkpt = cp.cuda.Event(); end_chkpt = cp.cuda.Event(); st_chkpt.record()    
        

        
        if texture_key == "sizes":
            tex_vals = np.empty((len(self),3))
            tex_vals[:,0] = np.log(self["sizes"]+1.0e-12) #255 #void_id
            tex_vals[:,1] = 255
            tex_vals[:,2] = 255
            
        elif "distance" in texture_key:
            raise ValueError("not implemented")

        elif "max_feret" in texture_key:
            if "max_feret" not in self.keys():
                self.calc_max_feret_dm()            
            tex_vals = np.empty((len(self),3))
            tex_vals[:,0] = self[texture_key]["norm_dia"].copy()
            tex_vals[:,1] = self[texture_key]["theta"].copy()            
            tex_vals[:,2] = self[texture_key]["phi"].copy()

        elif "number_density" in texture_key:
            if "number_density" not in self.keys():
                raise ValueError("number density is not computed")
            tex_vals = np.empty((len(self),3))
            tex_vals[:,0] = self[texture_key]
            tex_vals[:,1] = self["sizes"]
            tex_vals[:,2] = self["max_feret"]["norm_dia"] if "max_feret" in self.keys() else 0.0

        # ORIGINAL
        id_len = 0
        verts = []
        faces = []
        texture = []
        for iv in range(len(self)):
            surf = self._void2mesh(iv, tex_vals[iv], edge_thresh = edge_thresh)
            if not np.any(surf["faces"]):
                continue
            verts.append(surf["vertices"])
            faces.append(np.array(surf["faces"]) + id_len)
            texture.append(surf["texture"])
            id_len = id_len + len(surf["vertices"])


        # normalize colormap
        texture = np.concatenate(texture, axis = 0)
        for i3 in range(3):
            color = texture[:,i3]
            min_val = color.min()
            max_val = color.max()
            if max_val > min_val:
                texture[:,i3] = 255*(color - min_val)/(max_val - min_val)
            else:
                texture[:,i3] = 255


        surf = Surface(np.concatenate(verts, axis = 0), \
                       np.concatenate(faces, axis = 0), \
                       texture = texture.astype(np.uint8))
        
        end_chkpt.record(); end_chkpt.synchronize(); t_chkpt = cp.cuda.get_elapsed_time(st_chkpt,end_chkpt)
        print(f"\tTIME: compute void mesh {t_chkpt/1000.0:.2f} secs")
        return surf

    def export_void_mesh_mproc(self, texture_key, edge_thresh = 1.0):

        '''export with texture, slower but vis with color coding
        '''
        st_chkpt = cp.cuda.Event(); end_chkpt = cp.cuda.Event(); st_chkpt.record()    
        
        if texture_key == "sizes":
            tex_vals = np.empty((len(self),3))
            tex_vals[:,0] = np.log(self["sizes"]+1.0e-12) #255 #void_id
            tex_vals[:,1] = 255
            tex_vals[:,2] = 255
            
        elif "distance" in texture_key:
            raise ValueError("not implemented")

        elif "max_feret" in texture_key:
            if "max_feret" not in self.keys():
                self.calc_max_feret_dm()            
            tex_vals = np.empty((len(self),3))
            tex_vals[:,0] = self[texture_key]["norm_dia"].copy()
            tex_vals[:,1] = self[texture_key]["theta"].copy()            
            tex_vals[:,2] = self[texture_key]["phi"].copy()

        elif "number_density" in texture_key:
            if "number_density" not in self.keys():
                raise ValueError("number density is not computed")
            tex_vals = np.empty((len(self),3))
            tex_vals[:,0] = self[texture_key]
            tex_vals[:,1] = self["sizes"]
            tex_vals[:,2] = self["max_feret"]["norm_dia"] if "max_feret" in self.keys() else 0.0

        # ALTERNATIVE
        # remove empty voids (originating from those misclassified in coarse step)
        surf = void2mesh_mproc(self["x_voids"], self["cpts"], tex_vals, edge_thresh, self.b)
        end_chkpt.record(); end_chkpt.synchronize(); t_chkpt = cp.cuda.get_elapsed_time(st_chkpt,end_chkpt)
        print(f"\tTIME: compute void mesh {t_chkpt/1000.0:.2f} secs")
        return surf


    def calc_max_feret_dm(self):

        timer = TimerCPU("secs")
        timer.tic()
        arr = np.asarray([list(max_feret_dm(self["x_voids"][iv])) for iv in range(len(self))])
        self["max_feret"] = {}
        self["max_feret"]["dia"] = arr[:,0]
        self["max_feret"]["eq_sph"] = arr[:,3]
        self["max_feret"]["norm_dia"] = arr[:,4]
        self["max_feret"]["theta"] = arr[:,1]
        self["max_feret"]["phi"] = arr[:,2]
        timer.toc("calculate max feret diameter")
        return

    def calc_number_density(self, radius_val):
        timer = TimerCPU("secs")
        timer.tic()
        arr = num_density(self["cents"], radius_val)
        self['number_density'] = arr
        timer.toc("calculate number density")
        return

def single_void2mesh(void, spt, tex_val, edge_thresh = 0.0, b = 1):

    # make watertight
    void = np.pad(void, tuple([(2,2)]*3), mode = "constant", constant_values = 0)
        
    # try skimage measure
    verts, faces, _, __ = marching_cubes(void, 0.5)

    verts -= 2 # correct for padding
    for i3 in range(3):
        verts[:,i3] += spt[i3] # z, y, x to x, y, z
    
    # if b > 1, scale up the size
    verts *= b

    # Decimate
    if edge_thresh > 0:
        verts, faces, info = pymesh.collapse_short_edges_raw(verts, faces, edge_thresh, preserve_feature = True)
    
    # set texture
    texture = np.empty((len(verts),3), dtype = np.float32)
    texture[:,0] = float(tex_val[0])
    texture[:,1] = float(tex_val[1])
    texture[:,2] = float(tex_val[2])

    return verts, faces, texture

def void2mesh_mproc(x_voids, cpts, tex_vals, edge_thresh, b):

    # remove void sub-volumes which are empty (these originate from the coarse map where the subset reconstruction reveals there was no void in this bounding box)
    idxs = np.arange(len(x_voids))
    cond_list = np.asarray([1 if np.std(x_voids[idx]) > 0 else 0 for idx in idxs])
    idxs = idxs[cond_list == 1]
    x_voids = [x_voids[ii] for ii in idxs]
    cpts = cpts[idxs]
    tex_vals = tex_vals[idxs]
    
    # multiprocess marching cubes and mesh reduction
    func = functools.partial(single_void2mesh, edge_thresh = edge_thresh, b = b)
    pool = Pool(processes = cpu_count())
    outlist = pool.starmap(func, list(zip(x_voids, cpts, tex_vals)))
    pool.close()
    pool.join()
    
    # merge individual void mesh objects
    verts = []
    faces = []
    texture = []
    id_len = 0
    for item in outlist:
        vx, fc, tex = item
        verts.append(vx)
        faces.append(np.array(fc) + id_len)
        texture.append(tex)
        id_len = id_len + len(vx)


    # normalize colormap
    texture = np.concatenate(texture, axis = 0)
    for i3 in range(3):
        color = texture[:,i3]
        min_val = color.min()
        max_val = color.max()
        if max_val > min_val:
            texture[:,i3] = 255*(color - min_val)/(max_val - min_val)
        else:
            texture[:,i3] = 255


    surf = Surface(np.concatenate(verts, axis = 0), \
                    np.concatenate(faces, axis = 0), \
                    texture = texture.astype(np.uint8))
    
    return surf










class VoidLayers(Voids):

    def __init__(self):

        self["sizes"] = []
        self["cents"] = []
        self["cpts"] = []
        self["s_voids"] = []
        self["x_voids"] = []
        self["x_boundary"] = []
        self["surfaces"] = []
        self.vol_shape = (1,1,1)
        self.b = 1
        self.pad_bb = 1
        self.marching_cubes_algo = "skimage"#"vedo"
        return
        

    def add_layer(self, voids, z_max):

        '''
        
        
        '''
        if len(self["x_voids"]) == 0: # this is the first layer
            self["x_voids"] = voids["x_voids"]
            self["sizes"]  = voids["sizes"]
            self["cents"] = voids["cents"]
            self["cpts"] = voids["cpts"]
            self.b = voids.b
        else: # just append new layer
            self["x_voids"] += voids["x_voids"]
            self["sizes"] = np.concatenate([self["sizes"], voids["sizes"]], axis = 0)
            self["cents"] = np.concatenate([self["cents"], voids["cents"]+[z_max,0,0]], axis = 0)
            self["cpts"] = np.concatenate([self["cpts"], voids["cpts"]+[z_max,0,0]], axis = 0)
        
        for s_void in voids["s_voids"]:
            sz, sy, sx = s_void
            sz = slice(s_void[0].start + z_max, s_void[0].stop + z_max)
            s_void = (sz, sy, sx)
            self["s_voids"].append(s_void)
        
        # if "max_feret" in self.keys():
            # pass
        # if "number_density" in self.keys():
        #     self["number_density"] = np.concatenate(self["number_density"], voids["number_density"], axis = 0)

        return 
            


    def import_from_grid(self):
        raise NotImplementedError()
        
    def count_voids(self):
        raise NotImplementedError()

    
        

            
