#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
from operator import mod
from tomo_encoders.misc.voxel_processing import modified_autocontrast, TimerGPU
from tomo_encoders.reconstruction.recon import recon_patches_3d
import cupy as cp
import numpy as np
from skimage.filters import threshold_otsu
from tomo_encoders import Grid
from tomo_encoders.reconstruction.cuda_kernels import rec_all
from cupyx.scipy.fft import rfft, irfft
from tomo_encoders.reconstruction import retrieve_phase
from tomo_encoders.tasks.digital_zoom import segment_otsu, cylindrical_mask, edge_map, get_values_cyl_mask
from cupyx.scipy import ndimage
from tomo_encoders.misc.voxel_processing import modified_autocontrast
from tomo_encoders.structures.voids import Voids
from tomo_encoders.tasks.void_clustering import find_roi_voids, rank_voids_spherical, DBSCAN
from cupyx.scipy.ndimage import label
from epics import PV
from tomo_encoders.structures.voids import Surface
from tomo_encoders.misc.vis_vtk import update_vis_stream

class VoidMapper():
    """Class for tomography reconstruction, segmentatiion and mapping of voids in a stream of raw projection data.

    Parameters
    ----------
    ntheta : int
        The number of projections in the buffer (for simultaneous reconstruction)
    n, nz : int
        The pixel width and height of the projection.
    pars: dictionary contatining:
        center : float32
            Rotation center for reconstruction            
        idx, idy, idz: int32
            X-Y-Z ortho slices for reconstruction
        rotx, roty, rotz: float32
            Rotation angles for X-Y-Z slices
        fbpfilter: str
            Reconstruction filter
        dezinger: str
            None or radius for removing outliers
        energy: float32
            Beam energy
        dist: float32
            Source-detector distance
        alpha: float32
            Tuning parameter for phase retrieval
        pixelsize: float32
            Detector pixel size
    datatype: str
        Detector data type.
    """

    def __init__(self, ntheta, n, nz, pars, datatype):
        
        self.n = n
        self.nz = nz
        self.ntheta = ntheta        
        
        #CPU storage for the buffer
        self.data = np.zeros([ntheta, nz, n], dtype=datatype)
        self.theta = np.zeros([ntheta], dtype='float32')
        # GPU storage for dark and flat fields
        self.dark = cp.array(cp.zeros([nz, n]), dtype='float32')
        self.flat = cp.array(cp.ones([nz, n]), dtype='float32')
        # GPU storages for ortho-slices, and angles        
        self.obj = cp.zeros((nz, n, n), dtype='float32')# full 3d object reconstructed
        self.obj_orthoslice = cp.zeros([n, 3*n], dtype='float32')# ortho-slices are concatenated to one 2D array
        self.surf = None
        # reconstruction parameters 
        self.pars = pars

        # calculate chunk size fo gpu
        mem = cp.cuda.Device().mem_info[1]
        self.chunk = min(self.ntheta,int(np.ceil(mem/self.n/self.nz/32)))#cuda raw kernels do not work with huge sizes (issue in cupy?)
        

        # flag controlling appearance of new dark and flat fields   
        self.new_dark_flat = False
    
    def free(self):
        """Free GPU memory"""

        cp.get_default_memory_pool().free_all_blocks()

    def set_dark(self, data):
        """Copy dark field (already averaged) to GPU"""

        self.dark = cp.array(data.astype('float32'))        
        self.new_dark_flat = True
    
    def set_flat(self, data):
        """Copy flat field (already averaged) to GPU"""

        self.flat = cp.array(data.astype('float32'))
        self.new_dark_flat = True
    
    
    # def backprojection(self, data, theta):
    #     obj = cp.zeros((self.nz,self.n,self.n), dtype = 'float32')
    #     rec_all(obj, data, theta, self.pars["center"])
    #     return obj

    def fbp_filter(self, data):
        """FBP filtering of projections"""

        t = cp.fft.rfftfreq(self.n)
        if (self.pars['fbpfilter']=='Parzen'):
            wfilter = t * (1 - t * 2)**3    
        elif (self.pars['fbpfilter']=='Ramp'):
            wfilter = t
        elif (self.pars['fbpfilter']=='Shepp-logan'):
            wfilter = np.sin(t)
        elif (self.pars['fbpfilter']=='Butterworth'):# todo: replace by other
            wfilter = t / (1+pow(2*t,16)) # as in tomopy

        wfilter = cp.tile(wfilter, [self.nz, 1])    
        #data[:] = irfft(
           #wfilter*rfft(data,overwrite_x=True, axis=2), overwrite_x=True, axis=2)
        for k in range(data.shape[0]):# work with 2D arrays to save GPU memory
            data[k] = irfft(
                wfilter*rfft(data[k], overwrite_x=True, axis=1), overwrite_x=True, axis=1)

    def darkflat_correction(self, data):
        """Dark-flat field correction"""
        
        tmp = cp.maximum(self.flat-self.dark, 1e-6)
        for k in range(data.shape[0]):# work with 2D arrays to save GPU memory
            data[k] = (data[k]-self.dark)/tmp

    def minus_log(self, data):
        """Taking negative logarithm"""
        
        for k in range(data.shape[0]):# work with 2D arrays to save GPU memory
            data[k] = -cp.log(cp.maximum(data[k], 1e-6))
    
    def remove_outliers(self, data):
        """Remove outliers"""
        
        if(int(self.pars['dezinger'])>0):
            r = int(self.pars['dezinger'])            
            fdata = ndimage.median_filter(data,[1,r,r])
            ids = cp.where(cp.abs(fdata-data)>0.5*cp.abs(fdata))
            data[ids] = fdata[ids]        

    def phase(self, data):
        """Retrieve phase"""

        if(self.pars['alpha']>0):
            #print('retrieve phase')
            data = retrieve_phase.paganin_filter(
                data,  self.pars['pixelsize']*1e-4, self.pars['dist']/10, self.pars['energy'], self.pars['alpha'])
  

    def draw_orthoslice(self):

        self.obj_orthoslice[:self.n,         :self.n  ] = self.obj[self.pars['idz']]
        self.obj_orthoslice[:self.nz, self.n  :2*self.n] = self.obj[:,self.pars['idy'], :]
        self.obj_orthoslice[:self.nz , 2*self.n:3*self.n] = self.obj[:,:,self.pars['idx']]
        return self.obj_orthoslice.get()

    def recon_full(self, data, theta, ids, pars):
        """full 3D reconstruction of tomography object.

        Parameters
        ----------
        data : np.array(nproj,nz,n)
            Projection data 
        theta : np.array(nproj)
            Angles corresponding to the projection data
        ids : np.array(nproj)
            Ids of the data in the circular buffer array
        pars: dictionary contatining:
            center : float32
                Rotation center for reconstruction            
            fbpfilter: str
                Reconstruction filter
            dezinger: str
                None or radius for removing outliers
            energy: float32
                Beam energy
            dist: float32
                Source-detector distance
            alpha: float32
                Tuning parameter for phase retrieval
            pixelsize: float32
                Detector pixel size

        Return
        ----------
        obj: np.array([nz,n,n]) 
            Concatenated reconstructions for X-Y-Z orthoslices
        """
        
        # update data in the buffer
        self.data[ids] = data.reshape(data.shape[0], self.nz, self.n)
        self.theta[ids] = theta
        self.pars = pars.copy()
        self.new_dark_flat = False

        return self.update_recon(cp.array(self.data, dtype = cp.float32), cp.array(self.theta*np.pi/180.0, dtype = cp.float32))        


    def update_recon(self, data, theta):
        
        # Get pixel size and sample coordinates
        tomo0deg = PV("2bmS1:m2").get()
        tomo90deg = PV("2bmS1:m1").get()
        sampley = PV("2bmb:m25").get()
        # binning = PV('2bmbSP2:ROI1:BinX').get()            
        # pixel_size = PV("MCTOptics:ImagePixelSize").get()
        pixel_size = self.pars['pixelsize']
        
        

        size_thresh = self.pars['voidsize'] # micrometers
        binary_filter_size = tuple([int(self.pars['filtersize'])]*3)
        mask_size = self.pars['masksize']

        # reconstruction
        self.darkflat_correction(data)
        self.remove_outliers(data)
        self.phase(data)
        self.minus_log(data)
        self.fbp_filter(data)
        rec_all(self.obj, data, theta, self.pars["center"])
        self.obj[:] = ndimage.median_filter(self.obj, binary_filter_size)

        # binarization
        voxel_flat = get_values_cyl_mask(self.obj, 0.8)
        thresh = cp.float32(threshold_otsu(voxel_flat[::64].get()))
        obj_seg = (self.obj[:] < thresh).astype(cp.uint8)
        cylindrical_mask(obj_seg, mask_size, 0)
        obj_seg[:] = ndimage.binary_opening(obj_seg, structure = cp.ones(binary_filter_size))
        obj_edge = edge_map(obj_seg).get()


        # void export
        voids = Voids().count_voids(obj_seg, 1)
        if size_thresh > 0:
            voids.select_by_size(size_thresh, pixel_size_um = pixel_size)
        
        obj_seg[:] = 0.0
        icount = 0
        for iv, s_void in enumerate(voids["s_voids"]):
            obj_seg[s_void] = cp.array(voids["x_voids"][iv])
            icount += 1
        print(f'\tSTAT: number of voids after size select: {icount}')

        self.obj[obj_edge > 0] = cp.nan

        return        

        # timer = TimerGPU()
        # timer.tic()
        # surf = voids.export_void_mesh_with_texture("sizes")
        # # surf = voids.export_void_mesh()
        
        # # rescale mesh to sample coordinates
        # surf["vertices"][:,0] = np.asarray(surf["vertices"][:,0])*pixel_size + sampley
        # surf["vertices"][:,1] = np.asarray(surf["vertices"][:,1])*pixel_size + tomo90deg
        # surf["vertices"][:,2] = np.asarray(surf["vertices"][:,2])*pixel_size + tomo0deg
        
        # _ = timer.toc("new mesh exported")
            
        # if self.pars['findrois'] > 0:            
        #     if self.surf is None:
        #         self.surf = surf
        #     else:
        #         self.surf["vertices"] = np.concatenate([self.surf["vertices"], surf["vertices"]], axis = 0)
        #         self.surf["faces"] = np.concatenate([self.surf["faces"], surf["faces"] + len(self.surf["faces"])], axis = 0)
        #         self.surf["texture"] = np.concatenate([self.surf["texture"], surf["texture"]], axis = 0)
        #     update_vis_stream(self.surf, '/home/beams/TOMO/tekawade_beamtime_202204/tomoencoders_visout/')
        #     return 1
        # else:
        #     update_vis_stream(surf, '/home/beams/TOMO/tekawade_beamtime_202204/tomoencoders_visout/')
        #     return 0

        # if 0:
        #     eps = 100
        #     min_samples = 100
        #     dbs = DBSCAN(eps=eps,min_samples=min_samples)
        #     _ ,rvc = find_roi_voids(voids['cents'], \
        #                               estimators = [("DBSCAN", dbs),],\
        #                               keep_roi_voids_only = True, \
        #                               visualizer = False)

        #     rois = rank_voids_spherical(rvc,voids["cents"],SPHERICAL_RADIUS, self.pars['findrois'])
        #     if rois is not None:
        #         np.save("/data/tomoencoders_visout/roi_points.npy", rois)
                
        #     return 1
        # else:
        #     return 0
            




