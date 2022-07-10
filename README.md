# Tomo2Mesh:  
*Reconstruct, segment, and generate meshes for smart visualization of tomography raw data.*  

<p align="justify">  
Tomo2Mesh is an open-source project targeted towards real-time reconstruction, segmentation, and visualization of computed tomography (CT) data in mesh format. The CT reconstruction scheme is based on filtered back-projection of voxel subsets. The segmentation scheme uses a 3D convolutional neural network. To allow for fast, real-time reconstruction, voxel subsets are first identified by coarse reconstruction. The detail in specific regions of interest is improved through full reconstruction of voxel subsets in that region. Data structures are implemented to store and process voxel subsets. Finally, voxel data is labeled using connected components to detect disconnected regions such as voids whose morphological attributes (e.g., Feret diameter, principal axis orientation, local number density, size, etc.) can be measured also in real-time. Finally, a fast marching cubes implementation processes labeled voxel data into a triangular face mesh (vertices and faces) in .ply format for visualization in Paraview or other mesh visualization tools. The code provides a simple programming interface for detecting, classifying, and visualizing regions of interest based on morphology. For example, detected voids can be classified as round pores or extended cracks. Highly porous neighborhoods can be identified based on local number density. The mesh texture (or color) is assigned based these morphological attributes to allow smart visualization scenarios in real-time (e.g., show only long cracks). At the time of first release (July 2022), extraction of face mesh for visualization for raw CT data from a 2 megapixel camera would take between 1-5 minutes for most scenarios.
</p>  

The code is installable using pip or python.  
```  
cd Tomo2Mesh  
python setup.py install  
```  

The importable files that provide an API for reconstruction, segmentation and mesh generation are in tomo2mesh folder. To use on your data, you will need to implement a workflow using this API. The workflow will input raw CT projections and output a triangular mesh in .ply format with texture based on some context for visualization. Please check out the demos folder for example implementations. Publicly accessible datasets are provided via globus service.  

<p align="center">  
  <img width="800" src="images/illustration.png">  
</p>  

For more information on the algorithms and performance measurements of the code, please read the pre-print of our [paper](https://github.com/aniketkt/Tomo2Mesh/blob/main/images/paper.pdf).


## Developers  
Aniket Tekawade, Yashas Satapathy, Viktor Nikitin  
