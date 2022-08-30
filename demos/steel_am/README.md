# Porosity mapping in additively manufactured steel part  

**Data contributor:** Dr. Xuan Zhang, Argonne National Laboratory  
**Micro-CT Instrument** 1-ID beamline of Advanced Photon Source (APS)  

<p align="justify">  
This data is of a 3d-printed steel cylindrical specimen (nominal diameter 6-millimeter) scanned under monochromatic hard X-ray. The mosaic comprised of 3 horizontal and 6 vertical positions (total 18 scans). The field-of-view for each scan was approximately 2.2 mm wide and 1.4 mm tall with 1.17 micrometer voxel size. With an exposure time of 0.13 milliseconds for 3000 projections, scanning this mosaic took $\approx$ 2 hours not counting instrument-specific overheads in moving motors to reposition the sample between scans. This resulted in 3000 projections over 180 degrees sample rotation with raw image size 4096$\times$4096 (100 gigabytes of raw data at 16-bit precision) that would reconstruct to a 3D volume with $4096^3$ voxels. Two versions of the raw-data are shared. The original data (referred to as {4k} volume) with 1.17 micrometer pixel size (3000 projections) and a binned version ({2k} volume) with 2.34 micrometer pixel size (1500 projections).  
</p>  

<p align="justify">  
The specimen was printed using a GE Concept Laser Powder Bed Fusion printer, stress-relieved at 650 degC for 1 hour post printing and then creep-tested until rupture at 550 degC, under 275 MPa. The mosaic scans were performed to see through the fractured surface of one of the ruptured pieces. More about the specimen and the test can be found in previous work.  
</p>  

Li, Meimei, Wei-Ying Chen, and Xuan Zhang. "Effect of heat treatment on creep behavior of 316 L stainless steel manufactured by laser powder bed fusion." *Journal of Nuclear Materials* 559 (2022): 153469.  

To run the code, download the *steel_am_demo* folder which contains raw projection data and trained U-net model from the globus endpoint [here](https://app.globus.org/file-manager?destination_id=b6a3fd70-dc14-40ef-8091-dead0062a71e&destination_path=%2FRealTimePorosityMappingSupInfo%2F),

In the tomo2mesh/projects/steel_am/rw_utils.py file, update the base_fpath to wherever your dataset was downloaded:

```
base_fpath = 'path/to/your/folder/steel_am_demo/'
```

then go to the *demos* folder, run:  

```   
python demos/steel_am/tests/smartvis.py  
```   

It should output .ply files that can be viewed in a visualization software of your choice. We recommend Paraview to render the texture correctly. Here, the colormap scales with void size.  


<p align="center">  
  <img width="800" src="../../images/steel_am.png">  
</p>  
