#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOImage import (
    vtkBMPWriter,
    vtkJPEGWriter,
    vtkPNGWriter,
    vtkPNMWriter,
    vtkPostScriptWriter,
    vtkTIFFWriter
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkWindowToImageFilter
)

import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
import numpy as np
import time
import os

'''
source:
https://meshparty.readthedocs.io/en/latest/_modules/meshparty/trimesh_vtk.html
'''
def numpy_to_vtk_cells(mat):
    """function to convert a numpy array of integers to a vtkCellArray

    Parameters
    ----------
    mat : np.array
        MxN array to be converted

    Returns
    -------
    vtk.vtkCellArray
        representing the numpy array, has the same shaped cell (N) at each of the M indices

    """

    cells = vtk.vtkCellArray()

    # Seemingly, VTK may be compiled as 32 bit or 64 bit.
    # We need to make sure that we convert the trilist to the correct dtype
    # based on this. See numpy_to_vtkIdTypeArray() for details.
    isize = vtk.vtkIdTypeArray().GetDataTypeSize()
    req_dtype = np.int32 if isize == 4 else np.int64
    n_elems = mat.shape[0]
    n_dim = mat.shape[1]
    cells.SetCells(n_elems,
                   numpy_to_vtkIdTypeArray(
                       np.hstack((np.ones(n_elems)[:, None] * n_dim,
                                  mat)).astype(req_dtype).ravel(),
                       deep=1))
    return cells



def update_vis_stream(surf, parent_path):

    verts, faces, color = surf["vertices"], surf["faces"], surf["texture"]

    t = time.time()
    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(verts, deep=1))
    mesh.SetPoints(points)
    cells = numpy_to_vtk_cells(faces)
    mesh.SetPolys(cells)
    vtk_colors = numpy_to_vtk(color/255.0)
    # print(color)
    mesh.GetPointData().SetScalars(vtk_colors)
    print(f"time numpy to vtk object: {time.time() - t} seconds")



    colors = vtkNamedColors()

    # Set the background color.
    colors.SetColor('BkgColor', [26, 51, 102, 255])

    # create a rendering window and renderer
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)

    # # create a renderwindowinteractor
    # iren = vtkRenderWindowInteractor()
    # iren.SetRenderWindow(renWin)


    # mapper
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    
    # actor
    actor = vtkActor()
    actor.SetMapper(mapper)
    # actor.RotateX(90)
    # actor.RotateZ(90)

    # color the actor
    actor.GetProperty().SetColor(colors.GetColor3d('Yellow'))

    # assign actor to the renderer
    ren.AddActor(actor)
    ren.SetBackground(colors.GetColor3d('BkgColor'))

    # camera = vtk.vtkCamera()
    # camera.SetViewUp(-0.1377587370775003, 0.6088041868516177, 0.6088041868516177)
    # camera.SetPosition(747.8318948362745, -838.5552837914774, 1113.3030171868118)
    # camera.SetFocalPoint(306.0000000000005, 306.0000000000005, 143.5)
    # # camera.
    #     # self.camera.SetClippingRange(0.0, 100000)

    # ren.SetActiveCamera(camera) 

    renWin.SetWindowName('ImageWriter')
    renWin.SetSize(1024,1024)
    renWin.Render()


    ext = ['', '.png', '.jpg', '.ps', '.tiff', '.bmp', '.pnm']
    filenames = list(map(lambda x: os.path.join(parent_path,'ImageWriter') + x, ext))
    filenames[0] = filenames[0] + '1'
    for f in filenames:
        WriteImage(f, renWin, rgba=False)
  # create a renderwindowinteractor
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    iren.Initialize()
    iren.Start()
    

def WriteImage(fileName, renWin, rgba=True):
    '''
    Write the render window view to an image file.

    Image types supported are:
     BMP, JPEG, PNM, PNG, PostScript, TIFF.
    The default parameters are used for all writers, change as needed.

    :param fileName: The file name, if no extension then PNG is assumed.
    :param renWin: The render window.
    :param rgba: Used to set the buffer type.
    :return:
    '''

    import os

    if fileName:
        # Select the writer to use.
        path, ext = os.path.splitext(fileName)
        ext = ext.lower()
        if not ext:
            ext = '.png'
            fileName = fileName + ext
        if ext == '.bmp':
            writer = vtkBMPWriter()
        elif ext == '.jpg':
            writer = vtkJPEGWriter()
        elif ext == '.pnm':
            writer = vtkPNMWriter()
        elif ext == '.ps':
            if rgba:
                rgba = False
            writer = vtkPostScriptWriter()
        elif ext == '.tiff':
            writer = vtkTIFFWriter()
        else:
            writer = vtkPNGWriter()

        windowto_image_filter = vtkWindowToImageFilter()
        windowto_image_filter.SetInput(renWin)
        windowto_image_filter.SetScale(1)  # image quality
        if rgba:
            windowto_image_filter.SetInputBufferTypeToRGBA()
        else:
            windowto_image_filter.SetInputBufferTypeToRGB()
            # Read from the front buffer.
            windowto_image_filter.ReadFrontBufferOff()
            windowto_image_filter.Update()

        writer.SetFileName(fileName)
        writer.SetInputConnection(windowto_image_filter.GetOutputPort())
        writer.Write()
    else:
        raise RuntimeError('Need a filename.')




