# %% 
import sys, os
import numpy as np
sys.path.append('../..')
import Mesh

# %% An artificial 3D image of a sphere (i.e. a 3D numpy array of pixel data)
#       Any 3D image can be represented as a 3D matrix of data, which can be converted to a voxel mesh
#       using mesh.imread().
h = (0.1, 0.1, 0.2) # Voxel size
X, Y, Z = np.meshgrid(np.linspace(-1,1,int(2/h[2]+1)),np.linspace(-1,1,int(2/h[1]+1)),np.linspace(-1,1,int(2/h[0]+1)),indexing='ij')
img = Mesh.implicit.sphere(X, Y, Z, 0.75, [0,0,0])

# A full rectangular volume mesh.
#   I is a mesh() object with node coordinates stored in I.NodeCoords, node connectivity stored in I.NodeConn,
#   and voxel intensity values stored in I.ElemData['Image Data']. This mesh can be written to a .vtu, .inp, or 
#   various other mesh file types supported by the meshio package using the command `I.write(<filename>)`. A 
#   scaling factor for up- or downsampling images can be specified using the `scalefactor`` input. scalefactor=1 
#   (default) indicates no change. Values less than 1 will coarsen the image, values higher than 1 will interpolate 
#   voxel values to a finer voxel size. The order of interpolation can be specified through the `scaleorder` input 
#   (can be 1-5, default scaleorder=1), with higher values increasing computational cost. Scaling is done using 
#   scipy.ndimage.zoom(). 
I = Mesh.mesh.imread(img, h, scalefactor=1.1, scaleorder=1)

# A thresholded voxel mesh. 
#   The threshold input specifies a threshold which is used to discard voxels with values above or below the threshold
#   depending on whether threshold_direction is 1 (default) or -1, respectively. The option return_nodedata=True will 
#   average adjacent voxel values to obtain a value at the node. This typically increases runtime, so should only be 
#   used if needed.
Ithresh = Mesh.mesh.imread(img, h, threshold=0, threshold_direction=-1, return_nodedata=True)

# A smooth triangular surface mesh obtained from a voxel mesh.
#   The Marching Cubes (Lorensen & Cline, 1987; Chernyaev, 1995) is used to extract an isosurface from the voxel data
#   The 0 isosurface is chosen by default, but other values can be chosen using the `threshold` input. Nodal image data
#   must be used, requiring the use of the return_nodedata option in `mesh.imread()` (or some other means of assigning 
#   node data). MarchingCubes() returns NodeCoords and NodeConn arrays, which can be directly loaded into a mesh object
#   by unpacking with *.
S = Mesh.mesh(*Mesh.contour.MarchingCubes(Ithresh.NodeCoords, Ithresh.NodeConn, Ithresh.NodeData['Image Data']))

# Surface directly from image.
#   This will generally be more efficient that going to a voxel mesh first. This method also allows for 'linear' or 'cubic' 
#   interpolation (instead of just 'linear' or 'midpoint' with MarchingCubes()) for determining the placement of nodes on the 
#   isosurface. 
S = Mesh.mesh(*Mesh.contour.MarchingCubesImage(img, h, interpolation='cubic', threshold=0))

# %% Loading an image directly from files.
#       Instead of a 3D array of data, image files can be loaded using mesh.imread() by specifying a directory containing
#       a set of 2D images. The 2D images will be loaded in alpha-numeric order so they should be named appropriately. 
#       Currently supported file types are DICOMs (.dcm, .DCM), JPG (.jpg, .jpeg), PNG (.png), and TIFF (.tif, .tiff). 
#       DICOMs are loaded using the pydicom package, the other file types are loaded using opencv-python (cv2). If the 2D 
#       images contain multi-channel data (e.g. RGB), image data will be loaded into I.ElemData['Image Data'] as 
#       separate columns. Note that some file types (e.g. TIFFs) can contain 3D image data in a single file, this is 
#       currently not supported and will be interpreted as multichannel data for a 2D image. For such files, it is 
#       recommended to first load this data using the tifffile package, then use the 3D numpy array as shown above. 

imdir = r"C:\Users\toj\Downloads\SampleImage"
h = (.1,.1,2) # Voxel size, can be specified as a single number for cubic voxels or a tuple for rectangular voxels.
scalefactor = 1 
I = Mesh.mesh.imread(imdir, h, return_nodedata=True, scalefactor=scalefactor)

# %%
