Image-based Meshing
===================


Images in MyMesh
----------------
MyMesh can read from a variety of image file types, including DICOMs, tiffs,
jpgs, pngs, and more, utilizing image file readers 
`pydicom <https://pydicom.github.io/pydicom/stable/index.html>`_,
`tifffile <https://www.cgohlke.com/>`_, and
`opencv-python <https://github.com/opencv/opencv-python>`_. 

Images stored as a directory (or "z-stack") of two dimensional images, 
single-file 3D images (e.g. from the OME-TIFF format), or numpy arrays of 3D 
image data can all be read by :func:`mymesh.image.read` to be obtain a numpy
array of image data [#f1]_ (or tuple of arrays for multi-channel data), or passed 
directly to a mesh generator function (e.g. :func:`mymesh.image.VoxelMesh`). 
Images loaded from a directory containing a stack of image slices will be read
in lexicographical order by the image file names, and the files must have the 
proper file extensions.

Images stored in numpy arrays are assumed to be ordered such that their three
dimensions (0,1,2) correspond to (z,y,x). For example, if 
:func:`mymesh.image.read` is used to read a directory containing a stack of 
images and produces and array :code:`I`, then :code:`I[0,:,:]` will correspond
to the data contained by the first image in the stack. 

.. [#f1]
    A numpy array passed to :func:`mymesh.image.read` will only be modified 
    if the :code:`scalefactor` input is used to down- or up-sample the image,
    otherwise it will be returned unchanged.