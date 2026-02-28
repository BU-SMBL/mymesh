"""
.. currentmodule:: mymesh

Coarsening
==========

Mesh coarsening can be performed as part of a mesh quality improvement process,
or to reduce the resolution/complexity of a mesh for various other purposes.
The :func:`~mymesh.improvement.Contract` function supports coarsening of either
tetrahedral or triangular meshes by contracting edges to achieve the desired
edge length. For non-triangular or non-tetrahedral meshes, they can be first 
converted using :func:`~mymesh.converter.solid2tets` or :func:`~mymesh.converter.surf2tris` (see also :ref:`Conversion`).

See also: :ref:`Edge Contraction`, :func:`~mymesh.improvement.Contract`.

"""
#%%
import mymesh
from mymesh import *
import numpy as np

#%% 
# Surface Mesh Coarsening
# -----------------------

threshold = 100
bunny_img = mymesh.demo_image('bunny')
voxelsize = (0.337891, 0.337891, 0.5)

bunny_surf = image.SurfaceMesh(bunny_img, voxelsize, threshold, scalefactor=0.5, method='mc33')

# The feature angle option can be used to preserve sharp edges/corners by
# limiting coarsening in those areas, here it's disabled. 
bunny_coarse = improvement.Contract(bunny_surf, 8, FeatureAngle=None, verbose=True)

bunny_surf.plot(view='-x-z')
bunny_coarse.plot(view='-x-z')

#%%
# Label-preserving coarsening
# ---------------------------
# Labels can be assigned to different regions of the mesh to preserve the 
# interfaces between the regions during coarsening. This can be useful for
# preserving interfaces in multi-material meshes for finite element simulations

# Create a spherical mesh
S = implicit.TetMesh(implicit.sphere([0,0,0], 1), [-1,1,-1,1,-1,1], .1)

# Embed a torus in the mesh
S.NodeData['torus'] = implicit.torus([0,0,0],1,.5)(*S.NodeCoords.T)
S1 = S.Contour('torus', 0, threshold_direction=1, mixed_elements=False)
S1.ElemData['labels'] = np.zeros(S1.NElem)
S2 = S.Contour('torus', 0, threshold_direction=-1, mixed_elements=False)
S2.ElemData['labels'] = np.ones(S2.NElem)
S1.merge(S2)

# Coarsen
Sc = improvement.Contract(S1, 0.2, labels='labels', verbose=True)

visualize.Subplot((S1, Sc, S1.Clip(), Sc.Clip()), (2,2), scalars='labels',
                   show_edges=True, titles=['Original', 'Coarsened', '', ''],
                   view='-yz')

# sphinx_gallery_thumbnail_number = 2