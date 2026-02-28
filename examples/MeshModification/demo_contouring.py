"""
.. currentmodule:: mymesh

Contouring
==========

Contouring can be used to create meshes from images, implicit functions, or other scalar fields. 
It can also be used to modify existing meshes, for example by evaluating  implicit functions at the nodes of the mesh.

See also: :ref:`theory_contour`, :mod:`mymesh.contour`, :ref:`Constructive Solid Geometry`

"""
#%%
from mymesh import primitives, implicit, image, demo_image

#%%
# Starting with a cube, an implicit function of a sphere can be evaluated at 
# all nodes
cube = primitives.Grid([-0.9,0.9,-0.9,0.9,-0.9,0.9], .05)
func = implicit.sphere([0,0,0],1.1)

cube.NodeData['sphere'] = func(cube.NodeCoords[:,0], 
                               cube.NodeCoords[:,1], 
                               cube.NodeCoords[:,2])

cube.plot(scalars='sphere', clim=(-1.25, 1.25))
#%%
# Choosing a threshold of 0, which corresponds to the surface of our implicit 
# sphere, we can then use the :meth:`~mymesh.mesh.mesh.Contour` method to 
# "cut out" the sphere from the cube.
# Alternatively, one of several functions in the :mod:`~mymesh.contour` module
# can be used directly.

threshold = 0 
contoured = cube.Contour('sphere', threshold)
contoured.plot()
# %%
# The same approach can be used with more complicated shapes, for example 
# we can take the `Stanford bunny <https://graphics.stanford.edu/data/voldata/voldata.html#bunny>`_.
# and pattern it with a triply periodic minimal surface.
bunny_img = demo_image('bunny')

voxelsize = (0.337891, 0.337891, 0.5) # mm
threshold = 100
bunny_tet = image.TetMesh(bunny_img, voxelsize, threshold, scalefactor=0.5, interpolation='linear', voxel_mode='elem')
bunny_tet.plot(view='-x-z')

fischer_koch_S = implicit.wrapfunc(implicit.thicken(implicit.tpms('S', 20), 1))
bunny_tet.NodeData['f'] = fischer_koch_S(*bunny_tet.NodeCoords.T)
bunny_tet.plot(scalars='f', view='-x-z')

#%%
bunny_tetS = bunny_tet.Contour('f', 0, threshold_direction=-1)
bunny_tetS.plot(view='-x-z')
# sphinx_gallery_thumbnail_number = 5