"""
Pore size & Thickness measurements
==================================

Local (for every node) measurements of pore size and thickness can be obtained by performing ray-surface intersections between the surface node normals and the surface. 
This test is, by default, bidirectional, meaning it will find intersections in both the outward direction and the inward direction.
Inward intersections correspond to measures of thickness (the rays pass through the "inside" of the surface) while the outward intersections correspond to pore size measurements (the rays pass through the empty pore space and find the next intersection with the object).

Not all points on the surface will have pore size measurements (and for some geometries, such as a solid sphere, none will). 
For a hollow sphere, both the inner and outer surfaces will have measures of local thickness, the inner surface will have pore sizes approximately corresponding to the diameter of the spherical pore, and the outer surface won't have any pore size measurements

"""
#%% 
# Hollow Sphere
# -------------
import mymesh
from mymesh import rays, implicit
import numpy as np

hollow_sphere = implicit.diff(implicit.sphere([0,0,0],.5), 
                              implicit.sphere([0,0,0],.4))
HollowSphere = implicit.SurfaceMesh(hollow_sphere, [-1,1,-1,1,-1,1], 0.05)
HollowSphere.verbose = False

Pts = HollowSphere.NodeCoords
Rays = HollowSphere.NodeNormals

intersections, distances, ix_pts = rays.RaysSurfIntersection(Pts, Rays,  
                     HollowSphere.NodeCoords, HollowSphere.NodeConn,
                     Octree='generate')

# Identify the nearest intersections on either side of the surface
eps = 1e-14 # a small tolerance value to filter out near-zero self intersections
HollowSphere.NodeData['thickness'] = np.array([np.min(np.abs(d[d < -eps])) if np.any(d < -eps) else 0 for d in distances])
HollowSphere.NodeData['pore'] = np.array([np.min(d[d > eps]) if np.any(d > eps) else 0 for d in distances])

HollowSphere.Clip().plot(scalars='thickness', view='trimetric', clim=(.09, .11))
HollowSphere.Clip().plot(scalars='pore', view='trimetric')

#%% 
# Stanford Bunny
# --------------
threshold = 100
bunny_img = mymesh.demo_image('bunny') 
voxelsize = np.array((0.337891, 0.337891, 0.5)) # (mm)

# create a surface mesh of the imaged-object
bunny_surf = mymesh.image.SurfaceMesh(bunny_img, voxelsize, threshold,
                                      scalefactor=0.25) 
bunny_surf.verbose = False


Pts = bunny_surf.NodeCoords
Rays = bunny_surf.NodeNormals

intersections, distances, ix_pts = rays.RaysSurfIntersection(Pts, Rays,  
                     bunny_surf.NodeCoords, bunny_surf.NodeConn,
                     Octree='generate'
               )

# Identify the nearest intersections on either side of the surface
eps = 1e-14 # a small tolerance value to filter out near-zero self intersections
bunny_surf.NodeData['thickness'] = np.array([np.min(np.abs(d[d < -eps])) if np.any(d < -eps) else 0 for d in distances])
bunny_surf.NodeData['pore'] = np.array([np.min(d[d > eps]) if np.any(d > eps) else 0 for d in distances])

bunny_surf.plot(scalars='thickness', view='-x-z', clim=(0, 10))
bunny_surf.Clip(normal=[0,1,0]).plot(scalars='pore', view='-x-z', clim=(0, 150))
# sphinx_gallery_thumbnail_number = 2
# %%
