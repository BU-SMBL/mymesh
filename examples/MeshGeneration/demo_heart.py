"""
Implicit Heart
==============

This example generates a surface mesh of an implicit function and then performs 
Laplacian smoothing on it.

Defining an implicit function
-----------------------------
Implicit functions are defined as functions of three arguments (x, y, z), which 
can handle vectorized inputs. 
"""

#%%
from mymesh import implicit, mesh, improvement

def heart(x, y, z):
    # Taubin's Heart Surface
    return (x**2 + 9/4*y**2 + z**2 -1)**3 - x**2*z**3 - 9/80*y**2*z**3

#%%
# Creating a Surface
# ------------------
bounds = [-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]
h = 0.05 # element size
Heart = implicit.SurfaceMesh(heart, bounds, h)
Heart.plot(bgcolor='white', color='red')

#%%
# Smoothing
# ---------
# The heart surface has a bit of a ridge in the middle, this can be eliminated 
# by smoothing the mesh. This will also improve the quality of the elements in 
# the mesh and more evenly distribute the nodes.


Heart = improvement.LocalLaplacianSmoothing(Heart)
Heart.plot(bgcolor='white', color='red')
# sphinx_gallery_thumbnail_number = 2