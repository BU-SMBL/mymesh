"""
Pasta Shapes
============

This example creates pasta shapes with mymesh.primitives using sweep 
construction methods (:func:`~mymesh.primitives.Extrude` and 
:func:`~mymesh.primitives.Revolve`). These demonstrate how meshes 
of complex shapes can be constructed starting with a single straight line.

Please submit any new pasta constructions to toj@bu.edu
"""
#%%
from mymesh.primitives import Line, Revolve, Extrude
import numpy as np

#%%
ziti = Extrude(
            Revolve(
                Line([0,0,0.9],[0,0,1], n=5), 
                2*np.pi, np.pi/36, axis=0), 
        5, 1, axis=0)
ziti.plot(bgcolor='w', color='#EBCB8B')

penne = Extrude(
            Revolve(
                Line([0,0,0.9],[0,0,1], n=5), 
                2*np.pi, np.pi/36, axis=[1,1,1]), 
        5, 1, axis=0)
penne.plot(bgcolor='w', color='#EBCB8B')


gomiti = Revolve( # or elbow/macaroni
            Revolve(
                Line([0,0,1.5],[0,0,1.55], n=5), 
                2*np.pi, np.pi/36, axis=0, center=[0,0,1.25]), 
        np.pi, np.pi/36, axis=1, center=[0,0,0.5])
gomiti.plot(bgcolor='w', color='#EBCB8B')

fusilli = Revolve( # fusilli bucati to be specific
            Revolve(
                Line([0,0,.5], [0,0,.7], n=5),
                2*np.pi, np.pi/36, axis=1, center=[0,0,.5]),
        14*np.pi, np.pi/36, shift=5, axis=0)
fusilli.plot(bgcolor='w', color='#EBCB8B')
# sphinx_gallery_thumbnail_number = 4

# %%
