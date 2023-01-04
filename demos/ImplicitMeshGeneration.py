#%% Imports:
import sys, os
import numpy as np
sys.path.append('..\..')
from Mesh import *  
#%% Note on importing the Mesh package:
# When working from a different directory, to import
# the Mesh package, you must first add the parent directoy
# to the system path. If the files of Mesh are located at
# <path>/Mesh, then run the below code to import all files:
# import sys
# sys.path.append(<path>)
# from Mesh import *

#%% Define some parameters for the structure
L = 1 # Length scale for the TPMS periodicity
tFactor = 1 # Thickness factor for the scaffold. Surface will be offset to +/- tFactor/2
bounds = [0,1,0,1,0,1] # Bounds for the structure 
h = 0.04 # Element size
#%% Define a function for your implicit surface, this example uses the primitive TPMS
# Implicit function for the primitive TPMS
def func(x,y,z):
    # This function defines a function where negative values indicate the 'inside' of the surface 
    # and positive values indicate the 'outside' of the surface
    p = np.cos(2*np.pi*x/L) + np.cos(2*np.pi*y/L) + np.cos(2*np.pi*z/L)
    # Offsetting the surface in the positive and negative directions
    offp = ImplicitMesh.offset(p,tFactor/2)
    offn = ImplicitMesh.offset(p,-tFactor/2)
    # Join the offset surfaces
    f = ImplicitMesh.intersection(-ImplicitMesh.diff(offn,offp),ImplicitMesh.cube(x,y,z,*bounds))
    return f
#%% Evaluate the function on a uniform mesh grid
xlims = (bounds[0]-h,bounds[1]+h) # Add 1 voxel of padding to the bounds 
ylims = (bounds[2]-h,bounds[3]+h)
zlims = (bounds[4]-h,bounds[5]+h)
NodeCoords, NodeConn, NodeVals = ImplicitMesh.VoxelMesh(func,xlims,ylims,zlims,h,mode='notrim')
# Create a mesh object:
grid = mesh(NodeCoords,NodeConn)
grid.NodeData['v'] = NodeVals
# To write a file for visualization in paraview: grid.write(<path>/grid.vtu)
#%% Triangulate the surface
TriCoords,TriConn = MarchingCubes.MarchingCubes(grid.NodeCoords,grid.NodeConn,grid.NodeData['v'])
tri = mesh(TriCoords,TriConn)
# To write a file for visualization in paraview: tri.write(<path>/'surf.vtu')
#%% Improve the surface mesh:
tri.NodeCoords,tri.NodeConn = ImplicitMesh.SurfFlowOptimization(func,*tri,ZRIter=20,NZRIter=0,NZIter=0,Subdivision=False)
# To write a file for visualization in paraview: tri.write(<path>/'surf2.vtu')


#%% Generate a tetrahedral mesh with tetgen
tetcoords,tetconn = TetGen.tetgen(tri.NodeCoords, tri.NodeConn, switches=['-pq1.1/25','-Y','-o/150'], BoundingBox=False)
tet = mesh(tetcoords,tetconn)
# To write a file for visualization in paraview: tet.write(<path>/'tet.vtu')

# %%
