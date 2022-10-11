
"""
Created Sept 2022

@author: toj
"""
import numpy as np
from . import *

def Box(bounds,h,meshobj=False):
    
    nX = int(np.round((bounds[1]-bounds[0])/h))
    nY = int(np.round((bounds[3]-bounds[2])/h))
    nZ = int(np.round((bounds[5]-bounds[4])/h))

    xs = np.linspace(bounds[0],bounds[1],nX)
    ys = np.linspace(bounds[2],bounds[3],nY)
    zs = np.linspace(bounds[4],bounds[5],nZ)
    X, Y, Z = np.meshgrid(xs,ys,zs,indexing='ij')
    Xshape = X.shape
    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    ids = np.arange(len(x))
    Ids = np.reshape(ids,Xshape)
    VoxelCoords = np.vstack([x,y,z]).transpose()
    VoxelConn = [[] for i in range(nX-1) for j in range(nY-1) for k in range(nZ-1)] 
    l = 0
    for i in range(nX-1):
        for j in range(nY-1):
            for k in range(nZ-1):
                idxs = [Ids[i,j,k],Ids[i+1,j,k],Ids[i+1,j+1,k],Ids[i,j+1,k],Ids[i,j,k+1],Ids[i+1,j,k+1],Ids[i+1,j+1,k+1],Ids[i,j+1,k+1]]
                VoxelConn[l] = idxs
                l += 1
    BoxCoords,VoxelConn,_ = converter.removeNodes(VoxelCoords,VoxelConn)
    BoxConn = converter.quad2tri(converter.solid2surface(BoxCoords,VoxelConn))
    Box = mesh(BoxCoords,BoxConn)
    Box.cleanup()
    if meshobj:
        return Box
    return BoxCoords,BoxConn

def Sphere(center,radius,h):
    

    func = lambda x,y,z : ImplicitMesh.sphere(x,y,z,radius,center)
    

