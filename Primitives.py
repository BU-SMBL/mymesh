
"""
Created Sept 2022

@author: toj
"""
import numpy as np
import gc
from . import *

def Box(bounds,h,meshobj=False):
    """
    Box Generate a triangular surface mesh of a rectangular box. 

    Parameters
    ----------
    bounds : list
        Six element list, [xmin,xmax,ymin,ymax,zmin,zmax].
    h : float
        Approximate element size.
    meshobj : bool, optional
        If true, will return a mesh object, if false, will return 
        NodeCoords and NodeConn. By default False.

    Returns
    -------
    BoxCoords : list
        Node coordinates of the mesh. Returned if meshobj = False.
    BoxConn : list
        Nodal connectivities of the mesh. Returned if meshobj = False.
    Box : Mesh.mesh
        Mesh object containing the box mesh. Returned if meshobj = True.
    """    
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

def Grid(bounds, h, meshobj=False, exact_h=False):
    """
    Grid Generate a rectangular grid mesh.

    Parameters
    ----------
    bounds : list
        Six element list, [xmin,xmax,ymin,ymax,zmin,zmax].
    h : float
        Element size.
    meshobj : bool, optional
        If true, will return a mesh object, if false, will return 
        NodeCoords and NodeConn. By default False.
    exact_h : bool, optional
        If true, will make a mesh where the element size is exactly
        equal to what is specified, but the upper bounds may vary slightly.
        Otherwise, the bounds will be strictly enforced, but element size may deviate
        from the specified h. This may result in non-cubic elements. By default False.

    Returns
    -------
    GridCoords : list
        Node coordinates of the mesh. Returned if meshobj = False.
    GridConn : list
        Nodal connectivities of the mesh. Returned if meshobj = False.
    Grid : Mesh.mesh
        Mesh object containing the grid mesh. Returned if meshobj = True.
    """    

    
    if exact_h:
        xs = np.arange(bounds[0],bounds[1]+h,h)
        ys = np.arange(bounds[2],bounds[3]+h,h)
        zs = np.arange(bounds[4],bounds[5]+h,h)
        nX = len(xs)
        nY = len(ys)
        nZ = len(zs)
    else:
        nX = int(np.round((bounds[1]-bounds[0])/h))+1
        nY = int(np.round((bounds[3]-bounds[2])/h))+1
        nZ = int(np.round((bounds[5]-bounds[4])/h))+1
        xs = np.linspace(bounds[0],bounds[1],nX)
        ys = np.linspace(bounds[2],bounds[3],nY)
        zs = np.linspace(bounds[4],bounds[5],nZ)
        

    # VoxelCoords = np.hstack([
    #     np.repeat(xs,len(ys)*len(zs))[:,None],
    #     np.tile(np.repeat(ys,len(xs)),len(zs)).flatten()[:,None],
    #     np.tile(np.tile(zs,len(xs)).flatten(),len(ys)).flatten()[:,None]
    # ])
    VoxelCoords = np.hstack([
        np.repeat(xs,len(ys)*len(zs))[:,None],
        np.tile(np.repeat(ys,len(zs)),len(xs)).flatten()[:,None],
        np.tile(np.tile(zs,len(xs)).flatten(),len(ys)).flatten()[:,None]
    ])

    Ids = np.reshape(np.arange(len(VoxelCoords)),(nX,nY,nZ))
    
    VoxelConn = np.zeros(((nX-1)*(nY-1)*(nZ-1),8),dtype=int)

    VoxelConn[:,0] = Ids[:-1,:-1,:-1].flatten()
    VoxelConn[:,1] = Ids[1:,:-1,:-1].flatten()
    VoxelConn[:,2] = Ids[1:,1:,:-1].flatten()
    VoxelConn[:,3] = Ids[:-1,1:,:-1].flatten()
    VoxelConn[:,4] = Ids[:-1,:-1,1:].flatten()
    VoxelConn[:,5] = Ids[1:,:-1,1:].flatten()
    VoxelConn[:,6] = Ids[1:,1:,1:].flatten()
    VoxelConn[:,7] = Ids[:-1,1:,1:].flatten()
    
    if meshobj:
        from . import mesh
        return mesh(VoxelCoords,VoxelConn)
    return VoxelCoords, VoxelConn

def Sphere(center,radius,h):
    

    func = lambda x,y,z : ImplicitMesh.sphere(x,y,z,radius,center)
    

