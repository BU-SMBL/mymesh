
"""
Created Sept 2022

@author: toj
"""
import numpy as np
import gc
from . import converter, ImplicitMesh, mesh

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
    GridCoords, GridConn = Grid(bounds,h,exact_h=False)
    BoxConn = converter.quad2tri(converter.solid2surface(GridCoords,GridConn))
    BoxCoords,BoxConn,_ = converter.removeNodes(GridCoords,BoxConn)
    if meshobj:
        if 'mesh' in dir(mesh):
            Box = mesh.mesh(BoxCoords,BoxConn,'surf')
        else:
            Box = mesh(BoxCoords,BoxConn,'surf')
        Box.cleanup()
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
    if type(h) is tuple or type(h) is list:
        hx = h[0];hy = h[1]; hz = h[2]
    else:
        hx = h; hy = h; hz = h
    
    if exact_h:
        xs = np.arange(bounds[0],bounds[1]+hx,hx)
        ys = np.arange(bounds[2],bounds[3]+hy,hy)
        zs = np.arange(bounds[4],bounds[5]+hz,hz)
        nX = len(xs)
        nY = len(ys)
        nZ = len(zs)
    else:
        nX = int(np.round((bounds[1]-bounds[0])/hx))+1
        nY = int(np.round((bounds[3]-bounds[2])/hy))+1
        nZ = int(np.round((bounds[5]-bounds[4])/hz))+1
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
        if 'mesh' in dir(mesh):
            Grid = mesh.mesh(VoxelCoords,VoxelConn,'vol')
        else:
            Grid = mesh(VoxelCoords,VoxelConn,'vol')
        return Grid
    return VoxelCoords, VoxelConn

def Grid2D(bounds, h, z=0, meshobj=False, exact_h=False, ElemType='quad'):
    """
    Grid Generate a rectangular grid mesh.

    Parameters
    ----------
    bounds : list
        Six element list, [xmin,xmax,ymin,ymax].
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
    if type(h) is tuple or type(h) is list:
        hx = h[0];hy = h[1]
    else:
        hx = h; hy = h

    if bounds[0] > bounds[1]:
        bounds[0],bounds[1] = bounds[1], bounds[0]
    if bounds[2] > bounds[3]:
        bounds[2],bounds[3] = bounds[3], bounds[2]
    
    if exact_h:
        xs = np.arange(bounds[0],bounds[1]+hx,hx)
        ys = np.arange(bounds[2],bounds[3]+hy,hy)
        nX = len(xs)
        nY = len(ys)
    else:
        nX = int(np.round((bounds[1]-bounds[0])/hx))+1
        nY = int(np.round((bounds[3]-bounds[2])/hy))+1
        xs = np.linspace(bounds[0],bounds[1],nX)
        ys = np.linspace(bounds[2],bounds[3],nY)

    GridCoords = np.hstack([
        np.repeat(xs,len(ys))[:,None],
        np.tile(ys,len(xs)).flatten()[:,None],
        np.zeros((nX*nY,1))
    ])

    Ids = np.reshape(np.arange(len(GridCoords)),(nX,nY))
    
    GridConn = np.zeros(((nX-1)*(nY-1),4),dtype=int)

    GridConn[:,0] = Ids[:-1,:-1].flatten()
    GridConn[:,1] = Ids[1:,:-1].flatten()
    GridConn[:,2] = Ids[1:,1:].flatten()
    GridConn[:,3] = Ids[:-1,1:].flatten()

    if ElemType == 'tri':
        GridConn = converter.quad2tri(GridConn)
    
    if meshobj:
        if 'mesh' in dir(mesh):
            Grid = mesh.mesh(GridCoords,GridConn,'surf')
        else:
            Grid = mesh(GridCoords,GridConn,'surf')
        return Grid
    return GridCoords, GridConn

def Plane(pt, normal, bounds, h, meshobj=False, exact_h=False, ElemType='quad'):

    # Get rotation between the plane and the xy (z=0) plane
    normal = np.asarray(normal)/np.linalg.norm(normal)
    
    def quat_rotate(rotAxis,angle):
        q = [np.cos(angle/2),               # Quaternion Rotation
                rotAxis[0]*np.sin(angle/2),
                rotAxis[1]*np.sin(angle/2),
                rotAxis[2]*np.sin(angle/2)]
    
        R = [[2*(q[0]**2+q[1]**2)-1,   2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
                [2*(q[1]*q[2]+q[0]*q[3]), 2*(q[0]**2+q[2]**2)-1,   2*(q[2]*q[3]-q[0]*q[1])],
                [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 2*(q[0]**2+q[3]**2)-1]
                ]
        return R
    k = np.array([0,0,1])
    if np.all(normal == k) or np.all(normal == [0,0,-1]):
        R = np.eye(3)
    else:
        kxn = np.cross(k,normal)
        rotAxis = kxn/np.linalg.norm(kxn)
        angle = -np.arccos(np.dot(k,normal))
        R = quat_rotate(rotAxis,angle)

    if np.any(np.abs(normal) != [0,0,1]):
        axis1 = np.cross(normal, [0,0,1])
        axis2 = np.cross(normal, axis1)
    else:
        axis1 = np.cross(normal, [1,0,0])
        axis2 = np.cross(normal, axis1)
    
    axis1 /= np.linalg.norm(axis1)
    axis2 /= np.linalg.norm(axis2)

    BottomCorner = np.array([bounds[0], bounds[2], bounds[4]])
    TopCorner = np.array([bounds[1], bounds[3], bounds[5]])

    diagonal = np.linalg.norm(TopCorner-BottomCorner)
    corner1 = pt + (axis1 * (diagonal/2)) + (axis2 * (diagonal/2))
    corner2 = pt - (axis1 * (diagonal/2)) + (axis2 * (diagonal/2))
    corner3 = pt + (axis1 * (diagonal/2)) - (axis2 * (diagonal/2))
    corner4 = pt - (axis1 * (diagonal/2)) - (axis2 * (diagonal/2))

    Corners = np.array([
        np.clip(corner1, BottomCorner, TopCorner),
        np.clip(corner2, BottomCorner, TopCorner),
        np.clip(corner3, BottomCorner, TopCorner),
        np.clip(corner4, BottomCorner, TopCorner)
    ])
    xyCorners = np.dot(R,Corners.T).T

    n = (xyCorners[0] - xyCorners[1])
    n /= np.linalg.norm(n)
    if np.any(np.abs(n) != [0,1,0]):
        cross = np.cross([0,1,0],n)
        rotAxis = cross/np.linalg.norm(cross)
        angle = -np.arccos(np.dot([0,1,0],n))
        R2 = quat_rotate(rotAxis,angle)
    else:
        R2 = np.eye(3)

    GridCorners = np.dot(R2,xyCorners.T).T
    
    GridBounds = np.array([np.min(GridCorners,axis=0)[0],np.max(GridCorners,axis=0)[0],np.min(GridCorners,axis=0)[1],np.max(GridCorners,axis=0)[1]])

    GridCoords,GridConn = Grid2D(GridBounds, h, z=GridCorners[0,2], exact_h=exact_h, ElemType=ElemType)

    # Rotate xy plane to proper orientation
    PlaneCoords = np.dot(np.linalg.inv(R),np.dot(np.linalg.inv(R2),GridCoords.T)).T

    # Translate
    sd = np.dot(normal,PlaneCoords[0])-np.dot(normal,pt) 
    PlaneCoords = PlaneCoords - sd*normal


    if meshobj:
        if 'mesh' in dir(mesh):
            plane = mesh.mesh(PlaneCoords,GridConn,'surf')
        else:
            plane = mesh(PlaneCoords,GridConn,'surf')
        return plane
    return PlaneCoords, GridConn

# def Sphere(center,radius,h):
    

#     func = lambda x,y,z : ImplicitMesh.sphere(x,y,z,radius,center)
    

