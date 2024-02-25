
"""
Created Sept 2022

@author: toj
"""
import numpy as np
import gc
from . import converter, implicit, mesh, delaunay

def Box(bounds, h, meshobj=True, ElemType='quad'):
    """
    Generate a surface mesh of a rectangular box. 

    Parameters
    ----------
    bounds : list
        Six element list of bounds [xmin,xmax,ymin,ymax,zmin,zmax].
    h : float
        Approximate element size.
    meshobj : bool, optional
        If true, will return a mesh object, if false, will return 
        NodeCoords and NodeConn. By default False.
    ElemType : str, optional
        Specify the element type of the grid mesh. This can either be 'quad' for 
        a quadrilateral mesh or 'tri' for a triangular mesh, by default 'quad'.

    Returns
    -------
    BoxCoords : list
        Node coordinates of the mesh. Returned if meshobj = False.
    BoxConn : list
        Nodal connectivities of the mesh. Returned if meshobj = False.
    box : Mesh.mesh
        Mesh object containing the box mesh. Returned if meshobj = True (default).
    """    
    GridCoords, GridConn = Grid(bounds,h,exact_h=False)
    BoxConn = converter.solid2surface(GridCoords,GridConn)
    BoxCoords,BoxConn,_ = converter.removeNodes(GridCoords,BoxConn)
    if ElemType == 'tri':
        BoxConn = converter.quad2tri(BoxConn)
    if meshobj:
        if 'mesh' in dir(mesh):
            box = mesh.mesh(BoxCoords,BoxConn)
        else:
            box = mesh(BoxCoords,BoxConn)
        box.Type = 'surf'
        box.cleanup()
        return box
    return BoxCoords,BoxConn

def Grid(bounds, h, meshobj=True, exact_h=False, ElemType='hex'):
    """
    Generate a 3D rectangular grid mesh.

    Parameters
    ----------
    bounds : list
        Six element list, of bounds [xmin,xmax,ymin,ymax,zmin,zmax].
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
    ElemType : str, optional
        Specify the element type of the grid mesh. This can either be 'hex' for 
        a hexahedral mesh or 'tet' for a tetrahedral mesh, by default 'hex'.

    Returns
    -------
    GridCoords : list
        Node coordinates of the mesh. Returned if meshobj = False.
    GridConn : list
        Nodal connectivities of the mesh. Returned if meshobj = False.
    Grid : Mesh.mesh
        Mesh object containing the grid mesh. Returned if meshobj = True (default).
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
        

    GridCoords = np.hstack([
        np.repeat(xs,len(ys)*len(zs))[:,None],
        np.tile(np.repeat(ys,len(zs)),len(xs)).flatten()[:,None],
        np.tile(np.tile(zs,len(xs)).flatten(),len(ys)).flatten()[:,None]
    ])

    Ids = np.reshape(np.arange(len(GridCoords)),(nX,nY,nZ))
    
    GridConn = np.zeros(((nX-1)*(nY-1)*(nZ-1),8),dtype=int)

    GridConn[:,0] = Ids[:-1,:-1,:-1].flatten()
    GridConn[:,1] = Ids[1:,:-1,:-1].flatten()
    GridConn[:,2] = Ids[1:,1:,:-1].flatten()
    GridConn[:,3] = Ids[:-1,1:,:-1].flatten()
    GridConn[:,4] = Ids[:-1,:-1,1:].flatten()
    GridConn[:,5] = Ids[1:,:-1,1:].flatten()
    GridConn[:,6] = Ids[1:,1:,1:].flatten()
    GridConn[:,7] = Ids[:-1,1:,1:].flatten()

    if ElemType == 'tet':
        GridCoords, GridConn = converter.hex2tet(GridCoords, GridConn, method='1to6')
    
    if meshobj:
        if 'mesh' in dir(mesh):
            Grid = mesh.mesh(GridCoords,GridConn,'vol')
        else:
            Grid = mesh(GridCoords,GridConn,'vol')
        return Grid
    return GridCoords, GridConn

def Grid2D(bounds, h, z=0, meshobj=True, exact_h=False, ElemType='quad'):
    """
    Generate a rectangular grid mesh.

    Parameters
    ----------
    bounds : list
        Four element list, [xmin,xmax,ymin,ymax].
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
        Mesh object containing the grid mesh. Returned if meshobj = True (default).
    """    
    if type(h) is tuple or type(h) is list or type(h) is np.ndarray:
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

def Plane(pt, normal, bounds, h, meshobj=True, exact_h=False, ElemType='quad'):
    """
    Generate a 2D grid oriented on a plane

    Parameters
    ----------
    pt : list, np.ndarray
        Coordinates (x,y,z) of a point on the plane
    normal : list, np.ndarray
        Normal vector of the plane
    bounds : list
        Six element list, [xmin,xmax,ymin,ymax,zmin,zmax].
    h : _type_
        _description_
    meshobj : bool, optional
        _description_, by default True
    exact_h : bool, optional
        _description_, by default False
    ElemType : str, optional
        _description_, by default 'quad'

    Returns
    -------
    PlaneCoords : list
        Node coordinates of the mesh. Returned if meshobj = False.
    PlaneConn : list
        Nodal connectivities of the mesh. Returned if meshobj = False.
    plane : Mesh.mesh
        Mesh object containing the plane mesh. Returned if meshobj = True (default).
    """
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
    PlaneConn = GridConn


    if meshobj:
        if 'mesh' in dir(mesh):
            plane = mesh.mesh(PlaneCoords,PlaneConn,'surf')
        else:
            plane = mesh(PlaneCoords,PlaneConn,'surf')
        return plane
    return PlaneCoords, PlaneConn

def Extrude(line, distance, step, axis=2, ElemType='quad', meshobj=True):
    """
    Extrude a 2D line mesh to a 3D surface

    Parameters
    ----------
    line : mesh
        mesh object of 2D 
    distance : _type_
        _description_
    step : _type_
        _description_
    axis : int, optional
        _description_, by default 2
    ElemType : str, optional
        _description_, by default 'quad'
    meshobj : bool, optional
        _description_, by default True

    Returns
    -------
    NodeCoords : list
        Node coordinates of the mesh. Returned if meshobj = False.
    NodeConn : list
        Nodal connectivities of the mesh. Returned if meshobj = False.
    extruded : Mesh.mesh
        Mesh object containing the extruded mesh. Returned if meshobj = True (default).
    """    
    NodeCoords = np.array(line.NodeCoords)
    OriginalConn = np.array(line.NodeConn)
    NodeConn = np.empty((0,4))
    for i,s in enumerate(np.arange(step,distance+step,step)):
        temp = np.array(line.NodeCoords)
        temp[:,axis] += s
        NodeCoords = np.append(NodeCoords, temp, axis=0)

        NodeConn = np.append(NodeConn, np.hstack([OriginalConn+(i*len(temp)),np.fliplr(OriginalConn+((i+1)*len(temp)))]), axis=0)
        NodeConn = NodeConn.astype(int)
    if ElemType == 'tri':
        NodeConn = converter.quad2tri(NodeConn)
    if meshobj:
        if 'mesh' in dir(mesh):
            extruded = mesh.mesh(NodeCoords,NodeConn,'surf')
        else:
            extruded = mesh(NodeCoords,NodeConn,'surf')
        return extruded
    return NodeCoords, NodeConn

def Cylinder(bounds, resolution, axis=2, axis_step=None, meshobj=True):
    """
    Generate an axis-aligned cylindrical surface mesh

    Parameters
    ----------
    ----------
    bounds : list
        Six element list of bounds [xmin,xmax,ymin,ymax,zmin,zmax].
        NodeCoords and NodeConn. By default False.
    resolution : int
        Number of points in the circumference of the cylinder
    axis : int, optional
        Long axis of the cylinder (i.e. the circular ends will lie in the plane of the other two axes).
        Must be 0, 1, or 2 (x, y, z), by default 2
    axis_step : float, optional
        Element size in the <axis> direction, by default it will be set to the full length of the cylinder.
    meshobj : bool, optional
        If true, will return a mesh object, if false, will return 

    Returns
    -------
    NodeCoords : list
        Node coordinates of the mesh. Returned if meshobj = False.
    NodeConn : list
        Nodal connectivities of the mesh. Returned if meshobj = False.
    cyl : Mesh.mesh
        Mesh object containing the cylinder mesh. Returned if meshobj = True (default).

    """    
    bounds = np.asarray(bounds)
    if axis == 2:
        lbounds = bounds
        order = [0,1,2]
    elif axis == 1:
        lbounds = bounds[[0,1,4,5,2,3]]
        order = [0,2,1]
    elif axis == 0:
        lbounds = bounds[[4,5,2,3,0,1]]
        order = [2,1,0]
    else:
        raise Exception('Axis must be 0 (x), 1 (y), or 2 (z).')
    
    height = lbounds[5] - lbounds[4]
    if axis_step is None:
        axis_step = height

    a = (lbounds[1] - lbounds[0])/2   
    b = (lbounds[3] - lbounds[2])/2   
    ashift = lbounds[0] + a
    bshift = lbounds[2] + b

    t = np.linspace(0,2*np.pi,resolution+1)
    t = np.append(t, t[0])

    x = a*np.cos(t) + ashift
    y = b*np.sin(t) + bshift
    z = np.repeat(lbounds[4],len(x))
    xyz = [x,y,z]

    coords = np.column_stack(xyz)[:,order]
    conn = np.column_stack([np.arange(0,len(t)-1), np.arange(1,len(t))])

    if 'mesh' in dir(mesh):
        line = mesh.mesh(coords, conn)
    else:
        line = mesh(coords, conn)
    cyl = Extrude(line, height, axis_step, axis=axis, ElemType='tri')

    capconn = delaunay.ConvexHullFanTriangulation(np.arange(line.NNode))

    if 'mesh' in dir(mesh):
        cap1 = mesh.mesh(line.NodeCoords, np.fliplr(capconn))
        cap2 = mesh.mesh(np.copy(line.NodeCoords), capconn)
    else:
        cap1 = mesh(line.NodeCoords, np.fliplr(capconn))
        cap2 = mesh(np.copy(line.NodeCoords), capconn)
    cap2.NodeCoords[:,axis] += height

    cyl.merge(cap1)
    cyl.merge(cap2)
    cyl.cleanup()

    if meshobj:
        return cyl
    return cyl.NodeCoords, cyl.NodeConn
        
        

