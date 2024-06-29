# -*- coding: utf-8 -*-
# Created Sept 2022
# @author: toj

"""
Mesh generation for pre-defined shapes


.. currentmodule:: mymesh.primitives


Shapes
======
.. autosummary::
    :toctree: submodules/
    
    Box
    Grid
    Grid2D
    Plane
    Cylinder
    Sphere
    Torus

2D to 3D Constructions
======================
.. autosummary::
    :toctree: submodules/

    Extrude
    Revolve

"""
import numpy as np
import gc
from . import utils, converter, implicit, mesh, delaunay

def Box(bounds, h, ElemType='quad'):
    """
    Generate a surface mesh of a rectangular box. 

    Parameters
    ----------
    bounds : list
        Six element list of bounds [xmin,xmax,ymin,ymax,zmin,zmax].
    h : float
        Approximate element size.
    ElemType : str, optional
        Specify the element type of the grid mesh. This can either be 'quad' for a quadrilateral mesh or 'tri' for a triangular mesh, by default 'quad'.

    Returns
    -------
    box : mymesh.mesh
        Mesh object containing the box mesh. 

    .. note:: 
        Due to the ability to unpack the mesh object to NodeCoords and NodeConn, the NodeCoords and NodeConn array can be returned directly (instead of the mesh object) by running: ``NodeCoords, NodeConn = primitives.Box(...)``

    Examples
    --------
    .. plot::

        box = primitives.Box([0,1,0,1,0,1], 0.05, ElemType='tri')
        box.plot(bgcolor='w', show_edges=True)

    """    
    GridCoords, GridConn = Grid(bounds,h,exact_h=False)
    BoxConn = converter.solid2surface(GridCoords,GridConn)
    BoxCoords,BoxConn,_ = utils.RemoveNodes(GridCoords,BoxConn)
    if ElemType == 'tri':
        BoxConn = converter.quad2tri(BoxConn)

    if 'mesh' in dir(mesh):
        box = mesh.mesh(BoxCoords,BoxConn)
    else:
        box = mesh(BoxCoords,BoxConn)
    box.Type = 'surf'
    box.cleanup()
    return box

def Grid(bounds, h, exact_h=False, ElemType='hex'):
    """
    Generate a 3D rectangular grid mesh.

    Parameters
    ----------
    bounds : list
        Six element list, of bounds [xmin,xmax,ymin,ymax,zmin,zmax].
    h : float, tuple
        Element size. If provided as a three element tuple, indicates anisotropic element sizes in each direction.
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
    Grid : mymesh.mesh
        Mesh object containing the grid mesh.
        

    .. note:: 
        Due to the ability to unpack the mesh object to NodeCoords and NodeConn,
        the NodeCoords and NodeConn array can be returned directly (instead of the mesh object)
        by running: ``NodeCoords, NodeConn = primitives.Grid(...)``

    .. plot::

        box = primitives.Grid([0,1,0,1,0,1], 0.05)
        box.plot(bgcolor='w', show_edges=True)

    """    
    if type(h) is tuple or type(h) is list:
        hx = h[0];hy = h[1]; hz = h[2]
    else:
        hx = h; hy = h; hz = h
    if len(bounds) == 4:
        Grid = Grid2D(bounds, h, exact_h=exact_h)
        return Grid
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
    
    if 'mesh' in dir(mesh):
        Grid = mesh.mesh(GridCoords,GridConn,'vol')
    else:
        Grid = mesh(GridCoords,GridConn,'vol')
    return Grid

def Grid2D(bounds, h, z=0, exact_h=False, ElemType='quad'):
    """
    Generate a rectangular grid mesh.

    Parameters
    ----------
    bounds : list
        Four element list, [xmin,xmax,ymin,ymax].
    h : float
        Element size.
    exact_h : bool, optional
        If true, will make a mesh where the element size is exactly
        equal to what is specified, but the upper bounds may vary slightly.
        Otherwise, the bounds will be strictly enforced, but element size may deviate
        from the specified h. This may result in non-square elements. By default False.
    ElemType : str, optional
        Specify the element type of the grid mesh. This can either be 'quad' for 
        a quadrilateral mesh or 'tri' for a triangular mesh, by default 'quad'.

    Returns
    -------
    Grid : mymesh.mesh
        Mesh object containing the grid mesh.
        

    .. note::
        Due to the ability to unpack the mesh object to NodeCoords and NodeConn,
        the NodeCoords and NodeConn array can be returned directly (instead of the mesh object)
        by running: ``NodeCoords, NodeConn = primitives.Grid2D(...)``

    .. plot::

        box = primitives.Grid2D([0,1,0,1,], 0.05)
        box.plot(bgcolor='w', show_edges=True)

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
        z*np.ones((nX*nY,1))
    ])

    Ids = np.reshape(np.arange(len(GridCoords)),(nX,nY))
    
    GridConn = np.zeros(((nX-1)*(nY-1),4),dtype=int)

    GridConn[:,0] = Ids[:-1,:-1].flatten()
    GridConn[:,1] = Ids[1:,:-1].flatten()
    GridConn[:,2] = Ids[1:,1:].flatten()
    GridConn[:,3] = Ids[:-1,1:].flatten()

    if ElemType == 'tri':
        GridConn = converter.quad2tri(GridConn)
    
    if 'mesh' in dir(mesh):
        Grid = mesh.mesh(GridCoords,GridConn,'surf')
    else:
        Grid = mesh(GridCoords,GridConn,'surf')
    return Grid

def Plane(pt, normal, bounds, h, exact_h=False, ElemType='quad'):
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
    h : float
        Element size.
    exact_h : bool, optional
        If true, will make a mesh where the element size is exactly
        equal to what is specified, but the upper bounds may vary slightly.
        Otherwise, the bounds will be strictly enforced, but element size may deviate
        from the specified h. This may result in non-square elements. By default False.
    ElemType : str, optional
        Specify the element type of the grid mesh. This can either be 'quad' for 
        a quadrilateral mesh or 'tri' for a triangular mesh, by default 'quad'.

    Returns
    -------
    plane : mymesh.mesh
        Mesh object containing the plane mesh.
        

    .. note:: 
        Due to the ability to unpack the mesh object to NodeCoords and NodeConn,
        the NodeCoords and NodeConn array can be returned directly (instead of the mesh object)
        by running: ``NodeCoords, NodeConn = primitives.Extrude(...)``

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


    if 'mesh' in dir(mesh):
        plane = mesh.mesh(PlaneCoords,PlaneConn,'surf')
    else:
        plane = mesh(PlaneCoords,PlaneConn,'surf')
    return plane

def Cylinder(bounds, resolution, axis=2, axis_step=None, ElemType='tri', cap=True):
    """
    Generate an axis-aligned cylindrical surface mesh

    Parameters
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
    ElemType : str, optional
        Specify the element type of the walls of the cylinder mesh. This can either be 
        'quad' for a quadrilateral mesh or 'tri' for a triangular mesh, by default 'tri'.
        The ends of the cylinder will be triangular regardless of this input.
    cap : bool, optional
        If True, will close the ends of the cylinder, otherwise it will leave them open, by 
        default True.

    Returns
    -------
    cyl : mymesh.mesh
        Mesh object containing the cylinder mesh.
        

    .. note:: 
        Due to the ability to unpack the mesh object to NodeCoords and NodeConn,
        the NodeCoords and NodeConn array can be returned directly (instead of the mesh object)
        by running: ``NodeCoords, NodeConn = primitives.Cylinder(...)``
    
    Examples
    --------
    .. plot::

        cyl = primitives.Cylinder([0,1,0,1,0,1], 20, axis_step=0.25, axis=0)
        cyl.plot(bgcolor='w', show_edges=True)

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
    cyl = Extrude(line, height, axis_step, axis=axis, ElemType=ElemType)
    if cap:
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

    return cyl
               
def Sphere(center, radius, theta_resolution=10, phi_resolution=10, ElemType='tri'):
    """
    Generate a sphere (or ellipsoid)
    The total number of points will be phi_resolution*(theta_resolution-2) + 2

    Parameters
    ----------
    center : array_like
        Three element array of the coordinates of the center of the sphere.
    radius : scalar or array_like
        The radius of the sphere. Radius can be specified as a scalar radius of the sphere 
        or three element array of half-axes for an ellipsoid. 
    theta_resolution : int, optional
        Number of circular (or elliptical) cross sections sampled along the z axis, by default 10.
    phi_resolution : int, optional
        Number of circumferential points for each cross section, by default 10.
    ElemType : str, optional
        Specify the element type of the mesh. This can either be 'quad' for 
        a quadrilateral mesh or 'tri' for a triangular mesh, by default 'tri'.
        If 'quad' is specified, there will still be some triangles at z axis "poles".

    Returns
    -------
    sphere, mymesh.mesh
        Mesh object containing the cylinder mesh.
        

    .. note:: 
        Due to the ability to unpack the mesh object to NodeCoords and NodeConn, the NodeCoords and NodeConn array can be returned directly (instead of the mesh object) by running: ``NodeCoords, NodeConn = primitives.Sphere(...)``


    Examples
    ________
    .. plot::

        sphere = primitives.Sphere([0,0,0], 1)
        sphere.plot(bgcolor='w', show_edges=True)

    .. plot::

        ellipsoid = primitives.Sphere([0,0,0], (0.5,1,1.5),
        theta_resolution=20, phi_resolution=20)
        ellipsoid.plot(bgcolor='w', show_edges=True)

    """

    if isinstance(radius, (list, tuple, np.ndarray)):
        assert len(radius) == 3, 'radius must either be a scalar or a 3 element array_like.'
    elif np.isscalar(radius):
        radius = np.repeat(radius,3)
    else:
        raise TypeError('radius must either be a scalar or a 3 element array_like.')

    # Create cross section
    t = np.linspace(0,np.pi,theta_resolution)

    x = np.repeat(center[0],len(t))
    y = center[1] + radius[1]*np.sin(t)
    z = center[2] + radius[2]*np.cos(t)
    xyz = [x,y,z]

    coords = np.column_stack(xyz)
    conn = np.column_stack([np.arange(0,len(t)-1), np.arange(1,len(t))])

    if 'mesh' in dir(mesh):
        circle = mesh.mesh(coords, conn)
    else:
        circle = mesh(coords, conn)
    
    # Revolve cross section
    sphere = Revolve(circle, 2*np.pi, 2*np.pi/(phi_resolution), center=center, axis=2, ElemType=ElemType)

    # Perform z-scaling for ellipsoids
    sphere.NodeCoords[:,0] = (sphere.NodeCoords[:,0] - center[0])*radius[0]/radius[1] + center[0]

    return sphere

def Torus(center, R, r, axis=2, theta_resolution=20, phi_resolution=20, ElemType='tri'):
    """
    Generate a sphere (or ellipsoid)
    The total number of points will be phi_resolution*(theta_resolution-2) + 2

    Parameters
    ----------
    center : array_like
        Three element array of the coordinates of the center of the sphere.
    R : scalar
        The major axis of the torus. This is the distance from the center of the 
        torus to the center of the circular tube. 
    r : scalar
        The minor axis of the torus. This is the radius of the circular tube. 
    axis : int
        Axis of revolution of the torus. 0, 1, or 2 for x, y, z, respectively, by
        default 2
    theta_resolution : int, optional
        Number of circular cross sections rotated about the axis, by default 20.
    phi_resolution : int, optional
        Number of circumferential points for each circle section, by default 20.
    ElemType : str, optional
        Specify the element type of the mesh. This can either be 'quad' for 
        a quadrilateral mesh or 'tri' for a triangular mesh, by default 'tri'.
        If 'quad' is specified, there will still be some triangles at z axis "poles".

    Returns
    -------
    sphere, mymesh.mesh
        Mesh object containing the cylinder mesh.
        

    .. note:: 
        Due to the ability to unpack the mesh object to NodeCoords and NodeConn, the NodeCoords and NodeConn array can be returned directly (instead of the mesh object) by running: ``NodeCoords, NodeConn = primitives.Sphere(...)``


    Examples
    ________
    .. plot::

        torus = primitives.Torus([0,0,0], 1, .25, phi_resolution=50, ElemType='quad')
        torus.plot(bgcolor='w', show_edges=True)

    """

    # Create circle section
    t = np.linspace(0, 2*np.pi, theta_resolution)

    if axis == 2:
        x = np.repeat(center[0], len(t))
        y = center[1]+R + r*np.sin(t)
        z = center[2] + r*np.cos(t)
    elif axis == 1:
        x = center[0]+R + r*np.sin(t)
        y = center[1] + r*np.cos(t)
        z = np.repeat(center[2], len(t))
    elif axis == 0:
        x = center[0] + r*np.cos(t)
        y = np.repeat(center[1], len(t))
        z = center[2]+R + r*np.sin(t)
    xyz = [x,y,z]

    coords = np.column_stack(xyz)
    conn = np.column_stack([np.arange(0,len(t)-1), np.arange(1,len(t))])

    if 'mesh' in dir(mesh):
        circle = mesh.mesh(coords, conn)
    else:
        circle = mesh(coords, conn)
    torus = Revolve(circle, 2*np.pi, 2*np.pi/(phi_resolution), center=center, axis=axis, ElemType=ElemType)
    torus.cleanup()
    return torus

def Extrude(line, distance, step, axis=2, ElemType='quad'):
    """
    Extrude a 2D line mesh to a 3D surface

    Parameters
    ----------
    line : mesh
        mesh object of 2D line mesh
    distance : scalar
        Extrusion distance
    step : scalar
        Step size in the extrusion direction
    axis : int, optional
        Extrusion axis, either 0 (x), 1 (y), or 2 (z), by default 2
    ElemType : str, optional
        Specify the element type of the grid mesh. This can either be 'quad' for 
        a quadrilateral mesh or 'tri' for a triangular mesh, by default 'quad'.

    Returns
    -------
    extruded : mymesh.mesh
        Mesh object containing the extruded mesh.
        

    .. note:: 
        Due to the ability to unpack the mesh object to NodeCoords and NodeConn, the NodeCoords and NodeConn array can be returned directly (instead of the mesh object) by running: ``NodeCoords, NodeConn = primitives.Extrude(...)``

    Examples
    ________
    .. plot::

        x = np.linspace(0,1,100)
        y = np.sin(2*np.pi*x)
        coordinates = np.column_stack([x, y, np.zeros(len(x))])
        connectivity = np.column_stack([np.arange(len(x)-1), np.arange(len(x)-1)+1])
        line = mesh(coordinates, connectivity)
        extruded = primitives.Extrude(line, 1, 0.2)
        extruded.plot(bgcolor='w', show_edges=True)

    """    
    NodeCoords = np.array(line.NodeCoords)
    OriginalConn = np.asarray(line.NodeConn)
    NodeConn = np.empty((0,4))
    for i,s in enumerate(np.arange(step,distance+step,step)):
        temp = np.array(line.NodeCoords)
        temp[:,axis] += s
        NodeCoords = np.append(NodeCoords, temp, axis=0)

        NodeConn = np.append(NodeConn, np.hstack([OriginalConn+(i*len(temp)),np.fliplr(OriginalConn+((i+1)*len(temp)))]), axis=0)
    NodeConn = NodeConn.astype(int)
    if ElemType == 'tri':
        NodeConn = converter.quad2tri(NodeConn)
    if 'mesh' in dir(mesh):
        extruded = mesh.mesh(NodeCoords,NodeConn,'surf')
    else:
        extruded = mesh(NodeCoords,NodeConn,'surf')
    return extruded

def Revolve(line, angle, anglestep, center=[0,0,0], axis=2, ElemType='quad'):
    """
    Revolve a 2D line mesh to a 3D surface

    Parameters
    ----------
    line : mymesh.mesh
        Mesh object of 2d line mesh
    angle : scalar
        Angle (in radians) to revolve the line by. For a full rotation, angle=2*np.pi
    anglestep : scalar
        Step size (in radians) at which to sample the revolution.
    center : array_like, optional
        Three element vector denoting the center of revolution (i.e. a point on the axis),
        by default [0,0,0]
    axis : int or array_like, optional
        Axis of revolution. This can be specified as either 0 (x), 1 (y), or 2 (z) or a 
        three element vector denoting the axis, by default 2.
    ElemType : str, optional
        Specify the element type of the grid mesh. This can either be 'quad' for 
        a quadrilateral mesh or 'tri' for a triangular mesh, by default 'quad'. 
        If 'quad', some degenerate quads may be converted to tris.

    Returns
    -------
    revolve : mymesh.mesh
        Mesh object containing the revolved mesh.
        

    .. note:: 
        Due to the ability to unpack the mesh object to NodeCoords and NodeConn, the NodeCoords and NodeConn array can be returned directly (instead of the mesh object) by running: ``NodeCoords, NodeConn = primitives.Revolve(...)``

    """    
    if np.isscalar(axis):
        assert axis in (0, 1, 2), 'axis must be either 0, 1, or 2 (indicating x, y, z axes) or a 3 element vector.'
        if axis == 0:
            axis = [1, 0, 0]
        elif axis == 1:
            axis = [0, 1, 0]
        else:
            axis = [0, 0, 1]
    else:
        assert isinstance(axis, (list, tuple, np.ndarray)), 'axis must be either 0, 1, or 2 (indicating x, y, z axes) or a 3 element vector.'
        axis = axis/np.linalg.norm(axis)
    
    thetas = np.arange(0, angle+anglestep, anglestep)
    outer_prod = np.outer(axis,axis)
    cross_prod_matrix = np.zeros((3,3))
    cross_prod_matrix[0,1] = -axis[2]
    cross_prod_matrix[1,0] =  axis[2]
    cross_prod_matrix[0,2] =  axis[1]
    cross_prod_matrix[2,0] = -axis[1]
    cross_prod_matrix[1,2] = -axis[0]
    cross_prod_matrix[2,1] =  axis[0]

    rot_matrices = np.cos(thetas)[:,None,None]*np.repeat([np.eye(3)],len(thetas),axis=0) + np.sin(thetas)[:,None,None]*np.repeat([cross_prod_matrix],len(thetas),axis=0) + (1 - np.cos(thetas))[:,None,None]*np.repeat([outer_prod],len(thetas),axis=0)

    R = np.repeat([np.eye(4)],len(thetas),axis=0)
    R[:,:3,:3] = rot_matrices

    NodeCoords = np.array(line.NodeCoords)
    OriginalConn = np.asarray(line.NodeConn)
    NodeConn = np.empty((0,4))

    padded = np.hstack([NodeCoords, np.ones((len(NodeCoords),1))])
    T = np.array([
                [1, 0, 0, -center[0]],
                [0, 1, 0, -center[1]],
                [0, 0, 1, -center[2]],
                [0, 0, 0, 1],
                ])
    Tinv = np.linalg.inv(T)
    for i,r in enumerate(R[1:]):
        temp = np.linalg.multi_dot([Tinv,r,T,padded.T]).T[:,:3]
        NodeCoords = np.append(NodeCoords, temp, axis=0)

        NodeConn = np.append(NodeConn, np.hstack([OriginalConn+(i*len(temp)),np.fliplr(OriginalConn+((i+1)*len(temp)))]), axis=0)

    NodeConn = NodeConn.astype(int)

    if ElemType == 'tri':
        NodeConn = converter.quad2tri(NodeConn)
    NodeCoords, NodeConn = utils.DeleteDuplicateNodes(NodeCoords, NodeConn)
    NodeCoords, NodeConn = utils.CleanupDegenerateElements(NodeCoords, NodeConn, Type='surf')

    if 'mesh' in dir(mesh):
        revolve = mesh.mesh(NodeCoords,NodeConn,'surf')
    else:
        revolve = mesh(NodeCoords,NodeConn,'surf')
    return revolve