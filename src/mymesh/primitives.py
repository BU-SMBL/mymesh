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
    
    Line
    Multiline
    Box
    Grid
    Grid2D
    Plane
    Circle
    CirclePt
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
import gc, copy
from . import utils, converter, implicit, mesh, delaunay

def Line(pt1, pt2, h=None, n=None):
    """
    Generate a mesh of a straight line between two points 

    Parameters
    ----------
    pt1 : array_like
        Coordinates (x,y,z) of the first point of the line
    pt2 : array_like
        Coordinates (x,y,z) of the second point of the line
    h : float, optional
        Approximate element size. Only used if n is not provided. If neither h
        nor n are provided, element size is norm(pt2-pt1). By default, None.
    n : int, optional
        Number of segments of the line. If h is also provided, n takes
        precedence. If neither h nor n are provided, number of segments is 1. 
        By default, None.

    Returns
    -------
    line : mymesh.mesh
        Mesh object containing the line mesh. 

    .. note:: 
        Due to the ability to unpack the mesh object to NodeCoords and NodeConn, the NodeCoords and NodeConn array can be returned directly (instead of the mesh object) by running: ``NodeCoords, NodeConn = primitives.Line(...)``

    Examples
    --------
    .. plot::

        line = primitives.Line([0, 0, 0], [0.5, 0.5, 0])
        line.plot(bgcolor='w', show_edges=True)

    """    
    
    if n is None and h is not None:
        n = int(np.round((np.linalg.norm(pt2-pt1))/h))
    elif n is None and h is None:
        n = 1

    xs = np.linspace(pt1[0],pt2[0],n+1)
    ys = np.linspace(pt1[1],pt2[1],n+1)
    if len(pt1) > 2 and len(pt2) > 2:
        zs = np.linspace(pt1[2],pt2[2],n+1)
    else:
        zs = np.zeros(len(xs))

    LineCoords = np.column_stack([xs, ys, zs])
    LineConn = np.column_stack([np.arange(0,n), np.arange(1,n+1)])

    if 'mesh' in dir(mesh):
        line = mesh.mesh(LineCoords,LineConn)
    else:
        line = mesh(LineCoords,LineConn)
    line.Type = 'line'
    line.cleanup()
    return line

def Multiline(points, h=None, n=None, connect_ends=False):
    """
    Create a multi-point line by connecting a series of points

    Parameters
    ----------
    points : array_like
        Point coordinates (shape=(n,3))
    h : float, optional
        Element size, by default None. If specified, each line segment
        will be approximately divided into elements of size h. If neither h nor 
        n are specified, each line segment will be represented by a single 
        element. If both n and h are specified, n takes precedence.
    n : _type_, optional
        Number of elements for each segment, by default None. If specified, each 
        line segment will be divided into n elements. If neither h nor n are
        specified, each line segment will be represented by a single element. If 
        both n and h are specified, n takes precedence.
    connect_ends : bool, optional
        If true, the last point will be connected to the first point (e.g. to 
        create a closed loop), by default False.

    Returns
    -------
    line : mymesh.mesh
        Mesh of the multiline
    """    
    if connect_ends:
        points = np.append(points, [points[0]], axis=0)
    else:
        points = np.asarray(points)
    if 'mesh' in dir(mesh):
        line = mesh.mesh()
    else:
        line = mesh()
        
    for i in range(len(points)-1):
        line.merge(Line(points[i], points[i+1], h=h, n=n))
    
    line.Type ='line'
    line.cleanup()
    
    return line
        
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
        _,BoxConn = converter.quad2tri([],BoxConn)

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
    
    GridConn = np.empty(((nX-1)*(nY-1)*(nZ-1),8),dtype=int)

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
        _,GridConn = converter.quad2tri([],GridConn)
    
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

def Circle(center, radius, theta_resolution=20, radial_resolution=10, axis=2, ElemType=None, Type='surf'):
    """
    Construct a circle from a center and radius.

    Parameters
    ----------
    center : array_like
        Center of the circle
    radius : float
        Radius of the circle
    theta_resolution : int, optional
        Number of circumferential points, by default 20
    radial_resolution : int, optional
        Number of radial points from center to edge, by default 10
    axis : int, optional
        Axis perpendicular to the plane of the circle, specified as by integers
        (0=x, 1=y, 2=z), by default 2.
    ElemType : None, str, optional
        Element type of final mesh. If None, the result will contain predominantly
        quadrilateral elements with triangular elements at the center. If 'tri', 
        the result will be a purely triangular mesh, by default None.
    Type : str, optional
        Mesh type of the final mesh. This could be 'surf' for a surface mesh or
        'line' for a line mesh of just the circumference of the circle, by 
        default 'surf'.

    Returns
    -------
    circle : mymesh.mesh
        Mesh object containing the circle mesh.

    Examples
    --------
    .. plot::

        circle = primitives.Circle([0, 0, 0], 1)
        circle.plot(bgcolor='w', show_edges=True)
    """    
    if Type.lower() == 'line':
        radial_resolution = 2 
    pt2 = np.copy(center).astype(float)
    pt2[list({2,1,0}.difference({axis,}))[0]] += radius
    L = Line(center, pt2, n=radial_resolution-1)

    circle = Revolve(L, 2*np.pi, 2*np.pi/theta_resolution, center=center, axis=axis, ElemType=ElemType)

    if Type.lower() == 'line':
        if 'mesh' in dir(mesh):
            circle = mesh.mesh(circle.NodeCoords, converter.surf2edges(*circle), Type='line')
        else:
            circle = mesh(circle.NodeCoords, converter.surf2edges(*circle), Type='line')
        circle.cleanup()

    return circle

def CirclePt(center, point, theta_resolution=20, radial_resolution=10, axis=2, ElemType=None, Type='surf'):
    """
    Construct a circle from a center and point.

    Parameters
    ----------
    center : array_like
        Center of the circle
    point : array_like
        Coordinates of a point on the circle. 

        .. note:: 
            If the point is not coplanar with the center in the plane specified by
            `axis`, the result will not be a flat circle.

    theta_resolution : int, optional
        Number of circumferential points, by default 20
    radial_resolution : int, optional
        Number of radial points from center to edge, by default 10
    axis : int, optional
        Axis perpendicular to the plane of the circle, specified as by integers
        (0=x, 1=y, 2=z), by default 2.
    ElemType : None, str, optional
        Element type of final mesh. If None, the result will contain predominantly
        quadrilateral elements with triangular elements at the center. If 'tri', 
        the result will be a purely triangular mesh, by default None.
    Type : str, optional
        Mesh type of the final mesh. This could be 'surf' for a surface mesh or
        'line' for a line mesh of just the circumference of the circle, by 
        default 'surf'.
    Returns
    -------
    circle : mymesh.mesh
        Mesh object containing the circle mesh.
    """   
    if Type.lower() == 'line':
        radial_resolution = 2 
    L = Line(center, point, n=radial_resolution-1)

    circle = Revolve(L, 2*np.pi, 2*np.pi/theta_resolution, center=center, axis=axis, ElemType=ElemType)

    if Type.lower() == 'line':
        if 'mesh' in dir(mesh):
            circle = mesh.mesh(circle.NodeCoords, converter.surf2edges(*circle), Type='line')
        else:
            circle = mesh(circle.NodeCoords, converter.surf2edges(*circle), Type='line')
        circle.cleanup()

    return circle

def Cylinder(center, radius, height, theta_resolution=20, axial_resolution=10, radial_resolution=10, axis=2, cap=True, ElemType=None, Type='surf'):
    """
    Generate an axis-aligned cylindrical mesh

    Parameters
    ----------
    center : array_like
        Coordinates of the center of the circle at the base of the cylinder (x,y,z). 
    radius : scalar or array_like
        The radius of the cylinder. Radius can be specified as a scalar radius of 
        the cylinder or a two element array of half-axes for a cylinder with an
        elliptical cross-section. For elliptical cross-sections, the two elements
        correspond to the two axes in the plane perpendicular to <axis> (e.g. if
        axis=2, (r1, r2) specify the x, y radii, respectively; if axis=1, 
        (r1, r2) specify the x, z radii, respectively, etc.).
    height : float
        height of the cylinder. Cylinder will be extruded from <center> by <height>
        in the <axis> direction. To extrude in the negative-<axis> direction,
        a negative height can be given. 
    theta_resolution : int, optional
        Number of points in the circumference of the cylinder, by default 20.
    axial_resolution : int, optional
        Number of points along the axis of the cylinder, by default 10.
    radial_resolution : int, optional
        Number of radial points from center to edge, by default 10. Only relevant
        if Type='vol'.
    axis : int, optional
        Long axis of the cylinder (i.e. the circular ends will lie in the plane of the other two axes).
        Must be 0, 1, or 2 (x, y, z), by default 2
    cap : bool, optional
        If True, will close the ends of the cylinder, otherwise it will leave them open, by 
        default True.
    ElemType : str, optional
        Specify the element type of the mesh. If Type='surf' (default), ElemType
        can be 'tri', which will produce a purely triangular mesh or None, which
        will produce a quad-dominant mesh with some triangles. If Type='vol', 
        ElemType can be 'tet', which will produce a purely tetrahedral mesh or 
        None, which  will produce a hex-dominant mesh with some tetrahedra. By
        default, None. 
    Type : str, optional
        Mesh type of the final mesh. This could be 'surf' for a surface mesh or
        'vol' for a volumetric mesh, by default 'surf'.

    Returns
    -------
    cylinder : mymesh.mesh
        Mesh object containing the cylinder mesh.
        

    .. note:: 
        Due to the ability to unpack the mesh object to NodeCoords and NodeConn,
        the NodeCoords and NodeConn array can be returned directly (instead of the mesh object)
        by running: ``NodeCoords, NodeConn = primitives.Cylinder(...)``
    
    Examples
    --------
    .. plot::

        cylinder = primitives.Cylinder([0,0,0], 1, 2, 20, axis=2)
        cylinder.plot(bgcolor='w', show_edges=True)

    """    

    if isinstance(radius, (list, tuple, np.ndarray)):
        assert len(radius) == 2, 'radius must either be a scalar or a 2 element array_like.'
    elif np.isscalar(radius):
        radius = np.repeat(radius,2)
    else:
        raise TypeError('radius must either be a scalar or a 2 element array_like.')

    
    if cap or Type.lower() == 'vol':
        if not cap:
            warnings.warn('Cannot create an un-capped cylinder with Type="vol".')
        circle = Circle(center, radius[0], theta_resolution=theta_resolution,  radial_resolution=radial_resolution, axis=axis, Type='surf')
        if Type.lower() == 'vol':
            cylinder = Extrude(circle, height, height/axial_resolution, axis=axis, ElemType=ElemType)
        elif Type.lower() == 'surf':
            cylinder = Extrude(circle, height, height/axial_resolution, axis=axis)
            cylinder.NodeConn = converter.solid2surface(*cylinder)
            if ElemType == 'tri':
                cylinder.NodeCoords, cylinder.NodeConn = converter.surf2tris(*cylinder)
            cylinder.Type='surf'
            cylinder.cleanup()
        else:
            raise ValueError('Type must be "vol" or "surf".')
    
    else:
        circle = Circle(center, radius[0], theta_resolution=theta_resolution, radial_resolution=radial_resolution, axis=axis, Type='line')
        cylinder = Extrude(circle, height, height/axial_resolution, axis=axis, ElemType=ElemType)

    if radius[0] != radius[1]:
        keep_axis, warp_axis = list({0,1,2}.difference({axis,}))
        cylinder.NodeCoords[:,warp_axis] = (cylinder.NodeCoords[:,warp_axis] - center[warp_axis])*radius[1]/radius[0] + center[warp_axis]


    return cylinder

def Sphere(center, radius, theta_resolution=20, phi_resolution=20, radial_resolution=10, ElemType=None, Type='surf'):
    """
    Generate a sphere (or ellipsoid)
    The total number of points will be phi_resolution*(theta_resolution-2) + 2

    Parameters
    ----------
    center : array_like
        Three element array of the coordinates of the center of the sphere.
    radius : scalar or array_like
        The radius of the sphere. Radius can be specified as a scalar radius of 
        the sphere or three element array of half-axes for an ellipsoid. 
    theta_resolution : int, optionalperpendicular
        Number of circular (or elliptical) cross sections sampled along the z axis, by default 20.
    phi_resolution : int, optional
        Number of circumferential points for each cross section, by default 20.
    radial_resolution : int, optional
        Number of radial points from center to edge, by default 10. Only relevant
        if Type='vol'.
    ElemType : str, optional
        Specify the element type of the mesh. If Type='surf' (default), ElemType
        can be 'tri', which will produce a purely triangular mesh or None, which
        will produce a quad-dominant mesh with some triangles. If Type='vol', 
        ElemType can be 'tet', which will produce a purely tetrahedral mesh or 
        None, which  will produce a hex-dominant mesh with some tetrahedra. By
        default, None. 
    Type : str, optional
        Mesh type of the final mesh. This could be 'surf' for a surface mesh or
        'vol' for a volumetric mesh, by default 'surf'.

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

    if Type == 'surf':
        # Create cross section
        t = np.linspace(0,np.pi,theta_resolution)
        x = np.repeat(center[0],len(t))
        y = center[1] + radius[2]*np.sin(t)
        z = center[2] + radius[2]*np.cos(t)
        xyz = [x,y,z]

        coords = np.column_stack(xyz)
        conn = np.column_stack([np.arange(0,len(t)-1), np.arange(1,len(t))])

        if 'mesh' in dir(mesh):
            semicircle = mesh.mesh(coords, conn)
        else:
            semicircle = mesh(coords, conn)
    elif Type == 'vol':
        pt2 = np.copy(center).astype(float)
        pt2[2] -= radius[2]
        L = Line(center, pt2, n=radial_resolution-1)

        semicircle = Revolve(L, np.pi, np.pi/theta_resolution, center=center, axis=0, ElemType=None)
    else:
        raise ValueError('Type must be "surf" or "vol" for primitives.Sphere.')
    # Revolve cross section
    sphere = Revolve(semicircle, 2*np.pi, 2*np.pi/(phi_resolution), center=center, axis=2, ElemType=ElemType)

    # Perform x,y-scaling for ellipsoids
    sphere.NodeCoords[:,0] = (sphere.NodeCoords[:,0] - center[0])*radius[0]/radius[2] + center[0]
    sphere.NodeCoords[:,1] = (sphere.NodeCoords[:,1] - center[1])*radius[1]/radius[2] + center[1]

    return sphere

def Torus(center, R, r, axis=2, theta_resolution=20, phi_resolution=20, radial_resolution=10, ElemType=None, Type='surf'):
    """
    Generate a torus
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
    radial_resolution : int, optional
        Number of radial points from center to edge of the circular cross section, 
        by default 10. Only relevant if Type='vol'.
    ElemType : str, optional
        Specify the element type of the mesh. This can either be 'quad' for 
        a quadrilateral mesh or 'tri' for a triangular mesh, by default None.
        If 'quad' is specified, there will still be some triangles at z axis "poles".
    Type : str, optional
        Mesh type of the final mesh. This could be 'surf' for a surface mesh or
        'vol' for a volumetric mesh, by default 'surf'.
        
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

    if Type == 'surf':
        circle_type = 'line'
    elif Type == 'vol':
        circle_type = 'surf'
    
    if axis == 2:
        circle_center = [center[0], center[1]+R, center[2]] 
        circle_axis = 0
    elif axis == 1:
        circle_center = [center[0]+R, center[1], center[2]] 
        circle_axis = 2
    elif axis == 0:
        circle_center = [center[0], center[1], center[2]+R] 
        circle_axis = 1

    circle = Circle(circle_center, r, theta_resolution=theta_resolution, axis=circle_axis, Type=circle_type)
    if Type == 'vol':
        torus = Revolve(circle, 2*np.pi, 2*np.pi/(phi_resolution), center=center, axis=axis, ElemType=ElemType)
    else:
        torus = Revolve(circle, -2*np.pi, -2*np.pi/(phi_resolution), center=center, axis=axis, ElemType=ElemType)
    torus.cleanup()
    return torus

def Extrude(m, distance, step, axis=2, twist=0, twist_center=None, ElemType=None):
    """
    Extrude a 2D mesh to a 3D surface or volume

    Parameters
    ----------
    m : mymesh.mesh
        mesh object of 2D line mesh (m.Type='line') or 2D surface mesh 
        (m.Type='surf')
    distance : scalar
        Extrusion distance
    step : scalar
        Step size in the extrusion direction
    axis : int, optional
        Extrusion axis, either 0 (x), 1 (y), or 2 (z), by default 2
    twist : float, optional
        Amount to twist the initial geometry (in radians) over the course of the 
        extrusion, by default 0.
    twist_center : array_like, optional
        Center of rotation for twisting. If None is provided, the center of the
        initial geometry will be used.
    ElemType : str, optional
        Specify the element type of the extruded mesh. If m.Type='line', this
        can be None or 'tri' or if m.Type='surf', this can be None or 'tet'. 
        If 'tri' or 'tet', the mesh will be converted to a purely triangular/
        tetrahedral mesh, otherwise it will follow from the input mesh (input 
        line meshes will get quadrilateral elements, input surf meshes
        could get hexahedral, or wedge meshes). By default, None.

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
    NodeCoords = np.array(m.NodeCoords)
    if twist != 0 and twist_center is None:
        twist_center = np.array([
            (np.max(NodeCoords[:,0]) + np.min(NodeCoords[:,0]))/2,
            (np.max(NodeCoords[:,1]) + np.min(NodeCoords[:,1]))/2,
            (np.max(NodeCoords[:,2]) + np.min(NodeCoords[:,2]))/2
        ])
    elif isinstance(twist_center, (list, tuple)):
        twist_center = np.array(twist_center)
        
    if m.Type == 'line':
        OriginalConn = np.asarray(m.NodeConn)
        if ElemType is None:
            ElemType = 'quad'
        
        NodeConn = np.empty((0,4))
        steps = np.arange(step,distance+step,step)
        for i,s in enumerate(steps):
            temp = np.array(m.NodeCoords)
            temp[:,axis] += s
            if twist != 0:
                theta = twist*(i+1)/len(steps)
                rot = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
                axes = list({0,1,2}.difference({axis,}))
                temp[:,axes] = (rot@(temp-twist_center)[:,axes].T).T + twist_center[axes]
            NodeCoords = np.append(NodeCoords, temp, axis=0)

            NodeConn = np.append(NodeConn, np.hstack([OriginalConn+(i*len(temp)),np.fliplr(OriginalConn+((i+1)*len(temp)))]), axis=0)
        NodeConn = NodeConn.astype(int)
        if ElemType == 'tri':
            _,NodeConn = converter.quad2tri([],NodeConn)
        Type = 'surf'

    elif m.Type == 'surf':
        OriginalConn = copy.copy(m.NodeConn)
        NodeConn = []
        steps = np.arange(step,distance+step,step)
        for i,s in enumerate(steps):
            temp = np.array(m.NodeCoords)
            temp[:,axis] += s
            if twist != 0:
                theta = twist*(i+1)/len(steps)
                rot = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
                axes = list({0,1,2}.difference({axis,}))
                temp[:,axes] = (rot@(temp-twist_center)[:,axes].T).T + twist_center[axes]
            NodeCoords = np.append(NodeCoords, temp, axis=0)
            L = len(temp)
            NodeConn += [[n+((i)*L) for n in elem] + [n+((i+1)*L) for n in elem] for elem in OriginalConn]
        if ElemType == 'tet':
            NodeCoords,NodeConn = converter.solid2tets(NodeCoords,NodeConn)
        Type = 'vol'
    
    if 'mesh' in dir(mesh):
        extruded = mesh.mesh(NodeCoords,NodeConn,Type)
    else:
        extruded = mesh(NodeCoords,NodeConn,Type)

    return extruded

def Revolve(m, angle, anglestep, center=[0,0,0], shift=0, axis=2, ElemType=None):
    """
    Revolve a 2D mesh to a 3D surface or volume

    Parameters
    ----------
    m : mymesh.mesh
        Mesh object of 2d line mesh or 2d surface mesh
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
    shift : float, optional
        Offset along `axis` between the initial and final steps of the rotation,
        by default 0.
    ElemType : str, optional
        Specify the element type of the revolved mesh. If the input is a line 
        mesh (m.Type='line'), the element type can be None or 'tri'; if the input
        is a 2D surface (m.Type='surf'), the element type can be None or 'tet'. 
        If 'tri' or 'tet', the mesh will be converted to a purely triangular/
        tetrahedral mesh, otherwise it will follow from the input mesh (input 
        line meshes will get quad or quad-dominant meshes, input surf meshes
        could get hexahedral, or wedge-dominant meshes and may also contain
        pyramids or tetrahedra). By default, None.

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
            axis = np.array([1, 0, 0])
        elif axis == 1:
            axis = np.array([0, 1, 0])
        else:
            axis = np.array([0, 0, 1])
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

    NodeCoords = np.array(m.NodeCoords)
    

    padded = np.hstack([NodeCoords, np.ones((len(NodeCoords),1))])
    T = np.array([
                [1, 0, 0, -center[0]],
                [0, 1, 0, -center[1]],
                [0, 0, 1, -center[2]],
                [0, 0, 0, 1],
                ])
    Tinv = np.linalg.inv(T)
    if m.Type == 'line':
        OriginalConn = np.asarray(m.NodeConn)
        NodeConn = np.empty((0,4))
        for i,r in enumerate(R[1:]):
            temp = np.linalg.multi_dot([Tinv,r,T,padded.T]).T[:,:3]
            if shift != 0:
                temp += axis*shift*(i+1)/len(R[1:])

            NodeCoords = np.append(NodeCoords, temp, axis=0)

            NodeConn = np.append(NodeConn, np.hstack([OriginalConn+(i*len(temp)),np.fliplr(OriginalConn+((i+1)*len(temp)))]), axis=0)

        NodeConn = NodeConn.astype(int)

        if ElemType == 'tri':
            _,NodeConn = converter.quad2tri([],NodeConn)

        NodeCoords, NodeConn = utils.DeleteDuplicateNodes(NodeCoords, NodeConn)
        NodeCoords, NodeConn = utils.CleanupDegenerateElements(NodeCoords, NodeConn, Type='surf')
        Type='surf'
    elif m.Type == 'surf':
        NodeConn = []
        OriginalConn = copy.copy(m.NodeConn)
        for i,r in enumerate(R[1:]):
            temp = np.linalg.multi_dot([Tinv,r,T,padded.T]).T[:,:3]
            if shift != 0:
                temp += axis*shift*(i+1)/len(R[1:])

            NodeCoords = np.append(NodeCoords, temp, axis=0)

            L = len(temp)
            NodeConn += [[n+((i)*L) for n in elem[::-1]] + [n+((i+1)*L) for n in elem[::-1]] for elem in OriginalConn]

        if ElemType == 'tet':
            NodeCoords,NodeConn = converter.solid2tets(NodeCoords,NodeConn)

        NodeCoords, NodeConn = utils.DeleteDuplicateNodes(NodeCoords, NodeConn)
        NodeCoords, NodeConn = utils.CleanupDegenerateElements(NodeCoords, NodeConn, Type='vol')
        Type='vol'

    if 'mesh' in dir(mesh):
        revolve = mesh.mesh(NodeCoords,NodeConn,Type)
    else:
        revolve = mesh(NodeCoords,NodeConn,Type)
    return revolve