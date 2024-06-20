# -*- coding: utf-8 -*-
# Created on Fri Jan 14 17:43:57 2022
# @author: toj
"""
Implicit function meshing tools

An implicit function f(x,y,z) describes a surface in 3D where the surface is
located on the f(x,y,z) = 0 isosurface. The default convention used in this
module is that values less than zero are considered "inside" the surface,
and values above zero are considered "outside". 


.. currentmodule:: mymesh.implicit


Mesh Generation
===============
.. autosummary::
    :toctree: submodules/

    VoxelMesh
    SurfaceMesh
    TetMesh
    SurfaceNodeOptimization

Implicit Functions
==================
.. autosummary::
    :toctree: submodules/

    gyroid
    lidinoid
    primitive
    neovius
    diamond
    cylinder
    box
    xplane
    yplane
    zplane
    sphere

Implicit Function Operators
===========================
.. autosummary::
    :toctree: submodules/

    offset
    union
    diff
    thicken
    intersection
    unionf
    difff
    intersectionf
    thickenf
    unions
    diffs
    intersections
    thickens
    rMax
    rMin

Other Implicit Mesh Utilities
=============================
.. autosummary::
    :toctree: submodules/

    mesh2sdf
    mesh2udf
    grid2fun
    grid2grad
"""

# %%

import numpy as np
import sympy as sp
from scipy import optimize, interpolate
from scipy.spatial import KDTree
import sys, os, time, copy, warnings, bisect

from . import utils, converter, contour, quality, improvement, rays, octree, mesh, primitives

# Mesh generators
def VoxelMesh(func, bounds, h, threshold=0, threshold_direction=-1, mode='any', args=(), kwargs={}):
    """
    Generate voxel mesh of an implicit function

    Parameters
    ----------
    func : function
        Implicit function that describes the geometry of the object to be meshed. The function should be of the form v = f(x,y,z,*args,**kwargs) where x, y, and z are 1D numpy arrays of x, y and z coordinates and v is a 1D numpy array of function values. Additional arguments and keyword arguments may be passed through args and kwargs.
    bounds : array_like
        6 element array, list, or tuple with the minimum and maximum bounds in each direction that the function will be evaluated. This should be formatted as: [xmin, xmax, ymin, ymax, zmin, zmax]
    h : scalar, tuple
        Element side length. Can be specified as a single scalar value, or a three element tuple (or array_like).
    threshold : scalar
        Isovalue threshold to use for keeping/removing elements, by default 0.
    threshold_direction : signed integer
        If threshold_direction is negative (default), values less than or equal to the threshold will be considered "inside" the mesh and the opposite if threshold_direction is positive, by default -1.
    mode : str, optional
        Mode for determining which elements are kept, by default 'any'.
        Voxels will be kept if:
        'any' - if any node of a voxel is inside the surface, 
        'all' - if all nodes of a voxel are inside the surface, 
        'centroid' - if the centroid of the voxel is inside the surface. Centroids performs additional evaluation of the function, which could slow mesh generation for expensive to evaluate functions,
        'boundary' - if the voxel contains values above and below the threshold,
        'notrim' - all voxels are kept
    args : tuple, optional
        Tuple of additional positional arguments for func, by default ().
    kwargs : dict, optional
        Dictionary of additional keyword arguments for func, by default {}

    Returns
    -------
    voxel : mymesh.mesh
        Mesh object containing the voxel mesh. The values of the function at each node are stored in voxel.NodeData['func']

        .. note:: Due to the ability to unpack the mesh object to NodeCoords and NodeConn, the NodeCoords and NodeConn array can be returned directly (instead of the mesh object) by running: ``NodeCoords, NodeConn = implicit.VoxelMesh(...)``

    Examples
    --------
    .. plot::

        voxel = implicit.VoxelMesh(implicit.gyroid, [0,1,0,1,0,1], 0.05)
        voxel.plot(bgcolor='w', scalars=voxel.NodeData['func'])

    """        
    
    if not isinstance(h, (list, tuple, np.ndarray)):
        h = (h,h,h)

    # If func is a sympy-based function, convert it to numpy
    if isinstance(func, sp.Basic):
        x, y, z = sp.symbols('x y z', real=True)
        vector_func = sp.lambdify((x, y, z), func, 'numpy')
    elif isinstance(func(bounds[0],bounds[2],bounds[4]), sp.Basic):
        x, y, z = sp.symbols('x y z', real=True)
        vector_func = sp.lambdify((x, y, z), func(x,y,z), 'numpy')
    else:
        vector_func = func

    NodeCoords, GridConn = primitives.Grid(bounds, h, exact_h=False)
    Values = vector_func(NodeCoords[:,0], NodeCoords[:,1], NodeCoords[:,2], *args, **kwargs)
    if np.sign(threshold_direction) == 1:
        NodeVals = -1*Values
        threshold = -1*threshold
    else:
        NodeVals = Values

    if np.min(NodeVals) >= threshold:
        if 'mesh' in dir(mesh):
            voxel = mesh.mesh()
        else:
            voxel = mesh()
        voxel.NodeData['func'] = np.array([])
        return voxel
    if mode != 'notrim':
        if mode.lower() == 'centroid':
            centroids = utils.Centroids(NodeCoords, GridConn)
            Inside = vector_func(centroids[:,0], centroids[:,1], centroids[:,2]) <= threshold
            VoxelConn = GridConn[Inside]
        else:
            Inside = NodeVals <= threshold
            ElemInsides = Inside[GridConn]

            if mode.lower() == 'any':
                VoxelConn = GridConn[np.any(ElemInsides,axis=1)]
            elif mode.lower() == 'all':
                VoxelConn = GridConn[np.all(ElemInsides,axis=1)]
            elif mode.lower() == 'boundary':
                VoxelConn = GridConn[np.any(ElemInsides,axis=1) & ~np.all(ElemInsides,axis=1)]
            else:
                raise Exception('mode must be "any", "all", "centroid", "boundary", or "notrim".')

        NodeCoords, NodeConn, OriginalIds = utils.RemoveNodes(NodeCoords,VoxelConn)
        Values = Values[OriginalIds]
    else:   
        NodeConn = GridConn
    
    if 'mesh' in dir(mesh):
        voxel = mesh.mesh(NodeCoords, NodeConn)
    else:
        voxel = mesh(NodeCoords, NodeConn)
    voxel.NodeData['func'] = Values

    return voxel

def SurfaceMesh(func, bounds, h, threshold=0, threshold_direction=-1, method='mc', interpolation='linear', args=(), kwargs={}):
    """
    Generate a surface mesh of an implicit function 

    Parameters
    ----------
    func : function
        Implicit function that describes the geometry of the object to be meshed. The function should be of the form v = f(x,y,z,*args,**kwargs) where x, y, and z are 1D numpy arrays of x, y and z coordinates and v is a 1D numpy array of function values. For method='mc', x, y, and z will be 3D coordinate arrays and v must be 3D as well. Additional arguments and keyword arguments may be passed through args and kwargs.
    bounds : array_like
        6 element array, list, or tuple with the minimum and maximum bounds in each direction that the function will be evaluated. This should be formatted as: [xmin, xmax, ymin, ymax, zmin, zmax]
    h : scalar, tuple
        Element side length. Can be specified as a single scalar value, or a three element tuple (or array_like).
    threshold : scalar
        Isovalue threshold to use for keeping/removing elements, by default 0.
    threshold_direction : signed integer
        If threshold_direction is negative (default), values less than or equal to the threshold will be considered "inside" the mesh and the opposite if threshold_direction is positive, by default -1.
    method : str, optional
        Surface triangulation method, by default 'mc'.
        'mc' : Marching cubes (see contour.MarchingCubesImage) (default)

        'mc33' : Marching cubes 33 (see contour.MarchingCubes)

        'mt' : Marching tetrahedra (see contour.MarchingTetrahedra)
    interpolation : str, optional
        Method of interpolation used for placing the vertices on the approximated isosurface. This can be 'midpoint', 'linear', or 'cubic', by default 'linear'. If 'cubic' is selected, method is overridden to be 'mc'. 
    args : tuple, optional
        Tuple of additional positional arguments for func, by default ().
    kwargs : dict, optional
        Dictionary of additional keyword arguments for func, by default {}.

    Returns
    -------
    surface : mymesh.mesh
        Mesh object containing the surface mesh.

        .. note:: Due to the ability to unpack the mesh object to NodeCoords and NodeConn, the NodeCoords and NodeConn array can be returned directly (instead of the mesh object) by running: ``NodeCoords, NodeConn = implicit.SurfaceMesh(...)``

    Examples
    --------

    .. jupyter-execute::

        from mymesh import implicit

        surface = implicit.SurfaceMesh(implicit.gyroid, [0,1,0,1,0,1], 0.05)
        surface.view()
    
    .. plot::

        surface = implicit.SurfaceMesh(implicit.gyroid, [0,1,0,1,0,1], 0.05)
        surface.plot(bgcolor='w')
    """
    
    if not isinstance(h, (list, tuple, np.ndarray)):
        h = (h,h,h)

    if isinstance(func, sp.Basic):
        x, y, z = sp.symbols('x y z', real=True)
        vector_func = sp.lambdify((x, y, z), func, 'numpy')
    elif isinstance(func(bounds[0],bounds[2],bounds[4]), sp.Basic):
        x, y, z = sp.symbols('x y z', real=True)
        vector_func = sp.lambdify((x, y, z), func(x,y,z), 'numpy')
    else:
        vector_func = func

    if np.sign(threshold_direction) == 1:
        flip = True
    else:
        flip = False

    if method == 'mc' or interpolation=='cubic':
        if method != 'mc':
            warnings.warn('Using cubic interpolation overrides contour method to be marching cubes ("mc").')
        xs = np.arange(bounds[0],bounds[1]+h[0],h[0])
        ys = np.arange(bounds[2],bounds[3]+h[1],h[1])
        zs = np.arange(bounds[4],bounds[5]+h[2],h[2])

        X,Y,Z = np.meshgrid(xs,ys,zs,indexing='ij')
        F = vector_func(X,Y,Z,*args,**kwargs)
        SurfCoords, SurfConn = contour.MarchingCubesImage(F, h=h, threshold=threshold, flip=flip, method='original', interpolation=interpolation,VertexValues=True)
        SurfCoords[:,0] += bounds[0]
        SurfCoords[:,1] += bounds[2]
        SurfCoords[:,2] += bounds[4]
    elif method == 'mc33':
        voxel = VoxelMesh(vector_func, bounds, h, threshold=threshold, threshold_direction=threshold_direction, mode='boundary',*args,**kwargs)
        SurfCoords, SurfConn = contour.MarchingCubes(voxel.NodeCoords, voxel.NodeConn, voxel.NodeData['func'], method='33', threshold=threshold,flip=flip)
    elif method == 'mt':
        voxel = VoxelMesh(vector_func, bounds, h, threshold=threshold, threshold_direction=threshold, mode='boundary',*args,**kwargs)
        NodeCoords, NodeConn = converter.hex2tet(voxel.NodeCoords, voxel.NodeConn, method='1to6')
        SurfCoords, SurfConn = contour.MarchingTetrahedra(NodeCoords, NodeConn, voxel.NodeData['func'], method='surface', threshold=threshold,flip=flip)

    
    if 'mesh' in dir(mesh):
        surface = mesh.mesh(SurfCoords, SurfConn)
    else:
        surface = mesh(SurfCoords, SurfConn)
        
    return surface

def TetMesh(func, bounds, h, threshold=0, threshold_direction=-1, interpolation='linear', args=(), kwargs={}, background=None, snap2surf=True):
    """
    Generate a tetrahedral mesh of an implicit function 

    Parameters
    ----------
    func : function
        Implicit function that describes the geometry of the object to be meshed. The function should be of the form v = f(x,y,z,*args,**kwargs) where x, y, and z are 1D numpy arrays of x, y and z coordinates and v is a 1D numpy array of function values. For method='mc', x, y, and z will be 3D coordinate arrays and v must be 3D as well. Additional arguments and keyword arguments may be passed through args and kwargs.
    bounds : array_like
        6 element array, list, or tuple with the minimum and maximum bounds in each direction that the function will be evaluated. This should be formatted as: [xmin, xmax, ymin, ymax, zmin, zmax]
    h : scalar, tuple
        Element side length. Can be specified as a single scalar value, or a three element tuple (or array_like).
    threshold : scalar
        Isovalue threshold to use for keeping/removing elements, by default 0.
    threshold_direction : signed integer
        If threshold_direction is negative (default), values less than or equal to the threshold will be considered "inside" the mesh and the opposite if threshold_direction is positive, by default -1.
    interpolation : str, optional
        Method of interpolation used for placing the vertices on the approximated 
        isosurface. This can be 'midpoint', 'linear', or 'quadratic', by default
        'linear'. 
    args : tuple, optional
        Tuple of additional positional arguments for func, by default ().
    kwargs : dict, optional
        Dictionary of additional keyword arguments for func, by default {}.
    background : None or mymesh.mesh, optional
        Background tetrahedral mesh to use for evaluating the function and performing
        marching tetrahedra. If a mesh is provide, bounds and h will be ignored. 
        If None is provided, a uniform tetrahedral grid will be used, by default None.

    Returns
    -------
    tet : mymesh.mesh
        Mesh object containing the tetrahedral mesh.

        .. note:: Due to the ability to unpack the mesh object to NodeCoords and NodeConn, the NodeCoords and NodeConn array can be returned directly (instead of the mesh object) by running: ``NodeCoords, NodeConn = implicit.TetMesh(...)``

    Examples
    --------
    .. plot::

        tet = implicit.TetMesh(implicit.gyroid, [0,1,0,1,0,1], 0.05)
        tet.plot(bgcolor='w')
    """

    if not isinstance(h, (list, tuple, np.ndarray)):
        h = (h,h,h)

    if isinstance(func, sp.Basic):
        x, y, z = sp.symbols('x y z', real=True)
        vector_func = sp.lambdify((x, y, z), func, 'numpy')
    elif isinstance(func(bounds[0],bounds[2],bounds[4]), sp.Basic):
        x, y, z = sp.symbols('x y z', real=True)
        vector_func = sp.lambdify((x, y, z), func(x,y,z), 'numpy')
    else:
        vector_func = func

    if np.sign(threshold_direction) == 1:
        flip = True
    else:
        flip = False
        
    if snap2surf is True:
        snap = 0.1
    elif isinstance(snap2surf, (int, float)):
        snap = snap2surf
        snap2surf = True
    

    if background is None:
        voxel = VoxelMesh(vector_func, bounds, h, threshold=threshold, threshold_direction=threshold, mode='any', args=args, kwargs=kwargs)
        if snap2surf:
            voxel.NodeCoords, voxel.NodeConn, voxel.NodeData['func'] = contour.SnapGrid2Surf(voxel.NodeCoords, voxel.NodeConn, voxel.NodeData['func'], snap=snap)#, FixedNodes = voxel.SurfNodes)
        NodeCoords, NodeConn = converter.hex2tet(voxel.NodeCoords, voxel.NodeConn, method='1to6')
        
        if interpolation == 'quadratic':
            NodeCoords, NodeConn = converter.tet42tet10(NodeCoords, NodeConn)
            NodeVals = vector_func(NodeCoords[:,0], NodeCoords[:,1], NodeCoords[:,2])
        else:
            NodeVals = voxel.NodeData['func']
    else:
        NodeCoords, NodeConn = background
        NodeCoords = np.asarray(NodeCoords)
        NodeVals = vector_func(NodeCoords[:,0], NodeCoords[:,1], NodeCoords[:,2])
        if snap2surf:
            NodeCoords, NodeConn, NodeVals = contour.SnapGrid2Surf(NodeCoords, NodeConn, NodeVals, snap=snap, FixedNodes=background.SurfNodes)

        if interpolation == 'quadratic':
            if np.shape(NodeConn)[1] == 4:
                NodeCoords, NodeConn = converter.tet42tet10(NodeCoords, NodeConn)
            NodeVals = vector_func(NodeCoords[:,0], NodeCoords[:,1], NodeCoords[:,2])
    TetCoords, TetConn = contour.MarchingTetrahedra(NodeCoords, NodeConn, NodeVals, method='volume', threshold=threshold, flip=flip, interpolation=interpolation, cleanup=True)
    

    if 'mesh' in dir(mesh):
        tet = mesh.mesh(TetCoords, TetConn)
    else:
        tet = mesh(TetCoords, TetConn)
        
    return tet

def SurfaceNodeOptimization(M, func, h, iterate=1, threshold=0, FixedNodes=set(), FixEdges=False, finite_diff_step=1e-5, smooth=True, InPlace=False, springs=True):
    """
    Optimize the placement of surface node to lie on the "true" surface. This
    This simultaneously moves nodes towards the isosurface and redistributes
    nodes more evenly, thus smoothing the mesh without shrinkage or destruction
    of features. This method is consistes of using the Z-flow (and R-flow if 
    smooth=True) from :cite:p:`Ohtake2001a`.

    Parameters
    ----------
    M : mesh.mesh
        Mesh object
    func : function or sympy-type function
        Implicit function describing the mesh surface. This should be the same
        function used to create the mesh.
    h : float
        Element size of the surface mesh
    iterate : int, optional
        Number of iterations to perform, by default 1
    FixedNodes : set, optional
        Nodes to hold in place during repositioning, by default set()
    FixEdges : bool, optional
        Option to detect and hold in place exposed surface edges, by default False
    finite_diff_step : float, optional
        Small number used to calculate finite difference approximations to the 
        gradient. Only used if the function is not convertible to a 
        sympy-differentiable function, by default 1e-5.
    smooth : str, optional
        Option to perform smoothing. This can be either 'local' for local 
        Laplacian smoothing or 'tangential' for tangential Laplacian smoothing, 
        by default 'tangential'. For any other option, smoothing will not be 
        performed. Tangential smoothing differs from local in that nodes 
        are only repositioned in the tangent plane (based on the normal vector
        obtain from the gradient). 
    InPlace : bool, optional
        If False, will create a copy of the mesh, rather than altering node 
        positions of the original mesh object "in-place", by default False
    springs : bool, optional
        If True and the mesh is a volume mesh, internal nodes will be treated as 
        if they are connected by springs (see :func:`~mymesh.improvement.NodeSpringSmoothing`)
        to reduce risk of element distortion or inversion, by default True.

    Returns
    -------
    M : mesh.mesh
        Mesh with repositioned surface vertices

    """    

    if not InPlace:
        M = M.copy()
    if smooth == True:
        smooth = 'tangential'
        
    # Process nodes
    SurfNodes = set([n for elem in M.SurfConn for n in elem])
    if FixEdges:
        EdgeNodes = set(converter.surf2edges(M.NodeCoords, M.SurfConn).flatten())
    else:
        EdgeNodes = set()
    FreeNodes = np.array(list(SurfNodes.difference(EdgeNodes).difference(FixedNodes)))
    if len(FreeNodes) == 0:
        raise Exception('No free movable nodes.')

    # Process function
    def DiracDelta(x):
        if type(x) is np.ndarray:
            return (x == 0).astype(float)
        else:
            return float(x==0)

    x, y, z = sp.symbols('x y z', real=True)
    if callable(func):
        if isinstance(func(M.NodeCoords[0,0],M.NodeCoords[0,1],M.NodeCoords[0,2]), sp.Basic):
            F = sp.lambdify((x, y, z), func(x,y,z), 'numpy')
            
            Fx = sp.diff(func(x, y, z),x)
            Fy = sp.diff(func(x, y, z),y)
            Fz = sp.diff(func(x, y, z),z)
            Grad = sp.Matrix([Fx, Fy, Fz]).T
            gradF = sp.lambdify((x,y,z),Grad,['numpy',{'DiracDelta':DiracDelta}])

        else:
            F = func
            def gradF(X,Y,Z):
                gradx = (F(X+finite_diff_step/2,Y,Z) - F(X-finite_diff_step/2,Y,Z))/finite_diff_step
                grady = (F(X,Y+finite_diff_step/2,Z) - F(X,Y-finite_diff_step/2,Z))/finite_diff_step
                gradz = (F(X,Y,Z+finite_diff_step/2) - F(X,Y,Z-finite_diff_step/2))/finite_diff_step
                gradf = np.vstack((gradx,grady,gradz))
                return gradf

    elif isinstance(func, sp.Basic):
        F = sp.lambdify((x, y, z), func, 'numpy')

        Fx = sp.diff(func,x)
        Fy = sp.diff(func,y)
        Fz = sp.diff(func,z)
        Grad = sp.Matrix([Fx, Fy, Fz]).T
        gradF = sp.lambdify((x,y,z),Grad,['numpy',{'DiracDelta':DiracDelta}])

    else:
        raise TypeError('func must be a sympy function or callable function of three arguments (x,y,z).')

    M.NodeCoords = np.vstack([np.asarray(M.NodeCoords), [np.nan,np.nan,np.nan]])
    points = np.asarray(M.NodeCoords)[FreeNodes]

    if smooth is not None and smooth is not False:
        X = points[:,0]; Y = points[:,1]; Z = points[:,2]
        g = np.squeeze(gradF(X,Y,Z))
        r = utils.PadRagged(np.array(M.SurfNodeNeighbors,dtype=object)[FreeNodes].tolist())
        lengths = np.array([len(M.SurfNodeNeighbors[i]) for i in FreeNodes])


    for i in range(iterate):
        X = points[:,0]; Y = points[:,1]; Z = points[:,2]
        f = F(X,Y,Z) - threshold
        g = np.squeeze(gradF(X,Y,Z))
        fg = (f*g).T
        tau = h/(100*np.max(np.linalg.norm(fg,axis=1)))

        Zflow = -2*tau*fg

        # Rflow = np.zeros((len(NodeCoords),3))
        if smooth == 'tangential':
            Q = M.NodeCoords[r]
            U = (1/lengths)[:,None] * np.nansum(Q - points[:,None,:],axis=1)
            gnorm = np.linalg.norm(g,axis=0)
            gnorm[gnorm == 0] = 1
            NodeNormals = (g/gnorm).T
            Rflow = 1*(U - np.sum(U*NodeNormals,axis=1)[:,None]*NodeNormals)
        elif smooth == 'local':
            Q = M.NodeCoords[r]
            U = (1/lengths)[:,None] * np.nansum(Q - points[:,None,:],axis=1)
            Rflow = U
        else:
            Rflow = 0

        if springs and M.Type == 'vol':
            Forces = np.zeros((M.NNode, 3))
            Forces[FreeNodes] = Zflow + Rflow
            M = improvement.NodeSpringSmoothing(M, Displacements=Forces, options=dict(FixSurf=False, FixedNodes=FixedNodes, InPlace=True))
            NodeCoords = M.NodeCoords
            points = np.asarray(M.NodeCoords)[FreeNodes]
        else:
            points = points + Zflow + Rflow
            M.NodeCoords[FreeNodes] = points
    M.NodeCoords = M.NodeCoords[:-1]
    return M

# Implicit Function Primitives
def gyroid(x,y,z):
    """
    Implicit function approximation of the gyroid triply periodic minimal 
    surface (TPMS). This function uses sympy functions (sp.cos, sp.sin) to 
    enable symbolic differentiation. 
    
    For efficient vectorized evaluation, use:
    x, y, z = sp.symbols('x y z', real=True)
    vector_func = sp.lambdify((x, y, z), func(x,y,z), 'numpy')

    Parameters
    ----------
    x : scalar or np.ndarray
        x coordinate(s)
    y : scalar or np.ndarray
        y coordinate(s)
    z : scalar or np.ndarray
        z coordinate(s)

    Returns
    -------
    f : sympy expression
        implicit function evaluated with sympy
    """
    return sp.sin(2*np.pi*x)*sp.cos(2*np.pi*y) + sp.sin(2*np.pi*y)*sp.cos(2*np.pi*z) + sp.sin(2*np.pi*z)*sp.cos(2*np.pi*x)

def lidinoid(x,y,z):
    """
    Implicit function approximation of the lidinoid triply periodic minimal 
    surface (TPMS). This function uses sympy functions (sp.cos, sp.sin) to 
    enable symbolic differentiation. 
    
    For efficient vectorized evaluation, use:
    x, y, z = sp.symbols('x y z', real=True)
    vector_func = sp.lambdify((x, y, z), func(x,y,z), 'numpy')

    Parameters
    ----------
    x : scalar or np.ndarray
        x coordinate(s)
    y : scalar or np.ndarray
        y coordinate(s)
    z : scalar or np.ndarray
        z coordinate(s)

    Returns
    -------
    f : sympy expression
        implicit function evaluated with sympy
    """
    X = 2*np.pi*x
    Y = 2*np.pi*y
    Z = 2*np.pi*z
    f = 0.5*(sp.sin(2*X)*sp.cos(Y)*sp.sin(Z) + sp.sin(2*Y)*sp.cos(Z)*sp.sin(X) + sp.sin(2*Z)*sp.cos(X)*sp.sin(Y)) - 0.5*(sp.cos(2*X)*sp.cos(2*Y) + sp.cos(2*Y)*sp.cos(2*Z) + sp.cos(2*Z)*sp.cos(2*X)) + 0.15
    return f

def primitive(x,y,z):
    """
    Implicit function approximation of the primitive (Schwarz P) triply periodic 
    minimal surface (TPMS). This function uses sympy functions (sp.cos, sp.sin) 
    to enable symbolic differentiation. 
    
    For efficient vectorized evaluation, use:
    x, y, z = sp.symbols('x y z', real=True)
    vector_func = sp.lambdify((x, y, z), func(x,y,z), 'numpy')

    Parameters
    ----------
    x : scalar or np.ndarray
        x coordinate(s)
    y : scalar or np.ndarray
        y coordinate(s)
    z : scalar or np.ndarray
        z coordinate(s)

    Returns
    -------
    f : sympy expression
        implicit function evaluated with sympy
    """
    X = 2*np.pi*x
    Y = 2*np.pi*y
    Z = 2*np.pi*z
    return sp.cos(X) + sp.cos(Y) + sp.cos(Z)

def neovius(x,y,z):
    """
    Implicit function approximation of the neovius triply periodic minimal 
    surface (TPMS). This function uses sympy functions (sp.cos, sp.sin) to 
    enable symbolic differentiation. 
    
    For efficient vectorized evaluation, use:
    x, y, z = sp.symbols('x y z', real=True)
    vector_func = sp.lambdify((x, y, z), func(x,y,z), 'numpy')

    Parameters
    ----------
    x : scalar or np.ndarray
        x coordinate(s)
    y : scalar or np.ndarray
        y coordinate(s)
    z : scalar or np.ndarray
        z coordinate(s)

    Returns
    -------
    f : sympy expression
        implicit function evaluated with sympy
    """

    X = 2*np.pi*x
    Y = 2*np.pi*y
    Z = 2*np.pi*z
    return 3*(sp.cos(X) + sp.cos(Y) + sp.cos(Z)) + 4*sp.cos(X)*sp.cos(Y)*sp.cos(Z)

def diamond(x,y,z):
    """
    Implicit function approximation of the diamond (Schwarz D) triply periodic 
    minimal surface (TPMS). This function uses sympy functions (sp.cos, sp.sin) 
    to enable symbolic differentiation. 
    
    For efficient vectorized evaluation, use:
    x, y, z = sp.symbols('x y z', real=True)
    vector_func = sp.lambdify((x, y, z), func(x,y,z), 'numpy')

    Parameters
    ----------
    x : scalar or np.ndarray
        x coordinate(s)
    y : scalar or np.ndarray
        y coordinate(s)
    z : scalar or np.ndarray
        z coordinate(s)

    Returns
    -------
    f : sympy expression
        implicit function evaluated with sympy
    """
    return sp.sin(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.sin(2*np.pi*z) + sp.sin(2*np.pi*x)*sp.cos(2*np.pi*y)*sp.cos(2*np.pi*z) + sp.cos(2*np.pi*x)*sp.sin(2*np.pi*y)*sp.cos(2*np.pi*z) + sp.cos(2*np.pi*x)*sp.cos(2*np.pi*y)*sp.sin(2*np.pi*z)

def cylinder(center, radius):
    """
    Implicit function of a cylinder.

    Parameters
    ----------
    center : array_like
        2D vector specifying the x and y coordinates of the center of the 
        cylindrical cross section
    radius : float
        Radius of the cylinder

    Returns
    -------
    func : function
        Implicit function of three parameters (x, y, z)
    """    
    func = lambda x, y, z : (x-center[0])**2 + (y-center[1])**2 - radius**2
    return func

def box(x1,x2,y1,y2,z1,z2):    
    """
    Implicit function of a box.

    Parameters
    ----------
    x1 : float
        x coordinate lower bound
    x2 : float
        x coordinate upper bound
    y1 : float
        y coordinate lower bound
    y2 : float
        y coordinate upper bound
    z1 : float
        z coordinate lower bound
    z2 : float
        z coordinate upper bound

    Returns
    -------
    func : function
        Implicit function of three parameters (x, y, z)
    """
    func = lambda x, y, z : intersection(intersection(intersection(x1-x,x-x2),intersection(y1-y,y-y2)),intersection(z1-z,z-z2))
    return func

def xplane(x0, n=1):
    """
    Implicit function of a plane whose normal direction is along the x axis.

    Parameters
    ----------
    x0 : float
        Coordinate along the x axis of the plane. x0 = 0 corresponds to the 
        yz plane.
    n : int, optional
        Direction (1 or -1), or a scaling factor, by default 1. If n > 0, the 
        function will be negative when evaluated above x0.
        

    Returns
    -------
    func : function
        Implicit function of three parameters (x, y, z)
    """
    func = lambda x, y, z : n*(x0 - x)
    return func

def yplane(y0, n=1):
    """
    Implicit function of a plane whose normal direction is along the y axis.

    Parameters
    ----------
    y0 : float
        Coordinate along the y axis of the plane. y0 = 0 corresponds to the 
        xz plane.
    n : int, optional
        Direction (1 or -1), or a scaling factor, by default 1. If n > 0, the 
        function will be negative when evaluated above y0.
        

    Returns
    -------
    func : function
        Implicit function of three parameters (x, y, z).
    """
    func = lambda x, y, z : n*(y0 - y)
    return func

def zplane(z0, n=1):
    """
    Implicit function of a plane whose normal direction is along the z axis.

    Parameters
    ----------
    z0 : float
        Coordinate along the x axis of the plane. z0 = 0 corresponds to the 
        xy plane.
    n : int, optional
        Direction (1 or -1), or a scaling factor, by default 1. If n > 0, the 
        function will be negative when evaluated above z0.
        

    Returns
    -------
    func : function
        Implicit function of three parameters (x, y, z)
    """

    func = lambda x, y, z : n*(z0 - z)
    return func

def sphere(center, radius):
    """
    Implicit function of a sphere.

    Parameters
    ----------
    center : array_like
        3D coordinates ([x, y, z]) of the center of the sphere.
    radius : float
        radius of the sphere.

    Returns
    -------
    func : function
        Implicit function of three parameters (x, y, z).
    """    
    func = lambda x, y, z : (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2 - radius**2
    return func

# Implicit Function Operators
def offset(fval,value):
    """
    Offset function values by a prescribed amount. For a signed 
    distance function, this offsets the isosurface by a specified distance.

    Parameters
    ----------
    fval : scalar or np.ndarray
        Function value(s) to be offset
    value : scalar
        Offset value

    Returns
    -------
    offset_val : scalar or np.ndarray
        Offset value(s)
    """    
    offset_val = fval-value
    return offset_val

def union(fval1,fval2):
    """
    Boolean union of two values or sets of values. Negative values are assumed
    to be "inside". An R-function :func:`rMin` minimum is used to obtain a 
    continuously differentiable output.

    Parameters
    ----------
    fval1 : scalar or np.ndarray
        Value(s) of the first function
    fval2 : scalar or np.ndarray
        Value(s) of the second function

    Returns
    -------
    union_val : scalar or np.ndarray
        Union of the two sets of values
    """ 
    union_val = rMin(fval1,fval2)
    return union_val

def diff(fval1,fval2):
    """
    Boolean difference of two values or sets of values. Negative values are assumed
    to be "inside". An R-function :func:`rMax` maximum is used to obtain a 
    continuously differentiable output. Note that this operation is not 
    symmetric so the order of inputs matters.

    Parameters
    ----------
    fval1 : scalar or np.ndarray
        Value(s) of the first function
    fval2 : scalar or np.ndarray
        Value(s) of the second function

    Returns
    -------
    diff_val : scalar or np.ndarray
        Difference of the two sets of values
    """ 
    diff_val = rMax(fval1,-fval2)
    return diff_val

def diff_old(fval1, fval2):
    # warnings.warn('This function will be removed in the future.')
    return rMin(fval1,-fval2)
    
def intersection(fval1,fval2):
    """
    Boolean intersection of two values or sets of values. Negative values are 
    assumed to be "inside". An R-function :func:`rMax` maximum is used to 
    obtain a continuously differentiable output. 

    Parameters
    ----------
    fval1 : scalar or np.ndarray
        Value(s) of the first function
    fval2 : scalar or np.ndarray
        Value(s) of the second function

    Returns
    -------
    intersection_val : scalar or np.ndarray
        Intersection of the two sets of values
    """
    intersection_val = rMax(fval1,fval2)
    return intersection_val

def thicken(fval, t):
    """
    Thicken an isosurface by offsetting in both directions. The surface
    is offset in both directions by t/2.

    Parameters
    ----------
    fval : scalar or np.ndarray
        Function value(s) to be offset
    t : scalar
        Thickness value. For a signed distance function, this will correspond
        to an actual thickness, for other implicit functions, the offset distance
        depends on the function.

    Returns
    -------
    thick : scalar or np.ndarray
        Thickened value(s)
    """
    offp = offset(fval, t/2)
    offn = offset(fval, -t/2)
    thick = diff(offp, offn)
    return thick

def offsetf(f,value):
    """
    Offset function by a prescribed amount. For a signed 
    distance function, this offsets the isosurface by a specified distance.

    Parameters
    ----------
    f : callable
        Callable function of three variables (x, y, z). Function should be able
        to handle either scalar or vector inputs.
    value : scalar
        Offset value

    Returns
    -------
    offset_fun : callable
        Offset funcion
    """    
    offset_fun = lambda x,y,z : offset(f(x,y,z), value)
    return offset_fun

def unionf(f1,f2):
    """
    Boolean union of two functions. Negative values are assumed
    to be "inside". An R-function :func:`rMin` minimum is used to obtain a 
    continuously differentiable output.

    Parameters
    ----------
    f1 : callable
        Callable function of three variables (x, y, z). Function should be able
        to handle either scalar or vector inputs.
    f2 : callable
        Callable function of three variables (x, y, z). Function should be able
        to handle either scalar or vector inputs.

    Returns
    -------
    union_fun : callable
        Union function
    """ 
    union_fun = lambda x,y,z : rMin(f1(x,y,z),f2(x,y,z))
    return union_fun

def difff(f1,f2):
    """
    Boolean difference of two functions. Negative values are assumed
    to be "inside". An R-function :func:`rMax` maximum is used to obtain a 
    continuously differentiable output. Note that this operation is not 
    symmetric so the order of inputs matters.

    Parameters
    ----------
    f1 : callable
        Callable function of three variables (x, y, z). Function should be able
        to handle either scalar or vector inputs.
    f2 : callable
        Callable function of three variables (x, y, z). Function should be able
        to handle either scalar or vector inputs.

    Returns
    -------
    diff_fun : callable
        Difference of the two functions
    """ 
    diff_fun = lambda x,y,z : rMax(f1(x,y,z),-f2(x,y,z))
    return diff_fun

def intersectionf(f1,f2):
    """
    Boolean intersection of two functions. Negative values are assumed
    to be "inside". An R-function :func:`rMax` maximum is used to obtain a 
    continuously differentiable output. 

    Parameters
    ----------
    f1 : callable
        Callable function of three variables (x, y, z). Function should be able
        to handle either scalar or vector inputs.
    f2 : callable
        Callable function of three variables (x, y, z). Function should be able
        to handle either scalar or vector inputs.

    Returns
    -------
    intersection_fun : callable
        Intersection of the two functions
    """ 
    intersection_fun = lambda x,y,z : rMax(f1(x,y,z),f2(x,y,z))
    return intersection_fun

def thickenf(f, t):
    """
    Thicken an implicit function by offsetting in both directions. The surface
    is offset in both directions by t/2.

    Parameters
    ----------
    f : callable
        Callable function of three variables (x, y, z). Function should be able
        to handle either scalar or vector inputs.
    t : scalar
        Thickness value. For a signed distance function, this will correspond
        to an actual thickness, for other implicit functions, the offset distance
        depends on the function.

    Returns
    -------
    thick_fun : callable
        Thickened function
    """    
    offp = offsetf(f, t/2)
    offn = offsetf(f, -t/2)
    thick_fun = difff(offp, offn)
    return thick_fun

def unions(symfun1,symfun2):
    """
    Boolean union of two symbolic functions. Negative values are assumed
    to be "inside". An R-function :func:`rMins` minimum is used to obtain a 
    continuously differentiable output.

    Parameters
    ----------
    symfun1 : sympy function
        Symbolic sympy function of three variables (x, y, z). 
    symfun2 : sympy function
        Symbolic sympy function of three variables (x, y, z). 

    Returns
    -------
    union_sym : Scalar or np.ndarray
        Symbolic union function
    """ 
    union_sym = rMins(symfun1,symfun2)
    return union_sym

def diffs(symfun1,symfun2):
    """
    Boolean difference of two symbolic functions. Negative values are assumed
    to be "inside". An R-function :func:`rMaxs` minimum is used to obtain a 
    continuously differentiable output. Note that this operation is not 
    symmetric so the order of inputs matters.

    Parameters
    ----------
    symfun1 : sympy function
        Symbolic sympy function of three variables (x, y, z). 
    symfun2 : sympy function
        Symbolic sympy function of three variables (x, y, z). 

    Returns
    -------
    diff_sym : sympy function
        Symbolic difference function
    """ 
    diff_sym = rMaxs(symfun1,-symfun2)
    return diff_sym

def intersections(symfun1,symfun2):
    """
    Boolean intersection of two symbolic functions. Negative values are assumed
    to be "inside". An R-function :func:`rMaxs` minimum is used to obtain a 
    continuously differentiable output. 

    Parameters
    ----------
    symfun1 : sympy function
        Symbolic sympy function of three variables (x, y, z). 
    symfun2 : sympy function
        Symbolic sympy function of three variables (x, y, z). 

    Returns
    -------
    diff_sym : sympy function
        Symbolic difference function
    """ 
    intersection_sym = rMaxs(symfun1,symfun2)
    return intersection_sym

def thickens(symfun, t):
    offp = offset(symfun, t/2)
    offn = offset(symfun, -t/2)
    thick = diffs(offp, offn)
    return thick

def rMax(a,b,alpha=0,m=0,p=2):
    # R-Function :cite:p:`Shapiro1999` version of max(a,b) to yield a smoothly differentiable max - R0
    # Implicit Functions With Guaranteed Differential Properties - Shapiro & Tsukanov
    return 1/(1+alpha)*(a+b+(np.maximum(a**p+b**p - 2*alpha*a*b,0))**(1/p)) * (a**2 + b**2)**(m/2)

def rMin(a,b,alpha=0,m=0,p=2):
    # R-Function :cite:p:`Shapiro1999` version of min(a,b) to yield a smoothly differentiable min - R0
    # Implicit Functions With Guaranteed Differential Properties - Shapiro & Tsukanov
    return 1/(1+alpha)*(a+b-(np.maximum(a**p+b**p - 2*alpha*a*b,0))**(1/p)) * (a**2 + b**2)**(m/2)

def rMaxs(a,b,alpha=0,m=0,p=2):
    # R-Function version of max(a,b) to yield a smoothly differentiable max - R0
    # Implicit Functions With Guaranteed Differential Properties - Shapiro & Tsukanov
    return 1/(1+alpha)*(a+b+(a**p+b**p - 2*alpha*a*b)**(1/p)) * (a**2 + b**2)**(m/2)

def rMins(a,b,alpha=0,m=0,p=2):
    # R-Function version of min(a,b) to yield a smoothly differentiable min - R0
    # Implicit Functions With Guaranteed Differential Properties - Shapiro & Tsukanov
    return 1/(1+alpha)*(a+b-(a**p+b**p - 2*alpha*a*b)**(1/p)) * (a**2 + b**2)**(m/2)

def grid2fun(VoxelCoords,VoxelConn,Vals,method='linear',fill_value=None):
    """
    Converts a voxel grid mesh into a function that can be evaluated at any point within the bounds of the grid.

    Parameters
    ----------
    VoxelCoords : List of Lists
        List of nodal coordinates for the voxel mesh
    VoxelConn : List of Lists
       Nodal connectivity list for the voxel mesh.
    Vals : list
        List of values at each node or at each element.

    Returns
    -------
    fun : function
        Interpolation function, takes arguments (x,y,z), to return an
        evaluation of the function at the specified point.

    """
    
    if len(Vals) == len(VoxelCoords):
        Coords = np.asarray(VoxelCoords)
    elif len(Vals) == len(VoxelConn):
        Coords = utils.Centroids(VoxelCoords,VoxelConn)
    else:
        raise Exception('Vals must be the same length as either VoxelCoords or VoxelConn')
    # VoxelCoords = np.array(VoxelCoords)
    X = np.unique(Coords[:,0])
    Y = np.unique(Coords[:,1])
    Z = np.unique(Coords[:,2])
    
    points = (X,Y,Z)
    V = np.reshape(Vals,[len(X),len(Y),len(Z)])
    
    V[np.isnan(V)] = np.array([np.nan]).astype(int)[0]
    fun = lambda x,y,z : interpolate.RegularGridInterpolator(points,V,method=method,bounds_error=False,fill_value=None)(np.vstack([x,y,z]).T)
    
    return fun

def grid2grad(VoxelCoords,VoxelConn,NodeVals,method='linear'):
    """
    Converts a voxel grid mesh into a function. The function can be evaluated at any point within the bounds of the grid to return the gradient of the function.

    Parameters
    ----------
    VoxelCoords : List of Lists
        List of nodal coordinates for the voxel mesh
    VoxelConn : List of Lists
       Nodal connectivity list for the voxel mesh.
    NodeVals : list
        List of values at each node.

    Returns
    -------
    frad : function
        Interpolation function, takes arguments (x,y,z), to return an
        evaluation of the function gradient at the specified point.

    """
    
    VoxelCoords = np.array(VoxelCoords)
    X = np.unique(VoxelCoords[:,0])
    Y = np.unique(VoxelCoords[:,1])
    Z = np.unique(VoxelCoords[:,2])
    
    points = (X,Y,Z)
    V = np.reshape(NodeVals,[len(X),len(Y),len(Z)])
    # Assumes (and requires) that all voxels are cubic and the same size
    VoxelSize = abs(sum(VoxelCoords[VoxelConn[0][0]] - VoxelCoords[VoxelConn[0][1]]))
    G = np.gradient(V,VoxelSize)
    grad = lambda x,y,z : np.vstack([interpolate.interpn(points,G[i],np.vstack([x,y,z]).T,bounds_error=False,method=method) for i in range(len(G))]).T
    
    return grad

def FastMarchingMethod(VoxelCoords, VoxelConn, NodeVals):
    """
    FastMarchingMethod based on J.A. Sethian. A Fast Marching Level Set Method
    for Monotonically Advancing Fronts, Proc. Natl. Acad. Sci., 93, 4, 
    pp.1591--1595, 1996

    Parameters
    ----------
    VoxelCoords : list
        List of nodal coordinates for the voxel mesh.
    VoxelConn : list
        Nodal connectivity list for the voxel mesh.
    NodeVals : list
        List of value at each node.

    Returns
    -------
    T : list
        Lists of reinitialized node values.

    """
    warnings.warn('FastMarchingMethod is not fully functional.')

    # 3D
    N = 3
    # For now this is only for obtaining a signed distance function, so F = 1 everywhere
    F = 1
    NodeVals = np.array(NodeVals)
    # Get Neighbors
    if len(VoxelConn[0]) == 4:
        ElemType = 'quad'
    else:
        ElemType = 'hex'
    NodeNeighbors = utils.getNodeNeighbors(VoxelCoords, VoxelConn, ElemType=ElemType)
    xNeighbors = [[n for n in NodeNeighbors[i] if (VoxelCoords[n][1] == VoxelCoords[i][1]) and (VoxelCoords[n][2] == VoxelCoords[i][2])] for i in range(len(NodeNeighbors))]
    yNeighbors = [[n for n in NodeNeighbors[i] if (VoxelCoords[n][0] == VoxelCoords[i][0]) and (VoxelCoords[n][2] == VoxelCoords[i][2])] for i in range(len(NodeNeighbors))]
    zNeighbors = [[n for n in NodeNeighbors[i] if (VoxelCoords[n][0] == VoxelCoords[i][0]) and (VoxelCoords[n][1] == VoxelCoords[i][1])] for i in range(len(NodeNeighbors))]
    # Assumes (and requires) that all voxels are the same size
    h = abs(sum(np.array(VoxelCoords[VoxelConn[0][0]]) - np.array(VoxelCoords[VoxelConn[0][1]])))
    # Initialize Labels - Accepted if on the surface, Narrow Band if an adjacent node has a different sign (i.e. cross the surface), otherwise Far
    Accepted = set([i for i,v in enumerate(NodeVals) if v == 0])
    Narrow = [i for i,v in enumerate(NodeVals) if any(np.sign(NodeVals[NodeNeighbors[i]]) != np.sign(v)) and i not in Accepted]
    NarrowVals = []
    for i in Narrow:
        crosses = []
        for n in xNeighbors[i]:
            if np.sign(NodeVals[i]) != np.sign(NodeVals[n]):
                crosses.append(np.sign(NodeVals[i])*np.abs((0-NodeVals[i])*(VoxelCoords[n][0]-VoxelCoords[i][0])/(NodeVals[n]-NodeVals[i])))
        for n in yNeighbors[i]:
            if np.sign(NodeVals[i]) != np.sign(NodeVals[n]):
                crosses.append(np.sign(NodeVals[i])*np.abs((0-NodeVals[i])*(VoxelCoords[n][1]-VoxelCoords[i][1])/(NodeVals[n]-NodeVals[i])))
        for n in zNeighbors[i]:
            if np.sign(NodeVals[i]) != np.sign(NodeVals[n]):
                crosses.append(np.sign(NodeVals[i])*np.abs((0-NodeVals[i])*(VoxelCoords[n][2]-VoxelCoords[i][2])/(NodeVals[n]-NodeVals[i])))
        # NarrowVals.append(np.mean(crosses))
        NarrowVals.append(min(crosses))
    Far = set(range(len(NodeVals))).difference(Accepted.union(set(Narrow)))
    # Initialize Values (inf for Far, 0 for accepted)
    infty = 1e16 * max(NodeVals)
    T = infty*np.ones(len(NodeVals))
    for i in range(len(NarrowVals)):
        T[Narrow[i]] = NarrowVals[i]
    for i in Accepted:
        T[i] = 0
    
    Nar = sorted([t for i,t in enumerate(zip(NarrowVals,Narrow))], key=lambda x: x[0])
    while len(Far) + len(Nar) > 0:
        if len(Nar) > 0:
            pt = Nar[0][1]
        else:
            n = Far.pop()
            Nar.append((T[n],n))
        Accepted.add(pt)
        Nar.pop(0)
        for n in NodeNeighbors[pt]:
            if n in Far:
                Far.remove(n)
                Nar.insert(bisect.bisect_left(Nar, (T[n],n)), (T[n],n))
            if n not in Accepted:
                # Eikonal Update:
                Tx = min([T[x] for x in xNeighbors[n]]+[0])
                Ty = min([T[y] for y in yNeighbors[n]]+[0])
                Tz = min([T[z] for z in zNeighbors[n]]+[0])
                
                discriminant = sum([Tx,Ty,Tz])**2 - N*(sum([Tx**2,Ty**2,Tz**2]) - h**2/F**2)
                if discriminant > 0:
                    t = 1/N * sum([Tx,Ty,Tz]) + 1/N * np.sqrt(discriminant)
                else:
                    t = h/F + min([Tx,Ty,Tz])
                                
                Nar.pop(bisect.bisect_left(Nar, (T[n],n)))
                if t < T[n]: T[n] = t
                Nar.insert(bisect.bisect_left(Nar, (T[n],n)), (T[n],n))        
    T = [-1*t if np.sign(t) != np.sign(NodeVals[i]) else t for i,t in enumerate(T)]
    return T

def mesh2sdf(M, points, method='nodes+centroids'):
    """
    Generates a signed distance field for a mesh.

    Parameters
    ----------
    M : mesh.mesh
        Mesh object that will be used to define the distance field.
    points : array_like
        Points at which the signed distance field will be evaluated.
    method : str
        Method to be used 
        nodes 
        nodes+centroids
        centroids

    Returns
    -------
    NodeVals : list
        List of signed distance values evaluated at each node in the voxel grid.

    """
    if method == 'nodes':
        Normals = np.asarray(M.NodeNormals)
        SurfNodes = set(M.SurfNodes)
        Coords = np.array([n if i in SurfNodes else [10**32,10**32,10**32] for i,n in enumerate(M.NodeCoords)])
    elif method == 'centroids':
        Normals = np.asarray(M.ElemNormals)
        NodeCoords = np.array(M.NodeCoords)
        Coords = utils.Centroids(M.NodeCoords,M.NodeConn) #np.array([np.mean(NodeCoords[elem],axis=0) for elem in M.SurfConn])
    elif method == 'nodes+centroids':
        Normals = np.array(list(M.NodeNormals) + list(M.ElemNormals))
        NodeCoords = np.array(M.NodeCoords)
        SurfNodes = set(M.SurfNodes)
        Coords = np.append([n if i in SurfNodes else [10**32,10**32,10**32] for i,n in enumerate(M.NodeCoords)], utils.Centroids(M.NodeCoords,M.SurfConn),axis=0).astype(float)
    else:
        raise Exception('Invalid method - use "nodes", "centroids", or "nodes+centroids"')
    
    tree = KDTree(Coords)  
    Out = tree.query(points,1)
    ds = Out[0].flatten()
    cs = Out[1].flatten()
    rs = points - Coords[cs]
    signs = np.sign(np.sum(rs*Normals[cs,:],axis=1))# [np.sign(np.dot(rs[i],Normals[cs[i]])) for i in range(len(ds))]
    NodeVals = signs*ds
    
    return NodeVals

def mesh2udf(M, points):
    """
    Generates an unsigned distance field for a mesh.

    Parameters
    ----------
    M : mesh.mesh
        Mesh object that will be used to define the distance field.
    points : array_like
        Points at which the signed distance field will be evaluated.

    Returns
    -------
    NodeVals : list
        List of signed distance values evaluated at each node in the voxel grid.

    """
    Coords = np.asarray(M.NodeCoords)

    tree = KDTree(Coords)  
    Out = tree.query(points,1)
    NodeVals = Out[0].flatten()
    
    return NodeVals

def DoubleDualResampling(sdf,NodeCoords,NodeConn,DualCoords,DualConn,eps=1e-3,c=2):
    warnings.warn('DoubleDualResampling is not fully functional and may be unstable.')

    # Ohtake, Y., and Belyaev, A. G. (March 26, 2003). "Dual-Primal Mesh Optimization for Polygonized Implicit Surfaces With Sharp Features ." ASME. J. Comput. Inf. Sci. Eng. December 2002; 2(4): 277–284. https://doi.org/10.1115/1.1559153
    DualCoords,DualConn,gradP = DualMeshOptimization(sdf,NodeCoords,NodeConn,DualCoords,DualConn,eps=eps,return_grad=True)
    DualNeighbors,ElemConn = utils.getNodeNeighbors(DualCoords,DualConn,ElemType='polygon')
    NewNodeCoords = [[] for i in range(len(NodeCoords))]
    gradPnorms = [np.linalg.norm(gradP[i]) for i in range(len(gradP))]
    Normals = [gradP[j]/gradPnorms[j] if gradPnorms[j] > 0 else utils.CalcFaceNormal(DualCoords,[DualConn[ElemConn[j][0]]])[0] for j in range(len(DualCoords))]
    for i in range(len(NodeCoords)):
        Ps = DualConn[i]
        # Ns = [gradP[j]/gradPnorms[j] if gradPnorms[j] > 0 else utils.CalcFaceNormal(DualCoords,[Ps])[0] for j in Ps]
        ks = []
        for j,Pj in enumerate(Ps):
            NeighborPs = DualNeighbors[Pj][:3]
            # NNPs =[gradP[k]/gradPnorms[k] for k in NeighborPs]
            # ks.append(sum([np.arccos(np.dot(Ns[j],NNPs[k]))/(np.linalg.norm(DualCoords[Pj])*np.linalg.norm(DualCoords[NeighborPs[k]]))  for k in range(len(NNPs))]))
            
            ks.append(sum([np.arccos(min(np.dot(Normals[Pj],Normals[NeighborPs[k]]),1))/np.linalg.norm(np.subtract(DualCoords[Pj],DualCoords[NeighborPs[k]])) for k in range(len(NeighborPs))]))
            if np.isnan(ks[-1]):
                print('merp')
        
        weights = [1+c*ki for ki in ks]
        
        NewNodeCoords[i] = sum([np.multiply(weights[j],DualCoords[Ps[j]]) for j in range(len(Ps))])/sum(weights).tolist()
    return NewNodeCoords, NodeConn
    
def DualMeshOptimization(sdf,NodeCoords,NodeConn,DualCoords,DualConn,eps=1e-3,return_grad=False):
    # Ohtake, Y., and Belyaev, A. G. (March 26, 2003). "Dual-Primal Mesh Optimization for Polygonized Implicit Surfaces With Sharp Features ." ASME. J. Comput. Inf. Sci. Eng. December 2002; 2(4): 277–284. https://doi.org/10.1115/1.1559153
    warnings.warn('DualMeshOptimization is not fully functional and may be unstable.')
    
    def GradF(q,h):
        g = [-1,0,1]
        X = np.array([q[0]+h*x for x in g for y in g for z in g])
        Y = np.array([q[1]+h*y for x in g for y in g for z in g])
        Z = np.array([q[2]+h*z for x in g for y in g for z in g])
        F = sdf(X,Y,Z).reshape([3,3,3])
        dF = np.gradient(F,h)
        dFq = [dF[0][1,1,1],dF[1][1,1,1],dF[2][1,1,1]]
        return dFq
    def bisection(sdf, a, b, fa, fb, tol=eps):
        assert (fa < 0 and fb > 0) or (fa > 0 and fb < 0), 'Invalid bounds for bisection'
        
        thinking = True
        while thinking:
            c = np.mean([a,b],axis=0)
            fc = sdf(*c)
            # if fc == 0 or (np.linalg.norm(b-a)/2) < tol:
                # merp = 'meep'
            if abs(fc) < tol:
                thinking = False
            else:
                if np.sign(fc) == np.sign(fa):
                    a = c
                    fa = fc
                else:
                    b = c
                    fb = fc                
        return c
    def secant(sdf, a, b, fa, fb, tol=eps):
        assert (fa < 0 and fb > 0) or (fa > 0 and fb < 0), 'Invalid bounds for secant method'
        origA, origB, origFa, origFb = a, b, fa, fb
        thinking = True
        k = 0
        while thinking:
            k += 1
            c = np.subtract(b,fb*(np.subtract(b,a))/(fb-fa))
            fc = sdf(*c)
            if fc == 0 or abs(fc) < tol:
                thinking = False
            else:
                a,b = b,c
                fa,fb = fb,fc
            if k > 50 or fa == fb:
                thinking = False
                c = bisection(sdf, origA, origB, origFa, origFb, tol=tol)
        
        if not ((a[0] <= c[0] <= b[0] or a[0] >= c[0] >= b[0]) and (a[1] <= c[1] <= b[1] or a[1] >= c[1] >= b[1] ) and (a[2] <= c[2] <= b[2] or a[2] >= c[2] >= b[2])):
            c = bisection(sdf, origA, origB, origFa, origFb, tol=tol)
            
        return c
    ArrayCoords = np.array(NodeCoords)
    
    # _,ElemConn = utils.getNodeNeighbors(NodeCoords,NodeConn)
    # DualCoords,DualConn = converter.surf2dual(ArrayCoords,NodeConn,ElemConn=ElemConn)
    
    # Optimize dual mesh coordinates     
    gradP = [[] for i in range(len(DualCoords))]
    for c,P in enumerate(DualCoords):
        pts = ArrayCoords[NodeConn[c]]
        edgelengths = [np.linalg.norm(pts[1]-pts[0]),np.linalg.norm(pts[2]-pts[1]),np.linalg.norm(pts[0]-pts[2])]
        e = np.mean(edgelengths)
        lamb = e/2
        fP = sdf(*P)
        if abs(fP) < eps:
            if return_grad:
                gradP[c] = GradF(P,lamb/1000)
                # gradP[c] = gradF(*P)[0]
            continue
        
        Q = P
        fQ = fP

        it = 0
        thinking = True
        while thinking:
            dfQ = GradF(Q,lamb/1000)
            # dfQ = gradF(*Q)[0]
            d = -np.multiply(dfQ,fQ)
            d = d/np.linalg.norm(d)
            R = Q + lamb*d
            fR = sdf(*R)
            if fQ*fR < 0:
                P2 = bisection(sdf, Q, R, fQ, fR)
                thinking = False
            else: 
                Q = R
                fQ = sdf(*Q)
                it += 1
                if it == 3:
                    lamb = lamb/2
                if it > 500:
                    thinking = False
                    P2 = P
                    # raise Exception("Too many iterations - This probably shouldn't happen")
        #if np.linalg.norm(np.subtract(P2,P)) < e:

        S = P-2*P2
        fS = sdf(*S)
        if fP*fS < 0:
            P3 = bisection(sdf, P, S, fP, fS)
            if np.linalg.norm(np.subtract(P,P2)) < np.linalg.norm(np.subtract(P,P3)):
                P = P2
            else:
                P = P3
        else:
            P = P2
        # P = P2
        DualCoords[c] = P
        if return_grad: gradP[c] = GradF(P,lamb/10)
        # if return_grad: gradP[c] = gradF(*P)[0]
        
    if return_grad:
        return DualCoords, DualConn, gradP
    else:
        return DualCoords, DualConn
    
def AdaptiveSubdivision(sdf,NodeCoords,NodeConn,threshold=1e-3):
    # Ohtake, Y., and Belyaev, A. G. (March 26, 2003). "Dual-Primal Mesh Optimization for Polygonized Implicit Surfaces With Sharp Features ." ASME. J. Comput. Inf. Sci. Eng. December 2002; 2(4): 277–284. https://doi.org/10.1115/1.1559153
    def gradF(q,h=1e-6):
        if type(q) is list: q = np.array(q)
        if len(q.shape)==1: q = np.array([q])
        gradx = (sdf(q[:,0]+h/2,q[:,1],q[:,2]) - sdf(q[:,0]-h/2,q[:,1],q[:,2]))/h
        grady = (sdf(q[:,0],q[:,1]+h/2,q[:,2]) - sdf(q[:,0],q[:,1]-h/2,q[:,2]))/h
        gradz = (sdf(q[:,0],q[:,1],q[:,2]+h/2) - sdf(q[:,0],q[:,1],q[:,2]-h/2))/h
        gradf = np.vstack((gradx,grady,gradz)).T
        if len(gradf) == 1: gradf = gradf[0]
        return gradf
    
    NewNodeCoords = copy.copy(NodeCoords)
    NewNodeConn = copy.copy(NodeConn)
    ElemNeighbors = utils.getElemNeighbors(NodeCoords, NodeConn, mode='edge')

    ###
    Points = np.array(NodeCoords)[np.array(NodeConn)]
    cross = np.cross(Points[:,1]-Points[:,0],Points[:,2]-Points[:,0])
    norm = np.linalg.norm(cross,axis=1)
    ElemNormals = cross/norm[:,None]
    Area = norm/2
    splitCentroids = np.swapaxes(np.array([np.mean(Points,axis=1),
                          np.mean([Points[:,0], np.mean([Points[:,0], Points[:,1]],axis=0), np.mean([Points[:,0], Points[:,2]],axis=0)],axis=0),
                          np.mean([Points[:,1], np.mean([Points[:,0], Points[:,1]],axis=0), np.mean([Points[:,1], Points[:,2]],axis=0)],axis=0),
                          np.mean([Points[:,2], np.mean([Points[:,2], Points[:,1]],axis=0), np.mean([Points[:,0], Points[:,2]],axis=0)],axis=0),
                          ]),0,1)
    splitCentroids2 = splitCentroids.reshape((len(splitCentroids)*4,3))
    gradFCi = gradF(splitCentroids2)
    mCi = gradFCi/np.linalg.norm(gradFCi,axis=1)[:,None]
    mCi2 = mCi.reshape(splitCentroids.shape)
    en = np.array([Area[i]*sum(1-np.abs(np.dot(ElemNormals[i],mCi2[i].T))) for i in range(len(NodeConn))])
    for i,elem in enumerate(NodeConn):
        if en[i] > threshold:
            id01 = len(NewNodeCoords)
            NewNodeCoords.append(np.mean([NewNodeCoords[elem[0]],NewNodeCoords[elem[1]]],axis=0).tolist())
            id12 = len(NewNodeCoords)
            NewNodeCoords.append(np.mean([NewNodeCoords[elem[1]],NewNodeCoords[elem[2]]],axis=0).tolist())
            id20 = len(NewNodeCoords)
            NewNodeCoords.append(np.mean([NewNodeCoords[elem[2]],NewNodeCoords[elem[0]]],axis=0).tolist())
            NewNodeConn[i] = [
                [elem[0],id01,id20],
                [id01,elem[1],id12],
                [id20,id12,elem[2]],
                [id01,id12,id20]
                ]

    # Check for neighbors of split elements
    thinking = True
    mode = '1-4'
    while thinking:
        changes = 0
        for i,elem in enumerate(NewNodeConn):
            if type(elem[0]) is list:
                # Already subdivided
                continue
            nSplitNeighbors = 0
            SplitNeighbors = []
            for n in ElemNeighbors[i]:
                if type(NewNodeConn[n][0]) is list and len(NewNodeConn[n]) > 2: 
                    nSplitNeighbors += 1
                    SplitNeighbors.append(n)
            if mode == '1-4' and nSplitNeighbors > 1:
                changes += 1
                id01 = len(NewNodeCoords)
                NewNodeCoords.append(np.mean([NewNodeCoords[elem[0]],NewNodeCoords[elem[1]]],axis=0).tolist())
                id12 = len(NewNodeCoords)
                NewNodeCoords.append(np.mean([NewNodeCoords[elem[1]],NewNodeCoords[elem[2]]],axis=0).tolist())
                id20 = len(NewNodeCoords)
                NewNodeCoords.append(np.mean([NewNodeCoords[elem[2]],NewNodeCoords[elem[0]]],axis=0).tolist())
                NewNodeConn[i] = [
                    [elem[0],id01,id20],
                    [id01,elem[1],id12],
                    [id20,id12,elem[2]],
                    [id01,id12,id20]
                    ]
                
            elif mode == '1-2' and nSplitNeighbors == 1:
                changes += 1
                if elem[0] in NodeConn[SplitNeighbors[0]] and elem[1] in NodeConn[SplitNeighbors[0]]:
                    idx = len(NewNodeCoords)
                    NewNodeCoords.append(np.mean([NewNodeCoords[elem[0]],NewNodeCoords[elem[1]]],axis=0).tolist())
                    NewNodeConn[i] = [
                        [elem[0],idx,elem[2]],
                        [idx,elem[1],elem[2]]
                        ]
                elif elem[1] in NodeConn[SplitNeighbors[0]] and elem[2] in NodeConn[SplitNeighbors[0]]:
                    idx = len(NewNodeCoords)
                    NewNodeCoords.append(np.mean([NewNodeCoords[elem[1]],NewNodeCoords[elem[2]]],axis=0).tolist())
                    NewNodeConn[i] = [
                        [elem[1],idx,elem[0]],
                        [idx,elem[2],elem[0]]
                        ]
                else:
                    idx = len(NewNodeCoords)
                    NewNodeCoords.append(np.mean([NewNodeCoords[elem[2]],NewNodeCoords[elem[0]]],axis=0).tolist())
                    NewNodeConn[i] = [
                        [elem[0],elem[1],idx],
                        [elem[1],elem[2],idx]
                        ]
        if mode == '1-4' and changes == 0:
            # After all necessary 1-4 splits are completed, perform 1-2 splits
            mode = '1-2'
        elif mode == '1-2' and changes == 0:
            thinking = False
            
    NewNodeConn = [elem for elem in NewNodeConn if (type(elem[0]) != list)] + [e for elem in NewNodeConn if (type(elem[0]) == list) for e in elem]
    NewNodeCoords,NewNodeConn = utils.DeleteDuplicateNodes(NewNodeCoords,NewNodeConn)
            
    return NewNodeCoords,NewNodeConn
     
def DualPrimalOptimization(sdf,NodeCoords,NodeConn,eps=1e-3,nIter=2):
    # Ohtake, Y., and Belyaev, A. G. (March 26, 2003). "Dual-Primal Mesh Optimization for Polygonized Implicit Surfaces With Sharp Features ." ASME. J. Comput. Inf. Sci. Eng. December 2002; 2(4): 277–284. https://doi.org/10.1115/1.1559153
    warnings.warn('DualPrimalOptimization is not fully functional and may be unstable.')
       
    def PrimalMeshOptimization(DualCoords,DualConn,gradP,tau=10**3):
        ArrayCoords = np.zeros([len(DualConn),3])
        DualCoords = np.array(DualCoords)
        DualNeighbors,ElemConn = utils.getNodeNeighbors(DualCoords,DualConn,ElemType='polygon')
        centroids = utils.Centroids(DualCoords,DualConn)
        gradPnorms = [np.linalg.norm(gradP[i]) for i in range(len(gradP))]
        TransCoords = copy.copy(DualCoords)
        for j,Pis in enumerate(DualConn):
            
            # Transfrom Coordinates to local system centered on the centroid
            TransCoords[:,0] -= centroids[j][0]
            TransCoords[:,1] -= centroids[j][1]
            TransCoords[:,2] -= centroids[j][2]
            
            # Normal vector TODO: gradP[i] could = 0 at sharp features, in this case, need to use something else (maybe the element normal of the primal element corresponding to the dual node)
            # Ns = [np.divide(gradP[i],gradPnorms[i]) for i in Pis]
            Ns = [np.divide(gradP[i],gradPnorms[i]) if gradPnorms[i] > 0 else utils.CalcFaceNormal(DualCoords,[DualConn[ElemConn[i][0]]])[0] for i in Pis]
            
            r = np.linalg.norm(centroids[j]-DualCoords[Pis[0]])*2
            A = np.diag([sum([N[0]**2 for i,N in enumerate(Ns)]), 
                         sum([N[1]**2 for i,N in enumerate(Ns)]),
                         sum([N[2]**2 for i,N in enumerate(Ns)])])
            
            b = [sum([N[0]**2*TransCoords[Pis[i]][0] for i,N in enumerate(Ns)]), 
                 sum([N[1]**2*TransCoords[Pis[i]][1] for i,N in enumerate(Ns)]),
                 sum([N[2]**2*TransCoords[Pis[i]][2] for i,N in enumerate(Ns)])]
            x = np.linalg.lstsq(A,b,rcond=1/tau)[0]
            ArrayCoords[j] = x + centroids[j]
            # Reset TransCoords
            TransCoords[:,0] += centroids[j][0]
            TransCoords[:,1] += centroids[j][1]
            TransCoords[:,2] += centroids[j][2]
        return ArrayCoords.tolist()
    
    OptCoords = copy.copy(NodeCoords)
    OptConn = copy.copy(NodeConn)
    k = 0
    tau = 10**3
    for it in range(nIter):
        DualCoords, DualConn = converter.surf2dual(OptCoords,OptConn,sort='ccw')
        writeVTK('{:d}_Dual.vtk'.format(k),DualCoords,DualConn)
        OptCoords,_ = DoubleDualResampling(sdf,OptCoords,OptConn,DualCoords,DualConn)
        writeVTK('{:d}_PrimalResampled.vtk'.format(k),OptCoords,OptConn)
        DualCoords = utils.Centroids(OptCoords,OptConn)
        writeVTK('{:d}_Dual2.vtk'.format(k),DualCoords,DualConn)
        DualCoords, DualConn, gradP = DualMeshOptimization(sdf,OptCoords,OptConn,DualCoords,DualConn,eps=eps,return_grad=True) 
        writeVTK('{:d}_DualOpt.vtk'.format(k),DualCoords,DualConn)
        OptCoords = PrimalMeshOptimization(DualCoords,DualConn,gradP,tau=tau)
        writeVTK('{:d}_PrimalOpt.vtk'.format(k),OptCoords,OptConn)
        OptCoords,OptConn = AdaptiveSubdivision(sdf,OptCoords,OptConn)
        writeVTK('{:d}_PrimalOptSub.vtk'.format(k),OptCoords,OptConn)
        k += 1
        if k > 1 and tau > 10:
            tau = tau/10
    DualCoords, DualConn = converter.surf2dual(OptCoords,OptConn,sort='ccw')
    OptCoords,_ = DoubleDualResampling(sdf,OptCoords,OptConn,DualCoords,DualConn)
    writeVTK('{:d}_PrimalResampled.vtk'.format(k),OptCoords,OptConn)
    DualCoords, DualConn, gradP = DualMeshOptimization(sdf,OptCoords,OptConn,DualCoords,DualConn,eps=eps,return_grad=True) 
    writeVTK('{:d}_DualOpt.vtk'.format(k),DualCoords,DualConn)
    OptCoords = PrimalMeshOptimization(DualCoords,DualConn,gradP,tau=tau)
    writeVTK('{:d}_PrimalOpt.vtk'.format(k),OptCoords,OptConn)
    return OptCoords,OptConn

def SurfFlowOptimization(sdf,NodeCoords,NodeConn,h,ZRIter=50,NZRIter=50,NZIter=50,Subdivision=True,FixedNodes=set(), gradF=None):
    
    C = 0.1     # Positive Constant
    FreeNodes = list(set(range(len(NodeCoords))).difference(FixedNodes))
    if gradF is None:
        def gradF(q):
            hdiff = 1e-6    # Finite Diff Step Size
            if type(q) is list: q = np.array(q)
            if len(q.shape)==1: q = np.array([q])
            gradx = (sdf(q[:,0]+hdiff/2,q[:,1],q[:,2]) - sdf(q[:,0]-hdiff/2,q[:,1],q[:,2]))/hdiff
            grady = (sdf(q[:,0],q[:,1]+hdiff/2,q[:,2]) - sdf(q[:,0],q[:,1]-hdiff/2,q[:,2]))/hdiff
            gradz = (sdf(q[:,0],q[:,1],q[:,2]+hdiff/2) - sdf(q[:,0],q[:,1],q[:,2]-h/2))/hdiff
            gradf = np.vstack((gradx,grady,gradz)).T
            if len(gradf) == 1: gradf = gradf[0]
            return gradf
    def NFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=1):
        gradC = -gradF(Centroids)
        gradCnorm = np.linalg.norm(gradC,axis=1)
        m = np.divide(gradC,np.reshape(gradCnorm,(len(gradC),1)))
        
        # This is a slower but more straightforward version of what is done below
        # A = np.array([sum([Area[e] for e in ElemConn[i]]) for i in range(len(NodeCoords))])
        # tau = 1/(1000*max(A))
        # N1 = tau*np.array([1/sum(Area[T] for T in ElemConn[i]) * sum([Area[T]*np.dot((Centroids[T]-P),m[T])*m[T] for T in ElemConn[i]]) for i,P in enumerate(NodeCoords)])

        # Converting the ragged ElemConn array to a padded rectangular array (R) for significant speed improvements
        Area2 = np.append(Area,0)
        m2 = np.vstack([m,[0,0,0]])
        Centroids2 = np.vstack([Centroids,[0,0,0]])
        R = utils.PadRagged(ElemConn,fillval=-1)
        a = Area2[R]
        A = np.sum(a,axis=1)
        tau = tf*.75 # 1/(100*max(A))
        v = np.sum(m2[R]*(Centroids2[R] - NodeCoords[:,None,:]),axis=2)[:,:,None]*m2[R]
        C = np.sum(a[:,:,None]*v,axis=1)
        N = (tau/np.sum(Area2[R],axis=1))[:,None]*C
        return N
    def N2Flow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids):
        
        # Orthocenter coordinates: https://en.wikipedia.org/wiki/Triangle_center#Position_vectors
        tic = time.time()
        Points = NodeCoords[np.array(NodeConn)]
        a = np.linalg.norm(Points[:,1]-Points[:,2],axis=1)
        b = np.linalg.norm(Points[:,2]-Points[:,0],axis=1)
        c = np.linalg.norm(Points[:,1]-Points[:,0],axis=1)
        wA = a**4 - (b**2 - c**2)**2
        wB = b**4 - (c**2 - a**2)**2
        wC = c**4 - (a**2 - b**2)**2
        # Orthocenters
        H = (wA[:,None]*Points[:,0] + wB[:,None]*Points[:,1] + wC[:,None]*Points[:,2])/(wA + wB + wC)[:,None]
        H2 = np.vstack([H,[0,0,0]])
        # 
        lens = [len(e) for e in ElemConn]
        maxlens = max(lens)
        R = utils.PadRagged(ElemConn,fillval=-1)
        Mask0 = (R>=0).astype(int)
        Masknan = Mask0.astype(float)
        Masknan[Mask0 == 0] = np.nan
        
        PH = (H2[R] - NodeCoords[:,None,:])*Mask0[:,:,None]
        PHnorm = np.linalg.norm(PH,axis=2)
        e = PH/PHnorm[:,:,None]

        # For each point, gives the node connectivity of each incident element
        IncidentNodes = np.array(NodeConn)[R]*Masknan[:,:,None]
        ## TODO: This needs a speedup
        OppositeEdges = (((np.array([[np.delete(x,x==i) if i in x else [np.nan,np.nan] for x in IncidentNodes[i]] for i in range(len(IncidentNodes))])).astype(int)+1)*Mask0[:,:,None]-1)
        ##
        OppositeLength = np.linalg.norm(NodeCoords[OppositeEdges[:,:,0]] - NodeCoords[OppositeEdges[:,:,1]],axis=2)

        TriAntiGradient = e*OppositeLength[:,:,None]/2
        PointAntiGradient = np.nansum(TriAntiGradient,axis=1)
        degree = np.array([len(E) for E in ElemConn])
        N = 1/(5*degree[:,None]) * PointAntiGradient
        print(time.time()-tic)
        return N
    def ZFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=1):
        fP = sdf(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2])
        gradP = gradF(NodeCoords)
        # A = np.array([sum([Area[T] for T in ElemConn[i]]) for i in range(len(NodeCoords))])
        Area2 = np.append(Area,0)
        R = utils.PadRagged(ElemConn,fillval=-1)
        A = np.sum(Area2[R],axis=1)

        # tau = 1/(500*max(A))
        # Z = np.divide(-2*(tau*A)[:,None]*(fP[:,None]*gradP),np.linalg.norm(fP[:,None]*gradP,axis=1)[:,None],where=(fP!=0)[:,None])
        # tau = tf*1/(100*max(A*np.linalg.norm(fP[:,None]*gradP,axis=1)))
        tau = tf*h/(100*max(np.linalg.norm(fP[:,None]*gradP,axis=1)))
        Z = -2*tau*A[:,None]*fP[:,None]*gradP
        return Z
    def Z2Flow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids):
        fC = sdf(Centroids[:,0],Centroids[:,1],Centroids[:,2])
        gradC = -gradF(Centroids)
        Area2 = np.append(Area,0)
        fC = np.append(fC,0)
        gradC = np.vstack([gradC,[0,0,0]])
        R = utils.PadRagged(ElemConn,fillval=-1)
        A = np.sum(Area2[R],axis=1)
        tau = 1/(100*max(A))
        Z = 2*tau*np.sum(Area2[R][:,:,None]*gradC[R]*fC[R][:,:,None],axis=1)/3
        return Z
    def RFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids):
        ### Old slow version ###
        #     U = [1/len(N)*sum([np.subtract(NodeCoords[n],NodeCoords[i]) for n in N]) for i,N in enumerate(NodeNeighbors)]
        #     R = C*np.array([U[i] - np.dot(U[i],NodeNormals[i])*NodeNormals[i] for i in range(len(NodeCoords))])
        ###
        lens = np.array([len(n) for n in NodeNeighbors])
        r = utils.PadRagged(NodeNeighbors,fillval=-1)
        ArrayCoords = np.vstack([NodeCoords,[np.nan,np.nan,np.nan]])
        Q = ArrayCoords[r]
        U = (1/lens)[:,None] * np.nansum(Q - ArrayCoords[:-1,None,:],axis=1)
        R = C*(U - np.sum(U*NodeNormals,axis=1)[:,None]*NodeNormals)
        return R
    def NZRFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=1):
        N = NFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=tf)
        Z = ZFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=tf)
        R = RFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids)
        NZR = N + Z + R
        return NZR
    def ZRFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=1):
        Z = ZFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=tf)
        R = RFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids)
        ZR = Z + R
        return ZR
    def NZFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids, tf=1):
        N = NFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=tf)
        Z = ZFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=tf)
        NZ = N + Z
        return NZ
    def Flip(NodeCoords, NodeConn, ElemNormals, ElemNeighbors, Area, Centroids,threshold=1e-4):
        NodeCoords = np.array(NodeCoords)
        NewConn = copy.copy(NodeConn)
        gradC = gradF(Centroids)
        gradCnorm = np.linalg.norm(gradC,axis=1)
        m = np.divide(gradC,np.reshape(gradCnorm,(len(gradC),1)))
        NormalError = Area*np.array([(1-np.dot(ElemNormals[T],m[T])) for T in range(len(ElemNormals))])
        todo = np.where(NormalError > threshold)[0]
        for i in todo:
            restart = True
            while restart:
                for j in ElemNeighbors[i]:
                    tic = time.time()
                    if len(set(ElemNeighbors[i]).intersection(ElemNeighbors[j])) > 0:
                        # This condition checks if the flip will be legal
                        continue

                    Newi,Newj = improvement.FlipEdge(NodeCoords,NewConn,i,j)
                    [Ci,Cj] = utils.Centroids(NodeCoords,np.array([Newi,Newj]))
                    gradC = gradF(np.vstack([Ci,Cj]))
                    gradCnorm = np.linalg.norm(gradC,axis=1)
                    mi = gradC[0]/gradCnorm[0]
                    mj = gradC[1]/gradCnorm[1]
                    [Ni,Nj] = utils.CalcFaceNormal(NodeCoords,np.array([Newi,Newj]))
                    
                    Ai = np.linalg.norm(np.cross(NodeCoords[Newi[1]]-NodeCoords[Newi[0]],NodeCoords[Newi[2]]-NodeCoords[Newi[0]]))/2
                    Aj = np.linalg.norm(np.cross(NodeCoords[Newj[1]]-NodeCoords[Newj[0]],NodeCoords[Newj[2]]-NodeCoords[Newj[0]]))/2
                    Ei = Ai*(1-np.dot(Ni,mi))
                    Ej = Aj*(1-np.dot(Nj,mj))
                    # Ei = np.arccos(np.dot(Ni,mi))
                    # Ej = np.arccos(np.dot(Nj,mj))
                    OldError = NormalError[i] + NormalError[j]
                    NewError = Ei + Ej
                    if NewError < OldError:
                        NormalError[i] = Ei; NormalError[j] = Ej
                        NewConn[i] = Newi; NewConn[j] = Newj
                        
                        ENi = []; ENj = []
                        Si = set(Newi); Sj = set(Newj)
                        for k in np.unique(ElemNeighbors[i] + ElemNeighbors[j]):
                            if i in ElemNeighbors[k]: ElemNeighbors[k].remove(i)
                            if j in ElemNeighbors[k]: ElemNeighbors[k].remove(j)
                            if len(Si.intersection(NewConn[k])) == 2:
                                ENi.append(k)
                                ElemNeighbors[k].append(i)
                            if len(Sj.intersection(NewConn[k])) == 2:
                                ENj.append(k)
                                ElemNeighbors[k].append(j)

                        ElemNeighbors[i] = ENi; ElemNeighbors[j] = ENj
                        restart = True
                        break
                    else:
                        restart = False
        return NewConn, ElemNeighbors
    def Error(NodeCoords, ElemConn, ElemNormals, Area, Centroids):
        fP = sdf(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2])
        gradP = gradF(NodeCoords)
        gradPnorm = np.linalg.norm(gradP,axis=1)
        gradC = gradF(Centroids)
        gradCnorm = np.linalg.norm(gradC,axis=1)
        m = np.divide(gradC,np.reshape(gradCnorm,(len(gradC),1)))

        area = np.append(Area,0)
        R = utils.PadRagged(ElemConn,fillval=-1)
        A = np.sum(area[R],axis=1)

        VertexError = 1/(3*sum(Area)) * sum((fP**2/gradPnorm**2)*A)
        NormalError = 1/(sum(Area)) * sum(Area*(1-np.sum(ElemNormals*m,axis=1)))

        return VertexError, NormalError

    # edges = converter.surf2edges(NodeCoords,NodeConn)
    # if len(edges) > 0: warnings.warn('Input mesh should be closed and contain no exposed edges.')
    k = 0
    # mesh.mesh(NodeCoords,NodeConn).Mesh2Meshio().write(str(k)+'.vtu');k+=1

    if Subdivision: NodeCoords, NodeConn = AdaptiveSubdivision(sdf, NodeCoords, NodeConn,threshold=1e-3)
    NodeCoords,NodeConn = utils.DeleteDuplicateNodes(NodeCoords,NodeConn)
    NewCoords = np.array(NodeCoords)
    # mesh.mesh(NewCoords,NodeConn).Mesh2Meshio().write(str(k)+'.vtu');k+=1

    NodeNeighbors = utils.getNodeNeighbors(NewCoords, NodeConn) 
    ElemConn = utils.getElemConnectivity(NewCoords, NodeConn)
    ElemNeighbors = utils.getElemNeighbors(NodeCoords,NodeConn,mode='edge')
    # NodeConn, ElemNeighbors = improvement.ValenceImprovementFlips(NodeCoords,NodeConn,NodeNeighbors,ElemNeighbors)
    # vE = [];    nE = []   
    ElemNormals = utils.CalcFaceNormal(NewCoords, NodeConn)
    NodeNormals = np.array(utils.Face2NodeNormal(NewCoords, NodeConn, ElemConn, ElemNormals))
    
    tfs = np.linspace(1,0,ZRIter+1)
    for i in range(ZRIter):
        tf = tfs[i]
        Points = NewCoords[np.array(NodeConn)]
        Area = np.linalg.norm(np.cross(Points[:,1]-Points[:,0],Points[:,2]-Points[:,0]),axis=1)/2   
        Centroids = utils.Centroids(NewCoords, NodeConn)
        # v,n = Error(NewCoords, ElemConn, ElemNormals, Area, Centroids); vE.append(v); nE.append(n)
        NewCoords[FreeNodes] += ZRFlow(NewCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids, tf=tf)[FreeNodes]
        ElemNormals = utils.CalcFaceNormal(NewCoords, NodeConn)
        NodeNormals = np.array(utils.Face2NodeNormal(NewCoords, NodeConn, ElemConn, ElemNormals))
        # mesh.mesh(NewCoords,NodeConn).Mesh2Meshio().write(str(k)+'.vtu');k+=1
    for i in range(NZRIter):
        Points = NewCoords[np.array(NodeConn)]
        Area = np.linalg.norm(np.cross(Points[:,1]-Points[:,0],Points[:,2]-Points[:,0]),axis=1)/2   
        Centroids = utils.Centroids(NewCoords, NodeConn)
        # v,n = Error(NewCoords, ElemConn, ElemNormals, Area, Centroids); vE.append(v); nE.append(n)
        NewCoords[FreeNodes] += NZRFlow(NewCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids)[FreeNodes]
        ElemNormals = utils.CalcFaceNormal(NewCoords, NodeConn)
        NodeNormals = np.array(utils.Face2NodeNormal(NewCoords, NodeConn, ElemConn, ElemNormals))
        # mesh.mesh(NewCoords,NodeConn).Mesh2Meshio().write(str(k)+'.vtu');k+=1
    if NZIter > 0:
        if Subdivision: NewCoords, NodeConn = AdaptiveSubdivision(sdf, NewCoords.tolist(), NodeConn, threshold=1e-4)
        NewCoords = np.array(NewCoords)
        NodeNeighbors = utils.getNodeNeighbors(NewCoords, NodeConn)    
        ElemConn = utils.getElemConnectivity(NewCoords, NodeConn)    
        ElemNeighbors = utils.getElemNeighbors(NewCoords,NodeConn,mode='edge')
        ElemNormals = utils.CalcFaceNormal(NewCoords, NodeConn)
        tfs = np.linspace(1,0,NZIter+1)
    for i in range(NZIter):
        tf = tfs[i]
        Points = NewCoords[np.array(NodeConn)]
        Area = np.linalg.norm(np.cross(Points[:,1]-Points[:,0],Points[:,2]-Points[:,0]),axis=1)/2   
        Centroids = utils.Centroids(NewCoords, NodeConn)

        # v,n = Error(NewCoords, ElemConn, ElemNormals, Area, Centroids)
        # vE.append(v); nE.append(n)

        NewCoords[FreeNodes] += NZFlow(NewCoords, NodeConn, [], NodeNeighbors, ElemConn, Area, Centroids,tf=tf)[FreeNodes]
        
        NewElemNormals = np.array(utils.CalcFaceNormal(NewCoords, NodeConn))

        ### Check for near-intersections ###
        # Angles:
        Points = NewCoords[np.array(NodeConn)]
        v01 = Points[:,1]-Points[:,0]; l01 = np.linalg.norm(v01,axis=1)
        v12 = Points[:,2]-Points[:,1]; l12 = np.linalg.norm(v12,axis=1)
        v20 = Points[:,0]-Points[:,2]; l20 = np.linalg.norm(v20,axis=1)
        alpha = np.arccos(np.sum(v01*-v20,axis=1)/(l01*l20))
        beta = np.arccos(np.sum(v12*-v01,axis=1)/(l12*l01))
        gamma = np.arccos(np.sum(v20*-v12,axis=1)/(l20*l12))
        angles = np.vstack([alpha,beta,gamma]).T
        # Dihedrals:
        dihedrals = quality.SurfDihedralAngles(NewElemNormals,ElemNeighbors)
        # Normal Flipping:
        NormDot = np.sum(NewElemNormals * ElemNormals,axis=1)

        Risk = np.any(angles<5*np.pi/180,axis=1) | np.any(dihedrals > 175*np.pi/180,axis=1) | (NormDot < 0)
        Intersected = []
        if i >= NZIter-5:
            # NodeConn, ElemNeighbors = Flip(NewCoords,NodeConn,ElemNormals,ElemNeighbors,Area,Centroids)
            IntersectionPairs = rays.SurfSelfIntersection(NewCoords,NodeConn)
            Intersected = np.unique(IntersectionPairs).tolist()
            
        if np.any(Risk) or len(Intersected):
            # print('possible intersection')
            ArrayConn = np.array(NodeConn)
            AtRiskElems = np.where(Risk)[0].tolist() + Intersected
            NeighborhoodElems = np.unique([e for i in (ArrayConn[AtRiskElems]).flatten() for e in ElemConn[i]])
            PatchConn = ArrayConn[NeighborhoodElems] 
            BoundaryEdges = converter.surf2edges(NewCoords,PatchConn)
            FixedNodes = set([n for edge in BoundaryEdges for n in edge])
            NewCoords = np.array(improvement.LocalLaplacianSmoothing(NewCoords,PatchConn,2,FixedNodes=FixedNodes))

            # NodeConn = improvement.AngleReductionFlips(NewCoords,NodeConn,NodeNeighbors)
            NodeNeighbors,ElemConn = utils.getNodeNeighbors(NewCoords, NodeConn)    
            ElemNeighbors = utils.getElemNeighbors(NewCoords,NodeConn,mode='edge')
            ElemNormals = np.array(utils.CalcFaceNormal(NewCoords, NodeConn))
        else:
            ElemNormals = NewElemNormals
        # mesh.mesh(NewCoords,NodeConn).Mesh2Meshio().write(str(k)+'.vtu');k+=1
    # px.line(y=vE).show()
    # px.line(y=nE).show()
    return NewCoords.tolist(), NodeConn





