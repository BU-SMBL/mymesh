# -*- coding: utf-8 -*-
# Created on Sat Jan 15 12:02:26 2022
# @author: toj
"""
Delaunay triangulation and related methods

Currently this module consists of several basic implementations of algorithms
related to Delaunay triangulation, as well as interfaces to Delaunay 
triangulation with SciPy (which uses QHull) and Jonathan Shewchuk's Triangle.
Further development with improved Delaunay triangulation and tetrahedralization
capabilities are planned for the future.

Triangulation
=============
.. autosummary::
    :toctree: submodules/
    
    Triangulate
    BowyerWatson2d
    SciPy
    Triangle
    FanTriangulation

Tetrahedralization
==================
.. autosummary::
    :toctree: submodules/

    Tetrahedralize
    BowyerWatson3d

Hulls
=====
.. autosummary::
    :toctree: submodules/

    ConvexHull
    GiftWrapping2d
    QuickHull2d
    AlphaShape
    Alpha2d
    Alpha3d
    AlphaPeel3d    

Geometric Predicates
====================
.. autosummary::
    :toctree: submodules/

    orient2d
    circumcircle
    circumsphere
    convex2d
    circumsphere

    
"""
#%%
import numpy as np
import sys, copy, itertools, warnings, random
from . import utils, rays, converter, mesh, quality
from . import try_njit, check_numba, _MYMESH_USE_NUMBA
from scipy import spatial

def Triangulate(NodeCoords, Constraints=None, method=None, tol=1e-8, steiner=0):
    """
    Generate a triangulation for a 2D set of points. 

    Parameters
    ----------
    NodeCoords : array_like
        Coordinates of nodes to be triangulated. This can be an (n,3) or (n,2)
        array_like, however if given as an (n,3), the third dimension is ignored.
    Constraints : array_like, optional
        List of edge constraints that must be present in the final triangulation, 
        by default None. Edge constraints should be specified by node indices,
        for example [[0, 1], [1,2], ...]
    method : str, optional
        Triangulation method, by default 'BowyerWatson'.

        - 'BowyerWatson' - Generate a Delaunay triangulation by the Bowyer-Watson algorithm (:func:`BowyerWatson2d`)

        - 'scipy' - Use :external+scipy:class:`scipy.spatial.Delaunay`

        - 'Triangle' - Use Jonathon Shewchuk's Delaunay triangulator
    
    Returns
    -------
    T : mymesh.mesh
        Mesh object containing the triangulated mesh.
    """    
    
    Points = np.asarray(NodeCoords)
    if method is None:
        if (Constraints is None or len(Constraints) == 0):
            method = 'BowyerWatson'
        else:
            method = 'BowyerWatson'

    if (Constraints is None or len(Constraints) == 0):
        Points,_,idx = utils.DeleteDuplicateNodes(Points,[],return_idx=True, tol=tol)
        if method.lower() == 'bowyerwatson':
            NodeConn = idx[BowyerWatson2d(Points)]
        elif method.lower() == 'scipy':
            NodeConn = idx[SciPy(Points)]
        elif method.lower() == 'triangle':
            NodeCoords, NodeConn = Triangle(Points,steiner=steiner)
            NodeConn = idx[NodeConn]
        else:
            raise ValueError(f'Method "{method}" not supported for triangulation')
    else: 
        # Constrained Delaunay Triangulation - Sloan (1993)
        # Generate initial triangulation
        if method.lower() == 'bowyerwatson':
            NodeConn = BowyerWatson2d(Points, Constraints=Constraints)
        elif method.lower() == 'triangle':
            NodeCoords, NodeConn = Triangle(Points,Constraints=Constraints,steiner=steiner)
        else:
            raise ValueError(f'Method "{method}" not supported for constrained triangulation')

    if 'mesh' in dir(mesh):
        T = mesh.mesh(NodeCoords,NodeConn)
    else:
        T = mesh(NodeCoords,NodeConn)
    return T

def Tetrahedralize(NodeCoords, method=None, tol=1e-8):
    """
    Generate a Delaunay tetrahedralization for a 3D set of points. 

    Parameters
    ----------
    NodeCoords : array_like
        Coordinates of nodes to be triangulated. This can be an (n,3) or (n,2)
        array_like, however if given as an (n,3), the third dimension is ignored.
    Constraints : array_like, optional
        List of edge constraints that must be present in the final triangulation, 
        by default None. Edge constraints should be specified by node indices,
        for example [[0, 1], [1,2], ...]
    method : str, optional
        Triangulation method, by default 'BowyerWatson'.

        - 'BowyerWatson' - Generate a Delaunay triangulation by the Bowyer-Watson algorithm (:func:`BowyerWatson3d`)

        - 'scipy' - Use :external+scipy:class:`scipy.spatial.Delaunay`

    Returns
    -------
    T : mymesh.mesh
        Mesh object containing the tetrahedralized mesh.
    """    
    
    Points = np.asarray(NodeCoords)
    if method is None:
        method = 'scipy'

    Points,_,idx = utils.DeleteDuplicateNodes(Points,[],return_idx=True, tol=tol)
    if method.lower() == 'bowyerwatson':
        NodeConn = idx[BowyerWatson3d(Points)]
    elif method.lower() == 'scipy':
        NodeConn = idx[SciPy(Points)]
    else:
        raise ValueError(f'Invalid method "{method:s}".')


    if 'mesh' in dir(mesh):
        T = mesh.mesh(NodeCoords,NodeConn)
    else:
        T = mesh(NodeCoords,NodeConn)
    return T

def ConvexHull(NodeCoords, method=None, OrientSurf=True, nD=None):
    """
    Identify the convex hull of a set of points. For a 2D point set 
    (np.shape(NodeCoords) = (n,2)), a 2D convex hull of line elements will be 
    generated. For a 3D point set (np.shape(NodeCoords) = (n,2)), a 3D convex
    hull of triangle elements will be generated.

    Parameters
    ----------
    NodeCoords : array_like
        Coordinates of points around which the convex hull will be identified.
    method : str, optional
        Convex hull method, by default 'QuickHull' for 2D and 'scipy' for 3D.

        - 'QuickHull' - Use the quickhull algorithm (:func:`QuickHull2d`)
        
        - 'GiftWrapping' - Use the gift wrapping algorithm (:func:`GiftWrapping2d`)
        
        - 'scipy' or 'qhull' - Use qhull via :external+scipy:class:`scipy.spatial.ConvexHull`

        - 'BowyerWatson' - Extract the boundary of a Delaunay triangulation by the Bowyer-Watson algorithm (:func:`BowyerWatson2d` or :func:`BowyerWatson3d`). This option is mostly for theoretical interest.

        
    OrientSurf : str, optional
        Ensure the normals of the convex hull are consistently oriented outward,
        by default True. This is only relevant for method=='scipy', other methods are oriented regardless.

    Returns
    -------
    Hull : mymesh.mesh
        Mesh object containing the convex hull. :code:`Hull.Type='line'` for a 2D hull
        or :code:`Hull.Type='surf'` for a 3D hull.

    """    
    if nD is None:
        if np.shape(NodeCoords)[1] == 2:
            nD = 2
        elif np.all(NodeCoords[:,2] == 0):
            nD = 2
        elif np.shape(NodeCoords)[1] == 3:
            nD = 3
        else:
            raise ValueError("n>3 dimensional convex hulls aren't supported, use scipy.spatial.ConvexHull directly.")


    if nD == 2:
        if method is None:
            method = 'quickhull'
        if method.lower() == 'quickhull':
            hull = QuickHull2d(np.asarray(NodeCoords))
        elif method.lower() == 'giftwrapping':
            hull = GiftWrapping2d(np.asarray(NodeCoords),IncludeCollinear=False)
        elif method.lower() == 'scipy' or method.lower() == 'qhull':
            qhull = spatial.ConvexHull(np.asarray(NodeCoords[:,:2], dtype=np.float64))
            hull = qhull.simplices
        elif method.lower() == 'bowyerwatson':
            tri = BowyerWatson2d(NodeCoords)
            hull = converter.surf2edges(NodeCoords, tri)
        else:
            raise Exception(f'Invalid method: "{method:s}" for 2D convex hull.')
        if 'mesh' in dir(mesh):
            Hull = mesh.mesh(NodeCoords, hull, Type='line')
        else:
            Hull = mesh(NodeCoords, hull, Type='line')

    elif nD == 3:
        if method is None:
            method = 'scipy'
        if method.lower() == 'scipy':
            if OrientSurf:
                qhull = spatial.ConvexHull(np.asarray(NodeCoords, dtype=np.float64))
                
                tet = qhull.vertices[SciPy(np.asarray(NodeCoords, dtype=np.float64)[qhull.vertices], FixVol=True)]
                
                hull = np.asarray(converter.solid2surface(NodeCoords, tet))
            else:
                qhull = spatial.ConvexHull(np.asarray(NodeCoords, dtype=np.float64))
                hull = qhull.simplices
        elif method.lower() == 'bowyerwatson':
            tet = BowyerWatson3d(NodeCoords)
            hull = converter.solid2surface(NodeCoords, tet)
        else:
            raise Exception(f'Invalid method: "{method:s}" for 3D convex hull.')

        if 'mesh' in dir(mesh):
            Hull = mesh.mesh(NodeCoords, hull, Type='surf')
        else:
            Hull = mesh(NodeCoords, hull, Type='surf')
            
    else:
        raise ValueError('NodeCoords must contain two or three dimensional data with shape (n,2) or (n,3). Input NodeCoords has shape {str(np.shape(NodeCoords)):s}.')
    
    return Hull

def SciPy(NodeCoords, FixVol=True):
    """
    Wrapper for :external+scipy:class:`scipy.spatial.Delaunay` for 2D triangulation
    or 3D tetrahedralization.

    Parameters
    ----------
    NodeCoords : array_like
        (n,2) or (n,3) node coordinates for the triangulation. Triangulation is 
        only based on the coordinates in the first two dimensions, if an (n,3)
        is provided, the coordinates of the third column is ignored.
    FixVol : bool, optional
        By default some of the signed volumes of the tetrahedra are negative,
        this option reorders the node connectivity to correct this, by default 
        True.

    Returns
    -------
    NodeConn : np.ndarray
        mx3 array of node connectivity for the triangles
    """    
        
    out = spatial.Delaunay(NodeCoords,qhull_options='Qbb Qc Qz Q12 Qt')
    NodeConn = out.simplices
    if np.shape(NodeConn)[1] == 4 and FixVol:
        V = quality.tet_volume(NodeCoords, NodeConn)
        NodeConn[V < 0] = NodeConn[V < 0][:, [2, 1, 0, 3]]
    return NodeConn

def Triangle(NodeCoords,Constraints=None, steiner=0):
    """
    Interface to Jonathan Shewchuk's Triangle via a python wrapper (https://pypi.org/project/triangle/). To use, the python wrapper must be installed (`pip install triangle`).

    Parameters
    ----------
    NodeCoords : array_like
        Array of point coordinates
    Constraints : array_like, optional
        Edge connectivity array of node indices that indicate edges to be ensured
        by constrained Delaunay triangulation, by default None
    steiner : int, optional
        Maximum number of steiner points allowed

    Returns
    -------
    NodeConn : np.ndarray
        mx3 array of node connectivities for the Delaunay triangulation

    """    
    try:
        import triangle
    except:
        raise ImportError("This function interfaces with a python wrapper for Jonathan Shewchuk's Triangle. To install: pip install triangle")
    # Uses Triangle by Jonathan Shewchuk
    if Constraints is None or len(Constraints)==0:
        In = dict(vertices=NodeCoords)
        Out = triangle.triangulate(In,f'qS{steiner:d}')
    else:
        In = dict(vertices=NodeCoords,segments=Constraints)
        Out = triangle.triangulate(In,f'pqcS{steiner:d}')
    try:
        NodeConn = Out['triangles']
        # NodeCoords = Out['vertices']
        if len(Out['vertices']) != len(NodeCoords):
            # If constraints are improperly defined, extra points may be added, but these points most likely already exist
            for v in range(len(NodeCoords),len(Out['vertices'])):
                # print(v)
                All = np.all(np.abs(Out['vertices'][v]-NodeCoords)<1e-12,axis=1)
                if np.any(All):
                    NodeConn[NodeConn==v] = np.where(All)[0][0]
            if np.any(NodeConn >= len(NodeCoords)):
                NodeCoords = Out['vertices']
    except:
        warnings.warn('Error using Triangle, falling back to SciPy')
        NodeConn = SciPy(NodeCoords)

    return NodeCoords, NodeConn
    
def TetGen(NodeCoords, SurfConn, **kwargs):
    """
    Interface to Hang Si's Triangle via a python wrapper developed by the PyVista
    Project :cite:p:`Sullivan2019`. To use, the python wrapper must be installed 
    (`pip install tetgen`).

    Parameters
    ----------
    NodeCoords : array_like
        Array of point coordinates
    SurfConn : array_like
        Node connectivity of the surface mesh to be tetrahedralized. If this 
        surface isn't a triangular surface, it will be converted to one.
    **kwargs : optional
        Optional keyword arguments to tetgen's tetrahedralize function. See
        https://tetgen.pyvista.org/api.html for details and options.
        
        One recommended option is ``nobisect=True`` (equivalent to the -Y 
        command line switch) to preserve the surface of the input mesh. 
        Another is ``switches='-a<vol>'`` where <vol> is the target volume for 
        the tetrahedra (e.g. `switches='-a0.1'). By default, tetgen often 
        creates tetrahedra that get significantly larger far away from the 
        surface, so this option can help control the global element size.
    Returns
    -------
    NewCoords : np.ndarray
        mx3 array of node coordinates for the tetrahedralized mesh
    NewConn : np.ndarray
        mx4 array of node connectivities for the tetrahedralized mesh

    """  
    try:
        import tetgen
    except:
        raise ImportError("This function interfaces with the PyVista python wrapper for Hang Si's TetGen. To install: pip install tetgen")

    NodeCoords, SurfConn = converter.surf2tris(NodeCoords, SurfConn)

    assert len(mesh(NodeCoords, SurfConn, verbose=False).BoundaryNodes) == 0, 'The input mesh has unclosed boundary edges - TetGen will fail to tetrahedralize this input.'

    tet = tetgen.TetGen(NodeCoords, SurfConn)
    NewCoords, NewConn = tet.tetrahedralize(**kwargs)

    return NewCoords, NewConn

@try_njit(cache=True)
def GiftWrapping2d(points, IncludeCollinear=True):
    """
    Gift wrapping algorithm for computing the convex hull of a set of 2D points.

    :cite:`Jarvis1973`

    Parameters
    ----------
    NodeCoords : array_like
        Array of 2D point coordinates

    Returns
    -------
    Hull : np.ndarray
        Node connectivity of convex hull edges (shaoe=(m,2))
    """    

    mask = np.ones(len(points), dtype=np.bool)

    # Pick the starting point offset from the min corner of the bounding box
    eps = .1
    # p_origin = np.array([points[:,0].min(), points[:,1].min()]) - eps
    p0x = points[:,0].min() - eps
    p0y = points[:,1].min() - eps

    i_next = -1
    hull = []
    thetaTotal = 0
    thetaPrev = 0
    k = 0
    while k < 3 or i_next != hull[0]:
        Theta = np.inf
        D2 = np.inf
        for i in range(len(points)):
            if not mask[i] or (k > 1 and i == hull[-1]):
                continue
            d1 = points[i, 1] - p0y
            d0 = points[i, 0] - p0x
            theta = np.arctan2(d1, d0) - thetaTotal
            if theta < 0:
                theta += 2*np.pi
            if theta == Theta:
                d2 = d0**2 + d1**2
                if (IncludeCollinear and d2 < D2) or (not IncludeCollinear and d2 > D2):
                    # if same angle, use closer point for collinear or furthest point
                    Theta = theta
                    i_next = i
                    D2 = d2

            elif theta < Theta:
                D2 = d0**2 + d1**2
                Theta = theta
                i_next = i
                
        if k > 0:
            hull.append(i_next)
            # Point deletion optimization
            if k == 2:
                # calculate angles to the first hull point
                d1 = points[:, 1] - points[hull[0], 1]
                d0 = points[:, 0] - points[hull[0], 0]
                thetas = np.arctan2(d1, d0)
                thetas[thetas<0] += 2*np.pi
            elif k > 2:
                mask[mask] = thetas[mask] >= thetas[i_next]
                mask[hull[0]] = 1 # this is a bit dumb, but need to keep the start unmarked
            
            thetaTotal += Theta
            if thetaTotal > 2*np.pi:
                thetaTotal = thetaTotal % (2*np.pi)
            
        # mask[i_next] = 0
        # p_origin = points[i_next]
        p0x, p0y = points[i_next, 0], points[i_next, 1]
        k += 1
    hull = np.array(hull)[:-1]
    HullConn = np.column_stack((hull, np.roll(hull,-1)))
    return HullConn

@try_njit(cache=True)
def QuickHull2d(points):
    """
    QuickHull algorithm for computing the convex hull of a set of 2D points.

    :cite:`Barber1996`

    Parameters
    ----------
    NodeCoords : array_like
        Array of 2D point coordinates

    Returns
    -------
    Hull : np.ndarray
        Node connectivity of convex hull edges (shape=(m,2))
    """    
    minx = np.min(points[:,0])
    maxx = np.max(points[:,0])
    mins = np.nonzero(points[:,0] == minx)[0]
    maxs = np.nonzero(points[:,0] == maxx)[0]
    p1 = mins[np.argmin(points[mins,1])]
    p2 = maxs[np.argmax(points[maxs,1])]

    indices = np.arange(len(points))
    low = 0
    high = len(indices) - 1
    indices[low], indices[p1] = indices[p1], indices[low]
    if p2 == low:
        indices[high], indices[p1] = indices[p1], indices[high]
    else:
        indices[high], indices[p2] = indices[p2], indices[high]
    low += 1
    high -= 1

    stack = [(p1, p2, low, high)]
    hull = []
    k = 0
    eps = np.finfo(np.float64).eps
    while len(stack) > 0:
        p1, p2, low, high = stack.pop()
        mind = 0
        maxd = 0
        max_idx = -1
        min_idx = -1
        l, h = low, high # l, h are working versions of low and high
        h += 1
        vx, vy = points[p2,0] - points[p1,0], points[p2,1] - points[p1,1]
        norm_denom = 1/np.sqrt(vx**2 + vy**2)
        nx, ny = -vy*norm_denom, vx*norm_denom
        
        while l != h:
            pt_idx = indices[l]
            if pt_idx == p1 or pt_idx == p2:
                dist = 0
                l+=1
                continue
            elif ((points[pt_idx,0] == points[p1,0]) and (points[pt_idx,1] == points[p1,1])) or ((points[pt_idx,0] == points[p2,0]) and (points[pt_idx,1] == points[p2,1])):
                dist = 0
                l+=1
                continue
            else:
                # signed distance
                dist = (points[pt_idx,0] - points[p1,0])*nx + (points[pt_idx,1] - points[p1,1])*ny
            if dist > eps:
                # positive distance, move index to high
                indices[l], indices[h] = indices[h], indices[l]
                h -= 1
                if dist > maxd:
                    maxd = dist
                    max_idx = pt_idx
            else:
                # negative distance, move index to low
                l += 1
                if k == 0 and dist < mind:
                    mind = dist
                    min_idx = pt_idx

        if max_idx != -1:
            stack.append((p1, max_idx, h, high))
            stack.append((max_idx, p2, h, high))
        else:
            hull.append((p2, p1))

        if min_idx != -1:
            stack.append((p2, min_idx, low, l))
            stack.append((min_idx, p1, low, l))

        k += 1  
    return np.array(hull)

def FanTriangulation(NodeCoords, Hull=None):
    """
    Generate a fan triangulation of a two dimensional convex hull around the points.

    Parameters
    ----------
    NodeCoords: array_like
        Coordinates of points whose convex hull will be the basis of the fan
        triangulation. If 3 dimensional coordinates are given, the third coordinate
        will be ignored
    Hull : array_like
        Node connectivity of the 2D convex hull. If not provided, it will be 
        calculated internally.

    Returns
    -------
    NodeConn : np.ndarray
        Nodal connectivity of the triangulated hull.
    """
    NodeCoords = np.asarray(NodeCoords)
    if Hull is None:
        _, Hull = ConvexHull(NodeCoords[:,:2])
    else:
        Hull = np.asarray(Hull)
    HullShape = np.shape(Hull)
    assert len(HullShape) == 2, 'Hull must be a two-dimensional array of node connectivities.'
    assert HullShape[0] >= 3, 'Convex hull must contain at least 3 elements.'
    assert HullShape[1] == 2, 'Convex hull must be two dimensional, containing line elements (shape(Hull)=(m,2)).'
    
    idx = np.all(Hull!=Hull[0][0],axis=1)
    NodeConn = np.column_stack([np.repeat(Hull[0][0], len(Hull)-2), 
                                Hull[idx,0],
                                Hull[idx,1]
                            ])
    return NodeConn
    
def TriangleSplitting(NodeCoords, Hull=None):
    # This should be rewritten to use data structures like BowyerWatson
    assert len(NodeCoords) > 2, 'At least three points are required.'
    if NodeCoords.shape[1] == 2:
        Points = np.asarray(NodeCoords)
    else:
        warnings.warn('TriangleSplitting is only valid for points on a plane, the third dimension is ignored.')
        Points = np.asarray(NodeCoords)[:,:2]


    if Hull is None: Hull = GiftWrapping(Points)
    NodeConn = FanTriangulation(Hull)

    interior = np.setdiff1d(np.arange(len(NodeCoords)),Hull,assume_unique=True)
    for i in interior:
        alpha,beta,gamma = utils.BaryTris(Points[NodeConn],Points[i])
        
        # currently not using special treatment for nodes on boundaries
        inside = (alpha >= 0) & (beta >= 0) & (gamma >= 0)
        TriId = np.where(inside)[0]
        if len(TriId) > 1:
            a = 2
        else:
            TriId = TriId[0]
        Elem = copy.copy(NodeConn[TriId])
        NodeConn[TriId] = [Elem[0],Elem[1],i]
        NodeConn = np.append(NodeConn,[[Elem[1],Elem[2],i],[Elem[2],Elem[0],i]],axis=0)

    return NodeConn
        
def BowyerWatson2d(NodeCoords, Constraints=None):
    """
    Bowyer-Watson algorithm for 2D Delaunay triangulation

    :cite:p:`Bowyer1981`, :cite:p:`Watson1981`

    Parameters
    ----------
    NodeCoords : array_like
        (n,2) or (n,3) array of points to be triangulated. If three dimensional
        coordinates are given, the third coordinate will be ignored.

    Returns
    -------
    NodeConn : np.ndarray
        mx3 array of node connectivities for the Delaunay triangulation
    """
    if not check_numba():
        warnings.warn('Using numba is strongly recommended for efficiency of BowyerWatson2d.')

    NodeCoords = np.asarray(NodeCoords)
    assert NodeCoords.shape[0] >= 3, 'At least three points are required.'
    if NodeCoords.shape[1] == 2:
        TempCoords = NodeCoords
    else:
        warnings.warn('BowyerWatson2d is only valid for points on a plane, the third dimension is ignored.')
        TempCoords = NodeCoords[:,:2]

    nPts = len(NodeCoords)

    # Random insertion order for points
    # indices = list(range(nPts))
    # rng = np.random.default_rng()
    # rng.shuffle(indices)
    indices = _bin_sort_2d(NodeCoords)

    # Get super triangle - triangle with incircle that bounds the point set
    center = np.mean(TempCoords, axis=0)
    r = np.max(np.sqrt((TempCoords[:,0]-center[0])**2 + (TempCoords[:,1]-center[1])**2))
    R = 10*r

    super_triangle_points = np.array([
                                    [center[0], center[1]-2*R],
                                    [center[0]+R*np.sqrt(3), center[1]+R],
                                    [center[0]-R*np.sqrt(3), center[1]+R]
                            ])    
    TempCoords = np.hstack([np.vstack([TempCoords, super_triangle_points]), np.repeat(0,nPts+3)[:,None]])
    super_tri = (nPts, nPts+1, nPts+2)

    
    if 'mesh' in dir(mesh):
        m = mesh.mesh(TempCoords, [super_tri], Type='surf', verbose=False)
    else:
        m = mesh(TempCoords, [super_tri], Type='surf', verbose=False)
    
    d = m.mesh2dmesh()

    d = _bowyer_watson_loop_2d(d, indices)

    # Remove super triangle
    d.removeElems(d.getElemConn(nPts+2))
    d.removeElems(d.getElemConn(nPts+1))
    d.removeElems(d.getElemConn(nPts))
    # Insert constraints
    if Constraints is not None:
        d.NodeLabels = np.zeros(len(d.raw_NodeCoords), dtype=np.int64)
        d.NodeLabels[Constraints] = 1

        for segment in Constraints:
            d = _insert_segment_2d(d, segment)

    return d.NodeConn

def BowyerWatson3d(NodeCoords):
    """
    Bowyer-Watson algorithm for 2D Delaunay triangulation

    :cite:p:`Bowyer1981`, :cite:p:`Watson1981`

    Parameters
    ----------
    NodeCoords : array_like
        (n,2) or (n,3) array of points to be triangulated. If three dimensional
        coordinates are given, the third coordinate will be ignored.

    Returns
    -------
    NodeConn : np.ndarray
        mx3 array of node connectivities for the Delaunay triangulation
    """
    if not check_numba():
        warnings.warn('Using numba is strongly recommended for efficiency of BowyerWatson2d.')

    NodeCoords = np.asarray(NodeCoords)
    assert NodeCoords.shape[0] >= 3, 'At least three points are required.'
    if NodeCoords.shape[1] == 3:
        TempCoords = NodeCoords
    else:
        raise ValueError('BowyerWatson3d is only valid for three dimensional points.')

    nPts = len(NodeCoords)

    # Random insertion order for points
    indices = _bin_sort_3d(NodeCoords)

    # Get super tet - tet with incircle that bounds the point set
    center = np.mean(TempCoords, axis=0)
    r = np.max(np.sqrt((TempCoords[:,0]-center[0])**2 + (TempCoords[:,1]-center[1])**2))
    R = 10*r

    # Get super tetrahedron - tetrahedron with insphere that bounds the point set
    center = np.mean(NodeCoords, axis=0)
    r = np.max(np.sqrt((NodeCoords[:,0]-center[0])**2 + (NodeCoords[:,1]-center[1])**2 + (NodeCoords[:,2]-center[2])**2))
    R = r + 1000*r/10
    a = R*np.sqrt(24) # side length of tetrahedron

    super_tet_points = np.array([
                                [center[0]-a/2, center[1]-np.sqrt(3)*a/6, center[2]-R],
                                [center[0]+a/2, center[1]-np.sqrt(3)*a/6, center[2]-R],
                                [center[0],     center[1]+np.sqrt(3)*a/3, center[2]-R],
                                [center[0],     center[1], center[2]+np.sqrt(6)*a/3-R]
                            ])    
    TempCoords = np.vstack([NodeCoords, super_tet_points])
    super_tet = (nPts, nPts+1, nPts+2, nPts+3)

    
    if 'mesh' in dir(mesh):
        m = mesh.mesh(TempCoords, [super_tet], Type='vol', verbose=False)
    else:
        m = mesh(TempCoords, [super_tet], Type='vol', verbose=False)
    
    d = m.mesh2dmesh()

    d = _bowyer_watson_loop_3d(d, indices)

    # Remove super triangle
    d.removeElems(d.getElemConn(nPts+3))
    d.removeElems(d.getElemConn(nPts+2))
    d.removeElems(d.getElemConn(nPts+1))
    d.removeElems(d.getElemConn(nPts))
    # Insert constraints
    # if Constraints is not None:
    #     d.NodeLabels = np.zeros(len(d.raw_NodeCoords), dtype=np.int64)
    #     d.NodeLabels[Constraints] = 1

    #     for segment in Constraints:
    #         d = _insert_segment_2d(d, segment)

    return d.NodeConn

def AlphaShape(NodeCoords, alpha, method=None, Type='surf'):
    """
    Alpha shapes in 2D or 3D

    Parameters
    ----------
    NodeCoords : array_like
        Node coordinates. If coordinates are two dimensional (shape=(n,2)),
        a 2D alpha shape will be produce, for three dimensional coordinates,
        a 3D alpha shape will be produced.
    alpha : float or list of floats
        Alpha value. If given as a list of values, a corresponding list of 
        meshes will be returned
    method : str, optional
        Delaunay triangulation/tetrahedralization method used to determine the 
        alpha shape, by default None. See :func:`Triangulate`/
        :func:`Tetrahedralize` for more details and the default method.
    Type : str, optional
        Type of the returned Mesh or Meshes, by default 'surf'. Note that if  
        using Type='vol', tetrahedral meshes may have small unnoticed holes that 
        could pose problems for some applications. 

    Returns
    -------
    M : mymesh.mesh or list of mymesh.mesh
        Mesh of the alpha shape. If alpha is given as a list, a corresponding
        list of meshes will be returned.
    """    

    if np.shape(NodeCoords)[1] == 2:
        # 2D
        M = Alpha2d(NodeCoords, alpha, method, Type)
    elif np.shape(NodeCoords)[1] == 3:
        # 3D
        M = Alpha3d(NodeCoords, alpha, method, Type)
    return M

def Alpha2d(NodeCoords, alpha, method='scipy', Type='line'):
    """
    2D Alpha shapes

    Parameters
    ----------
    NodeCoords : array_like
        Node coordinates
    alpha : float or list of floats
        Alpha value. If given as a list of values, a corresponding list of 
        meshes will be returned
    method : str, optional
        Delaunay triangulation method used to determine the alpha shape, by 
        default None. See :func:`Triangulate` for more details and the default 
        method.
    Type : str, optional
        Type of the returned Mesh or Meshes, by default 'line'. Note that if  
        using Type='surf', triangular meshes may have small unnoticed holes that 
        could pose problems for some applications. 

    Returns
    -------
    M : mymesh.mesh or list of mymesh.mesh
        Mesh of the alpha shape. If alpha is given as a list, a corresponding
        list of meshes will be returned.
    """    
    T = Triangulate(NodeCoords, method=method)
    T.verbose=False
    R = quality.tri_circumradius(T.NodeCoords, T.NodeConn)
    if isinstance(alpha, (list, tuple, np.ndarray)):
        M = []
        for a in alpha:
            thresh = 1/a if a != 0 else np.inf
            m = T.Threshold(R, (0,thresh), 'in', InPlace=False)
            if Type.lower() == 'line':
                M.append(m.Boundary)
            else:
                M.append(m)
    else:
        thresh = 1/alpha if alpha != 0 else np.inf
        T.Threshold(R, (0,thresh), 'in', InPlace=True)
        if Type.lower() == 'line':
            M = T.Boundary
        else:
            M = T
    return M

def Alpha3d(NodeCoords, alpha, method=None, Type='surf'):
    """
    3D Alpha shapes

    Parameters
    ----------
    NodeCoords : array_like
        Node coordinates
    alpha : float or list of floats
        Alpha value. If given as a list of values, a corresponding list of 
        meshes will be returned
    method : str, optional
        Delaunay tetrahedralization method used to determine the alpha shape, by 
        default None. See :func:`Tetrahedralize` for more details and the 
        default method.
    Type : str, optional
        Type of the returned Mesh or Meshes, by default 'surf'. Note that if  
        using Type='vol', tetrahedral meshes may have small unnoticed holes that 
        could pose problems for some applications. 

    Returns
    -------
    M : mymesh.mesh or list of mymesh.mesh
        Mesh of the alpha shape. If alpha is given as a list, a corresponding
        list of meshes will be returned.
    """    
    T = Tetrahedralize(NodeCoords, method=method)
    T.verbose=False
    R = quality.tet_circumradius(T.NodeCoords, T.NodeConn)
    
    if isinstance(alpha, (list, tuple, np.ndarray)):
        M = []
        for a in alpha:
            m = T.Threshold(R, (0,1/a), 'in', InPlace=False)
            if Type.lower() == 'surf':
                M.append(m.Surface)
            else:
                M.append(m)
    else:
        T.Threshold(R, (0,1/alpha), 'in', InPlace=True)
        if Type.lower() == 'surf':
            M = T.Surface
        else:
            M = T
    return M

def AlphaPeel3d(NodeCoords, alpha, method='scipy', Type='surf'):
    """
    3D Alpha shapes

    Parameters
    ----------
    NodeCoords : array_like
        Node coordinates
    alpha : float or list of floats
        Alpha value. If given as a list of values, a corresponding list of 
        meshes will be returned
    method : str, optional
        Delaunay tetrahedralization method used to determine the alpha shape, by 
        default 'scipy'. See :func:`Tetrahedralize` for more details.
    Type : str, optional
        Type of the returned Mesh or Meshes, by default 'surf'. Note that if  
        using Type='vol', tetrahedral meshes may have small unnoticed holes that 
        could pose problems for some applications. 

    Returns
    -------
    M : mymesh.mesh or list of mymesh.mesh
        Mesh of the alpha shape. If alpha is given as a list, a corresponding
        list of meshes will be returned.
    """    
    T = Tetrahedralize(NodeCoords, method=method)
    T.verbose=False
    R = quality.tet_circumradius(T.NodeCoords, T.NodeConn)

    _, SurfElem = converter.solid2surface(T.NodeCoords, T.NodeConn, return_SurfElem=True)
    ElemNeighbors = T.ElemNeighbors

    Peelable = set(SurfElem[R[SurfElem] > 1/alpha])
    Peeled = set()
    while len(Peelable) > 0:

        NextLayer = []
        Peeled.update(Peelable)
        Peelable = {elem for i in Peelable for elem in ElemNeighbors[i] if R[elem] > 1/alpha and elem not in Peeled}

    T.removeElems(Peeled)
    if Type.lower() == 'surf':
        return T.Surface
    
    return T

## Utils ##
@try_njit(inline='always', cache=True)
def _bin_sort_2d(points):
    # based on sloan 1992
    P = np.empty(points.shape, dtype=np.float32)
    # Psort = np.empty(points.shape, dtype=np.float32)
    indices = np.empty(len(points),dtype=np.uint64)
    n = int(np.ceil(len(P)**(1/4)))
    b = np.empty(len(P), dtype=np.uint32)
    bin_counts = np.zeros(n*n, dtype=np.uint32)
    xmax, xmin = points[:,0].max(), points[:,0].min()
    ymax, ymin = points[:,1].max(), points[:,1].min()
    dmax = np.maximum(xmax-xmin,ymax-ymin)
    invdmax = 1/dmax
    
    _xmax = (xmax - xmin)*invdmax
    _ymax = (ymax - ymin)*invdmax

    for idx in range(len(P)):
        P[idx, 0] = (points[idx, 0] - xmin) * invdmax
        P[idx, 1] = (points[idx, 1] - ymin) * invdmax
        i = int(0.99 * n * P[idx,1]/_ymax)
        j = int(0.99 * n * P[idx,0]/_xmax)

        if i%2 == 0:
            b[idx] = i * n + j
        else:
            b[idx] = (i + 1) * n - j - 1
        bin_counts[b[idx]] += 1
    
    bin_starts = np.zeros(n*n, dtype=np.uint64)
    for bin_idx in range(1, n*n):
        bin_starts[bin_idx] = bin_counts[bin_idx-1] + bin_starts[bin_idx-1]
    
    for idx,bidx in enumerate(b):
        indices[bin_starts[bidx]] = idx
        bin_starts[bidx] += 1
    
    return indices

@try_njit
def _bin_sort_3d(points):
    # based on sloan 1992
    P = np.empty(points.shape, dtype=np.float32)
    # Psort = np.empty(points.shape, dtype=np.float32)
    indices = np.empty(len(points),dtype=np.uint64)
    n = int(np.ceil(len(P)**(1/4)))
    b = np.empty(len(P), dtype=np.uint32)
    bin_counts = np.zeros(n*n*n, dtype=np.uint32)
    xmax, xmin = points[:,0].max(), points[:,0].min()
    ymax, ymin = points[:,1].max(), points[:,1].min()
    zmax, zmin = points[:,2].max(), points[:,2].min()
    dmax = np.maximum(np.maximum(xmax-xmin, ymax-ymin), zmax-zmin)
    invdmax = 1/dmax
    
    _xmax = (xmax - xmin)*invdmax
    _ymax = (ymax - ymin)*invdmax
    _zmax = (zmax - zmin)*invdmax

    for idx in range(len(P)):
        P[idx, 0] = (points[idx, 0] - xmin) * invdmax
        P[idx, 1] = (points[idx, 1] - ymin) * invdmax
        P[idx, 2] = (points[idx, 2] - zmin) * invdmax
        i = int(0.99 * n * P[idx,2]/_zmax) 
        j = int(0.99 * n * P[idx,1]/_ymax)
        k = int(0.99 * n * P[idx,0]/_xmax)
        ######################################
        # this is a pretty lazy extension to 3D, could be better
        if i%2 == 0:
            b[idx] = i * n + j + (k*n**2)
        else:
            b[idx] = (i + 1) * n - j - 1 + (k*n**2)
        ######################################
        bin_counts[b[idx]] += 1
    
    bin_starts = np.zeros(n*n*n, dtype=np.uint64)
    for bin_idx in range(1, n*n*n):
        bin_starts[bin_idx] = bin_counts[bin_idx-1] + bin_starts[bin_idx-1]
    
    for idx,bidx in enumerate(b):
        indices[bin_starts[bidx]] = idx
        bin_starts[bidx] += 1
    
    return indices

@try_njit(inline='always', cache=True)
def _bowyer_watson_loop_2d(d, indices, nsample=1):
    for i in indices:
        newPt = d.raw_NodeCoords[i]
        tri_id = _walk_2d(d, newPt, tri_id=d.NElem-1, nsample=nsample)
        # Breadth first search of adjacent triangles to find all invalid triangles
        bad_triangles, cavity_edges = _build_cavity_2d(d, tri_id, newPt)
        
        # Remove triangles and edges
        d.removeElems(bad_triangles)

        # Create new triangles and edges
        for e in cavity_edges:
            d.addElem([e[0], e[1], i])
        
    return d

@try_njit(inline='always')#, cache=True)
def _bowyer_watson_loop_3d(d, indices, nsample=1):
    for i in indices:
        newPt = d.raw_NodeCoords[i]
        tet_id = _walk_3d(d, newPt, tet_id=d.NElem-1)#, nsample=nsample)
        # Breadth first search of adjacent tets to find all invalid tets
        bad_tets, cavity_faces = _build_cavity_3d(d, tet_id, newPt)
        
        # Remove tets
        d.removeElems(bad_tets)

        # Create new triangles and edges
        for f in cavity_faces:
            d.addElem(np.array([f[2], f[1], f[0], i]))
        
    return d

@try_njit(inline='always', cache=True)
def _walk_2d(d, newPt, tri_id=None, nsample=1):

    if tri_id is None:
        tri_id = np.random.randint(0,d.NElem)
        tri = d.raw_NodeConn[tri_id]
        if nsample > 1:
            # Try multiple start points and choose the closest
            minL = (d.raw_NodeCoords[tri[0],0] - newPt[0])**2 + (d.raw_NodeCoords[tri[0],1] - newPt[1])**2 # squared distance
            if nsample > d.NElem:
                nsample = d.NElem
            for i in range(nsample-1):
                t_id = np.random.randint(0,d.NElem)
                t = d.raw_NodeConn[t_id]
                L = (d.raw_NodeCoords[t[0],0] - newPt[0])**2 + (d.raw_NodeCoords[t[0],1] - newPt[1])**2 # squared distance
                if L < minL:
                    tri = t
                    tri_id = t_id
    else:
        tri = d.raw_NodeConn[tri_id]
    alpha, beta, gamma = utils.BaryTri(d.raw_NodeCoords[tri], newPt, d=2)
    while not (alpha >= 0 and beta >= 0 and gamma >= 0):
        # find node with smallest (most negative) barycentric coordinate
        if alpha <= beta and alpha <= gamma:
            # alpha is min
            edge_n1 = tri[1]
            edge_n2 = tri[2]
        elif beta <= alpha and beta <= gamma:
            # beta is min
            edge_n1 = tri[0]
            edge_n2 = tri[2]
        else:
            # gamma is min
            edge_n1 = tri[0]
            edge_n2 = tri[1]
        
        # step into the neighboring triangle 
        # directly using the linked lists rather than getElemConn to minimize
        # overhead and enable early exits
        next_tri_id = -1
        i = d.ElemConn_head[edge_n1]
        while i != -1:
            next_elem = d.ElemConn_elem[i]
            # iterate through elem conn for the first node in the edge
            if next_elem != tri_id:
                # skip the current triangle
                t = d.raw_NodeConn[next_elem]
                if (t[0] == edge_n2) or (t[1] == edge_n2) or (t[2] == edge_n2):
                    # element is opposite a shared edge, step into it
                    next_tri_id = next_elem
                    break                    
            i = d.ElemConn_next[i] 
        if next_tri_id != -1:
            tri_id = next_tri_id
            tri = t            
            
        alpha, beta, gamma = utils.BaryTri(d.raw_NodeCoords[tri], newPt)
    return tri_id

@try_njit(inline='always', cache=True)
def _build_cavity_2d(d, tri_id, newPt):

    tri = d.raw_NodeConn[tri_id]
    # Queue contains the id of a triangle followed by the two vertices that define an edge of that triangle
    queue = [(tri_id, tri[0], tri[1]),
             (tri_id, tri[1], tri[2]),
             (tri_id, tri[2], tri[0])]
    visited = [tri_id,]
    bad_triangles = [tri_id,]
    cavity_edges = []
   
    # super triangle nodes
    super_cutoff = d.NNode - 3 # nodes >= super_cutoff are part of the super triangle

    while len(queue) > 0:

        prev_t_id, e0, e1 = queue.pop() # triangle ID, edge vertex 1, edge vertex 2

        next_t_id = -1
        i = d.ElemConn_head[e0]
        while i != -1:
            next_elem = d.ElemConn_elem[i]
            # iterate through elem conn for the first node in the edge
            if next_elem != prev_t_id:
                # skip the current triangle
                t = d.raw_NodeConn[next_elem]
                if (t[0] == e1) or (t[1] == e1) or (t[2] == e1):
                    # element is opposite a shared edge, step into it
                    next_t_id = next_elem
                    break                    
            i = d.ElemConn_next[i] 
        if next_t_id == -1:
            # boundary edge
            cavity_edges.append((e0, e1))
            continue

        if next_t_id in visited:
            # triangle has already been checked
            if prev_t_id in bad_triangles and next_t_id not in bad_triangles:
                cavity_edges.append((e0, e1))
            continue

        tri = d.raw_NodeConn[next_t_id]

        if ((tri[0] >= super_cutoff) ^ (tri[1] >= super_cutoff) ^ (tri[2] >= super_cutoff)) and (e0 < super_cutoff and e1 < super_cutoff):
            # ^ = XOR
            # TODO: verify that this is necessary/correct
            # triangle is connected to super triangle, mark boundary
            cavity_edges.append((e0, e1))
            visited.append(next_t_id)
            continue
            
        # test circumcircle
        # manual determinant of matrix [[A,B,C],[D,E,F],[G,H,I]]
        A = d.raw_NodeCoords[tri[0], 0] - newPt[0]
        B = d.raw_NodeCoords[tri[0], 1] - newPt[1]
        C = (d.raw_NodeCoords[tri[0], 0] - newPt[0])**2 + (d.raw_NodeCoords[tri[0], 1] - newPt[1])**2

        D = d.raw_NodeCoords[tri[1], 0] - newPt[0]
        E = d.raw_NodeCoords[tri[1], 1] - newPt[1]
        F = (d.raw_NodeCoords[tri[1], 0] - newPt[0])**2 + (d.raw_NodeCoords[tri[1], 1] - newPt[1])**2

        G = d.raw_NodeCoords[tri[2], 0] - newPt[0]
        H = d.raw_NodeCoords[tri[2], 1] - newPt[1]
        I = (d.raw_NodeCoords[tri[2], 0] - newPt[0])**2 + (d.raw_NodeCoords[tri[2], 1] - newPt[1])**2

        det = A*(E*I-F*H) - B*(D*I-F*G) + C*(D*H-E*G)

        if det > 0:
            # point in cicrumcircle of tri; add edges to queue (except for the edge that was just used)
            bad_triangles.append(next_t_id)
            
            if (
            (tri[0] == e0 or tri[0] == e1) and (tri[1] == e0 or tri[1] == e1)
            ):
                # old edge is (tri[0], tri[1])
                queue.append((next_t_id, tri[1], tri[2]))
                queue.append((next_t_id, tri[2], tri[0]))
            elif(
            (tri[1] == e0 or tri[1] == e1) and (tri[2] == e0 or tri[2] == e1)
            ):
                # old edge is (tri[1], tri[2])
                queue.append((next_t_id, tri[0], tri[1]))
                queue.append((next_t_id, tri[2], tri[0]))
            else:
                # old edge is (tri[0], tri[2])
                queue.append((next_t_id, tri[0], tri[1]))
                queue.append((next_t_id, tri[1], tri[2]))

        else:
            # boundary between a valid and invalid triangle
            cavity_edges.append((e0, e1))
            
        visited.append(next_t_id)
    return bad_triangles, cavity_edges

@try_njit
def _insert_segment_2d(D, segment):
    # Other constraint edges should be marked by labeling their nodes with D.NodeLabels[i] = 1
    n0, n1 = segment
    elems = D.getElemConn(n0)

    flip_queue = []
    next_elem = -1
    # Find first edge intersection:
    for e in elems:
        a, b, c = D.NodeConn[e]

        if a == n0:
            if b == n1 or c == n1:
                # segment is already in the mesh
                return D
            if segment_intersect2d(D.raw_NodeCoords[n0], 
                                    D.raw_NodeCoords[n1], 
                                    D.raw_NodeCoords[b], 
                                    D.raw_NodeCoords[c]):
                # intersection - (b, c) will need to be flipped
                next_elem = D.get_TriEdgeNeighbor(e, b, c)
                prev_edge = (b, c)
                flip_queue.append(prev_edge)
                break
        elif b == n0:
            if a == n1 or c == n1:
                # segment is already in the mesh
                return D
            if segment_intersect2d(D.raw_NodeCoords[n0], 
                                    D.raw_NodeCoords[n1], 
                                    D.raw_NodeCoords[c], 
                                    D.raw_NodeCoords[a]):
                # intersection - (c, a) will need to be flipped
                next_elem = D.get_TriEdgeNeighbor(e, c, a)
                prev_edge = (c, a)
                flip_queue.append(prev_edge)
                break
        elif c == n0:
            if a == n1 or b == n1:
                # segment is already in the mesh
                return D
            if segment_intersect2d(D.raw_NodeCoords[n0], 
                                    D.raw_NodeCoords[n1], 
                                    D.raw_NodeCoords[a], 
                                    D.raw_NodeCoords[b]):
                # intersection - (a, b) will need to be flipped
                next_elem = D.get_TriEdgeNeighbor(e, a, b)
                prev_edge = (a, b)
                flip_queue.append(prev_edge)
                break

    if next_elem == -1:
        # the intersected edge has no neighbor - this should be an impossible scenario for properly defined mesh
        raise Exception('Segment intersects mesh boundary - unexpected scenario')
    # Walk from the first element to the other end of the segment, finding all intersections
    while next_elem != -1:
        a, b, c = D.NodeConn[next_elem]
        
        if n1 == a or n1 == b or n1 == c:
            # This element contains the other end of the segment, terminate search
            break

        # set n = node opposite the edge, the two segments to check are (n, s1) and (n,s2)
        if a != prev_edge[0] and a != prev_edge[1]:
            # a is the node opposite the previous edge
            s1 = b
            s2 = c
            n = a
        elif b != prev_edge[0] and b != prev_edge[1]:
            # b is the node opposite the previous edge
            s1 = c
            s2 = a
            n = b
        elif c != prev_edge[0] and c != prev_edge[1]:
            # c is the node opposite the previous edge
            s1 = a
            s2 = b
            n = c 
        else:
            raise Exception("This shouldn't happen")

        if segment_intersect2d(D.raw_NodeCoords[n0], 
                                D.raw_NodeCoords[n1], 
                                D.raw_NodeCoords[s1], 
                                D.raw_NodeCoords[n]):
            # intersection - (s1, n) must be flipped
            next_elem = D.get_TriEdgeNeighbor(next_elem, s1, n)
            prev_edge = (s1, n)
            flip_queue.append(prev_edge)
        else:
            # intersection - (s2, n) must be flipped
            next_elem = D.get_TriEdgeNeighbor(next_elem, s2, n)
            prev_edge = (s2, n)
            flip_queue.append(prev_edge)

    # Perform flips to eliminate intersections
    new_edges = []
    while len(flip_queue) > 0:
        edge = flip_queue.pop(0)
        new_edge = D.TriFlipEdge(edge[0], edge[1])
        if new_edge[0] == -1:
            # flip failed
            flip_queue.append(edge)
        else:
            # First part of this check is probably problematic in edge cases
            # without it, the intersection test says edges starting/ending at the constraint nodes intersect
            if (n0 in new_edge or n1 in new_edge) or not segment_intersect2d(D.raw_NodeCoords[n0], 
                            D.raw_NodeCoords[n1], 
                            D.raw_NodeCoords[new_edge[0]], 
                            D.raw_NodeCoords[new_edge[1]]):
                new_edges.append(new_edge)
            else:
                flip_queue.append(new_edge)

    # Perform flips to restore Delaunay criteria where possible
    nswaps = 1
    passes = 0
    while nswaps > 0:
        nswaps = 0
        passes += 1
        for i,edge in enumerate(new_edges):
            n1, n2 = edge
            if n1 == segment[0]:
                if n2 == segment[1]:
                    continue
            elif n2 == segment[0]:
                if n1 == segment[1]:
                    continue
            elif D.NodeLabels is not None and D.NodeLabels[n1] == D.NodeLabels[n2] == 1:
                # This edge is also a constraint
                continue
            
            
            elems = D.get_TriEdgeConn(n1, n2)
            
            a, b, c = D.NodeConn[elems[0]]
            d, e, f = D.NodeConn[elems[1]]

            if a != n1 and a != n2:
                n3 = a
            elif b != n1 and b != n2:
                n3 = b
            else:
                n3 = c

            if d != n1 and d != n2:
                n4 = d
            elif e != n1 and e != n2:
                n4 = e
            else:
                n4 = f

            if circumcircle(D.raw_NodeCoords[a], 
                            D.raw_NodeCoords[b], 
                            D.raw_NodeCoords[c], 
                            D.raw_NodeCoords[n4]) or \
                circumcircle(D.raw_NodeCoords[d], 
                                D.raw_NodeCoords[e], 
                                D.raw_NodeCoords[f], 
                                D.raw_NodeCoords[n3]):
                # delaunay condition not satisfied

                if not convex2d(D.raw_NodeCoords[np.array([n1,n3,n2,n4]),:]):
                    continue

                # perform flip 
                # NOTE: if I in-lined the convexity test I wouldn't need to repeat the orientation tests
                D.removeElems(elems)
                if orient2d(D.raw_NodeCoords[n3], D.raw_NodeCoords[n4], D.raw_NodeCoords[n1]) > 0:
                    D.addElem(np.array([n3, n4, n1]))
                else:
                    D.addElem(np.array([n1, n4, n3]))

                if orient2d(D.raw_NodeCoords[n3], D.raw_NodeCoords[n4], D.raw_NodeCoords[n2]) > 0:
                    D.addElem(np.array([n3, n4, n2]))
                else:
                    D.addElem(np.array([n2, n4, n3]))
                
                new_edges[i] = (n3, n4)
                nswaps += 1
    return D

@try_njit
def _walk_3d(d, newPt, tet_id=None):

    if tet_id is None:
        tet_id = np.random.randint(0,d.NElem)
        tet = d.raw_NodeConn[tet_id]
        # if nsample > 1:
        #     # Try multiple start points and choose the closest
        #     pass
            # minL = (d.raw_NodeCoords[tri[0],0] - newPt[0])**2 + (d.raw_NodeCoords[tri[0],1] - newPt[1])**2 # squared distance
            # if nsample > d.NElem:
            #     nsample = d.NElem
            # for i in range(nsample-1):
            #     t_id = np.random.randint(0,d.NElem)
            #     t = d.raw_NodeConn[t_id]
            #     L = (d.raw_NodeCoords[t[0],0] - newPt[0])**2 + (d.raw_NodeCoords[t[0],1] - newPt[1])**2 # squared distance
            #     if L < minL:
            #         tri = t
            #         tri_id = t_id
    else:
        tet = d.raw_NodeConn[tet_id]
    alpha, beta, gamma, delta = utils.BaryTet(d.raw_NodeCoords[tet], newPt)
    while not (alpha >= 0 and beta >= 0 and gamma >= 0 and delta >= 0):
        # find node with smallest (most negative) barycentric coordinate
        if alpha <= beta and alpha <= gamma and alpha <= delta:
            # alpha is min
            face_n1 = tet[1]
            face_n2 = tet[2]
            face_n3 = tet[3]
        elif beta <= alpha and beta <= gamma and beta <= delta:
            # beta is min
            face_n1 = tet[0]
            face_n2 = tet[2]
            face_n3 = tet[3]
        elif gamma <= alpha and gamma <= beta and gamma <= delta:
            # gamma is min
            face_n1 = tet[0]
            face_n2 = tet[1]
            face_n3 = tet[3]
        else:
            # delta is min
            face_n1 = tet[0]
            face_n2 = tet[1]
            face_n3 = tet[2]

        # step into the neighboring tet 
        ##################
        # directly using the linked lists rather than getElemConn to minimize
        # overhead and enable early exits

        next_tet_id = -1
        i = d.ElemConn_head[face_n1]
        while i != -1:
            next_elem = d.ElemConn_elem[i]
            # iterate through elem conn for the first node in the edge
            if next_elem != tet_id:
                # skip the current triangle
                t = d.raw_NodeConn[next_elem]
                if ((t[0]==face_n2) or (t[1]==face_n2) or \
                    (t[2]==face_n2) or (t[3]==face_n2)) and \
                ((t[0]==face_n3) or (t[1]==face_n3) or \
                    (t[2]==face_n3) or (t[3]==face_n3)):
                    # element is opposite a shared edge, step into it
                    next_tet_id = next_elem
                    break                    
            i = d.ElemConn_next[i] 

        ######
        if next_tet_id != -1:
            tet_id = next_tet_id
            tet = d.raw_NodeConn[tet_id]            
            
        alpha, beta, gamma, delta = utils.BaryTet(d.raw_NodeCoords[tet], newPt)
    return tet_id

@try_njit
def _build_cavity_3d(d, tet_id, newPt):
    tet = d.raw_NodeConn[tet_id]
    # Queue contains the id of a tetrahedron followed by the three vertices that define a face of that tet
    queue = [(tet_id, tet[0], tet[1], tet[3]),
             (tet_id, tet[1], tet[2], tet[3]),
             (tet_id, tet[2], tet[0], tet[3]),
             (tet_id, tet[2], tet[1], tet[0])]
    visited = [tet_id,]
    bad_tets = [tet_id,]
    cavity_faces = []
   
    # super tet nodes
    super_cutoff = d.NNode - 4 # nodes >= super_cutoff are part of the super tet

    while len(queue) > 0:

        prev_t_id, f0, f1, f2 = queue.pop() # tetrahedron ID, face vertex 1, face vertex 2, face vertex 3

        next_t_id = -1
        i = d.ElemConn_head[f0]
        while i != -1:
            next_elem = d.ElemConn_elem[i]
            # iterate through elem conn for the first node in the edge
            if next_elem != prev_t_id:
                # skip the current triangle
                t = d.raw_NodeConn[next_elem]
                if (f1 in t and f2 in t): # This can be faster with explicit checks
                    # element is opposite a shared element, step into it
                    next_t_id = next_elem
                    break                    
            i = d.ElemConn_next[i] 
        if next_t_id == -1:
            # boundary face
            cavity_faces.append((f0, f1, f2))
            continue

        if next_t_id in visited:
            # tet has already been checked
            if prev_t_id in bad_tets and next_t_id not in bad_tets:
                cavity_faces.append((f0, f1, f2))
            continue

        tet = d.raw_NodeConn[next_t_id]

        # if ((tet[0] >= super_cutoff) ^ (tet[1] >= super_cutoff) ^ (tet[2] >= super_cutoff) ^ (tet[3] >= super_cutoff)) and (f0 < super_cutoff and f1 < super_cutoff and f2 < super_cutoff):
        #     # ^ = XOR
        #     # TODO: verify that this is necessary/correct
        #     # tet is connected to super tet, mark boundary
        #     cavity_faces.append((f0, f1, f2))
        #     visited.append(next_t_id)
        #     continue
            
        # test circumcircle
        pass

        if circumsphere(d.raw_NodeCoords[t[0]], d.raw_NodeCoords[t[1]], d.raw_NodeCoords[t[2]], d.raw_NodeCoords[t[3]], newPt):
            # point in cicrumsphere of tet; add faces to queue (except for the face that was just used)
            bad_tets.append(next_t_id)
            
            if (
                (tet[2] == f0 or tet[2] == f1 or tet[2] == f2) and \
                (tet[1] == f0 or tet[1] == f1 or tet[1] == f2) and \
                (tet[0] == f0 or tet[0] == f1 or tet[0] == f2)
            ):
                # old face is (tet[2], tet[1], tet[0])
                queue.append((next_t_id, tet[0], tet[1], tet[3]))
                queue.append((next_t_id, tet[1], tet[2], tet[3]))
                queue.append((next_t_id, tet[2], tet[0], tet[3]))

            elif (
                (tet[0] == f0 or tet[0] == f1 or tet[0] == f2) and \
                (tet[1] == f0 or tet[1] == f1 or tet[1] == f2) and \
                (tet[3] == f0 or tet[3] == f1 or tet[3] == f2)
            ):
                # old face is (tet[0], tet[1], tet[3])
                queue.append((next_t_id, tet[1], tet[2], tet[3]))
                queue.append((next_t_id, tet[2], tet[0], tet[3]))
                queue.append((next_t_id, tet[2], tet[1], tet[0]))

            elif (
                (tet[1] == f0 or tet[1] == f1 or tet[1] == f2) and \
                (tet[2] == f0 or tet[2] == f1 or tet[2] == f2) and \
                (tet[3] == f0 or tet[3] == f1 or tet[3] == f2)
            ):
                # old face is (tet[1], tet[2], tet[3])
                queue.append((next_t_id, tet[2], tet[0], tet[3]))
                queue.append((next_t_id, tet[2], tet[1], tet[0]))
                queue.append((next_t_id, tet[0], tet[1], tet[3]))

            else:
                # old edge is (tet[2], tri[0], tet[3])
                queue.append((next_t_id, tet[2], tet[1], tet[0]))
                queue.append((next_t_id, tet[0], tet[1], tet[3]))
                queue.append((next_t_id, tet[1], tet[2], tet[3]))

        else:
            # boundary between a valid and invalid triangle
            cavity_faces.append((f0, f1, f2))
            
        visited.append(next_t_id)
    return bad_tets, cavity_faces

## Predicates ##
@try_njit(inline='always')
def orient2d(a, b, c):
    """
    Two dimensional orientation test. 
    Determines whether three points in the plane are clockwise, counterclockwise, or colinear.

    Parameters
    ----------
    a : np.ndarray
        Two dimensional coordinates of the first point (shape=(2,))
    b : np.ndarray
        Two dimensional coordinates of the second point (shape=(2,))
    c : np.ndarray
        Two dimensional coordinates of the third point (shape=(2,))

    Returns
    -------
    o : float
        Cross product of (b-a) x (c-a). ``o < 0`` indicates clockwise, 
        ``o > 0`` indicates counterclockwise, ``o == 0`` indicates colinear.
    """    
    # Cross product u x v between u = b - a and v = c - a
    o = (b[0] - a[0])*(c[1] - a[1]) - (c[0] - a[0])*(b[1] - a[1])
    # o > 0 -> CCW
    # o = 0 -> colinear
    # o < 0 -> CW
    return o

@try_njit(inline='always')
def circumcircle(a, b, c, d):
    r"""
    Two dimensional point in triangular circumcircle test.
    Tests if the point :math:`d` is in the circumcircle of triangle :math:`abc`

    The :ref:`determinant test <Circumcircle Test>`:

    .. math::

        \det{\begin{bmatrix} 
        a_x & a_y & a_x^2 + a_y^2 & 1 \\
        b_x & b_y & b_x^2 + b_y^2 & 1 \\
        c_x & c_y & c_x^2 + c_y^2 & 1 \\
        d_x & d_y & d_x^2 + d_y^2 & 1 
        \end{bmatrix}} > 0

    is simplified by moving the triangle so that the point d is at the origin:

    .. math::

        \det{\begin{bmatrix} 
        a_x - d_x & a_y - d_y & (a_x - d_x)^2 + (a_y - d_y)^2 \\
        b_x - d_x & b_y - d_y & (b_x - d_x)^2 + (b_y - d_y)^2 \\
        c_x - d_x & c_y - d_y & (c_x - d_x)^2 + (c_y - d_y)^2 
        \end{bmatrix}} > 0

    Parameters
    ----------
    a : np.ndarray
        Two dimensional coordinates of the first point of the triangle (shape=(2,))
    b : np.ndarray
        Two dimensional coordinates of the second point of the triangle (shape=(2,))
    c : np.ndarray
        Two dimensional coordinates of the third point of the triangle (shape=(2,))
    d : np.ndarray
        Two dimensional coordinates of the point to be compared to the circumcircle

    Returns
    -------
    bool
        True if point d is in the circumcircle of triangle abc
    """    

    # manual determinant of matrix [[A,B,C],[D,E,F],[G,H,I]]
    A = a[0] - d[0]
    B = a[1] - d[1]
    C = (a[0] - d[0])**2 + (a[1] - d[1])**2

    D = b[0] - d[0]
    E = b[1] - d[1]
    F = (b[0] - d[0])**2 + (b[1] - d[1])**2

    G = c[0] - d[0]
    H = c[1] - d[1]
    I = (c[0] - d[0])**2 + (c[1] - d[1])**2

    det = A*(E*I-F*H) - B*(D*I-F*G) + C*(D*H-E*G)
    return det > 0 

@try_njit(inline='always')
def circumsphere(a, b, c, d, e):
    r"""
    Three dimensional point in tetrahedral circumsphere test.
    Tests if the point e is in the circumcircle of tetrahedron abcd

    The :ref:`determinant test <Circumcircle Test>`:

    .. math::

        \det{\begin{bmatrix} 
        a_x & a_y & a_z & a_x^2 + a_y^2 + a_z^2 & 1 \\
        b_x & b_y & b_z & b_x^2 + b_y^2 + b_z^2 & 1 \\
        c_x & c_y & c_z & c_x^2 + c_y^2 + c_z^2 & 1 \\
        d_x & d_y & d_z & d_x^2 + d_y^2 + d_z^2 & 1 \\
        e_x & e_y & e_z & e_x^2 + e_y^2 + e_z^2 & 1 \\
        \end{bmatrix}} > 0

    is simplified by moving the tetraheron so that the point e is at the origin:

    .. math::

        \det{\begin{bmatrix} 
        a_x - e_x & a_y - e_y & a_z - e_z & (a_x - e_x)^2 + (a_y - e_y)^2 + (a_z - e_z)^2 \\
        b_x - e_x & b_y - e_y & b_z - e_z & (b_x - e_x)^2 + (b_y - e_y)^2 + (b_z - e_z)^2 \\
        c_x - e_x & c_y - e_y & c_z - e_z & (c_x - e_x)^2 + (c_y - e_y)^2 + (c_z - e_z)^2 \\
        d_x - e_x & d_y - e_y & d_z - e_z & (d_x - e_x)^2 + (d_y - e_y)^2 + (d_z - e_z)^2 \\
        \end{bmatrix}} > 0


    Parameters
    ----------
    a : np.ndarray
        Three dimensional coordinates of the first point of the tetrahedron (shape=(3,))
    b : np.ndarray
        Three dimensional coordinates of the second point of the tetrahedron (shape=(3,))
    c : np.ndarray
        Three dimensional coordinates of the third point of the tetrahedron (shape=(3,))
    d    : np.ndarray
        Three dimensional coordinates of the third point of the tetrahedron (shape=(3,))
    e : np.ndarray
        Three dimensional coordinates of the point to be compared to the circumsphere (shape=(3,))

    Returns
    -------
    bool
        True if point e is in the circumcircle of tetrahedron abcd
    """   

    # manual determinant of matrix [[A,B,C,D],[E,F,G,H],[I,J,K,L],[M,N,O,P]]
    A = a[0] - e[0]
    B = a[1] - e[1]
    C = a[2] - e[2]
    D = (a[0] - e[0])**2 + (a[1] - e[1])**2 + (a[2] - e[2])**2

    E = b[0] - e[0]
    F = b[1] - e[1]
    G = b[2] - e[2]
    H = (b[0] - e[0])**2 + (b[1] - e[1])**2 + (b[2] - e[2])**2

    I = c[0] - e[0]
    J = c[1] - e[1]
    K = c[2] - e[2]
    L = (c[0] - e[0])**2 + (c[1] - e[1])**2 + (c[2] - e[2])**2

    M = d[0] - e[0]
    N = d[1] - e[1]
    O = d[2] - e[2]
    P = (d[0] - e[0])**2 + (d[1] - e[1])**2 + (d[2] - e[2])**2

    # TODO:
    det = A*(F*(K*P - L*O) - G*(J*P - L*N) + H*(J*O - K*N)) - \
            B*(E*(K*P - L*O) - G*(I*P - L*M) + H*(I*O - K*M)) + \
            C*(E*(J*P - L*N) - F*(I*P - L*M) + H*(I*N - J*M)) - \
            D*(E*(J*O - K*N) - F*(I*O - K*M) + G*(I*N - J*M))
    return det < 0 

@try_njit(inline='always')
def convex2d(points):
    """
    Two dimensional convexity test.
    Determines whether n points form a strictly convex hull. 
    The presence of three collinear points is considered not strictly convex.

    Parameters
    ----------
    points : np.ndarray
        Two dimensional point coordinates for n > 2 points (shape=(n,2))

    Returns
    -------
    convex : bool
        True of the points are strictly convex
    """    
    assert len(points) >= 3, 'At least 3 points are needed to test convexity'
    
    # test first point
    o1 = orient2d(points[0], points[1], points[2])
    if o1 == 0:
        # collinear, not strictly convex
        return False
    sign1 = o1 > 0
    # test middle points
    for i in range(1, len(points)-2):
        o2 = orient2d(points[i], points[i+1], points[i+2])
        if o2 == 0 or sign1 != (o2 > 0):
            # collinear or concave, not strictly convex
            return False
    # test second-to-last segment
    o2 = orient2d(points[-2], points[-1], points[0])
    if o2 == 0 or sign1 != (o2 > 0):
        # collinear or concave, not strictly convex
        return False
    o2 = orient2d(points[-1], points[0], points[1])
    if o2 == 0 or sign1 != (o2 > 0):
        # collinear or concave, not strictly convex
        return False
    return True

@try_njit(inline='always')
def segment_intersect2d(a, b, c, d):
    # segment a b intersection with segment c d

    ux = b[0] - a[0]
    uy = b[1] - a[1]

    vx = c[0] - a[0]
    vy = c[1] - a[1]

    # orientation test 1 (c vs ab)
    o1 = ux*vy - vx*uy
    if o1 == 0:
        # c collinear with ab
        if (min(a[0], b[0]) <= c[0] <= max(a[0], b[0])) and  \
            (min(a[1], b[1]) <= c[1] <= max(a[1], b[1])):
            return True

    wx = d[0] - a[0]
    wy = d[1] - a[1]
    # orientation test 2 (d vs ab)
    o2 = ux*wy - wx*uy
    if o2 == 0:
        # d collinear with ab
        if (min(a[0], b[0]) <= d[0] <= max(a[0], b[0])) and  \
            (min(a[1], b[1]) <= d[1] <= max(a[1], b[1])):
            return True
        return False

    if ((o2 > 0) and (o1 > 0)) or ((o2 < 0) and (o1 < 0)):
        # c and d on the same side of ab
        return False 
    
    qx = d[0] - c[0]
    qy = d[1] - c[1]

    rx = -vx
    ry = -vy

    # orientation test 3 (a vs cd)
    o3 = qx*ry - rx*qy
    if o3 == 0:
        # a collinear with cd
        if (min(c[0], d[0]) <= a[0] <= max(c[0], d[0])) and  \
            (min(c[1], d[1]) <= a[1] <= max(c[1], d[1])):
            return True
    
    sx = b[0] - c[0]
    sy = b[1] - c[1] 

    # orientation test 4 (b vs cd)
    o4 = qx*sy - sx*qy
    if o4 == 0:
        # b collinear with cd
        if (min(c[0], d[0]) <= b[0] <= max(c[0], d[0])) and  \
            (min(c[1], d[1]) <= b[1] <= max(c[1], d[1])):
            return True
        return False
    
    if ((o4 > 0) and (o3 > 0)) or ((o4 < 0) and (o3 < 0)):
        # a and b on the same side of cd
        return False 

    return True

