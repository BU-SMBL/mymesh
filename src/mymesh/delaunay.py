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

Convex Hull
===========
.. autosummary::
    :toctree: submodules/

    ConvexHull
    GiftWrapping
    

"""
#%%
import numpy as np
import sys, copy, itertools, warnings, random
from . import utils, rays, converter, mesh
from . import try_njit, check_numba, _MYMESH_USE_NUMBA
from scipy import spatial

def Triangulate(NodeCoords,Constraints=None,method=None,tol=1e-8):
    """
    Generate a triangulation for a 2D set of points. This will be a strictly
    convex triangulation.

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
        Triangulation method, by default 'BowyerWatson' if no constraints are 
        provided and 'Triangle' if constraints are provided.

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
            method = 'Triangle'

    if (Constraints is None or len(Constraints) == 0):
        Points,_,idx = utils.DeleteDuplicateNodes(Points,[],return_idx=True, tol=tol)
        if method.lower() == 'bowyerwatson':
            NodeConn = idx[BowyerWatson2d(Points)]
        elif method.lower() == 'scipy':
            NodeConn = idx[SciPy(Points)]
        elif method.lower() == 'triangle':
            NodeConn = idx[Triangle(Points)]
        else:
            raise ValueError(f'Invalid method "{method:s}".')
    else: 
        # Constrained Delaunay Triangulation - Sloan (1993)
        # Generate initial triangulation
        if method.lower() == 'triangle':
            method = 'triangle'
        else:
            raise ValueError('Currently only method="Triangle" is supported for constrained triangulation.')
        
        NodeConn = Triangle(Points,Constraints=Constraints)

    if 'mesh' in dir(mesh):
        T = mesh.mesh(NodeCoords,NodeConn)
    else:
        T = mesh(NodeCoords,NodeConn)
    return T

def Tetrahedralize(NodeCoords, method=None, tol=1e-8):
    """
    Generate a Delaunay tetrahedralization for a 3D set of points. This will be 
    a strictly convex tetrahedralization.

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
        method = 'bowyerwatson'

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

def ConvexHull(NodeCoords,method='scipy'):
    """
    Identify the convex hull of a set of points. For a 2D point set 
    (np.shape(NodeCoords) = (n,2)), a 2D convex hull of line elements will be 
    generated. For a 3D point set (np.shape(NodeCoords) = (n,2)), a 3D convex
    hull of triangle elements will be generated.

    Parameters
    ----------
    NodeCoords : array_like
        Coordinates of points around which the convex hull will be identified.
        If shape = (n,2), a 2D convex hull will be identified, or if shape = (n,3)
        a 3D convex hull will be identified. 
    method : str, optional
        Convex hull method, by default 'scipy'.

        - 'scipy' - Use :external+scipy:class:`scipy.spatial.ConvexHull`

        - 'BowyerWatson' - Generate a Delaunay triangulation by the Bowyer-Watson algorithm (:func:`BowyerWatson2d` or :func:`BowyerWatson3d`)

        - 'GiftWrapping' - Use the gift wrapping algorithm (:func:`GiftWrapping`)

    Returns
    -------
    Hull : mymesh.mesh
        Mesh object containing the convex hull. :code:`Hull.Type='line'` for a 2D hull
        or :code:`Hull.Type='surf'` for a 3D hull.

    """    
    nD = np.shape(NodeCoords)[1]

    if nD == 2:
        if method.lower() == 'giftwrapping':
            hull = GiftWrapping(NodeCoords,IncludeCollinear=True)
        elif method.lower() == 'scipy':
            qhull = spatial.ConvexHull(np.asarray(NodeCoords, dtype=np.float64))
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
        if method.lower() == 'scipy':
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

def SciPy(NodeCoords):
    """
    Wrapper for :external+scipy:class:`scipy.spatial.Delaunay` for 2D triangulation
    or 3D tetrahedralization.

    Parameters
    ----------
    NodeCoords : array_like
        (n,2) or (n,3) node coordinates for the triangulation. Triangulation is 
        only based on the coordinates in the first two dimensions, if an (n,3)
        is provided, the coordinates of the third column is ignored.

    Returns
    -------
    NodeConn : np.ndarray
        mx3 array of node connectivity for the triangles
    """    
        
    out = spatial.Delaunay(NodeCoords,qhull_options='Qbb Qc Qz Q12 Qt')
    NodeConn = out.simplices
    return NodeConn

def Triangle(NodeCoords,Constraints=None):
    """
    Interface to Jonathan Shewchuk's Triangle via a python wrapper (https://pypi.org/project/triangle/). To use, the python wrapper must be installed (`pip install triangle`).

    Parameters
    ----------
    NodeCoords : array_like
        Array of point coordinates
    Constraints : array_like, optional
        Edge connectivity array of node indices that indicate edges to be ensured
        by constrained Delaunay triangulation, by default None

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
    else:
        In = dict(vertices=NodeCoords,segments=Constraints)
    try:
        Out = triangle.triangulate(In,'pc')
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
                a = 2
                NodeCoords = Out['vertices']
    except:
        NodeConn = SciPy(NodeCoords)

    return NodeConn
    
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

def GiftWrapping(NodeCoords,IncludeCollinear=True):
    """
    Gift wrapping algorithm for computing the convex hull of a set of 2D points.

    :cite:`Jarvis1973`

    Parameters
    ----------
    NodeCoords : array_like
        List of 2D point coordinates

    Returns
    -------
    Hull : np.ndarray
        List of point indices that form the convex hull, in counterclockwise order
    """    

    assert len(NodeCoords) > 2, 'At least three points are required.'
    if NodeCoords.shape[1] == 2:
        Points = np.asarray(NodeCoords)
    else:
        warnings.warn('GiftWrapping is only valid for points on a plane, the third dimension is ignored.')
        Points = np.asarray(NodeCoords)[:,:2]

    sortidx = Points[:,1].argsort()[::-1]
    Points = Points[sortidx,:] # sorting from max y to min y (TODO:for some reason if the first point comes before the second point, there are problems)
    
    indices = np.arange(len(Points))
    firstP = np.where(Points[:,1]==np.min(Points[:,1]))[0] # Minimum y coordinate point
    if len(firstP) > 0:
        # if there are multiple points at the same min y coordinate, choose the one with the max x coordinate
        firstP = firstP[np.where(Points[firstP,0]==np.max(Points[firstP,0]))[0][0]]
    nextP = -1
    Hull = [firstP]
    mask = np.repeat(True,len(Points))
    mask[firstP] = False
    thetaTotal = 0
    theta = np.arctan2(Points[mask,1]-Points[Hull[-1],1],Points[mask,0]-Points[Hull[-1],0]) 
    mask[firstP] = True

    while nextP != firstP:

        idxs = np.where(theta == theta.min())[0]
        if len(idxs) > 0:
            
            # Check for collinear vertices on the boundary
            dists = np.linalg.norm(Points[indices[mask][idxs]] - Points[Hull[-1]],axis=1)
            if IncludeCollinear:
                # includes closest point first
                idx = idxs[dists.argmin()]
            else:
                # Skip to furthest point
                idx = idxs[dists.argmax()]
        else:
            idx = idxs[0]
        thetaTotal += theta[idx]
        nextP = indices[mask][idx]
        mask[nextP] = False
        Hull.append(nextP)
        # Polar coordinate angles of all (non-hull) points, centered at the most recently added hull point
        theta = np.arctan2(Points[mask,1]-Points[Hull[-1],1],Points[mask,0]-Points[Hull[-1],0]) - thetaTotal
        theta[theta<0] += 2*np.pi

    Hull = sortidx[Hull[:-1]]
    HullConn = np.column_stack([Hull, np.roll(Hull,-1)])
    return HullConn

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
    
    NodeConn = np.column_stack([np.repeat(Hull[0][0], len(Hull)-2), 
                                Hull[1:-1,0],
                                Hull[1:-1,1]
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
        
def BowyerWatson2d(NodeCoords):
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
    if check_numba():
        import numba
        from numba.typed import Dict
    else:
        warnings.warn('Using numba is strongly recommended for efficiency of BowyerWatson2d. Activate with `mymesh.use_numba(True)`')
        Dict = dict

    NodeCoords = np.asarray(NodeCoords)
    assert NodeCoords.shape[0] >= 3, 'At least three points are required.'
    if NodeCoords.shape[1] == 2:
        TempCoords = NodeCoords
    else:
        warnings.warn('BowyerWatson2d is only valid for points on a plane, the third dimension is ignored.')
        TempCoords = NodeCoords[:,:2]

    nPts = len(NodeCoords)

    # Random insertion order for points
    indices = list(range(nPts))
    rng = np.random.default_rng()
    rng.shuffle(indices)

    # Get super triangle - triangle with incircle that bounds the point set
    center = np.mean(TempCoords, axis=0)
    r = np.max(np.sqrt((TempCoords[:,0]-center[0])**2 + (TempCoords[:,1]-center[1])**2))
    R = r + 1*r/10

    super_triangle_points = np.array([
                                    [center[0], center[1]-2*R],
                                    [center[0]+R*np.sqrt(3), center[1]+R],
                                    [center[0]-R*np.sqrt(3), center[1]+R]
                            ])    
    TempCoords = np.hstack([np.vstack([TempCoords, super_triangle_points]), np.repeat(0,nPts+3)[:,None]])
    super_tri = (nPts, nPts+1, nPts+2)

    ElemTable = Dict()
    # Elem table links elements to tuples of (oriented) edges
    # e.g. ElemTable[(0,1,2)] = ((0,1),(1,2),(2,0))
    ElemTable[super_tri] = ((nPts, nPts+1), (nPts+1, nPts+2), (nPts+2, nPts)) 

    EdgeTable = Dict()
    # Edge table links oriented (half) edges to their one connected element
    # e.g. EdgeTable[(0,1)] = (0,1,2)
    EdgeTable[(nPts, nPts+1)] = super_tri
    EdgeTable[(nPts+1, nPts+2)] = super_tri
    EdgeTable[(nPts+2, nPts)] = super_tri

    for i in indices:
        newPt = TempCoords[i]

        tri = _walk_2d(TempCoords, ElemTable, EdgeTable, newPt, nsample=1)
        # Breadth first search of adjacent triangles to find all invalid triangles
        # Initiate a queue of the edges
        bad_triangles, cavity_edges = _build_cavity_2d(TempCoords, ElemTable, EdgeTable, tri, newPt)
        # Remove triangles and edges
        for t in bad_triangles:
            for e in ElemTable[t]:
                del EdgeTable[e]
            del ElemTable[t]

        # Create new triangles and edges
        for e in cavity_edges:
            t = (e[0], e[1], i)
            edges = (e, (e[1], i), (i, e[0]))
            ElemTable[t] = edges
            for edge in edges:
                EdgeTable[edge] = t

    NodeConn = np.array(list(ElemTable.keys()))
    Super = np.any(NodeConn == nPts, axis=1) | np.any(NodeConn == (nPts+1), axis=1) | np.any(NodeConn == (nPts+2), axis=1)
    NodeConn = NodeConn[~Super]
    
    return NodeConn

def BowyerWatson3d(NodeCoords):
    """
    Bowyer-Watson algorithm for 3D Delaunay tetrahedralization
    :cite:p:`Bowyer1981`, :cite:p:`Watson1981`, :cite:p:`Marot2019`

    Parameters
    ----------
    NodeCoords : array_like
        nx3 set of points to be tetrahedralized

    Returns
    -------
    NodeConn : np.ndarray
        mx4 array of node connectivities for the Delaunay tetrahedralization
    """
    if check_numba():
        import numba
        from numba.typed import Dict
    else:
        Dict = dict

    NodeCoords = np.asarray(NodeCoords)
    assert NodeCoords.shape[0] >= 3, 'At least three points are required.'
    assert NodeCoords.shape[1] == 3, 'BowyerWatson3d is only valid for three dimensional points.'

    nPts = len(NodeCoords)

    # Random insertion order for points
    indices = list(range(nPts))
    rng = np.random.default_rng()
    rng.shuffle(indices)

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

    ElemTable = Dict()
    # Elem table links elements to tuples of (oriented) faces
    # e.g. ElemTable[(0,1,2,3)] = ((2,0,1),(1,0,3),(3,0,2),(2,1,3)))
    # Faces are stragically numbered s.t. the minimum node number is in the 
    # center to allow for flipping to find the face's twin
    ElemTable[super_tet] = (
        (super_tet[2], super_tet[0], super_tet[1]), 
        (super_tet[1], super_tet[0], super_tet[3]), 
        (super_tet[3], super_tet[0], super_tet[2]), 
        (super_tet[2], super_tet[1], super_tet[3])
        )

    EdgeTable = Dict()
    # Edge table links oriented (half) faces to their one connected element
    # e.g. EdgeTable[(2,0,1)] = (0,1,2,3)
    EdgeTable[(super_tet[2], super_tet[0], super_tet[1])] = super_tet
    EdgeTable[(super_tet[1], super_tet[0], super_tet[3])] = super_tet
    EdgeTable[(super_tet[3], super_tet[0], super_tet[2])] = super_tet
    EdgeTable[(super_tet[2], super_tet[1], super_tet[3])] = super_tet
 
    for i in indices:
        newPt = TempCoords[i]
        tet = _walk_3d(TempCoords, ElemTable, EdgeTable, newPt, nsample=1)
        # Breadth first search of adjacent tets to find all invalid tets
        # Initiate a queue of the faces
        bad_tets, cavity_edges = _build_cavity_3d(TempCoords, ElemTable, EdgeTable, tet, newPt)
        # Remove tets and faces
        for t in bad_tets:
            for e in ElemTable[t]:
                del EdgeTable[e]
            del ElemTable[t]

        # Create new tets and faces
        for e in cavity_edges:
            t = (e[0], e[1], e[2], i)
            e1 = (e[0], i, e[1])
            e2 = (e[0], e[2], i)
            e3 = (e[1], i, e[2])
            min_e1_idx = e1.index(min(e1))
            min_e2_idx = e2.index(min(e2))
            min_e3_idx = e3.index(min(e3))

            # Ordering each new edge to have the smallest value in the middle
            e1 = e1 if min_e1_idx == 1 else (e1[1], e1[2], e1[0]) if min_e1_idx == 2 else (e1[2], e1[0], e1[1])
            e2 = e2 if min_e2_idx == 1 else (e2[1], e2[2], e2[0]) if min_e2_idx == 2 else (e2[2], e2[0], e2[1])
            e3 = e3 if min_e3_idx == 1 else (e3[1], e3[2], e3[0]) if min_e3_idx == 2 else (e3[2], e3[0], e3[1])

            edges = (e, e1, e2, e3)

            ElemTable[t] = edges
            for edge in edges:
                EdgeTable[edge] = t

    NodeConn = np.array(list(ElemTable.keys()))
    Super = np.any(NodeConn == nPts, axis=1) | np.any(NodeConn == (nPts+1), axis=1) | np.any(NodeConn == (nPts+2), axis=1) | np.any(NodeConn == (nPts+3), axis=1)
    NodeConn = NodeConn[~Super]
    

    return NodeConn

## Utils ##
@try_njit
def _walk_2d(TempCoords, ElemTable, EdgeTable, newPt, nsample=1):
    # Walking algorithm to find triangle containing the new point
    tri = list(ElemTable.keys())[np.random.randint(0,len(ElemTable))]
    minL = np.linalg.norm(TempCoords[tri[0]]-newPt)
    for i in range(min(nsample-1,len(ElemTable)-1)):
        t = list(ElemTable.keys())[np.random.randint(0,len(ElemTable))]
        L = np.linalg.norm(TempCoords[t[0]]-newPt)
        if L < minL:
            tri = t
    alpha, beta, gamma = utils.BaryTri(TempCoords[np.array(list(tri))], newPt)
    while not (alpha >= 0 and beta >= 0 and gamma >= 0):
        # find node with smallest (most negative) barycentric coordinate
        bcoords = [alpha,beta,gamma]
        nodeid = tri[bcoords.index(min(bcoords))]

        # find edge opposite that node
        if nodeid not in ElemTable[tri][0]:
            edge = ElemTable[tri][0]
        elif nodeid not in ElemTable[tri][1]:
            edge = ElemTable[tri][1]
        else:
            edge = ElemTable[tri][2]

        # step to the neighboring triangle across that edge
        tri = EdgeTable[edge[::-1]]
        alpha, beta, gamma = utils.BaryTri(TempCoords[np.array(list(tri))], newPt)
    return tri

@try_njit
def _build_cavity_2d(TempCoords, ElemTable, EdgeTable, tri, newPt):
    bad_triangles = [tri]
    cavity_edges = []
    queue = set(ElemTable[tri])
    s1 = len(TempCoords) - 1; s2 = len(TempCoords) - 2; s3 = len(TempCoords) - 3;
    while len(queue) > 0:
        edge = queue.pop()
        twin = edge[::-1]

        if twin in EdgeTable:
            t = EdgeTable[twin]
            # Check if triangle is connected to the super triangle
            if sum([s1 in t, s2 in t, s3 in t]) == 1 and sum([s1 in twin, s2 in twin, s3 in twin]) == 0:
                # mark this edge as a cavity boundary
                cavity_edges.append(edge)
                continue
            # test circumcircle
            mat = np.array([
                [TempCoords[t[0], 0] - newPt[0], TempCoords[t[0], 1] - newPt[1], (TempCoords[t[0], 0] - newPt[0])**2 + (TempCoords[t[0], 1] - newPt[1])**2],
                [TempCoords[t[1], 0] - newPt[0], TempCoords[t[1], 1] - newPt[1], (TempCoords[t[1], 0] - newPt[0])**2 + (TempCoords[t[1], 1] - newPt[1])**2],
                [TempCoords[t[2], 0] - newPt[0], TempCoords[t[2], 1] - newPt[1], (TempCoords[t[2], 0] - newPt[0])**2 + (TempCoords[t[2], 1] - newPt[1])**2]
            ])
            invalid = np.linalg.det(mat) > 0
            if invalid:
                # mark invalid triangles for deletion
                bad_triangles.append(t)
                # add adjacent neighbors to queue
                queue.update([e for e in ElemTable[t] if e != twin])
            else:
                # mark this edge as a cavity boundary
                cavity_edges.append(edge)

        else:
            # boundary edge, add to cavity
            cavity_edges.append(edge)
    return bad_triangles, cavity_edges


# TODO: Traversals in 3d probably won't work right because half-face pairs can't 
# necessarily be obtained just by reversing the order
@try_njit
def _walk_3d(TempCoords, ElemTable, EdgeTable, newPt, nsample=1):
    # Walking algorithm to find tets containing the new point
    tet = list(ElemTable.keys())[np.random.randint(0,len(ElemTable))]
    minL = np.linalg.norm(TempCoords[tet[0]]-newPt)
    for i in range(min(nsample-1,len(ElemTable)-1)):
        t = list(ElemTable.keys())[np.random.randint(0,len(ElemTable))]
        L = np.linalg.norm(TempCoords[t[0]]-newPt)
        if L < minL:
            tet = t
    alpha, beta, gamma, delta = utils.BaryTet(TempCoords[np.array(list(tet))], newPt)
    while not (alpha >= 0 and beta >= 0 and gamma >= 0 and delta >= 0):
        # find node with smallest (most negative) barycentric coordinate
        bcoords = [alpha,beta,gamma,delta]
        nodeid = tet[bcoords.index(min(bcoords))]

        # find edge opposite that node
        edge = [e for e in ElemTable[tet] if nodeid not in e][0]

        # step to the neighboring tet across that face
        tet = EdgeTable[edge[::-1]]
        alpha, beta, gamma, delta = utils.BaryTet(TempCoords[np.array(list(tet))], newPt)
    return tet

@try_njit
def _build_cavity_3d(TempCoords, ElemTable, EdgeTable, tet, newPt):
    # TODO: it seems like some of the tets get visited more than once - maybe not anymore?
    bad_tets = set((tet,)) #[tet]
    cavity_edges = []
    valid_set = set()
    invalid_set = set()
    queue = set(ElemTable[tet])
    while len(queue) > 0:
        edge = queue.pop()
        twin = edge[::-1]

        if twin in EdgeTable:
            t = EdgeTable[twin]
            if t in invalid_set:
                continue
            elif t in valid_set:
                cavity_edges.append(edge)
                continue

            # test circumsphere
            x0, y0, z0 = TempCoords[t[0]]
            x1, y1, z1 = TempCoords[t[1]]
            x2, y2, z2 = TempCoords[t[2]]
            x3, y3, z3 = TempCoords[t[3]]
            x, y, z = newPt

            mat = np.array([
                [x0, y0, z0, x0**2+y0**2+z0**2, 1],
                [x1, y1, z1, x1**2+y1**2+z1**2, 1],
                [x2, y2, z2, x2**2+y2**2+z2**2, 1],
                [x3, y3, z3, x3**2+y3**2+z3**2, 1],
                [x,  y,  z,  x**2+y**2+z**2,    1],
            ])

            invalid = np.linalg.det(mat) < 0
            if invalid:
                invalid_set.add(t)
                # mark invalid tets for deletion
                bad_tets.add(t)
                # add adjacent neighbors to queue
                queue.update([e for e in ElemTable[t] if e != twin])
            else:
                valid_set.add(t)
                # mark this edge as a cavity boundary
                cavity_edges.append(edge)

        else:
            # boundary edge, add to cavity
            cavity_edges.append(edge)
    return list(bad_tets), cavity_edges

