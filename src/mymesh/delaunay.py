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
    AlphaShape
    Alpha2d
    Alpha3d
    AlphaPeel3d
    GiftWrapping
    

"""
#%%
import numpy as np
import sys, copy, itertools, warnings, random
from . import utils, rays, converter, mesh, quality
from . import try_njit, check_numba, _MYMESH_USE_NUMBA
from scipy import spatial

def Triangulate(NodeCoords,Constraints=None,method=None,tol=1e-8, steiner=0):
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
            NodeCoords, NodeConn = Triangle(Points,steiner=steiner)
            NodeConn = idx[NodeConn]
        else:
            raise ValueError(f'Invalid method "{method:s}".')
    else: 
        # Constrained Delaunay Triangulation - Sloan (1993)
        # Generate initial triangulation
        if method.lower() == 'triangle':
            method = 'triangle'
        else:
            raise ValueError('Currently only method="Triangle" is supported for constrained triangulation.')
        
        NodeCoords, NodeConn = Triangle(Points,Constraints=Constraints,steiner=steiner)

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
        Triangulation method, by default 'scipy'.

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

def ConvexHull(NodeCoords,method='scipy',OrientSurf=True):
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
    OrientSurf : str, optional
        Ensure the normals of the convex hull are consistently oriented outward,
        by default True.

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
        Out = triangle.triangulate(In,f'pqS{steiner:d}')
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
        
def BowyerWatson2d(NodeCoords, nsample=3):
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

    d = _bowyer_watson_loop_2d(d, indices, nsample)

    NodeConn = d.NodeConn
    Super = np.any(NodeConn == nPts, axis=1) | \
        np.any(NodeConn == (nPts+1), axis=1) | \
        np.any(NodeConn == (nPts+2), axis=1)
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
    dmax = np.maximum(xmax-xmin, xmax-ymin)
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

# from numba import objmode
# import time
@try_njit(inline='always', cache=True)
def _bowyer_watson_loop_2d(d, indices, nsample):
    # with objmode(walk_time='float64'): walk_time = 0
    # with objmode(bcavity_time='float64'): bcavity_time = 0
    # with objmode(dcavity_time='float64'): dcavity_time = 0
    for i in indices:
        newPt = d.raw_NodeCoords[i]
        # with objmode(t0='float64'): t0 = time.time()
        tri_id = _walk_2d(d, newPt, tri_id=d.NElem-1, nsample=nsample)
        # with objmode(walk_time='float64'): walk_time += time.time() - t0
        # Breadth first search of adjacent triangles to find all invalid triangles
        # with objmode(t0='float64'): t0 = time.time()
        bad_triangles, cavity_edges = _build_cavity_2d(d, tri_id, newPt)
        # with objmode(bcavity_time='float64'): bcavity_time += time.time() - t0
        
        # with objmode(t0='float64'): t0 = time.time()
        # Remove triangles and edges
        d.removeElems(bad_triangles)

        # Create new triangles and edges
        for e in cavity_edges:
            d.addElem([e[0], e[1], i])
        
        # with objmode(dcavity_time='float64'): dcavity_time += time.time() - t0
    # with objmode(): print(walk_time, bcavity_time, dcavity_time, bcavity_time+dcavity_time)
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

