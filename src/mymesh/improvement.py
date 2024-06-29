# -*- coding: utf-8 -*-
# Created on Wed Jan 26 09:27:53 2022
# @author: toj
"""
Mesh quality improvement

Many of these functions are still being improved for both efficiency and 
robustness.


.. currentmodule:: mymesh.improvement


Mesh smoothing/node repositioning
=================================
.. autosummary::
    :toctree: submodules/

    LocalLaplacianSmoothing
    TangentialLaplacianSmoothing
    SmartLaplacianSmoothing
    GlobalLaplacianSmoothing
    GeoTransformSmoothing
    NodeSpringSmoothing
    SegmentSpringSmoothing

Local mesh topology
===================
.. autosummary::
    :toctree: submodules/

    TetContract
    TetSplit
    TetFlip
    TetImprove


"""

import numpy as np
import sys, warnings, time, random, copy, tqdm, heapq
from collections import deque
from . import converter, utils, quality, rays, octree, mesh, implicit
from scipy import sparse, spatial
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize

## Mesh smoothing/node repositioning
def LocalLaplacianSmoothing(M, options=dict()):
    """
    Performs iterative Laplacian smoothing, repositioning each node to the 
    center of its adjacent nodes.

    Parameters
    ----------
    M : mymesh.mesh
        Mesh object to smooth
    options : dict
        Smoothing options. Available options are:

        method : str
            'simultaneous' or 'sequential'. Specifies if smoothing is performed
            on all nodes at the same time, or one after another. Simultaneous
            laplacian smoothing will move nodes to the center of their neighbors'
            initial positions, while sequential will use the current positions of
            previously smoothed nodes, by default 'simultaneous'.
        iterate : int or str
            Fixed number of iterations to perform, or 'converge' to iterate until
            convergence, by default 'converge'.
        tolerance : float
            Convergence tolerance. For local Laplacian smoothing, iteration
            will terminate if the largest movement of a node is less than the
            specified tolerance, by default 1e-6.
        maxIter : int
            Maximum number of iterations when iterate='converge', By default 10.
        FixedNodes : set or array_like
            Set of nodes that are held fixed during iteration, by default none
            are fixed.
        FixFeatures : bool
            If true, feature nodes on edges or corners (identified by
            :func:`~mymesh.utils.DetectFeatures`) will be held in place, by default False.
        FixSurf : bool
            If true, all nodes on the surface will be held in place and only 
            interior nodes will be smoothed, by default False.

        

    Returns
    -------
    Mnew : mymesh.mesh
        Mesh object with the new node locations.

    """    
    
    NodeCoords, NodeConn = M
    NodeCoords = np.copy(NodeCoords)
    NodeNeighbors = M.NodeNeighbors
    ElemConn = M.ElemConn
    SurfConn = M.SurfConn
    
    # Process inputs
    SmoothOptions = dict(method='simultaneous',
                        iterate = 'converge',
                        tolerance = 1e-6,
                        maxIter = 10,
                        FixedNodes = set(),
                        FixFeatures = False,
                        FixSurf = False,
                        qualityFunc = quality.MeanRatio
                    )

    NodeCoords, NodeConn, SmoothOptions = _SmoothingInputParser(NodeCoords, NodeConn, SurfConn, SmoothOptions, options)
    FreeNodes = SmoothOptions['FreeNodes']
    FixedNodes = SmoothOptions['FixedNodes']
    tolerance = SmoothOptions['tolerance']
    iterate = SmoothOptions['iterate']
    qualityFunc = SmoothOptions['qualityFunc']
    maxIter = SmoothOptions['maxIter']
    method = SmoothOptions['method']

    lens = np.array([len(n) for n in NodeNeighbors])
    r = utils.PadRagged(NodeNeighbors,fillval=-1)
    ArrayCoords = np.vstack([NodeCoords,[np.nan,np.nan,np.nan]])
    
    if SmoothOptions['iterate'] == 'converge':
        condition = lambda i, U : (i == 0) | (i < maxIter) | (np.max(U) < tolerance)
    elif isinstance(SmoothOptions['iterate'], (int, np.integer)):
        condition = lambda i, U : i < SmoothOptions['iterate']
    else:
        raise ValueError('options["iterate"] must be "converge" or an integer.')
    
    i = 0
    U = np.zeros(len(NodeCoords))
    while condition(i, U[FreeNodes]):
        i += 1
        Q = ArrayCoords[r]
        U = (1/lens)[:,None] * np.nansum(Q - ArrayCoords[:-1,None,:],axis=1)
        ArrayCoords[FreeNodes] += U[FreeNodes]

    NewCoords = ArrayCoords[:-1]

    if 'mesh' in dir(mesh):
        Mnew = mesh.mesh(NewCoords, NodeConn)
    else:
        Mnew = mesh(NewCoords, NodeConn)

    return Mnew

def TangentialLaplacianSmoothing(M, options=dict()):
    """
    Performs tangential Laplacian smoothing :cite:p:`Ohtake2003`, repositioning 
    each node to the center of its adjacent nodes in the plane tangent to the 
    surface. Primarily for use on surface meshes, interior nodes of a volume 
    mesh will be fixed.

    Parameters
    ----------
    M : mymesh.mesh
        Mesh object to smooth
    options : dict
        Smoothing options. Available options are:

        method : str
            'simultaneous' or 'sequential'. Specifies if smoothing is performed
            on all nodes at the same time, or one after another. Simultaneous
            laplacian smoothing will move nodes to the center of their neighbors'
            initial positions, while sequential will use the current positions of
            previously smoothed nodes, by default 'simultaneous'.
        iterate : int or str
            Fixed number of iterations to perform, or 'converge' to iterate until
            convergence, by default 'converge'.
        tolerance : float
            Convergence tolerance. For local Laplacian smoothing, iteration
            will terminate if the largest movement of a node is less than the
            specified tolerance, by default 1e-6.
        maxIter : int
            Maximum number of iterations when iterate='converge', By default 10.
        FixedNodes : set or array_like
            Set of nodes that are held fixed during iteration, by default none
            are fixed.
        FixFeatures : bool
            If true, feature nodes on edges or corners (identified by
            :func:`~mymesh.utils.DetectFeatures`) will be held in place, by default False.
        FixSurf : bool
            If true, all nodes on the surface will be held in place and only 
            interior nodes will be smoothed, by default False.
        

    Returns
    -------
    Mnew : mymesh.mesh
        Mesh object with the new node locations.
    """    

    
    NodeCoords, NodeConn = M
    NodeCoords = np.copy(NodeCoords)
    NodeNeighbors = M.Surface.NodeNeighbors
    SurfConn = M.Surface.NodeConn
    
    # Process inputs
    SmoothOptions = dict(method='simultaneous',
                        iterate = 'converge',
                        tolerance = 1e-6,
                        maxIter = 10,
                        FixedNodes = set(),
                        FixFeatures = False,
                        FixSurf = False,
                        qualityFunc = quality.MeanRatio
                    )

    NodeCoords, NodeConn, SmoothOptions = _SmoothingInputParser(NodeCoords, NodeConn, SurfConn, SmoothOptions, options)
    FreeNodes = SmoothOptions['FreeNodes']
    FixedNodes = SmoothOptions['FixedNodes']
    tolerance = SmoothOptions['tolerance']
    iterate = SmoothOptions['iterate']
    qualityFunc = SmoothOptions['qualityFunc']
    maxIter = SmoothOptions['maxIter']
    method = SmoothOptions['method']

    lens = np.array([len(n) for n in NodeNeighbors])
    r = utils.PadRagged(NodeNeighbors,fillval=-1)
    idx = np.unique(SurfConn)
    FreeNodes = list(set(idx).difference(FixedNodes))

    ArrayCoords = np.vstack([NodeCoords,[np.nan,np.nan,np.nan]])
    
    NodeNormals = M.NodeNormals
    
    if SmoothOptions['iterate'] == 'converge':
        condition = lambda i, U : (i == 0) | (i < maxIter) | (np.max(U) < tolerance)
    elif isinstance(SmoothOptions['iterate'], (int, np.integer)):
        condition = lambda i, U : i < SmoothOptions['iterate']
    else:
        raise ValueError('options["iterate"] must be "converge" or an integer.')
    
    i = 0
    U = np.zeros(len(NodeCoords))
    while condition(i, U[FreeNodes]):
        i += 1

        Q = ArrayCoords[r]
        U = (1/lens)[:,None] * np.nansum(Q - ArrayCoords[:-1,None,:],axis=1)
        R = 1*(U - np.sum(U*NodeNormals,axis=1)[:,None]*NodeNormals)
        ArrayCoords[FreeNodes] += R[FreeNodes]

    NewCoords = np.copy(NodeCoords)
    NewCoords[idx] = ArrayCoords[idx]

    if 'mesh' in dir(mesh):
        Mnew = mesh.mesh(NewCoords, NodeConn)
    else:
        Mnew = mesh(NewCoords, NodeConn)

    return Mnew

def SmartLaplacianSmoothing(M, target='mean', TangentialSurface=True, labels=None, options=dict()):
    """
    Performs smart Laplacian smoothing :cite:p:`Freitag1997a`, repositioning 
    each node to the center of its adjacent nodes only if doing so doesn't
    reduce the element quality.

    Parameters
    ----------
    M : mymesh.mesh
        Mesh object to smooth
    target : str
        Determines whether criteria for repositioning a node, by default 'mean'. 

        'mean' - repositioning is allowed if the average quality of the connected
        elements doesn't decrease.

        'min' - repositioning is allowed if the minimum quality of the connected
        elements doesn't decrease.
    TangentialSurface : bool, optional
        Option to use tangential laplacian smoothing on the surface (and interfaces, 
        if labels are provided), by default True.
    options : dict, optional
        Smoothing options. Available options are:

        method : str
            'simultaneous' or 'sequential'. Specifies if smoothing is performed
            on all nodes at the same time, or one after another. Simultaneous
            laplacian smoothing will move nodes to the center of their neighbors'
            initial positions, while sequential will use the current positions of
            previously smoothed nodes, by default 'sequential'.
        iterate : int or str
            Fixed number of iterations to perform, or 'converge' to iterate until
            convergence, by default 'converge'.
        tolerance : float
            Convergence tolerance. For local Laplacian smoothing, iteration
            will terminate if the largest movement of a node is less than the
            specified tolerance, by default 1e-6.
        maxIter : int
            Maximum number of iterations when iterate='converge', By default 100.
        FixedNodes : set or array_like
            Set of nodes that are held fixed during iteration, by default none
            are fixed.
        FixFeatures : bool
            If true, feature nodes on edges or corners (identified by
            :func:`~mymesh.utils.DetectFeatures`) will be held in place, by default False.
        FixSurf : bool
            If true, all nodes on the surface will be held in place and only 
            interior nodes will be smoothed, by default False.
        qualityFunc : function
            Function used for computing quality. It is assumed that a larger
            number corresponds to higher quality, be default quality.MeanRatio
        InPlace : bool
            If True, the input mesh is modified directly, otherwise a new copy of the mesh
            is created, by default False.

    Returns
    -------
    Mnew : mymesh.mesh
        Mesh object with the new node locations.
    """    

    NodeCoords, NodeConn = M
    NodeCoords = np.copy(NodeCoords)
    NodeConn = np.asarray(NodeConn)
    NodeNeighbors = copy.copy(M.NodeNeighbors)
    ElemConn = M.ElemConn
    # SurfConn = M.SurfConn

    if labels is None:
        SurfConn = np.array(M.SurfConn, dtype=int)
        EdgeNodes = np.array([],dtype=int)
        EdgeNodeNeighbors = []
    else:
        if isinstance(labels, str):
            if labels in M.ElemData.keys():
                label_str = labels
                labels = M.ElemData[label_str]
            else:
                raise ValueError('If provided as a string, labels must correspond to an entry in M.ElemData')
        else:
            label_str = 'labels'
        assert len(labels) == M.NElem, 'labels must correspond to the number of elements.'
        if 'mesh' in dir(mesh):
            MultiSurface = mesh.mesh(NodeCoords,verbose=False)
        else:
            MultiSurface = mesh(NodeCoords,verbose=False)
        ulabels = np.unique(labels)
        label_nodes = np.zeros((len(ulabels),M.NNode),dtype=int) # For each label, boolean indicator of whether each node is touching an element with that label
        mesh_nodes = np.arange(M.NNode)
        for i,label in enumerate(ulabels):
            if 'mesh' in dir(mesh):
                m = mesh.mesh(NodeCoords, NodeConn[labels == label], verbose=False)
            else:
                m = mesh(NodeCoords, NodeConn[labels == label], verbose=False)
            
            MultiSurface.addElems(m.SurfConn)
            label_nodes[i,np.unique(m.NodeConn)] = 1
        MultiSurface.NodeConn = MultiSurface.Faces  # This prevents doubling of surface elements at interfaces
        SurfConn = MultiSurface.NodeConn
        # Identify nodes on edges shared by more than two triangles
        JunctionEdges = MultiSurface.Edges[np.array([len(conn) > 2 for conn in MultiSurface.EdgeElemConn])]
        EdgeNodes = np.unique(JunctionEdges)
        EdgeNodeNeighbors = utils.getNodeNeighbors(MultiSurface.NodeCoords, JunctionEdges)
    # Process inputs
    SmoothOptions = dict(method='sequential',
                        iterate = 'converge',
                        tolerance = 1e-6,
                        maxIter = 100,
                        FixedNodes = set(),
                        FixFeatures = False,
                        FixSurf = False,
                        qualityFunc = quality.MeanRatio,
                        InPlace = False
                    )

    NodeCoords, NodeConn, SmoothOptions = _SmoothingInputParser(NodeCoords, NodeConn, SurfConn, SmoothOptions, options)
    FreeNodes = SmoothOptions['FreeNodes']
    FixedNodes = SmoothOptions['FixedNodes']
    tolerance = SmoothOptions['tolerance']
    iterate = SmoothOptions['iterate']
    qualityFunc = SmoothOptions['qualityFunc']
    maxIter = SmoothOptions['maxIter']
    method = SmoothOptions['method']
    InPlace = SmoothOptions['InPlace']

    # Initialize
    lens = np.array([len(n) for n in NodeNeighbors])
    NodeConn = np.asarray(NodeConn)
    if TangentialSurface:
        SurfNodes = set([n for elem in SurfConn for n in elem])
        if labels is None:
            NodeNormals = M.NodeNormals
        else:
            NodeNormals = MultiSurface.NodeNormals
        SurfNodeNeighbors = utils.getNodeNeighbors(NodeCoords, SurfConn)
        for i in SurfNodes:
            NodeNeighbors[i] = SurfNodeNeighbors[i]
        for i in EdgeNodes:
            NodeNeighbors[i] = EdgeNodeNeighbors[i]

    q = qualityFunc(NodeCoords, NodeConn)
    qmin = np.nanmin(q)
    qmean = np.nanmean(q)

    # Smoothing Functions
    def SequentialSmoother(NodeCoords, q):
        
        for inode in FreeNodes:
            oldnode = np.copy(NodeCoords[inode])
            oldq = q[ElemConn[inode]]
            # NodeCoords[inode] = np.mean(NodeCoords[NodeNeighbors[inode]], axis=0)

            Q = NodeCoords[NodeNeighbors[inode]]
            if TangentialSurface and inode in SurfNodes:
                u = (1/lens[inode]) * np.sum(Q - NodeCoords[inode],axis=0)
                U = 1*(u - np.sum(u*NodeNormals[inode],axis=0)*NodeNormals[inode])
            else:
                U = (1/lens[inode]) * np.sum(Q - NodeCoords[inode],axis=0)
            NodeCoords[inode] += U

            q[ElemConn[inode]] = qualityFunc(NodeCoords, NodeConn[ElemConn[inode]])

            newqmin = np.min(q[ElemConn[inode]])
            oldqmin = np.min(oldq)
            if target == 'mean':
                oldqmean = np.mean(oldq)
                newqmean = np.mean(q[ElemConn[inode]])

                if (newqmean < oldqmean) | ((newqmin < oldqmin) & (newqmin < 0)):
                    # If mean gets worse or a negative min gets worse
                    NodeCoords[inode] = oldnode
                    q[ElemConn[inode]] = oldq
            elif target == 'min':
                # if min gets worse
                if (oldqmin < newqmin):
                    NodeCoords[inode] = oldnode
                    q[ElemConn[inode]] = oldq

        return NodeCoords, q

    if method == 'sequential':
        smoother = SequentialSmoother
    else:
        raise ValueError(f'Invalid method "{str(method):s}", must be "sequential".')

    # Iterate
    qmin_hist = [qmin]
    qmean_hist = [qmean]
    
    if SmoothOptions['iterate'] == 'converge':
        condition = lambda i, q, qmin, qmean : (i == 0) | (i < maxIter) & ((np.sum(np.abs(np.min(q) - qmin)) > tolerance) | (np.sum(np.abs(np.mean(q) - qmean)) > tolerance))
    elif isinstance(SmoothOptions['iterate'], (int, np.integer)):
        condition = lambda i, q, qmin, qmean : i < SmoothOptions['iterate']
    else:
        raise ValueError('options["iterate"] must be "converge" or an integer.')
    
    i = 0
    while condition(i, q, qmin, qmean):
        i += 1

        qmin = np.min(q)
        qmean = np.mean(q)
        qmin_hist.append(qmin)
        qmean_hist.append(qmean)
        NodeCoords, q = smoother(NodeCoords, q)

    if InPlace:
        Mnew = M
    else:
        Mnew = M.copy()
    Mnew.NodeCoords = NodeCoords

    return Mnew

# Needs update:   
def GlobalLaplacianSmoothing(NodeCoords, NodeConn,FeatureNodes=[],FixedNodes=set(),FeatureWeight=1,BaryWeight=1/3):
    # Ji, Z., Liu, L. and Wang, G., 2005, December. A global laplacian 
    # smoothing approach with feature preservation. In Ninth International 
    # Conference on Computer Aided Design and Computer Graphics
    
    NodeNeighbors = utils.getNodeNeighbors(NodeCoords,NodeConn)
    
    NNode = len(NodeCoords)
    NFeature = len(FeatureNodes)
    NElem = len(NodeConn)
    
    # Vertex Weights (NNode x NNode)

    Lrows = []
    Lcols = []
    Lvals = []
    for row in range(NNode):
        Lrows.append(row)
        Lcols.append(row)
        Lvals.append(1)
        for col in NodeNeighbors[row]:
            Lrows.append(row)
            Lcols.append(col)
            Lvals.append(-1/len(NodeNeighbors[row]))
    L = sparse.coo_matrix((Lvals,(Lrows,Lcols)))
    # L = np.zeros([NNode,NNode])
    # for row in range(NNode):
    #     # Vertex Weights
    #     L[row,row] = 1
    #     for col in NodeNeighbors[row]:
    #         L[row,col] = -1/len(NodeNeighbors[row]) 
            
    # Feature Weights (NFeature x NNode)
    if NFeature > 0:
        Frows = [row for row in FeatureNodes]
        Fcols = [col for col in FeatureNodes]
        Fvals = [FeatureWeight for i in range(NFeature)]
        F = sparse.coo_matrix((Fvals,(Frows,Fcols)))    
    else:
        F = sparse.coo_matrix(np.zeros([0,NNode]))
    # F = np.zeros([NFeature,NNode])
    # for row in FeatureNodes:
    #     F[row,row] = FeatureWeight
    
    # Barycenter Weights (NElem x NNode)
    Zrows = [e for e in range(NElem) for i in range(len(NodeConn[0]))]
    Zcols = [n for elem in NodeConn for n in elem]
    Zvals = [BaryWeight for e in range(NElem) for i in range(len(NodeConn[0]))]
    Z = sparse.coo_matrix((Zvals,(Zrows,Zcols)))
    # Z = np.zeros([NElem,NNode])
    # for row in range(len(NodeConn)):
    #     for col in NodeConn[row]:
    #         Z[row,col] = BaryWeight
    A = sparse.vstack((L,F,Z)).tocsc()
    At = A.transpose()
    AtA = At*A
    # Vertex b Matrix (NNode x 1)
    # bL = np.zeros([NNode,1])
    bL = sparse.coo_matrix(np.zeros([NNode,1]))

    NewCoords = np.zeros(np.shape(NodeCoords))
    # For each dimension:
    for d in range(len(NodeCoords[0])):        
            
        # Feature b Matrix (NFeature x 1)
        # bF = np.zeros([NFeature,1])
        if NFeature > 0:
            bFcols = np.zeros(NFeature,dtype=int)
            bFrows = list(FeatureNodes)
            bFvals = [FeatureWeight*NodeCoords[i][d] for i in bFrows]
            # for i,f in enumerate(FeatureNodes):
                # bF[i] = FeatureWeight*NodeCoords[f][d]
            bF = sparse.coo_matrix((bFvals,(bFrows,bFcols)))
        else:
            bF = sparse.coo_matrix(np.zeros([0,1]))
        # Bary b Matrix (NElem x 1)
        bZcols = np.zeros(NElem,dtype=int)
        bZrows = np.arange(len(NodeConn),dtype=int)
        bZvals = [BaryWeight*sum([NodeCoords[node][d] for node in elem]) for elem in NodeConn]
        bZ = sparse.coo_matrix((bZvals,(bZrows,bZcols)))
        # bZ = np.zeros([NElem,1])
        # for i,elem in enumerate(NodeConn):
        #     bZ[i] = BaryWeight*sum([NodeCoords[node][d] for node in elem])
            
        b = sparse.vstack([bL,bF,bZ])
        NewCoords[:,d] = spsolve(AtA, sparse.csc_matrix(At*b))
    NewCoords = NewCoords.tolist()
    NewCoords = [NodeCoords[i] if i in FixedNodes else coord for i,coord in enumerate(NewCoords)]
    return NewCoords

def GeoTransformSmoothing(M, sigma_min=None, sigma_max=None, eta=None, rho=None, qualityThreshold=.2, options=dict()):

    # For method=sequential only, elements with quality less than qualityThreshold will be considered
    NodeCoords, NodeConn = M
    NodeCoords = np.copy(NodeCoords)
    NodeNeighbors = M.NodeNeighbors
    ElemConn = M.ElemConn
    SurfConn = M.SurfConn
    # Process inputs
    convergence_lookback = 10
    SmoothOptions = dict(method='simultaneous',
                        iterate = 'converge',
                        tolerance = 1e-6,
                        maxIter = 100,
                        FixedNodes = set(),
                        FixFeatures = False,
                        FixSurf = True,
                        qualityFunc = quality.MeanRatio
                    )

    NodeCoords, NodeConn, SmoothOptions = _SmoothingInputParser(NodeCoords, NodeConn, SurfConn, SmoothOptions, options)
    FreeNodes = SmoothOptions['FreeNodes']
    FixedNodes = SmoothOptions['FixedNodes']
    tolerance = SmoothOptions['tolerance']
    iterate = SmoothOptions['iterate']
    qualityFunc = SmoothOptions['qualityFunc']
    method = SmoothOptions['method']
    if 'maxIter' not in options.keys() and method=='sequential':
        maxIter = len(NodeConn)
    else:
        maxIter = SmoothOptions['maxIter']
    

    # Initialize
    
    if method == 'simultaneous':
        RElemConn = utils.PadRagged(ElemConn)
        
        if sigma_min is None: sigma_min = 0
        if sigma_max is None: sigma_max = 2
        if eta is None: eta = 0
        if rho is None: rho = 0.1
        q = qualityFunc(NodeCoords, NodeConn)
        qmin = np.min(q)
        qmean = np.mean(q)
        args = ()
        
    elif method == 'sequential':
        ElemNeighbors = utils.getElemNeighbors(NodeCoords, NodeConn, mode='node')
        FixedNodes = set(FixedNodes)
        if sigma_min is None: sigma_min = 1e-2
        if sigma_max is None: sigma_max = 1e-2
        if eta is None: eta = 0
        if rho is None: rho = 0.75
        qual = qualityFunc(NodeCoords, NodeConn)

        # Heapq structure stores quality and a pointer to the elemid
        q = list(zip(qual, range(len(NodeConn)), range(len(NodeConn))))
        heapq.heapify(q)
        qmin = q[0][0]
        qmean = 0 # mean is ignored for sequential

        lookup = {i:i for i in range(len(NodeConn))}            # Lookup table relating element ids to heap ids
        lookup_nextid = len(lookup)                             # counter to ensure identifiers stay unique
        visit_counts = np.zeros(len(NodeConn))                  # Tracker of the number of times an element has been visited

        args = (lookup, visit_counts, lookup_nextid)
    else:
        raise ValueError(f'Invalid method "{str(method):s}", must be "simultaneous" or "sequential".')

    # Smoothing functions
    def SimultaneousSmoother(NodeCoords, q):
        
        points = np.asarray(NodeCoords)[np.asarray(NodeConn)]

        D = np.stack([
            points[:,1] - points[:,0], 
            points[:,2] - points[:,0],
            points[:,3] - points[:,0]
            ]).swapaxes(0,1)

        n1 = np.cross(points[:,3,:] - points[:,1,:], points[:,2,:] - points[:,1,:])
        n2 = np.cross(points[:,3,:] - points[:,2,:], points[:,0,:] - points[:,2,:])
        n3 = np.cross(points[:,1,:] - points[:,3,:], points[:,0,:] - points[:,3,:])
        n4 = np.cross(points[:,1,:] - points[:,0,:], points[:,2,:] - points[:,0,:])

        N1 = n1/np.sqrt(np.linalg.norm(n1, axis=1))[:,None]
        N2 = n2/np.sqrt(np.linalg.norm(n2, axis=1))[:,None]
        N3 = n3/np.sqrt(np.linalg.norm(n3, axis=1))[:,None]
        N4 = n4/np.sqrt(np.linalg.norm(n4, axis=1))[:,None]
        N = np.stack([N1, N2, N3, N4]).swapaxes(0,1)

        # Calculate transformation factor sigma
        sigma = sigma_min + (sigma_max - sigma_min)*(1 - q)

        # Transform
        p_trans = points + sigma[:,None,None] * N

        # Scale
        D_trans = np.stack([
            p_trans[:,1] - p_trans[:,0], 
            p_trans[:,2] - p_trans[:,0],
            p_trans[:,3] - p_trans[:,0]
            ]).swapaxes(0,1)

        c_trans = np.mean(p_trans,axis=1) # element centroids of transformed tets

        xi = (np.linalg.det(D)/np.linalg.det(D_trans))**(1/3) # Volume preservation scaling

        p_scal = c_trans[:,None,:] + xi[:,None,None]*(p_trans - c_trans[:,None,:])

        # Relax
        p_relax = rho*p_scal + (1 - rho)*points

        # Weighted averaging
        weights = ((1 - q)**eta)[:,None,None] * np.ones(points.shape)
        NewCoords = np.zeros_like(NodeCoords)
        w = np.zeros_like(NodeCoords)
        np.add.at(NewCoords, NodeConn.flatten(), weights.reshape(-1,3)*p_relax.reshape(-1,3))
        np.add.at(w, NodeConn.flatten(), weights.reshape(-1,3))
        NewCoords = NewCoords/w

        # Reset fixed nodes
        NewCoords[FixedNodes] = NodeCoords[FixedNodes]

        # Reset any inverted elements inversions
        V = quality.tet_volume(NewCoords, NodeConn)
        if np.any(V <= 0):
            V = np.append(V, np.inf)
            while np.any(V <= 0):
                affected = np.where(np.any(V[RElemConn] <= 0,axis=1))[0]
                NewCoords[affected] = NodeCoords[affected]
                affected_elems = np.unique([e for a in affected for e in ElemConn[a]])
                V[affected_elems] = quality.tet_volume(NewCoords, NodeConn[affected_elems])

        # Update quality
        qnew = qualityFunc(NewCoords, NodeConn)
        qmin = np.min(np.array(q))   # This is the old min
        qmean = np.mean(np.array(q)) # This is the old mean
        return NewCoords, qnew, qmin, qmean, ()

    def SequentialSmoother(NodeCoords, q, lookup, visit_counts, lookup_nextid):
        
        qual, lookup_key, elemid = heapq.heappop(q)
        while lookup_key != lookup[elemid]:
            # Skip inactive entries
            qual, lookup_key, elemid = heapq.heappop(q)
            
            
        visit_counts[elemid] += 1
        repeat_factor = .01*np.sqrt(visit_counts[elemid])

        points = NodeCoords[NodeConn[elemid]]

        D = np.array([
            points[1] - points[0], 
            points[2] - points[0],
            points[3] - points[0]
        ])
        
        n1 = np.cross(points[3] - points[1], points[2] - points[1])
        n2 = np.cross(points[3] - points[2], points[0] - points[2])
        n3 = np.cross(points[1] - points[3], points[0] - points[3])
        n4 = np.cross(points[1] - points[0], points[2] - points[0])

        N1 = n1/np.sqrt(np.linalg.norm(n1))
        N2 = n2/np.sqrt(np.linalg.norm(n2))
        N3 = n3/np.sqrt(np.linalg.norm(n3))
        N4 = n4/np.sqrt(np.linalg.norm(n4))
        N = np.vstack([N1, N2, N3, N4])

        # Calculate transformation factor sigma
        sigma = sigma_min + (sigma_max - sigma_min)*(1 - qual)

        # Transform
        p_trans = points + sigma * N

        # Scale
        D_trans = np.array([
            p_trans[1] - p_trans[0], 
            p_trans[2] - p_trans[0],
            p_trans[3] - p_trans[0]
            ])

        c_trans = np.mean(p_trans, axis=0) # element centroids of transformed tets

        xi = (np.linalg.det(D)/np.linalg.det(D_trans))**(1/3) # Volume preservation scaling

        p_scal = c_trans + xi*(p_trans - c_trans)

        # Relax
        p_relax = rho*p_scal + (1 - rho)*points

        NodeCoords[NodeConn[elemid]] = p_relax
        for i,node in enumerate(NodeConn[elemid]):
            if node not in FixedNodes:
                NodeCoords[node] = p_relax[i]
        # Update data structures
        if any(quality.tet_volume(NodeCoords, NodeConn[ElemNeighbors[elemid]]) < 0):
            # If any inversions occured, don't modify nodes
            key = (qual+repeat_factor, lookup_key, elemid)
            heapq.heappush(q, key) # Return this element to the heap
            NodeCoords[NodeConn[elemid]] = points
            
        else:
            # Calculate new quality for the element and neighbors
            newquals = qualityFunc(NodeCoords, NodeConn[[elemid, *ElemNeighbors[elemid]]])

            if newquals[0] < qualityThreshold:
                # Update this element's entry
                key = (newquals[0]+repeat_factor, lookup_key, elemid)
                heapq.heappush(q,key)

            # Update neighboring element entries
            for i,neighbor in enumerate(ElemNeighbors[elemid]):
                # If the neighbor entries fall below the quality threshold, add new entries to heap
                repeat_factor = .01*np.sqrt(visit_counts[neighbor])
                if newquals[i+1] < qualityThreshold:
                    # Define new heap entry
                    key = (newquals[i+1]+repeat_factor, lookup_nextid, neighbor) 

                    # Update the lookup id in the neighbors lookup entry
                    lookup[neighbor] = lookup_nextid

                    heapq.heappush(q, key)
                    lookup_nextid += 1

        qmin = qual
        qmean = 0
        args = (lookup, visit_counts, lookup_nextid)
        return NodeCoords, q, qmin, qmean, args

    if method == 'simultaneous':
        smoother = SimultaneousSmoother
        if SmoothOptions['iterate'] == 'converge':
            condition = lambda i, q, qmin_hist, qmean_hist : (i == 0) | (i < maxIter) & ((np.sum(np.abs(np.min(q) - qmin_hist[-convergence_lookback:])) > tolerance) | (np.sum(np.abs(np.mean(q) - qmean_hist[-convergence_lookback:])) > tolerance))
        elif isinstance(SmoothOptions['iterate'], (int, np.integer)):
            condition = lambda i, q, qmin_hist, qmean_hist : i < SmoothOptions['iterate']

    elif method == 'sequential':
        smoother = SequentialSmoother
        condition = lambda i, q, qmin_hist, qmean_hist : (i < convergence_lookback) | (i < maxIter) & (np.sum(np.abs(q[0][0] - qmin_hist[-convergence_lookback:])) > tolerance)


    else:
        raise ValueError(f'Invalid method "{str(method):s}", must be "simultaneous" or "sequential".')
    
    
    qmin_hist = [qmin]
    qmean_hist = [qmean]
    
    i = 0
    # Iterate
    while condition(i, q, qmin_hist, qmean_hist):
        i += 1

        # print(f'{qmin:.6f}, {qmean:.6f}')
        
        NodeCoords, q, qmin, qmean, args = smoother(NodeCoords, q, *args)
        qmin_hist.append(qmin)
        qmean_hist.append(qmean)

    if 'mesh' in dir(mesh):
        Mnew = mesh.mesh(NodeCoords, NodeConn)
    else:
        Mnew = mesh(NodeCoords, NodeConn)

    return Mnew

# Needs update:
def SegmentSpringSmoothing(M, StiffnessFactor=1, Forces=None, Displacements=None, L0Override='min', CellCentered=True, FaceCentered=True, return_KF=False, options=dict()):
    
    """
    SegmentSpringSmoothing - 
    Blom, F.J., 2000. Considerations on the spring analogy. International journal for numerical methods in fluids, 32(6), pp.647-668.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.
    NodeNeighbors : list, optional
        List of node neighboring nodes for each node in the mesh.
        If provided with ElemConn, will avoid the need to recalculate, by default None.
        If only one is provided, both will be recalculated.
    ElemConn : list, optional
        List of elements connected to each node.
        If provided with NodeNeighbors, will avoid the need to recalculate, by default None.
        If only one is provided, both will be recalculated.
    StiffnessFactor : float, optional
        Specifies a scaling factor for the stiffness of the springs. The default is 1.
    FixedNotes : list or set, optional
        Set of nodes to be held fixed. The default is set().
    Forces : list, optional
        Set of applied forces. If specified, forces must be specified for every node, 
        with a force of [0,0,0] applied to unloaded nodes. The default is None.
    L0Override : str or float, optional
        Override value for assigning the length of springs whose initial length is 0.
        'min' : 0-length springs will be assigned to be equal to the shortest non-0-length spring in the mesh.
        'max' : 0-length springs will be assigned to be equal to the longest spring in the mesh.
        float : 0-length springs will be assigned to be equal to the specified float.
        The default is 'min'.
    CellCentered : bool, optional
        If true, will add cell (element)-centered springs, adding springs between each node in an element to
        the element centrod, by default True.
    FaceCentered : bool, optional
        If true, will add face-centered springs, adding springs between each node in an element face to
        the face centrod, by default True.
    return_KF : bool, optional
        If true, will return a tuple (K,F) containing the matrices (in scipy sparse formats) K and F of the
        linear spring equation KU=F which is solved to find the the nodal displacements, by default False.

    Returns
    -------
    Xnew : list
        Updated list of nodal coordinates.
    dXnew : list
        List of nodal displacements to go from NodeCoords -> Xnew
    KF : tuple of sparse matrices, optional
        If return_KF is true, the tuple of sparse matrice KF=(K,F) is returned.
    """

    
    NodeCoords, NodeConn = M
    if type(NodeConn) is np.ndarray: NodeConn = NodeConn.tolist()
    NodeCoords = np.copy(NodeCoords)
    NodeNeighbors = M.NodeNeighbors
    ElemConn = M.ElemConn
    SurfConn = M.SurfConn
    
    # Process inputs
    SmoothOptions = dict(method='simultaneous',
                        iterate = 'converge',
                        tolerance = 1e-3,
                        maxIter = 20,
                        FixedNodes = set(),
                        FixFeatures = False,
                        FixSurf = True,
                        qualityFunc = quality.MeanRatio
                    )

    NodeCoords, NodeConn, SmoothOptions = _SmoothingInputParser(NodeCoords, NodeConn, SurfConn, SmoothOptions, options)
    FreeNodes = SmoothOptions['FreeNodes']
    FixedNodes = SmoothOptions['FixedNodes']
    tolerance = SmoothOptions['tolerance']
    iterate = SmoothOptions['iterate']
    qualityFunc = SmoothOptions['qualityFunc']
    maxIter = SmoothOptions['maxIter']
    method = SmoothOptions['method']

    # if NodeNeighbors is None or ElemConn is None:
    #     NodeNeighbors,ElemConn = utils.getNodeNeighbors(NodeCoords,NodeConn)
    if Forces is None or len(Forces) == 0:
        Forces = np.zeros((len(NodeCoords),3))
    else:
        assert len(Forces) == len(NodeCoords), 'Forces must be assigned for every node'
    
    if Displacements is None or len(Displacements) == 0:
        Displacements = np.zeros((len(NodeCoords),3))
    else:
        assert len(Displacements) == len(NodeCoords), 'Displacements must be assigned for every node'


    TempCoords = np.append(NodeCoords, [[np.nan,np.nan,np.nan]], axis=0)
    # NodeCoords = np.array(NodeCoords)
    RNeighbors = utils.PadRagged(NodeNeighbors+[[-1]])
    Points = TempCoords[RNeighbors]
    lengths = np.sqrt((TempCoords[:,0,None]-Points[:,:,0])**2 + (TempCoords[:,1,None]-Points[:,:,1])**2 + (TempCoords[:,2,None]-Points[:,:,2])**2)

    if L0Override == 'min':
        minL = np.nanmin(lengths[lengths!=0])
        lengths[lengths==0] = minL
    elif L0Override == 'max':
        maxL = np.nanmax(lengths[lengths!=0])
        lengths[lengths==0] = maxL
    elif isinstance(L0Override, (int,float)):
        lengths[lengths==0] = L0Override
    else:
        raise Exception("Invalid L0Override value. Must be 'min', 'max', an int, or a float")
    
    k = StiffnessFactor/lengths

    # Set Right hand side of fixed or prescribed disp nodes
    Forces = np.asarray(Forces)
    Forces[FixedNodes] = [0,0,0]

    DispNodes = np.where(np.any(Displacements != 0, axis=1))[0]
    Forces[DispNodes] = Displacements[DispNodes]
    
    Krows_diag = np.arange(len(NodeCoords))
    Kcols_diag = np.arange(len(NodeCoords))
    Kvals_diag = np.nansum(k[:-1],axis=1) 
    if CellCentered:
        centroids = M.Centroids
        centroids = np.append(centroids,[[np.nan,np.nan,np.nan]],axis=0)
        RElemConn = utils.PadRagged(ElemConn)
        ElemConnCentroids = centroids[RElemConn]
        ElemConnCenterDist = np.sqrt((NodeCoords[:,0,None]-ElemConnCentroids[:,:,0])**2 + (NodeCoords[:,1,None]-ElemConnCentroids[:,:,1])**2 + (NodeCoords[:,2,None]-ElemConnCentroids[:,:,2])**2)
        kcenters = StiffnessFactor/ElemConnCenterDist
        Kvals_diag += np.nansum(kcenters,axis=1)
    if FaceCentered:
        Faces = M.Faces
        fcentroids = utils.Centroids(NodeCoords,Faces)
        fcentroids = np.append(fcentroids,[[np.nan,np.nan,np.nan]],axis=0)
        FConn = utils.getElemConnectivity(NodeCoords,Faces)
        RFConn = utils.PadRagged(FConn)
        FConnCentroids = fcentroids[RFConn]
        FConnCenterDist = np.sqrt((NodeCoords[:,0,None]-FConnCentroids[:,:,0])**2 + (NodeCoords[:,1,None]-FConnCentroids[:,:,1])**2 + (NodeCoords[:,2,None]-FConnCentroids[:,:,2])**2)
        fkcenters = StiffnessFactor/FConnCenterDist
        Kvals_diag += np.nansum(fkcenters,axis=1)

    # Set stiffness matrix for prescribed displacement nodes
    Kvals_diag[FixedNodes] = 1
    Kvals_diag[DispNodes] = 1
    UnfixedNodes = np.array(list(set(range(len(NodeCoords))).difference(FixedNodes).difference(DispNodes)))
    
    template = (RNeighbors[:-1]>=0)[UnfixedNodes]
    flattemplate = template.flatten()
    Krows_off = (template.astype(int)*UnfixedNodes[:,None]).flatten()[flattemplate]
    Kcols_off = RNeighbors[UnfixedNodes].flatten()[flattemplate]
    Kvals_off = -k[UnfixedNodes].flatten()[flattemplate]
    
    Krows = np.concatenate((Krows_diag,Krows_off))
    Kcols = np.concatenate((Kcols_diag,Kcols_off))
    Kvals = np.concatenate((Kvals_diag,Kvals_off))

    if CellCentered:
        RNodeConn = utils.PadRagged(NodeConn,fillval=-1)
        RNodeConn = np.append(RNodeConn,-1*np.ones((1,RNodeConn.shape[1]),dtype=int),axis=0)
        pretemplate = RNodeConn[RElemConn]
        # template = ((pretemplate >= 0) & (pretemplate != np.arange(len(NodeCoords))[:,None,None]))[UnfixedNodes]
        template = (pretemplate >= 0)[UnfixedNodes]
        flattemplate = template.flatten()
        Krows_Ccentered = (template.astype(int)*UnfixedNodes[:,None,None]).flatten()[flattemplate]
        Kcols_Ccentered = pretemplate[UnfixedNodes][template].flatten()
        Kvals_Ccentered = -np.repeat(kcenters[UnfixedNodes][:,:,None],template.shape[2],2)[template]/template.shape[2]

        Krows = np.concatenate((Krows,Krows_Ccentered))
        Kcols = np.concatenate((Kcols,Kcols_Ccentered))
        Kvals = np.concatenate((Kvals,Kvals_Ccentered))

    if FaceCentered:
        RFaces = utils.PadRagged(Faces,fillval=-1)
        RFaces = np.append(RFaces,-1*np.ones((1,RFaces.shape[1]),dtype=int),axis=0)
        pretemplate = RFaces[RFConn]
        # template = ((pretemplate >= 0) & (pretemplate != np.arange(len(NodeCoords))[:,None,None]))[UnfixedNodes]
        template = (pretemplate >= 0)[UnfixedNodes]
        flattemplate = template.flatten()
        Krows_Fcentered = (template.astype(int)*UnfixedNodes[:,None,None]).flatten()[flattemplate]
        Kcols_Fcentered = pretemplate[UnfixedNodes][template].flatten()
        Kvals_Fcentered = -np.repeat(fkcenters[UnfixedNodes][:,:,None],template.shape[2],2)[template]/template.shape[2]

        Krows = np.concatenate((Krows,Krows_Fcentered))
        Kcols = np.concatenate((Kcols,Kcols_Fcentered))
        Kvals = np.concatenate((Kvals,Kvals_Fcentered))

    K = sparse.coo_matrix((Kvals,(Krows,Kcols)))
    F = sparse.csc_matrix(Forces)
    dXnew = spsolve(K.tocsc(), F).toarray()
    
    Xnew = NodeCoords + dXnew
    Xnew[FixedNodes] = np.array(NodeCoords)[FixedNodes] # Enforce fixed nodes

    if 'mesh' in dir(mesh):
        Mnew = mesh.mesh(Xnew, NodeConn)
    else:
        Mnew = mesh(Xnew, NodeConn)

    if return_KF:
        return Mnew, (K,F)
    return Mnew

def NodeSpringSmoothing(M, Stiffness=1, Forces=None, Displacements=None, options=dict()):
    """
    Perform node spring smoothing, with or without mesh deformation. Uses the
    node spring analogy :cite:p:`Blom2000` to redistribute nodes and achieve
    equilibrium. If Forces and/or Displacements are prescribed, this an be used
    to deform a mesh while keeping nodes spread apart. 
    
    .. note::
        Element inversions aren't strictly prevented and may result from
        large deformations.

    Parameters
    ----------
    M : mymesh.mesh
        Mesh object to smooth
    Stiffness : float, optional
        Spring stiffness, by default 1. If no forces are applied, the choice
        of stiffness is irrelevant.
    Forces : np.ndarray or NoneType, optional
        nx3 array of applied forces, where n is the number of nodes in the mesh.
        If provided, there must be forces assigned to all nodes, by default None.
    Displacments : np.ndarray or NoneType, optional
        nx3 array of applied forces, where n is the number of nodes in the mesh.
        If provided, there must be forces assigned to all nodes, by default None.
        Nodes with non-zero displacements will be held fixed at their displaced
        position, while nodes with zero displacement in x, y, and z will be 
        free to move (to prescribe a displacement of 0 to hold a node in place,
        add that node to options['FixedNodes'].)
    options : dict
        Smoothing options. Available options are:

        iterate : int or str
            Fixed number of iterations to perform, or 'converge' to iterate until
            convergence, by default 'converge'.
        tolerance : float
            Convergence tolerance. For local Laplacian smoothing, iteration
            will terminate if the largest movement of a node is less than the
            specified tolerance, by default 1e-3.
        maxIter : int
            Maximum number of iterations when iterate='converge', By default 20.
        FixedNodes : set or array_like
            Set of nodes that are held fixed during iteration, by default none
            are fixed.
        FixFeatures : bool
            If true, feature nodes on edges or corners (identified by
            :func:`~mymesh.utils.DetectFeatures`) will be held in place, by default False.
        FixSurf : bool
            If true, all nodes on the surface will be held in place and only 
            interior nodes will be smoothed, by default True.

    Returns
    -------
    Mnew : mymesh.mesh
        Mesh object with the new node locations.
    """    
    # Blom, F.J., 2000. Considerations on the spring analogy. International journal for numerical methods in fluids, 32(6), pp.647-668.
    
    NodeCoords, NodeConn = M
    NodeCoords = np.copy(NodeCoords)
    NodeNeighbors = M.NodeNeighbors
    SurfConn = M.SurfConn
    
    # Process inputs
    SmoothOptions = dict(method='simultaneous',
                        iterate = 'converge',
                        tolerance = 1e-3,
                        maxIter = 20,
                        FixedNodes = set(),
                        FixFeatures = False,
                        FixSurf = False,
                        qualityFunc = quality.MeanRatio,
                        InPlace = False
                    )

    NodeCoords, NodeConn, SmoothOptions = _SmoothingInputParser(NodeCoords, NodeConn, SurfConn, SmoothOptions, options)
    FreeNodes = SmoothOptions['FreeNodes']
    FixedNodes = SmoothOptions['FixedNodes']
    tolerance = SmoothOptions['tolerance']
    iterate = SmoothOptions['iterate']
    qualityFunc = SmoothOptions['qualityFunc']
    maxIter = SmoothOptions['maxIter']
    method = SmoothOptions['method']
    InPlace = SmoothOptions['InPlace']
    if Forces is None or len(Forces) == 0:
        Forces = np.zeros((len(NodeCoords),3))
    else:
        assert len(Forces) == len(NodeCoords), 'Forces must be assigned for every node'

    if Displacements is None or len(Displacements) == 0:
        Displacements = np.zeros((len(NodeCoords),3))
    else:
        assert len(Displacements) == len(NodeCoords), 'Displacements must be assigned for every node'
    
    NodeCoords += Displacements
    FixedNodes = np.unique(np.append(FixedNodes, np.where(np.any(Displacements != 0, axis=1))))

    k = Stiffness

    RNeighbors = utils.PadRagged(NodeNeighbors, fillval=-1)
    RNeighbors = np.append(RNeighbors, [np.repeat(-1, RNeighbors.shape[1])], axis=0)

    X = np.append(NodeCoords, [[np.nan, np.nan, np.nan]], axis=0)
    Forces = np.append(Forces, [[0, 0, 0]], axis=0)
    unattached = np.all(RNeighbors == -1, axis=1)

    thinking = True
    iteration = 0
    while thinking:
        
        with np.errstate(divide='ignore', invalid='ignore'):
            Xnew = (np.nansum(k*X[RNeighbors],axis=1) + Forces)/(np.sum(k*(RNeighbors != -1),axis=1))[:,None]
            Xnew[unattached] = X[unattached]
            Xnew[FixedNodes] = X[FixedNodes]

        iteration += 1
        # print(np.linalg.norm(X[FreeNodes]-Xnew[FreeNodes]))
        if iteration > maxIter or np.linalg.norm(X[FreeNodes]-Xnew[FreeNodes]) < tolerance:
            thinking = False
        else:
            X = np.copy(Xnew)
    if InPlace:
        M.NodeCoords = Xnew[:-1]
        Mnew = M
        
    else:
        if 'mesh' in dir(mesh):
            Mnew = mesh.mesh(Xnew[:-1], NodeConn)
        else:
            Mnew = mesh(Xnew[:-1], NodeConn)

    return Mnew

# Needs update: 
def FixInversions(NodeCoords, NodeConn, FixedNodes=set(), maxfev=1000):
    """
    FixInversions Mesh optimization to reposition nodes in order to maximize the minimal area
    of elements connected to each node, with the aim of eliminating any inverted elements
    TODO: Need better convergence criteria to ensure no more inversions but not iterate more than necessary
    
    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.
    FixedNodes : set (or list), optional
        Set of nodes to hold fixed, by default set()
    maxfev : int, optional
        _description_, by default 1000

    Returns
    -------
    NewCoords : list
        Updated list of nodal coordinates.
    """
    V = quality.Volume(NodeCoords, NodeConn)
    if min(V) > 0:
        return NodeCoords
    
    InversionElems = np.where(np.asarray(V) < 0)[0]
    InversionConn = [NodeConn[i] for i in InversionElems]
    InversionNodes = np.unique([n for elem in InversionConn for n in elem])
    ProblemNodes = list(set(InversionNodes).difference(FixedNodes))
    if len(ProblemNodes) == 0:
        warnings.warn('Fixed nodes prevent any repositioning.')
        return NodeCoords
    ElemConn = utils.getElemConnectivity(NodeCoords, NodeConn)
    NeighborhoodElems = np.unique([e for i in ProblemNodes for e in ElemConn[i]])
    NeighborhoodConn = [NodeConn[i] for i in NeighborhoodElems]

    ArrayCoords = np.array(NodeCoords)

    def fun(x):
        ArrayCoords[ProblemNodes] = x.reshape((int(len(x)/3),3))
        v = quality.Volume(ArrayCoords,NeighborhoodConn)
        # print(sum(v<0))
        return -min(v)

    x0 = ArrayCoords[ProblemNodes].flatten()

    out = minimize(fun,x0,method='Nelder-Mead',options=dict(adaptive=True,xatol=.01,fatol=.01))#,maxfev=maxfev))
    if not out.success:
        warnings.warn('Unable to eliminate all element inversions.')
    ArrayCoords[ProblemNodes] = out.x.reshape((int(len(out.x)/3),3))

    NewCoords = ArrayCoords.tolist()
    return NewCoords

## Local Mesh Topology Operations
def TetContract(M, h, FixedNodes=set(), verbose=True, cleanup=True, maxIter=5, labels=None, FeatureAngle=25, sizing=None):
    """
    Edge contraction for tetrahedral mesh coarsening and quality improvement. 
    Edges with edge length less than `h` will attempt to be contracted. Surfaces
    and features are preserved by prioritizing surface/feature nodes over 
    interior nodes when deciding which node to remove in the edge collapse
    operation. Features (edges, corners), as determined by 
    :func:`~mymesh.utils.DetectFeatures`, are held fixed. An edge is only 
    contracted if doing so doesn't invert any elements or reduce the quality
    by creating a new element with a lower quality than was present in the 
    local edge neighborhood before the contraction. Edges are processed in a
    heap sorted by edge length, with shorter edges being contracted first. New
    edges created in the edge contraction process are added to the heap, thus 
    most edges are contracted in the first iteration, however some non-viable
    contractions may become viable after the first pass, thus multiple iterations
    can be performed until all possible edges are contracted.

    Parameters
    ----------
    M : mymesh.mesh
        Tetrahedral mesh to be contracted
    h : float
        Edge length below which wil be contracted. Using 4/5 of the target
        edge length is often recommended.
    FixedNodes : set or array_like, optional
        Indices of nodes to be held fixed, by default {}
    verbose : bool, optional
        If true, will display progress, by default True
    cleanup : bool, optional
        If true, unused nodes will be removed from the mesh and nodes will be
        renumbered, by default True. 
    maxIter : int, optional
        Maximum number of times to iterate through the mesh attempting to 
        contract edges, by default 10. Most edges will be contracted in the 
        first pass, however some contracts may not be permitted until other
        nearby elements have been modified. 
    labels : str or array_like, optional
        Element labels used to identify separate regions (e.g. materials) within
        a mesh, by default None. If provided as a string, the string must
        refer to an entry in `M.ElemData`, otherwise must be an array_like with
        the number of entries equal to the number of elements (`M.NElem`).
        Providing labels will preserve the interface and interface features
        between regions of differening labels. The labels of the new mesh will
        be stored in the ElemData of the returned mesh, either in 
        ElemData['labels'] (if labels were provided as an array), or the entry
        matching the original ElemData entry (if labels were provided as a 
        string).
    FeatureAngle : int, optional
        Angle (in degrees) used to identify features, by default 25. See
        :func:`~mymesh.utils.DetectFeatures` for more information.

    Returns
    -------
    Mnew : mymesh.mesh
        Coarsened tetrahedral mesh

    """

    cutoff = h # contraction will be attempted on edges below this length
    NewCoords = np.array(M.NodeCoords)
    NewConn = np.array(M.NodeConn, dtype=int)
    
    Edges = np.sort(M.Edges,axis=1)
    EdgeConn = M.EdgeConn
    EdgeElemConn = M.EdgeElemConn
    if labels is None:
        SurfConn = np.array(M.SurfConn, dtype=int)
        SurfEdges = np.sort(M.Surface.Edges)
        JunctionNodes = np.array([],dtype=int)
    else:
        if isinstance(labels, str):
            if labels in M.ElemData.keys():
                label_str = labels
                labels = M.ElemData[label_str]
            else:
                raise ValueError('If provided as a string, labels must correspond to an entry in M.ElemData')
        else:
            label_str = 'labels'
        assert len(labels) == M.NElem, 'labels must correspond to the number of elements.'
        if 'mesh' in dir(mesh):
            MultiSurface = mesh.mesh(verbose=False)
        else:
            MultiSurface = mesh(verbose=False)
        ulabels = np.unique(labels)
        label_nodes = np.zeros((len(ulabels),M.NNode),dtype=int) # For each label, boolean indicator of whether each node is touching an element with that label
        mesh_nodes = np.arange(M.NNode)
        for i,label in enumerate(ulabels):
            if 'mesh' in dir(mesh):
                m = mesh.mesh(NewCoords, NewConn[labels == label], verbose=False)
            else:
                m = mesh(NewCoords, NewConn[labels == label], verbose=False)
            
            MultiSurface.addElems(m.SurfConn)
            label_nodes[i,np.unique(m.NodeConn)] = 1
        MultiSurface.NodeConn = MultiSurface.Faces  # This prevents doubling of surface elements at interfaces
        SurfConn = MultiSurface.NodeConn
        SurfEdges = np.sort(MultiSurface.Edges)
        JunctionEdges = MultiSurface.Edges[np.array([len(conn) > 2 for conn in MultiSurface.EdgeElemConn])]
        JunctionNodes = np.unique(JunctionEdges)

    SurfEdgeSet = set(tuple(e) for e in SurfEdges)
        
    # Detect Features
    # TODO: Some redundant calculation (edges) occurs in DetectFeatures
    surface_nodes = np.unique(SurfEdges)
    surfnodeset = set(surface_nodes)
    fixed_nodes = np.array(list(FixedNodes),dtype=int)
    feat_edges, feat_corners = utils.DetectFeatures(NewCoords,SurfConn,angle=FeatureAngle)

    # 0 : interior; 1 : surface; 2 : feature edge; 3 : feature corner; 4 : fixed node
    FeatureRank = np.zeros(len(NewCoords))
    FeatureRank[surface_nodes] = 1
    FeatureRank[feat_edges]    = 2
    FeatureRank[JunctionNodes] = 2
    FeatureRank[feat_corners]  = 3
    FeatureRank[fixed_nodes]   = 4
    ElemConn = utils.getElemConnectivity(NewCoords, NewConn)
    ElemConnTable = {i : set([tuple(NewConn[e]) for e in ElemConn[i]]) for i in range(len(ElemConn))}
    SurfEdgeTable = {tuple(edge) : True if tuple(edge) in SurfEdgeSet else False for edge in Edges}

    # Get edge lengths
    EdgeDiff = NewCoords[Edges[:,0]] - NewCoords[Edges[:,1]]
    EdgeLengths = np.sqrt(EdgeDiff[:,0]**2 + EdgeDiff[:,1]**2 + EdgeDiff[:,2]**2)

    if type(sizing) is str and sizing == 'auto':
        sizing = 2*h

    if sizing is None:
        node_cutoff = None
        cutoff = np.repeat(h, M.NEdge)
    elif isinstance(sizing, (float, int)):
        if labels is None:
            Surface = M.Surface
        else:
            Surface = MultiSurface

        udf = implicit.mesh2udf(Surface, M.NodeCoords)
        node_cutoff = (sizing - h)*udf + h
        cutoff = np.mean(node_cutoff[Edges], axis=1)

    # Create edge heap
    heap = [(EdgeLengths[i], tuple(edge)) for i,edge in enumerate(Edges) if (EdgeLengths[i] < cutoff[i]) and (EdgeLengths[i] > 0)]
    inheap = {tuple(edge):True if (EdgeLengths[i] < cutoff[i]) and (EdgeLengths[i] > 0) else False for i,edge in enumerate(Edges)}
    inmesh = {tuple(edge):True for i,edge in enumerate(Edges)}
    heapq.heapify(heap)

    def do_collapse(targetnode, collapsenode):
        # Get affected elements
        AffectedConn = np.array(list(ElemConnTable[collapsenode]))
        
        # Update elements
        collapsed = np.any(AffectedConn==targetnode, axis=1) & np.any(AffectedConn==collapsenode, axis=1)
        UpdatedConn = AffectedConn[~collapsed]
        UpdatedConn[UpdatedConn == collapsenode] = targetnode
        if len(UpdatedConn) == 0:
            success = False
            return success

        # Check volume
        new_vol = quality.tet_volume(NewCoords, UpdatedConn)
        old_qual = quality.MeanRatio(NewCoords, AffectedConn)
        new_qual = quality.MeanRatio(NewCoords, UpdatedConn)
        
        if any(new_vol <= 0) or np.min(new_qual) < np.min(old_qual):
            success = False
            return success

        # Flip is valid, update data structures
        new_elems = [tuple(e) for e in UpdatedConn]
        old_elems = [tuple(e) for e in AffectedConn]

        for node in set([n for elem in old_elems for n in elem]):
            # Remove old nodes from the table  
            ElemConnTable[node].difference_update(old_elems)
            ElemConnTable[node].update([elem for elem in new_elems if node in elem])
            
        # Add new elements to the target node
        ElemConnTable[targetnode].update(new_elems)
        if labels is not None:
            label_nodes[:,targetnode] = label_nodes[:,collapsenode] | label_nodes[:,targetnode]

        # Remove all elements from the collapsed node
        ElemConnTable[collapsenode] = set()
        
        # Add new edges to the heap
        for edgenode in set(UpdatedConn[UpdatedConn != targetnode]):
            newedge = tuple(sorted((edgenode, targetnode)))
            inmesh[newedge] = True
            if newedge in inheap.keys() and inheap[newedge]:
                continue

            # All nodes that are newly connected to the target node
            L = np.linalg.norm(NewCoords[targetnode] - NewCoords[edgenode])
            if SurfEdgeTable[tuple(sorted((edgenode, collapsenode)))]:
                SurfEdgeTable[newedge] = True
            else:
                SurfEdgeTable[newedge] = False
            # if L < cutoff:
            if (sizing is None and L < h) or (sizing is not None and L < (node_cutoff[newedge[0]]+node_cutoff[newedge[1]])/2):
                heapq.heappush(heap, (L, newedge))
                inheap[newedge] = True

        success = True
        return success

    loop = 0; valid = 1;
    if verbose and 'tqdm' in sys.modules:
        tqdm_loaded = True
    else:
        tqdm_loaded = False
    while valid > 0 and len(heap) > 0 and loop < maxIter:
        loop += 1
        if verbose: print(f'TetContract Iteration {loop:d}:', end='')
        valid = 0
        invalid = 0
        if verbose and tqdm_loaded:
            progress = tqdm.tqdm(total=len(heap))
        while len(heap) > 0:
            l, edge = heapq.heappop(heap)

            inheap[edge] = False
            if verbose and tqdm_loaded:
                progress.update(1)
                L1 = len(heap)

            node1 = edge[0]
            node2 = edge[1]

            # Validity checks
            if len(ElemConnTable[node1]) == 0 or len(ElemConnTable[node2]) == 0:
                # Edge has already been removed
                continue
            elif FeatureRank[node1] == FeatureRank[node2] == 4:
                # Both nodes are fixed, can't collapse this edge
                continue
            elif FeatureRank[node1] > 0 and FeatureRank[node2] > 0 and not SurfEdgeTable[edge]:
                # Both nodes are on the surface, but they're not connected by a
                # surface edge -> connected through the body -> invalid
                continue
            elif FeatureRank[node1] > 1 and FeatureRank[node2] > 1:
                # This holds all edges and corners fixed
                # TODO: I'd prefer not to have this test, but another one that 
                # prevents that inconsistencies that this prevents
                continue

            # Determine which node to collapse
            if FeatureRank[node1] < FeatureRank[node2]:
                collapsenode = node1
                targetnode = node2
                equal_rank = False
            elif FeatureRank[node1] > FeatureRank[node2]:
                collapsenode = node2
                targetnode = node1
                equal_rank = False
            else:
                # if ranks are equal, try node1, but if fails, can retry with node 2
                collapsenode = node1
                targetnode = node2
                equal_rank = True 

            # Check if collapse is valid:
            success = do_collapse(targetnode, collapsenode)
            # If rank is equal, try the other node of the edge
            if equal_rank and not success:
                success = do_collapse(collapsenode, targetnode)
            
            if success:
                valid += 1
                inmesh[edge] = False
            else:
                invalid += 1
            
            if verbose and tqdm_loaded:
                L2 = len(heap)
                progress.total += max(0, L2-L1)
            
        NewConn = np.array(list(set([elem for i in ElemConnTable.keys() for elem in ElemConnTable[i]])))
        if valid > 0:
            Edges = np.array([key for key in inmesh.keys() if inmesh[key]])
            EdgeDiff = NewCoords[Edges[:,0]] - NewCoords[Edges[:,1]]
            EdgeLengths = np.sqrt(EdgeDiff[:,0]**2 + EdgeDiff[:,1]**2 + EdgeDiff[:,2]**2)

            # Create edge heap
            if sizing is None:
                heap = [(EdgeLengths[i], tuple(edge)) for i,edge in enumerate(Edges) if (EdgeLengths[i] < h) and (EdgeLengths[i] > 0)]
                inheap = {tuple(edge):True if (EdgeLengths[i] < h) and (EdgeLengths[i] > 0) else False for i,edge in enumerate(Edges)}
            else:
                cutoff = np.mean(node_cutoff[Edges], axis=1)
                # Create edge heap
                heap = [(EdgeLengths[i], tuple(edge)) for i,edge in enumerate(Edges) if (EdgeLengths[i] < cutoff[i]) and (EdgeLengths[i] > 0)]
                inheap = {tuple(edge):True if (EdgeLengths[i] < cutoff[i]) and (EdgeLengths[i] > 0) else False for i,edge in enumerate(Edges)}
           
            heapq.heapify(heap)

    if labels is not None:
        # Apply labels to the coarsened mesh
        filler = np.max(labels) + 1
        new_labels = np.repeat(filler, len(NewConn))
        for i,label in enumerate(ulabels):
            new_labels[np.all(label_nodes[i][NewConn],axis=1)] = label

    if cleanup:
        NewCoords, NewConn,_ = utils.RemoveNodes(NewCoords, NewConn)
     
    if 'mesh' in dir(mesh):
        Mnew = mesh.mesh(NewCoords, NewConn, verbose=M.verbose)
    else:
        Mnew = mesh(NewCoords, NewConn, verbose=M.verbose)
    if labels is not None:
        Mnew.ElemData[label_str] = new_labels

    return Mnew

def TetSplit(M, h, verbose=True, labels=None, sizing=None, QualitySizing=False):
    """
    Edge splitting of tetrahedral meshes. Edges with length greater than the 
    specified edge length (`h`) will be split by placing a new node at the 
    midpoint of the edge. Tetrahedral edge splitting is inherently interface
    and feature preserving as nodes are only added, not removed or moved. 

    This method is inspired by :cite:`Faraj2016` and :cite:`Hu2018`.

    Parameters
    ----------
    M : mymesh.mesh
        Tetrahedral mesh to be contracted
    h : float
        Edge length above which will be split. Using 4/3 of the target
        edge length is often recommended.
    verbose : bool, optional
        If true, will display progress, by default True
    labels : str or array_like, optional
        Element labels used to identify separate regions (e.g. materials) within
        a mesh, by default None. If provided as a string, the string must
        refer to an entry in `M.ElemData`, otherwise must be an array_like with
        the number of entries equal to the number of elements (`M.NElem`).
        Providing labels will preserve the interface and interface features
        between regions of differening labels. The labels of the new mesh will
        be stored in the ElemData of the returned mesh, either in 
        ElemData['labels'] (if labels were provided as an array), or the entry
        matching the original ElemData entry (if labels were provided as a 
        string).
    sizing : str, float, callable, or None
        Option for non-uniform element sizing. 

        float - Specify a second target edge length (hi) to be used for edges 
        far from the boundary. This will be used following :cite:t:`Faraj2016`
        to generate an adaptive sizing field h_node = (hi-h)*D(node)+h, where 
        D is a distance field from boundaries/interface evaluated at each
        node. The target edge length is then taken as the average of it's 
        two nodes

        'auto' - Uses hi = 2*h as the second target edge length and is used
        as described above for floats.

        callable - Uses a callable, vectorized function of three inputs (f(x,y,z))
        where x, y, z can be either scalar or vector coordinates and specifies the 
        target edge length for each node and takes the average for an edge

        None - Uniform sizing

    Returns
    -------
    Mnew : mymesh.mesh
        Tetrahedral mesh after edge splitting

    """
    
    # Setup Data Structures
    NodeCoords = np.asarray(M.NodeCoords)
    NodeConn = np.asarray(M.NodeConn)
    N = len(NodeCoords)
    Edges = [tuple(edge) for edge in np.sort(M.Edges,axis=1)]
    Elems = [tuple(elem) for elem in M.NodeConn]
    EdgeTable = {edge:tuple(Elems[ielem] for ielem in M.EdgeElemConn[i]) for i,edge in enumerate(Edges)}
    NodeTable = {i:coord for i,coord in enumerate(NodeCoords)}
    if labels is not None:
        if isinstance(labels, str):
            if labels in M.ElemData.keys():
                label_str = labels
                labels = M.ElemData[label_str]
            else:
                raise ValueError('If provided as a string, labels must correspond to an entry in M.ElemData')
        else:
            label_str = 'labels'
        assert len(labels) == M.NElem, 'labels must correspond to the number of elements.'
        ElemLabels = {elem:labels[i] for i,elem in enumerate(Elems)}
        
    ## Get edge lengths
    EdgeDiff = NodeCoords[M.Edges[:,0]] - NodeCoords[M.Edges[:,1]]
    EdgeLengths = np.sqrt(EdgeDiff[:,0]**2 + EdgeDiff[:,1]**2 + EdgeDiff[:,2]**2)

    if type(sizing) is str and sizing == 'auto':
        sizing = 2*h

    if sizing is None:
        node_cutoff = None
        cutoff = np.repeat(h, M.NEdge)
    elif isinstance(sizing, (float, int)):
        if labels is None:
            Surface = M.Surface
        else:
            if 'mesh' in dir(mesh):
                MultiSurface = mesh.mesh(verbose=False)
            else:
                MultiSurface = mesh(verbose=False)

            ulabels = np.unique(labels)
            label_nodes = np.zeros((len(ulabels),M.NNode),dtype=int) # For each label, boolean indicator of whether each node is touching an element with that label
            mesh_nodes = np.arange(M.NNode)
            for i,label in enumerate(ulabels):
                if 'mesh' in dir(mesh):
                    m = mesh.mesh(NodeCoords, NodeConn[labels == label], verbose=False)
                else:
                    m = mesh(NodeCoords, NodeConn[labels == label], verbose=False)
                
                MultiSurface.addElems(m.SurfConn)
                label_nodes[i,np.unique(m.NodeConn)] = 1
            Surface = MultiSurface

        udf = implicit.mesh2udf(Surface, M.NodeCoords)
        node_cutoff = (sizing - h)*udf + h
        cutoff = np.mean(node_cutoff[Edges], axis=1)


    # Create edge heap
    heap = [(-EdgeLengths[i], tuple(edge)) for i,edge in enumerate(Edges) if (EdgeLengths[i] > cutoff[i]) and (EdgeLengths[i] > 0)]
    inmesh = {tuple(edge):True for i,edge in enumerate(Edges)}
    heapq.heapify(heap)


    def do_split(L, edge, N):
        
        inmesh[edge] = False
        edge1 = (edge[0],N)
        edge2 = (edge[1],N)
        EdgeTable[edge1] = tuple()
        EdgeTable[edge2] = tuple()
        inmesh[edge1] = True
        inmesh[edge2] = True
        NewEdges = set()
        
        for elem in EdgeTable[edge]:
            # for each connected element
            a, b, c, d = elem
            lookup_key = np.sum(np.isin(elem, edge) * 2**np.arange(0,4)[::-1])

            if lookup_key == 3:
                new_elems = ((a,b,c,N), (d,b,a,N))
                # Split Edges - add one new element
                EdgeTable[(c,N)] += (new_elems[0],)
                EdgeTable[(d,N)] += (new_elems[1],)
                
                # Other New Edges - add both new elements
                NewEdges.update({(a,N), (b,N)})
                if (a,N) in EdgeTable.keys():
                    EdgeTable[(a,N)] += new_elems
                else:
                    EdgeTable[(a,N)] = new_elems
                if (b,N) in EdgeTable.keys():
                    EdgeTable[(b,N)] += new_elems
                else:
                    EdgeTable[(b,N)] = new_elems
                # Existing Edges - remove old element and add new element(s)
                ab = tuple(sorted((a,b)))
                ac = tuple(sorted((a,c)))
                ad = tuple(sorted((a,d)))
                bc = tuple(sorted((b,c)))
                bd = tuple(sorted((b,d)))

                EdgeTable[ab] = tuple(e for e in EdgeTable[ab] if e != elem) + new_elems
                EdgeTable[bc] = tuple(e for e in EdgeTable[bc] if e != elem) + (new_elems[0],)
                EdgeTable[ac] = tuple(e for e in EdgeTable[ac] if e != elem) + (new_elems[0],)
                EdgeTable[bd] = tuple(e for e in EdgeTable[bd] if e != elem) + (new_elems[1],)
                EdgeTable[ad] = tuple(e for e in EdgeTable[ad] if e != elem) + (new_elems[1],)
            elif lookup_key == 5:
                new_elems = ((a,b,c,N), (c,d,a,N))
                # Split Edges - add one new element
                EdgeTable[(b,N)] += (new_elems[0],)
                EdgeTable[(d,N)] += (new_elems[1],)
                
                # Other New Edges - add both new elements
                NewEdges.update({(a,N), (c,N)})
                if (a,N) in EdgeTable.keys():
                    EdgeTable[(a,N)] += new_elems
                else:
                    EdgeTable[(a,N)] = new_elems
                if (c,N) in EdgeTable.keys():
                    EdgeTable[(c,N)] += new_elems
                else:
                    EdgeTable[(c,N)] = new_elems
                # Existing Edges - remove old element and add new element(s)
                ab = tuple(sorted((a,b)))
                ac = tuple(sorted((a,c)))
                ad = tuple(sorted((a,d)))
                bc = tuple(sorted((b,c)))
                cd = tuple(sorted((c,d)))

                EdgeTable[ac] = tuple(e for e in EdgeTable[ac] if e != elem) + new_elems
                EdgeTable[ab] = tuple(e for e in EdgeTable[ab] if e != elem) + (new_elems[0],)
                EdgeTable[bc] = tuple(e for e in EdgeTable[bc] if e != elem) + (new_elems[0],)
                EdgeTable[cd] = tuple(e for e in EdgeTable[cd] if e != elem) + (new_elems[1],)
                EdgeTable[ad] = tuple(e for e in EdgeTable[ad] if e != elem) + (new_elems[1],)
            elif lookup_key == 9:
                new_elems = ((a,b,c,N), (b,d,c,N))
                # Split Edges - add one new element
                EdgeTable[(a,N)] += (new_elems[0],)
                EdgeTable[(d,N)] += (new_elems[1],)
                
                # Other New Edges - add both new elements
                NewEdges.update({(b,N), (c,N)})
                if (b,N) in EdgeTable.keys():
                    EdgeTable[(b,N)] += new_elems
                else:
                    EdgeTable[(b,N)] = new_elems
                if (c,N) in EdgeTable.keys():
                    EdgeTable[(c,N)] += new_elems
                else:
                    EdgeTable[(c,N)] = new_elems
                
                # Existing Edges - remove old element and add new element(s)
                ab = tuple(sorted((a,b)))
                ac = tuple(sorted((a,c)))
                ad = tuple(sorted((a,d)))
                bc = tuple(sorted((b,c)))
                bd = tuple(sorted((b,d)))
                cd = tuple(sorted((c,d)))

                EdgeTable[bc] = tuple(e for e in EdgeTable[bc] if e != elem) + new_elems
                EdgeTable[ab] = tuple(e for e in EdgeTable[ab] if e != elem) + (new_elems[0],)
                EdgeTable[ac] = tuple(e for e in EdgeTable[ac] if e != elem) + (new_elems[0],)
                EdgeTable[bd] = tuple(e for e in EdgeTable[bd] if e != elem) + (new_elems[1],)
                EdgeTable[cd] = tuple(e for e in EdgeTable[cd] if e != elem) + (new_elems[1],)
            elif lookup_key == 6:
                new_elems = ((a,d,b,N), (a,c,d,N))
                
                # Split Edges - add one new element
                EdgeTable[(b,N)] += (new_elems[0],)
                EdgeTable[(c,N)] += (new_elems[1],)
                
                # Other New Edges - add both new elements
                NewEdges.update({(a,N), (d,N)})
                if (a,N) in EdgeTable.keys():
                    EdgeTable[(a,N)] += new_elems
                else:
                    EdgeTable[(a,N)] = new_elems
                if (d,N) in EdgeTable.keys():
                    EdgeTable[(d,N)] += new_elems
                else:
                    EdgeTable[(d,N)] = new_elems
                
                # Existing Edges - remove old element and add new element(s)
                ab = tuple(sorted((a,b)))
                ac = tuple(sorted((a,c)))
                ad = tuple(sorted((a,d)))
                bd = tuple(sorted((b,d)))
                cd = tuple(sorted((c,d)))

                EdgeTable[ad] = tuple(e for e in EdgeTable[ad] if e != elem) + new_elems
                EdgeTable[ab] = tuple(e for e in EdgeTable[ab] if e != elem) + (new_elems[0],)
                EdgeTable[bd] = tuple(e for e in EdgeTable[bd] if e != elem) + (new_elems[0],)
                EdgeTable[ac] = tuple(e for e in EdgeTable[ac] if e != elem) + (new_elems[1],)
                EdgeTable[cd] = tuple(e for e in EdgeTable[cd] if e != elem) + (new_elems[1],)
            elif lookup_key == 10:
                new_elems = ((a,d,b,N), (b,d,c,N))

                # Split Edges - add one new element
                EdgeTable[(a,N)] += (new_elems[0],)
                EdgeTable[(c,N)] += (new_elems[1],)
                
                # Other New Edges - add both new elements
                NewEdges.update({(b,N), (d,N)})
                if (b,N) in EdgeTable.keys():
                    EdgeTable[(b,N)] += new_elems
                else:
                    EdgeTable[(b,N)] = new_elems
                if (d,N) in EdgeTable.keys():
                    EdgeTable[(d,N)] += new_elems
                else:
                    EdgeTable[(d,N)] = new_elems
                
                # Existing Edges - remove old element and add new element(s)
                ab = tuple(sorted((a,b)))
                ad = tuple(sorted((a,d)))
                bc = tuple(sorted((b,c)))
                bd = tuple(sorted((b,d)))
                cd = tuple(sorted((c,d)))

                EdgeTable[bd] = tuple(e for e in EdgeTable[bd] if e != elem) + new_elems
                EdgeTable[ab] = tuple(e for e in EdgeTable[ab] if e != elem) + (new_elems[0],)
                EdgeTable[ad] = tuple(e for e in EdgeTable[ad] if e != elem) + (new_elems[0],)
                EdgeTable[bc] = tuple(e for e in EdgeTable[bc] if e != elem) + (new_elems[1],)
                EdgeTable[cd] = tuple(e for e in EdgeTable[cd] if e != elem) + (new_elems[1],)
            elif lookup_key == 12:
                new_elems = ((a,c,d,N), (b,d,c,N))
                # Split Edges - add one new element
                EdgeTable[(a,N)] += (new_elems[0],)
                EdgeTable[(b,N)] += (new_elems[1],)
                
                # Other New Edges - add both new elements
                NewEdges.update({(c,N), (d,N)})
                if (c,N) in EdgeTable.keys():
                    EdgeTable[(c,N)] += new_elems
                else:
                    EdgeTable[(c,N)] = new_elems
                if (d,N) in EdgeTable.keys():
                    EdgeTable[(d,N)] += new_elems
                else:
                    EdgeTable[(d,N)] = new_elems
                
                # Existing Edges - remove old element and add new element(s)
                ac = tuple(sorted((a,c)))
                ad = tuple(sorted((a,d)))
                bc = tuple(sorted((b,c)))
                bd = tuple(sorted((b,d)))
                cd = tuple(sorted((c,d)))

                EdgeTable[cd] = tuple(e for e in EdgeTable[cd] if e != elem) + new_elems
                EdgeTable[ac] = tuple(e for e in EdgeTable[ac] if e != elem) + (new_elems[0],)
                EdgeTable[ad] = tuple(e for e in EdgeTable[ad] if e != elem) + (new_elems[0],)
                EdgeTable[bc] = tuple(e for e in EdgeTable[bc] if e != elem) + (new_elems[1],)
                EdgeTable[bd] = tuple(e for e in EdgeTable[bd] if e != elem) + (new_elems[1],)
        
            # Update labels - new elements get the label of their parent element
            if labels is not None:
                for e in new_elems:
                    ElemLabels[e] = ElemLabels[elem]
        # Define new node coordinates
        NodeTable[N] = (NodeTable[edge[0]] + NodeTable[edge[1]])/2
        N += 1

        # Add new edges to heap if they're above the threshold
        # Note lengths are negative so longest edges get sorted first in the heap
        if (sizing is None and L/2 < -h) or (sizing is not None and L/2 < -(node_cutoff[edge[0]]+node_cutoff[edge[1]])/2):
            # Split edges
            heapq.heappush(heap,(L/2, edge1))
            heapq.heappush(heap,(L/2, edge2))
        for e in NewEdges:
            # other new edges
            inmesh[e] = True
            L = - np.linalg.norm(NodeTable[e[0]]-NodeTable[e[1]])
            if (sizing is None and L < -h) or (sizing is not None and L < - (node_cutoff[e[0]]+node_cutoff[e[1]])/2):
                heapq.heappush(heap,(L, e))
        return N
    
    if verbose:
        print('TetSplit:', end='')
        if 'tqdm' in sys.modules:
            tqdm_loaded = True
            progress = tqdm.tqdm(total=len(heap))
        else:
            tqdm_loaded = False
    
    while len(heap) > 0:
        
        L, edge = heapq.heappop(heap)
        if verbose and tqdm_loaded:
            progress.update(1)
            L1 = len(heap)
        N = do_split(L, edge, N)
        if verbose and tqdm_loaded:
            L2 = len(heap)
            progress.total += max(0, L2-L1)

    NewConn = np.array(list(set(elem for edge in EdgeTable.keys() for elem in EdgeTable[edge] if inmesh[edge])))
    NewCoords = np.array([NodeTable[key] for key in NodeTable.keys()])

    
    if 'mesh' in dir(mesh):
        Mnew = mesh.mesh(NewCoords, NewConn, verbose=M.verbose)
    else:
        Mnew = mesh(NewCoords, NewConn, verbose=M.verbose)
    if labels is not None:
        new_labels = np.array([ElemLabels[tuple(elem)] for elem in NewConn])
        Mnew.ElemData[label_str] = new_labels
    
    return Mnew

def TetFlip(M, iterate='converge', QualityMetric='Skewness', target='min', flips=['4-4','3-2','2-3'], verbose=False):

    NodeCoords = M.NodeCoords
    NodeConn = M.NodeConn
    Faces = M.Faces
    FaceConn = M.FaceConn
    FaceElemConn = M.FaceElemConn
    Edges = M.Edges
    EdgeConn = M.EdgeConn
    EdgeElemConn = M.EdgeElemConn

    if QualityMetric == 'Skewness':
        qualfunc = lambda NodeCoords, NodeConn, V : 1 - quality.tet_vol_skewness(NodeCoords,NodeConn, V)

    # Sort and prep hash table ids/dictionary keys
    SortElem = [tuple(elem) for elem in np.sort(NodeConn, axis=1).tolist()]
    SortFace = [tuple(face) for face in np.sort(Faces, axis=1).tolist()]
    SortFaceNormals = utils.CalcFaceNormal(NodeCoords, SortFace)
    SortEdge = [tuple(edge) for edge in np.sort(Edges, axis=1).tolist()]

    volume = quality.Volume(NodeCoords, NodeConn, ElemType='tet')
    qual = qualfunc(NodeCoords, NodeConn, volume)

    # Construct element, face, and edge tables
    ElemTable = {SortElem[i] : {'status'  : True, # Status indicates whether this element is currently in the mesh
                                'elem'    : elem, # elem gives the properly oriented element connectivity (may not be necessary)
                                'volume'  : volume[i], # Element volume, helps ensure flips are valid
                                'quality' : qual[i],
                                'faces'   : tuple([SortFace[j] for j in FaceConn[i]]), # faces gives the dict keys to the face table
                                'edges'   : tuple([SortEdge[j] for j in EdgeConn[i]])  # edges gives the dict keys to the edge table
                                } for i,elem in enumerate(NodeConn)}
    FaceTable = {SortFace[i] : {'elems'   : tuple([SortElem[j] for j in FaceElemConn[i] if not np.isnan(j)]),
                                'normal'  : SortFaceNormals[i]
                                } for i,face in enumerate(Faces)}
    EdgeTable = {SortEdge[i] : {'elems'   : tuple([SortElem[j] for j in EdgeElemConn[i]])} for i,edge in enumerate(Edges)}

    n44 = 0
    n32 = 0
    n23 = 0
    # Visit all tets
    ElemTableKeys = list(ElemTable.keys())

    if iterate == 'converge':
        condition = lambda i, n44, n32, n23 : ((n44 + n32 + n23) > 0) | (i == 0)
    else:
        condition = lambda i, n44, n32, n23 : i < iterate
    i = 0
    while condition(i, n44, n32, n23):
        i += 1
        n44 = 0; n32 = 0; n23 = 0
        for key in ElemTableKeys:
            if not ElemTable[key]['status']:
                # skip if the element isn't active in the mesh
                continue
                
            keyset = set(key)
            ### Attempt edge removal ###
            # 4-4 flip
            if '4-4' in flips:
                success = _Tet44Flip(key, NodeCoords, ElemTable, FaceTable, EdgeTable, qualfunc,target=target)
                if success: 
                    n44 += 1
                    continue

            # 3-2 flip
            if '3-2' in flips:
                success = _Tet32Flip(key, NodeCoords, ElemTable, FaceTable, EdgeTable, qualfunc, target=target)
                if success: 
                    n32 += 1
                    continue
                    
            ###########################

            ## Attempt face removal ###
            # 2-3 flip
            if '2-3' in flips:
                success = _Tet23Flip(key, NodeCoords, ElemTable, FaceTable, EdgeTable, qualfunc, target=target)
                if success: 
                    n23 += 1
                    continue
            
            ############################

        if verbose: 
            if '3-2' in flips: print(f'3-2 Flips: {n32:d}')
            if '2-3' in flips: print(f'2-3 Flips: {n23:d}')
            if '4-4' in flips: print(f'4-4 Flips: {n44:d}')
    # Extract updated mesh

    NewConn = [ElemTable[key]['elem'] for key in ElemTable.keys() if ElemTable[key]['status']]

    if 'mesh' in dir(mesh):
        tet = mesh.mesh(NodeCoords, NewConn)
    else:
        tet = mesh(NodeCoords, NewConn)
    return tet

def TetImprove(M, h, schedule='scfS', repeat=1, labels=None, smoother='SmartLaplacianSmoothing', smooth_kwargs={}, verbose=True, FeatureAngle=25, ContractIter=5):
    """
    Tetrahedral mesh quality improvement using multiple local operations. 

    Parameters
    ----------
    M : mymesh.mesh
        Tetrahedral mesh to be improved
    h : float
        Target element size/edge length
    schedule : str, optional
        Order of operations to perform, specified as a string with each 
        character indicating an operation, by default 'scfS'.

        Possible operations:
            's' - Splitting (:func:`TetSplit`)
            'c' - Contraction (:func:`TetContract`)
            'f' - Flipping (:func:`TetFlip`)
            'S' - Smoothing 

    repeat : int, optional
        Number of times to repeat the schedule, by default 1
    labels : str, array_like, or NoneType, optional
        Element labels indicating different regions. If specified,
        region interfaces will be preserved. This can be specified as 
        an array_like with M.NElem entries or a string corresponding to
        an entry in M.ElemData, by default None.
    smoother : str, optional
        Specify which smoothing operation to use, by default 'SmartLaplacianSmoothing'
    smooth_kwargs : dict, optional
        Key word arguments to be passed to the smoother, by default {}
    verbose : bool, optional
        If True, will display progress, by default True
    FeatureAngle : int, optional
        FeatureAngle : int, optional
        Angle (in degrees) used to identify features, by default 25. See
        :func:`~mymesh.utils.DetectFeatures` for more information., by default 25
    ContractIter : int, optional
        Maximum number of iterations to perform in the contraction step, by default 5

    Returns
    -------
    _type_
        _description_
    """    
    M.verbose=False
    for loop in range(repeat):
        for operation in schedule:

            if operation == 's':
                # Split
                M = TetSplit(M, 4/3*h, verbose=verbose, labels=labels, sizing=None, QualitySizing=False)
                M.verbose=False
            elif operation == 'c':
                # Contract
                M = TetContract(M, 4/5*h, verbose=verbose, labels=labels, FeatureAngle=FeatureAngle, maxIter=ContractIter)
                M.verbose=False
            # elif operation == 'f':
            #     # Flip
            #     M = TetFlip(M, flips=['3-2','2-3'], verbose=verbose)
            #     M.verbose=False
            elif operation == 'S':
                if smoother == 'SmartLaplacianSmoothing':
                    M = SmartLaplacianSmoothing(M, TangentialSurface=True, labels=labels, options=dict(FixFeatures=True))
                    M.verbose=False
    return M

## Need to be updated or removed
def CollapseSlivers(NodeCoords, NodeConn, skewThreshold=0.9, FixedNodes=set(), verbose=False):
    """
    Collapse sliver elements above a skewnessthreshold (default 0.9)
 
    Parameters
    ----------
    NodeCoords : array_like
        Node coordinates
    NodeConn : list, array_like
        Nodal connectivites
    skewThreshold : float, optional
        Skewness threshold to determine whether an element is a sliver,
        by default 0.9
    FixedNodes : set, optional
        Set of nodes to be held in place and not affected by
        the sliver collapse operation, by default set().
    verbose : bool, optional
        If True, will report , by default False

    Returns
    -------
    NewCoords : array_like
        New node coordinate array
    NewConn : array_like
        New node connectivity array
    """    
    if type(FixedNodes) is list: FixedNodes = set(FixedNodes)
    ArrayCoords = np.asarray(NodeCoords)
    skew = quality.Skewness(ArrayCoords,NodeConn,verbose=False)
    Slivers = np.where(skew>skewThreshold)[0]
    if len(Slivers) > 0:

        NewConn = copy.copy(NodeConn)
        RNodeConn = utils.PadRagged(NodeConn)
        
        Edges, EdgeConn, EdgeElem = converter.solid2edges(ArrayCoords,NodeConn,return_EdgeConn=True,return_EdgeElem=True)
        Edges = np.asarray(Edges)
        REdgeConn = utils.PadRagged(EdgeConn)
        
        Pt1 = ArrayCoords[Edges[:,0]]; Pt2 = ArrayCoords[Edges[:,1]]
        EdgeLen = np.append(np.linalg.norm(Pt1-Pt2,axis=1),np.infty)
        ElemEdgeLen = EdgeLen[REdgeConn]
        EdgeSort = np.argsort(ElemEdgeLen,axis=1)

        for sliver in Slivers:
            for edge in REdgeConn[sliver][EdgeSort[sliver]]:
                check = [Edges[edge][0] in FixedNodes, Edges[edge][1] in FixedNodes]
                if check[0] and check[1]:
                    continue
                elif check[1]:
                    RNodeConn[RNodeConn == Edges[edge][0]] = Edges[edge][1]
                elif check[0]:
                    RNodeConn[RNodeConn == Edges[edge][1]] = Edges[edge][0]
                else:
                    ArrayCoords[Edges[edge][0]] = (ArrayCoords[Edges[edge][0]]+ArrayCoords[Edges[edge][1]])/2
                    RNodeConn[RNodeConn == Edges[edge][0]] = Edges[edge][1]
                break
        NewCoords,NewConn = utils.DeleteDegenerateElements(ArrayCoords,NewConn,strict=True)
        NewCoords = NewCoords
    else:
        NewCoords = NodeCoords
        NewConn = NodeConn
    return NewCoords, NewConn

def SliverPeel(NodeCoords, NodeConn, skewThreshold=0.95, FixedNodes=set()):
    """
    SliverPeel Peel highly skewed surface slivers off of a tetrahedral mesh.
    Only slivers above the skewThreshold with two surface faces will be peeled.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.
    skewThreshold : float, optional
        Skewness threshold above which slivers will be peeled from the surface, by default 0.95.

    Returns
    -------
    NewConn : list
        New nodal connectivity list.
    """
    skew = quality.Skewness(NodeCoords,NodeConn)
    Faces,FaceConn,FaceElem = converter.solid2faces(NodeCoords,NodeConn,return_FaceConn=True,return_FaceElem=True)
    FaceElemConn,UFaces, UFaceConn, UFaceElem, idx, inv = converter.faces2faceelemconn(Faces,FaceConn,FaceElem,return_UniqueFaceInfo=True)
    UFaceConn = utils.PadRagged(UFaceConn)
    ElemNormals = np.array(utils.CalcFaceNormal(NodeCoords,Faces))

    where = np.where(np.isnan(FaceElemConn))
    SurfElems = np.asarray(FaceElemConn)[where[0],1-where[1]].astype(int)
    SurfConn = converter.solid2surface(NodeCoords,NodeConn)
    SurfNodes = set([n for e in SurfConn for n in e])

    ElemIds = np.arange(len(NodeConn))
    Skew01 = skew>skewThreshold         # Boolean check for if element skewness is greater than threshold
    # Surf01 = np.in1d(ElemIds,SurfElems) # Check if element is on the surface
    Surf01 = np.array([all([n in SurfNodes for n in elem]) for i,elem in enumerate(NodeConn)])
    X = np.isnan(np.append(FaceElemConn,[[np.inf,np.inf]],axis=0)[UFaceConn])
    Face01 = np.sum(X,axis=(1,2))==2 # Check if elem has 2 faces on surface
    # For elems with two surface faces, check if their normals are pointing the same direction
    SurfFacePairs = utils.PadRagged(FaceConn)[Face01][np.any(X,axis=2)[Face01]].reshape((sum(Face01),2))
    dots = np.sum(ElemNormals[SurfFacePairs[:,0]]*ElemNormals[SurfFacePairs[:,1]],axis=1)
    Face01[Face01] = dots > np.sqrt(3)/2
    Feature01 = np.array([sum([n in FixedNodes for n in elem]) < 2 for i,elem in enumerate(NodeConn)])


    NewConn = np.array(NodeConn,dtype=object)[~(Skew01 & Face01 & Feature01)].tolist()
    return NewConn

    
def ResolveSurfSelfIntersections(NodeCoords,SurfConn,FixedNodes=set(),octree='generate',maxIter=10):

    if type(FixedNodes) != set:
        FixedNodes = set(FixedNodes)

    NewCoords = np.array(NodeCoords)
    SurfConn = np.asarray(SurfConn)
    if octree == 'generate':
        octree = octree.Surface2Octree(NewCoords,SurfConn)
    IntersectionPairs = rays.SurfSelfIntersection(NewCoords,SurfConn,octree=octree)
    Intersected = np.unique(IntersectionPairs).tolist()
    
    count = 0
    while len(Intersected) > 0 and count < maxIter:
        print(count)
        ElemConn = utils.getElemConnectivity(NewCoords, SurfConn)
        NeighborhoodElems = np.unique([e for i in (SurfConn[Intersected]).flatten() for e in ElemConn[i]])
        PatchConn = SurfConn[NeighborhoodElems]
        BoundaryEdges = converter.surf2edges(NewCoords,PatchConn) 

        NewCoords = np.asarray(LocalLaplacianSmoothing(NewCoords,PatchConn,2,FixedNodes=FixedNodes.union(set([n for edge in BoundaryEdges for n in edge]))))

        IntersectionPairs = rays.SurfSelfIntersection(NewCoords,SurfConn,octree=octree)
        Intersected = np.unique(IntersectionPairs).tolist()
        count += 1
        if len(Intersected) > 0 and count > maxIter:
            warnings.warn('Unable to resolve surface mesh self intersections.')
    return NewCoords

def FlipEdge(NodeCoords,NodeConn,i,j):
    Si = set(NodeConn[i])
    Sj = set(NodeConn[j])
    shared = list(Si.intersection(Sj))
    assert len(shared)==2, 'Elements {:d} & {:d} are not properly connected for an edge flip.'.format(i,j)
    NotIni = Sj.difference(Si).pop()
    NotInj = Si.difference(Sj).pop()
    
    Newi = [n if n != shared[0] else NotIni for n in NodeConn[i]]
    Newj = [n if n != shared[1] else NotInj for n in NodeConn[j]]
    # SetConn = [set(elem) for elem in NewConn]
    # if set(Newi) not in SetConn and set(Newj) not in SetConn:
    return Newi,Newj

def ValenceImprovementFlips(NodeCoords,NodeConn,NodeNeighbors,ElemNeighbors):

    Array = np.array(NodeCoords)
    NewConn = copy.copy(NodeConn)
    lens = np.array([len(n) for n in NodeNeighbors])
    R = -1*np.ones([len(NodeNeighbors),2*max(lens)],dtype=int)
    for i,n in enumerate(NodeNeighbors): R[i,:lens[i]] = n

    # Contains the number of node neighbors for each node stored at the nodes location within NodeConn
    TriValences = np.sum(R[np.array(NewConn)]>=0,axis=2)
    flips = 1
    while flips>0:
        flips = 0
        for i in range(len(ElemNeighbors)):
            if any(TriValences[i] < 5) or any(TriValences[i] > 7):
                for j in ElemNeighbors[i]:
                    if len(set(ElemNeighbors[i]).intersection(ElemNeighbors[j])) > 0:
                        # This condition checks if the flip will be legal
                        continue
                    shared = list(set(NewConn[i]).intersection(NewConn[j]))
                    TempConn = copy.copy(NewConn)
                    Newi,Newj = FlipEdge(NodeCoords,NewConn,i,j)
                    if np.all(np.round(np.cross(Array[Newj[0]]-Array[Newj[1]],Array[Newj[0]]-Array[Newj[2]]),15) == [0,0,0]) or np.all(
                        np.round(np.cross(Array[Newi[0]]-Array[Newi[1]],Array[Newi[0]]-Array[Newi[2]]),15) == [0,0,0]):
                        # This condition checks if the flip leads to degenerate triangles
                        continue


                    TempConn[i] = Newi
                    TempConn[j] = Newj
                    Newshared = list(set(Newi).intersection(Newj))

                    if not rays.SegmentSegmentIntersection(Array[shared],Array[Newshared]):
                        # This condition checks if the flipped segment intersects with the old segment.
                        # Flips that don't pass this check could cause element inversion
                        continue

                    tempNeighbors = [
                        copy.copy(NodeNeighbors[shared[0]]),
                        copy.copy(NodeNeighbors[shared[1]]),
                        copy.copy(NodeNeighbors[Newshared[0]]),
                        copy.copy(NodeNeighbors[Newshared[1]]),
                    ]
                    tempNeighbors[0].remove(shared[1])
                    tempNeighbors[1].remove(shared[0])
                    tempNeighbors[2].append(Newshared[1])
                    tempNeighbors[3].append(Newshared[0])

                    ## TODO: This could be done more efficiently without copying all of R
                    newR = np.copy(R)
                    newR[[shared[0],shared[1],Newshared[0],Newshared[1]]] = -1*np.ones((4,R.shape[1]))
                    newR[shared[0],:len(tempNeighbors[0])] = tempNeighbors[0]
                    newR[shared[1],:len(tempNeighbors[1])] = tempNeighbors[1]
                    newR[Newshared[0],:len(tempNeighbors[2])] = tempNeighbors[2]
                    newR[Newshared[1],:len(tempNeighbors[3])] = tempNeighbors[3]
                    NewTriValences = np.sum(newR[np.array(TempConn)]>=0,axis=2)
                    if (np.max([TriValences[i],TriValences[j]]) - np.min([TriValences[i],TriValences[j]])) > (
                        np.max([NewTriValences[i],NewTriValences[j]]) - np.min([NewTriValences[i],NewTriValences[j]])):
                        flips += 1
                        NewConn = TempConn
                        R = newR
                        TriValences = NewTriValences
                        NodeNeighbors[shared[0]] = tempNeighbors[0]
                        NodeNeighbors[shared[1]] = tempNeighbors[1]
                        NodeNeighbors[Newshared[0]] = tempNeighbors[2]
                        NodeNeighbors[Newshared[1]] = tempNeighbors[3]
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
                        break
        # print(flips)


    return NewConn, ElemNeighbors

def AngleReductionFlips(NodeCoords,NodeConn,NodeNeighbors=None,FixedNodes=[]):
    NewCoords = np.array(NodeCoords)
    NewConn = copy.copy(NodeConn)
    if not NodeNeighbors: NodeNeighbors = utils.getNodeNeighbors(NodeCoords,NodeConn)
    thinking = True
    iter = 0
    while thinking:
        iter += 1
        flips = 0
        
        Edges, EdgeConn, EdgeElem = converter.solid2edges(NewCoords,NewConn,return_EdgeConn=True,return_EdgeElem=True)
        UEdges, UIdx, UInv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
        UEdgeElem = np.asarray(EdgeElem)[UIdx]
        UEdgeConn = UInv[utils.PadRagged(EdgeConn)]
        EECidx = (UEdgeElem[UEdgeConn] == np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)).astype(int)
        EdgeElemConn = -1*(np.ones((len(UEdges),2),dtype=int))
        EdgeElemConn[UEdgeConn,EECidx] = np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)
    
        NewCoords = np.asarray(NewCoords)
        NewConn = np.asarray(NewConn)
        EdgeVectors = NewCoords[UEdges[:,1]] - NewCoords[UEdges[:,0]]
        EdgeLengths = np.linalg.norm(EdgeVectors,axis=1)

        ElemVectors = EdgeVectors[UEdgeConn]
        ElemLengths = EdgeLengths[UEdgeConn]

        OppositeAngles = np.zeros(ElemLengths.shape)
        OppositeAngles[:,0] = np.arccos(np.sum(ElemVectors[:,2]*-ElemVectors[:,1],axis=1)/(ElemLengths[:,1]*ElemLengths[:,2]))
        OppositeAngles[:,1] = np.arccos(np.sum(ElemVectors[:,0]*-ElemVectors[:,2],axis=1)/(ElemLengths[:,0]*ElemLengths[:,2]))
        OppositeAngles[:,2] = np.arccos(np.sum(ElemVectors[:,1]*-ElemVectors[:,0],axis=1)/(ElemLengths[:,1]*ElemLengths[:,0]))
        
        EdgeOppositeAngles = np.empty((len(UEdges),2))
        EdgeOppositeAngles[UEdgeConn,EECidx] = OppositeAngles

        NonDelaunay = np.where(np.sum(EdgeOppositeAngles,axis=1) > np.pi)[0]    # Indices of edges between two triangles that don't pass the 2D delaunay criteria (alpha+gamma<180)
        EffectedElems = set()
        nskipped = 0
        for k in NonDelaunay:  
            
            [i,j] = EdgeElemConn[k]
            Edge = UEdges[k]

            if i in EffectedElems or j in EffectedElems:
                continue
            elif Edge[0] in FixedNodes and Edge[1] in FixedNodes:
                continue

            Newi,Newj = FlipEdge(NodeCoords,NewConn,i,j)
            NewEdge = list(set(Newi).intersection(Newj))

            if np.any(np.all(NewEdge==UEdges,axis=1)) or np.any(np.all(NewEdge[::-1]==UEdges,axis=1)):
                # Edge already exists
                nskipped += 1
                continue

            if np.all(np.round(np.cross(NewCoords[Newj[0]]-NewCoords[Newj[1]],NewCoords[Newj[0]]-NewCoords[Newj[2]]),15) == [0,0,0]) or np.all(
                np.round(np.cross(NewCoords[Newi[0]]-NewCoords[Newi[1]],NewCoords[Newi[0]]-NewCoords[Newi[2]]),15) == [0,0,0]):
                # This condition checks if the flip leads to degenerate triangles
                a = 1
                nskipped += 1
                continue            

            if not rays.SegmentSegmentIntersection(NewCoords[Edge],NewCoords[NewEdge]):
                # This condition checks if the flipped segment intersects with the old segment.
                # Flips that don't pass this check could cause element inversion
                a = 2
                nskipped += 1
                continue

            if len(NodeNeighbors[UEdges[k,0]]) == 3 or len(NodeNeighbors[UEdges[k,1]]) == 3:
                # If one of the edge nodes involved has only 3 neighbors, it can cause a topological error
                nskipped += 1
                continue
            
            NewConn[i] = Newi; NewConn[j] = Newj
            EffectedElems.update((i,j))
            
            Edges[k] = NewEdge

            flips += 1
        if flips < len(NonDelaunay)-nskipped:
            thinking = True
            NodeNeighbors = utils.getNodeNeighbors(NewCoords, NewConn)    
        else:
            thinking = False

    return NewConn

def Split(NodeCoords, NodeConn, h, iterate='converge', criteria=['AbsLongness','RelLongness'],thetal=145):
    # Clark, B., Ray, N., & Jiao, X. (2013). Surface mesh optimization, adaption, and untangling with high-order accuracy. Proceedings of the 21st International Meshing Roundtable, IMR 2012, 385–402. https://doi.org/10.1007/978-3-642-33573-0_23
    # Jiao, X., Colombi, A., Ni, X., & Hart, J. (2010). Anisotropic mesh adaptation for evolving triangulated surfaces. Engineering with Computers, 26(4), 363–376. https://doi.org/10.1007/s00366-009-0170-1
    
    # criteria:
        # AbsLongness
        # RelLongness
        # AbsLargeAngle
    
    
    ### Parameters ###
    l = h       # Desired Edge Length
    phil = 0.005
    phiu = 0.07
    r = 0.25    # Small fraction of longest edge in incident triangles (Contraction - Relative shortness)
    R = 0.45    # Fraction of the longest edge in the mesh (Contraction - Relative small triangle)
    rho = np.sqrt(phil/phiu)  
    s = l*rho   # Minimum shortness threshold (Edge splitting - Relative longness)
    S = R*l     # Maximum shortness threshold (Contraction - Absolute small triangle)
    L = 1.5*l   # Maximum longness threshold (Edge splitting - Absolute longness)
    thetas = np.pi/12 # (15 deg) Minimum angle threshold (Contraction - Absolute small angle)
    thetal = thetal*np.pi/180 # (145 deg) Maximum angle threshold (Edge Splitting - Relative longness)
    ### ########## ###
    if type(criteria) is str: criteria = [criteria]
    def do_split(NewCoords,NewConn,EdgeSort,ConnSort,i):
        center = np.mean([NewCoords[EdgeSort[i,0]],NewCoords[EdgeSort[i,1]]],axis=0)
        NewId = len(NewCoords)
        NewCoords = np.vstack([NewCoords,center])
        # Get the node not belonging to the edge
        elem0 = NewConn[ConnSort[i,0]]
        elem1 = NewConn[ConnSort[i,1]]
        if type(elem0[0]) is list or type(elem1[0]) is list:
            return NewCoords, NewConn
        diff0 = set(elem0).difference(EdgeSort[i])
        diff1 = set(elem1).difference(EdgeSort[i])
        if len(diff0) == 0 or len(diff1) == 0:
            return NewCoords, NewConn
        NotShared0 = diff0.pop()
        NotShared1 = diff1.pop()
        # cycle the element definition so that it starts with the non-shared node
        #TODO: Probable Bottleneck
        while elem0[0] != NotShared0: elem0 = [elem0[-1]]+elem0[0:-1]
        while elem1[0] != NotShared1: elem1 = [elem1[-1]]+elem1[0:-1]
        
        if ConnSort[i,0] >= 0: NewConn[ConnSort[i,0]] = [[elem0[0],elem0[1],NewId],[elem0[0],NewId,elem0[2]]]
        if ConnSort[i,1] >= 0: NewConn[ConnSort[i,1]] = [[elem1[0],elem1[1],NewId],[elem1[0],NewId,elem1[2]]]
        return NewCoords, NewConn

    NewCoords = np.array(NodeCoords)
    NewConn = copy.copy(NodeConn)
    if type(NewConn) is np.ndarray: NewConn = NewConn.tolist()

    if iterate == 'converge': iterate = np.inf

    iter = 0
    while iter < iterate:
        # print(iter)
        iter += 1
        Edges, EdgeConn, EdgeElem = converter.solid2edges(NewCoords,NewConn,return_EdgeConn=True,return_EdgeElem=True)
        UEdges, UIdx, UInv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
        UEdgeElem = np.asarray(EdgeElem)[UIdx]
        UEdgeConn = UInv[utils.PadRagged(EdgeConn)]
        EECidx = (UEdgeElem[UEdgeConn] == np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)).astype(int)
        EdgeElemConn = -1*(np.ones((len(UEdges),2),dtype=int))
        EdgeElemConn[UEdgeConn,EECidx] = np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)

        # EdgeVectors = NewCoords[UEdges[:,1]] - NewCoords[UEdges[:,0]]
        # EdgeLengths = np.linalg.norm(EdgeVectors,axis=1)

        # ElemVectors = EdgeVectors[UEdgeConn]
        # ElemLengths = EdgeLengths[UEdgeConn]

        # OppositeAngles = -1*np.ones(ElemLengths.shape)
        # # sign = (EECidx-.5)*2
        # OppositeAngles[:,0] = np.clip(np.sum(ElemVectors[:,2]*-ElemVectors[:,1],axis=1)/(ElemLengths[:,1]*ElemLengths[:,2]),-1,1)
        # OppositeAngles[:,1] = np.clip(np.sum(ElemVectors[:,0]*-ElemVectors[:,2],axis=1)/(ElemLengths[:,0]*ElemLengths[:,2]),-1,1)
        # OppositeAngles[:,2] = np.clip(np.sum(ElemVectors[:,1]*-ElemVectors[:,0],axis=1)/(ElemLengths[:,1]*ElemLengths[:,0]),-1,1)
        # OppositeAngles = np.arccos(OppositeAngles)


        ###
        Edges = np.asarray(Edges); EdgeConn = np.asarray(EdgeConn)
        EdgeVectors = NewCoords[Edges[:,1]] - NewCoords[Edges[:,0]]
        EdgeLengths = np.linalg.norm(EdgeVectors,axis=1)

        ElemVectors = EdgeVectors[EdgeConn]
        ElemLengths = EdgeLengths[EdgeConn]

        OppositeAngles = -1*np.ones(ElemLengths.shape)
        # sign = (EECidx-.5)*2
        with np.errstate(divide='ignore', invalid='ignore'):
            OppositeAngles[:,0] = np.clip(np.sum(ElemVectors[:,2]*-ElemVectors[:,1],axis=1)/(ElemLengths[:,1]*ElemLengths[:,2]),-1,1)
            OppositeAngles[:,1] = np.clip(np.sum(ElemVectors[:,0]*-ElemVectors[:,2],axis=1)/(ElemLengths[:,0]*ElemLengths[:,2]),-1,1)
            OppositeAngles[:,2] = np.clip(np.sum(ElemVectors[:,1]*-ElemVectors[:,0],axis=1)/(ElemLengths[:,1]*ElemLengths[:,0]),-1,1)
            OppositeAngles = np.arccos(OppositeAngles)

        ###
        EdgeOppositeAngles =  -1*np.ones((len(UEdges),2))
        EdgeOppositeAngles[UEdgeConn,EECidx] = OppositeAngles

        sortkey = np.argsort(EdgeLengths[UIdx])[::-1]
        LengthSort = EdgeLengths[UIdx][sortkey]
        AngleSort = EdgeOppositeAngles[sortkey]
        EdgeSort = np.asarray(UEdges)[sortkey]
        ConnSort = np.array(EdgeElemConn)[sortkey]

        todobool = np.repeat(False,len(sortkey))

        if 'RelLongness' in criteria: 
            RelLongness = (l < LengthSort) & np.any(AngleSort > thetal,axis=1) & (np.min(np.hstack([ElemLengths[ConnSort[:,0]],ElemLengths[ConnSort[:,1]]]),axis=1) >= s)
            todobool = todobool | RelLongness
        if 'AbsLongness' in criteria: 
            AbsLongness = (L < LengthSort) & (LengthSort == np.max(np.hstack([ElemLengths[ConnSort[:,0]],ElemLengths[ConnSort[:,1]]]),axis=1))
            todobool = todobool | AbsLongness
        if 'AbsLargeAngle' in criteria:
            AbsLargeAngle = np.any(AngleSort >= thetal,axis=1)
            todobool = todobool | AbsLargeAngle

        todo = np.where(todobool)[0]
        # Splits
        SplitElems = set()
        count = 0
        for i in todo:
            if ConnSort[i,0] in SplitElems or ConnSort[i,1] in SplitElems:
                continue
            NewCoords,NewConn = do_split(NewCoords,NewConn,EdgeSort,ConnSort,i)
            SplitElems.update(ConnSort[i])
            count += 1
        NewConn = [elem if (type(elem[0]) != list) else elem[0] for elem in NewConn] + [elem[1] for elem in NewConn if (type(elem[0]) == list)]
        if count == 0:
            break
    NewCoords = NewCoords.tolist()

    return NewCoords, NewConn

def Contract(NodeCoords, NodeConn, h, iterate='converge', FixedNodes=set(), FixFeatures=False):
    # Clark, B., Ray, N., & Jiao, X. (2013). Surface mesh optimization, adaption, and untangling with high-order accuracy. Proceedings of the 21st International Meshing Roundtable, IMR 2012, 385–402. https://doi.org/10.1007/978-3-642-33573-0_23
    # Jiao, X., Colombi, A., Ni, X., & Hart, J. (2010). Anisotropic mesh adaptation for evolving triangulated surfaces. Engineering with Computers, 26(4), 363–376. https://doi.org/10.1007/s00366-009-0170-1
    ### Parameters ###
    l = h       # Desired Edge Length
    phil = 0.005
    phiu = 0.07
    r = 0.25    # Small fraction of longest edge in incident triangles (Contraction - Relative shortness)
    R = 0.45    # Fraction of the longest edge in the mesh (Contraction - Relative small triangle)
    rho = np.sqrt(phil/phiu)  
    s = l*rho   # Minimum shortness threshold (Edge splitting - Relative longness)
    S = R*l     # Maximum shortness threshold (Contraction - Absolute small triangle)
    L = 1.5*l   # Maximum longness threshold (Edge splitting - Absolute longness)
    thetas = np.pi/12 # (15 deg) Minimum angle threshold (Contraction - Absolute small angle)
    thetal = 145*np.pi/180 # (145 deg) Maximum angle threshold (Edge Splitting - Relative longness)
    ### ########## ###
    if iterate == 'converge': iterate = np.inf
    
    
    NewCoords = np.array(NodeCoords)
    NewConn = np.array(NodeConn)
    Old_NElem = len(NewConn)
    iter = 0
    degenerate = set()
    edges,corners = utils.DetectFeatures(NewCoords,NewConn)
    if FixFeatures:
        FixedNodes.update(edges)
        FixedNodes.update(corners)
    # FeatureRank = [2 if i in corners else 1 if i in edges else 0 for i in range(len(NewCoords))]
    FeatureRank = np.zeros(len(NewCoords))
    FeatureRank[list(corners)] = 2
    FeatureRank[list(edges)] = 1
    
    while iter < iterate:
        iter += 1
        NodeNeighbors = utils.getNodeNeighbors(NewCoords,NewConn)
        ElemConn = utils.getElemConnectivity(NewCoords,NewConn)
        Edges, EdgeConn, EdgeElem = converter.solid2edges(NewCoords,NewConn,return_EdgeConn=True,return_EdgeElem=True)
        UEdges, UIdx, UInv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
        UEdgeElem = np.asarray(EdgeElem)[UIdx]
        UEdgeConn = UInv[utils.PadRagged(EdgeConn)]
        EECidx = (UEdgeElem[UEdgeConn] == np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)).astype(int)
        EdgeElemConn = -1*(np.ones((len(UEdges),2),dtype=int))
        EdgeElemConn[UEdgeConn,EECidx] = np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)
    
        NewCoords = np.asarray(NewCoords)
        NewConn = np.asarray(NewConn)
        EdgeVectors = NewCoords[UEdges[:,1]] - NewCoords[UEdges[:,0]]
        EdgeLengths = np.linalg.norm(EdgeVectors,axis=1)

        ElemVectors = EdgeVectors[UEdgeConn]
        ElemLengths = EdgeLengths[UEdgeConn]
        with np.errstate(divide='ignore', invalid='ignore'):
            OppositeAngles = np.zeros(ElemLengths.shape)
            OppositeAngles[:,0] = np.arccos(np.clip(np.sum(ElemVectors[:,2]*-ElemVectors[:,1],axis=1)/(ElemLengths[:,1]*ElemLengths[:,2]),-1,1))
            OppositeAngles[:,1] = np.arccos(np.clip(np.sum(ElemVectors[:,0]*-ElemVectors[:,2],axis=1)/(ElemLengths[:,0]*ElemLengths[:,2]),-1,1))
            OppositeAngles[:,2] = np.arccos(np.clip(np.sum(ElemVectors[:,1]*-ElemVectors[:,0],axis=1)/(ElemLengths[:,1]*ElemLengths[:,0]),-1,1))
        
        EdgeOppositeAngles = np.empty((len(UEdges),2))
        EdgeOppositeAngles[UEdgeConn,EECidx] = OppositeAngles

        # # Feature Edges:
        # FeatureEdgeIds = [i for i,edge in enumerate(UEdges) if (edge[0] in edges or edge[0] in corners) and (edge[1] in edges or edge[1] in corners)]
        # maxFeatureEdgeLength = max(EdgeLengths[FeatureEdgeIds])
        # EdgeLengths[FeatureEdgeIds] -= maxFeatureEdgeLength

        sortkey = np.argsort(EdgeLengths)
        LengthSort = EdgeLengths[sortkey]
        AngleSort = EdgeOppositeAngles[sortkey]
        EdgeSort = np.asarray(UEdges)[sortkey]
        ConnSort = np.array(EdgeElemConn)[sortkey]
        MeshMaxLen = LengthSort[-1]

        TriMaxLen = np.max([ElemLengths[ConnSort[:,0]],ElemLengths[ConnSort[:,1]]],axis=2).T
        MaxTriMaxLen = np.max(TriMaxLen,axis=1)
        TriMinLen = np.min([ElemLengths[ConnSort[:,0]],ElemLengths[ConnSort[:,1]]],axis=2).T
        MinTriMinLen = np.min(TriMinLen,axis=1)
        
        AbsSmallAngle = np.any((AngleSort < thetas) & (TriMaxLen < l),axis=1)
        RelShortness = LengthSort < r*MaxTriMaxLen
        AbsSmallTriangle = (LengthSort == MinTriMinLen) & (MaxTriMaxLen < S)
        RelSmallTriangle = (MaxTriMaxLen < R*MeshMaxLen) & (MaxTriMaxLen < l)
        todo = np.where(AbsSmallAngle | RelShortness | AbsSmallTriangle | RelSmallTriangle)[0]
        k=0
        impacted = set()
        for edgeID in todo:
            edge = EdgeSort[edgeID]
            if edge[0] in FixedNodes and edge[1] in FixedNodes:
                continue
            elif edge[0] in impacted or edge[1] in impacted:
                continue
            elif edge[0] in FixedNodes:
                newpoint = NewCoords[edge[0]]
            elif edge[1] in FixedNodes:
                newpoint = NewCoords[edge[1]]
            elif FeatureRank[edge[0]] == FeatureRank[edge[1]]:
                newpoint = np.mean([NewCoords[edge[0]],NewCoords[edge[1]]],axis=0)
            elif FeatureRank[edge[0]] > FeatureRank[edge[1]]:
                newpoint = NewCoords[edge[0]]
            else:
                newpoint = NewCoords[edge[1]]
            OldCoords = [copy.copy(NewCoords[edge[0]]),copy.copy(NewCoords[edge[1]])]
            OldNormals = utils.CalcFaceNormal(NewCoords,[NewConn[e] for e in (ElemConn[edge[0]]+ElemConn[edge[1]])])
            
            NewCoords[edge[0]] = newpoint
            NewCoords[edge[1]] = newpoint
            NewNormals = utils.CalcFaceNormal(NewCoords,[NewConn[e] for e in (ElemConn[edge[0]]+ElemConn[edge[1]])])
            if any([np.dot(OldNormals[i],NewNormals[i]) < np.sqrt(3)/2 for i in range(len(OldNormals)) if not(np.any(np.isnan(NewNormals[i])) or np.any(np.isnan(OldNormals[i])))]):
                NewCoords[edge[0]] = OldCoords[0]
                NewCoords[edge[1]] = OldCoords[1]
            else:
                if FeatureRank[edge[0]] >= FeatureRank[edge[1]]:
                    remove = edge[1]
                    keep = edge[0]
                else:
                    keep = edge[1]
                    remove = edge[0]
                NewConn[NewConn==remove] = keep
                EdgeSort[EdgeSort==remove] = keep
                impacted.update([keep,remove])
                impacted.update(NodeNeighbors[keep])
                impacted.update(NodeNeighbors[remove])
                k+=1
        NewCoords,NewConn = utils.DeleteDegenerateElements(NewCoords,NewConn,strict=True)
        if k == 0:
            break

    NewCoords,NewConn = utils.DeleteDuplicateNodes(NewCoords,NewConn)

    if type(NewCoords) is np.ndarray: NewCoords = NewCoords.tolist()
    return NewCoords, NewConn

def TetSUS(NodeCoords, NodeConn, ElemConn=None, method='BFGS', FreeNodes='inverted', FixedNodes=set(), iterate=1, verbose=True):
    """
    Simultaneous untangling and smoothing for tetrahedral mehses. Optimization-based smoothing for untangling inverted elements.

    Escobar, et al. 2003. “Simultaneous untangling and smoothing of tetrahedral meshes.”

    Parameters
    ----------
    NodeCoords : array_like
        Node coordinates
    NodeConn : array_like
        Node connectivity. This should be mx4 for a purely tetrahedral mesh.
    ElemConn : list, optional
        Option to provide pre-computed element connectivity, (``mesh.ElemConn`` of ``utils.getElemConnectivity()``). If not provided it will be computed, by default None.
    method : str, optional
        Optimization method for ``scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>``_, by default 'BFGS'. 
    FreeNodes : str/set/array_like, optional
        Nodes allowed to move during the optimization. This can be a set, array_like, or a string. If a str, this can be "all" or "inverted" to operate on all nodes or only the nodes connected to inverted elements, by default 'inverted'. Any fixed nodes will be removed from the set of free nodes
    FixedNodes : set/array_like, optional
        Nodes to hold fixed during the optimization. These will be removed from the set of free nodes, by default set().
    iterate : int, optional
        Number of passes over the free nodes in the mesh, by default 1.
    verbose : bool, optional
        If True, will use a tqdm progress bar to indicate the progress of each iteration.

    Returns
    -------
    NewCoords : np.ndarray
        New node coordinates.
    NodeConn : np.ndarray
        Node connectivity, unchanged/passed through from input. 
    """    

    NodeCoords = np.asarray(NodeCoords)
    NewConn = np.asarray(NodeConn)

    if type(FreeNodes) is str:
        if FreeNodes.lower() == 'all': 
            FreeNodes = set(NewConn.flatten())
        elif FreeNodes.lower() == 'inverted':
            V = quality.Volume(NodeCoords, NewConn)
            FreeNodes = set(list(NewConn[V <= 0].flatten()))

    elif type(FreeNodes) is np.ndarray:
        FreeNodes = set(FreeNodes.tolist())

    elif isinstance(FreeNodes, (list, tuple)): 
        FreeNodes = set(FreeNodes)

    FreeNodes = np.array(list(FreeNodes.difference(FixedNodes)),dtype=int)

    if ElemConn is None:
        ElemConn = utils.getElemConnectivity(NodeCoords, NewConn)
    assert np.shape(NewConn) == (len(NewConn), 4), 'Mesh must be purely tetrahedral, with only 4 node elements in NodeConn.'
    
    Winv = np.array([
                [ 1.        , -0.57735027, -0.40824829],
                [ 0.        ,  1.15470054, -0.40824829],
                [ 0.        ,  0.        ,  1.22474487]])

    def func(NodeCoords, NodeConn, nodeid):
        p = 1 # p-norm
        x = NodeCoords[:,0][NodeConn]
        y = NodeCoords[:,1][NodeConn]
        z = NodeCoords[:,2][NodeConn]

        A = np.moveaxis(np.array([
            [x[:,1] - x[:,0], x[:,2] - x[:,0], x[:,3] - x[:,0]],
            [y[:,1] - y[:,0], y[:,2] - y[:,0], y[:,3] - y[:,0]],
            [z[:,1] - z[:,0], z[:,2] - z[:,0], z[:,3] - z[:,0]],
        ]), 2, 0)

        # Jacobian matrix
        S = np.matmul(A, Winv)

        # Frobenius norm
        Snorm = np.linalg.norm(S, axis=(1,2), ord='fro')

        sigma = np.linalg.det(S)

        eps = np.finfo(float).eps
        delta = np.sqrt(eps*(eps - sigma.min())) if sigma.min() < eps else 0
        h = 0.5 * (sigma + np.sqrt(sigma**2 + 4*delta**2))

        a = (NodeConn == nodeid).astype(int)

        zero = np.zeros_like(a[:,0])

        dSdx = np.matmul(np.moveaxis(np.array([
            [a[:,1] - a[:,0], a[:,2] - a[:,0], a[:,3] - a[:,0]],
            [zero, zero, zero],
            [zero, zero, zero]
        ]), 2, 0), Winv)
        dsigmadx = np.linalg.det(dSdx)

        dSdy = np.matmul(np.moveaxis(np.array([
            [zero, zero, zero],
            [a[:,1] - a[:,0], a[:,2] - a[:,0], a[:,3] - a[:,0]],
            [zero, zero, zero]
        ]), 2, 0), Winv)
        dsigmady = np.linalg.det(dSdy)

        dSdz = np.matmul(np.moveaxis(np.array([
            [zero, zero, zero],
            [zero, zero, zero],
            [a[:,1] - a[:,0], a[:,2] - a[:,0], a[:,3] - a[:,0]]
        ]), 2, 0), Winv)
        dsigmadz = np.linalg.det(dSdz)

        Snorm2 = Snorm**2
        eta = Snorm2 / (3 * h**(2/3))
        K = np.linalg.norm(eta, ord=p)

        # deta/dalpha = [deta/dx, deta/dy, deta/dz]
        detadalpha = np.vstack([           
            2*eta*(
                np.trace(np.matmul(dSdx.swapaxes(1,2), S), axis1=1, axis2=2)/Snorm2 - dsigmadx/(3*np.sqrt(sigma**2 + 4*delta**2))
            ),
            2*eta*(
                np.trace(np.matmul(dSdy.swapaxes(1,2), S), axis1=1, axis2=2)/Snorm2 - dsigmady/(3*np.sqrt(sigma**2 + 4*delta**2))
            ),
            2*eta*(
                np.trace(np.matmul(dSdz.swapaxes(1,2), S), axis1=1, axis2=2)/Snorm2 - dsigmadz/(3*np.sqrt(sigma**2 + 4*delta**2))
            )
        ])

        # Chain rule: dK/dalpha = dK/deta * deta/dalpha
        dKdeta = eta * np.abs(eta)**(p-2) / np.linalg.norm(eta, ord=p)**(p-1)
        dKdalpha = np.matmul(dKdeta, detadalpha.T)

        return K, dKdalpha

    def q(NodeCoords,NodeConn):
        x = NodeCoords[:,0][NodeConn]
        y = NodeCoords[:,1][NodeConn]
        z = NodeCoords[:,2][NodeConn]

        A = np.moveaxis(np.array([
            [x[:,1] - x[:,0], x[:,2] - x[:,0], x[:,3] - x[:,0]],
            [y[:,1] - y[:,0], y[:,2] - y[:,0], y[:,3] - y[:,0]],
            [z[:,1] - z[:,0], z[:,2] - z[:,0], z[:,3] - z[:,0]],
        ]), 2, 0)

        # Jacobian matrix
        S = np.matmul(A, Winv)

        # Frobenius norm
        Snorm = np.linalg.norm(S, axis=(1,2), ord='fro')

        sigma = np.linalg.det(S)

        qeta = 3*sigma**(2/3)/Snorm**2

        return qeta

    def obj(x, nodeid):
        
        NewCoords[nodeid] = x
        LocalConn = NewConn[ElemConn[nodeid]]
        f, jac = func(NewCoords, LocalConn, nodeid)
        # print(np.nanmean(q(NewCoords, NewConn)))

        return f, jac
    
    qeta = q(NodeCoords, NodeConn)
    qeta2 = np.append(qeta, np.nan)
    nodeqs = np.nanmean(qeta2[utils.PadRagged(ElemConn, fillval=-1).astype(int)[FreeNodes,:]], axis=1)

    NewCoords = np.copy(NodeCoords)

    nodeids = FreeNodes[nodeqs.argsort()]
    for i in range(iterate):
        if verbose:
            iterable = tqdm.tqdm(nodeids, desc=f'Iteration {i:d}/{iterate:d}:')
        else:
            iterable = nodeids
        for nodeid in iterable:
            # print(nodeid)
            x0 = NewCoords[nodeid]
            out = minimize(obj, x0, jac=True, args=(nodeid), method='L-BFGS-B', options=dict(maxiter=10))
            NewCoords[nodeid] = out.x

    return NewCoords, NodeConn

## Utilities

def _SmoothingInputParser(NodeCoords, NodeConn, SurfConn, SmoothOptions, UserOptions):

    NodeCoords = np.asarray(NodeCoords)
    NodeConn = np.asarray(NodeConn)
    SurfConn = np.asarray(SurfConn)
    for key in UserOptions.keys(): SmoothOptions[key] = UserOptions[key]

    # Process input options
    if type(SmoothOptions['FixedNodes']) is not set: 
            SmoothOptions['FixedNodes'] = set(SmoothOptions['FixedNodes'])
    if SmoothOptions['FixFeatures']:
        edges, corners = utils.DetectFeatures(NodeCoords,SurfConn)
        SmoothOptions['FixedNodes'].update(edges)
        SmoothOptions['FixedNodes'].update(corners)

    if SmoothOptions['FixSurf']:
        SmoothOptions['FixedNodes'].update(SurfConn.flatten())
    idx = set([n for elem in NodeConn for n in elem])
    

    SmoothOptions['FreeNodes'] = np.array(list(set(idx).difference(SmoothOptions['FixedNodes'])),dtype=int)
    SmoothOptions['FixedNodes'] = np.array(list(SmoothOptions['FixedNodes']),dtype=int)

    SmoothOptions['method'] = SmoothOptions['method'].lower()

    return NodeCoords, NodeConn, SmoothOptions
    
def _Tet32Flip(elemkey, NodeCoords, ElemTable, FaceTable, EdgeTable, qualfunc, target='min'):

    success = False

    key = elemkey 
    keyset = set(key)
    for i,edge in enumerate(ElemTable[key]['edges']):
        valid = True
        adjacent_tets = EdgeTable[edge]['elems']
        
        if len(adjacent_tets) != 3:
            # Not flippable
            continue
        
        tet1, tet2 = [tet for tet in adjacent_tets if tet != key]

        pts = keyset.union(tet1).union(tet2)
        if len(pts) != 5:
            # the three tets must form a hull of 5 points
            continue

        current_quality = [ElemTable[key]['quality'], ElemTable[tet1]['quality'], ElemTable[tet2]['quality']]

        a, e = edge # Nodes of shared edge
        b, c, d = pts.difference(edge) # b, c, d should be sorted automatically from the set

        # Outer faces
        face1key = tuple(sorted((a,c,d)))
        face2key = tuple(sorted((c,d,e)))
        face3key = tuple(sorted((a,b,c)))
        face4key = tuple(sorted((b,c,e)))
        face5key = tuple(sorted((a,b,d)))
        face6key = tuple(sorted((b,d,e)))

        # Convexity test
        normals = np.repeat([FaceTable[face1key]['normal'],
                            FaceTable[face2key]['normal'],
                            FaceTable[face3key]['normal'],
                            FaceTable[face4key]['normal'],
                            FaceTable[face5key]['normal'],
                            FaceTable[face6key]['normal'],
                        ], 2, axis=0)

        pts = NodeCoords[np.array([b, e,
                                    a, b,
                                    d, e,
                                    a, d,
                                    c, e,
                                    a, c
                                    ])]
        ds = -np.sum(normals * NodeCoords[np.array([a, a, c, c, a, a, b, b, a, a, b, b])], axis=1)
        dist_signs = np.sign(np.sum(normals * pts, axis=1) + ds).reshape(6,2)
        if not np.all(np.all((dist_signs < 0),axis=1) | np.all((dist_signs > 0), axis=1)):
            # Not convex (includes coplanar faces as nonconvex as such a flip would produce 0 volume elements)
            break
        

        newface = tuple(sorted((b,c,d)))
        if newface in FaceTable.keys():
            normal = FaceTable[newface]['normal']
        else:
            normal = utils.CalcFaceNormal(NodeCoords, [newface])[0]


        # Test which way the face is facing to allow for appropriate tet construction
        if (np.dot(NodeCoords[a], normal) - np.dot(NodeCoords[b], normal)) > 0:
            # face is facing `a`
            elem1 = (*newface, a)
            elem2 = (*newface[::-1], e)
        else:
            # face is facing `e`
            elem1 = (*newface[::-1], a)
            elem2 = (*newface, e)

        elem1key = tuple(sorted(elem1)) 
        elem2key = tuple(sorted(elem2)) 
        
        elems = (elem1, elem2)
        elemkeys = (elem1key, elem2key)

        for elem,elemkey in zip(elems, elemkeys):
            if elemkey not in ElemTable.keys():
                vol = quality.Volume(NodeCoords, [elem], ElemType='tet')[0]
                ElemTable[elemkey] = {
                    'status'  : False,
                    'elem'    : elem,
                    'volume'  : vol
                    }
                if vol > 0:
                    ElemTable[elemkey]['quality'] = qualfunc(NodeCoords, [elem], [vol])[0]
                    ElemTable[elemkey]['faces'] = [
                        (elemkey[0], elemkey[1], elemkey[2]), 
                        (elemkey[0], elemkey[1], elemkey[3]), 
                        (elemkey[1], elemkey[2], elemkey[3]), 
                        (elemkey[0], elemkey[2], elemkey[3])
                        ]
                    ElemTable[elemkey]['edges'] = [
                        (elemkey[0], elemkey[1]), 
                        (elemkey[1], elemkey[2]), 
                        (elemkey[0], elemkey[2]), 
                        (elemkey[0], elemkey[3]), 
                        (elemkey[1], elemkey[3]), 
                        (elemkey[2], elemkey[3])
                        ]
                else:
                    # Invalid flip, no need to keep processing this configuration
                    valid = False
                    break
            elif ElemTable[elemkey]['volume'] <= 0:
                valid = False
                break


        if valid:
            proposed_quality = [ElemTable[elemkey]['quality'] for elemkey in elemkeys]

            if ((target == 'min') & (min(proposed_quality) > min(current_quality))) | ((target == 'mean') & (np.mean(proposed_quality) > np.mean(current_quality))):
                oldset = {key, tet1, tet2}
                # Flip leads to a quality improvement - accept the flip and update edges/faces
                # a, b, c, d, e are still defined from checking step
                # key is still the reference to the current element and other_tet is the reference to its flipping neighbor
                ElemTable[elem1key]['status']  = True
                ElemTable[elem2key]['status']  = True
                ElemTable[key]['status']       = False
                ElemTable[tet1]['status']      = False
                ElemTable[tet2]['status']      = False

                ### Update Face Table ###
                # New face
                if newface not in FaceTable.keys():
                    FaceTable[newface] = {'elems'  : (elem1key, elem2key),
                                          'normal' : normal
                                        }
                else:
                    FaceTable[newface]['elems'] = (elem1key, elem2key)

                # Outer faces
                FaceTable[face1key]['elems'] = tuple((elem for elem in FaceTable[face1key]['elems'] if elem not in oldset)) + tuple([elem1key])
                FaceTable[face2key]['elems'] = tuple((elem for elem in FaceTable[face2key]['elems'] if elem not in oldset)) + tuple([elem2key])
                FaceTable[face3key]['elems'] = tuple((elem for elem in FaceTable[face3key]['elems'] if elem not in oldset)) + tuple([elem1key])
                FaceTable[face4key]['elems'] = tuple((elem for elem in FaceTable[face4key]['elems'] if elem not in oldset)) + tuple([elem2key])
                FaceTable[face5key]['elems'] = tuple((elem for elem in FaceTable[face5key]['elems'] if elem not in oldset)) + tuple([elem1key])
                FaceTable[face6key]['elems'] = tuple((elem for elem in FaceTable[face6key]['elems'] if elem not in oldset)) + tuple([elem2key])

                ### Update Edge Table ###
                edge1key = tuple(sorted((a,b)))
                edge2key = tuple(sorted((a,c)))
                edge3key = tuple(sorted((a,d)))
                edge4key = tuple(sorted((e,b)))
                edge5key = tuple(sorted((e,c)))
                edge6key = tuple(sorted((e,d)))
                edge7key = tuple(sorted((b,c)))
                edge8key = tuple(sorted((c,d)))
                edge9key = tuple(sorted((d,c)))

                # All edges already exist
                
                EdgeTable[edge1key]['elems'] = tuple((elem for elem in EdgeTable[edge1key]['elems'] if elem not in oldset)) + tuple([elem1key])
                EdgeTable[edge2key]['elems'] = tuple((elem for elem in EdgeTable[edge2key]['elems'] if elem not in oldset)) + tuple([elem1key])
                EdgeTable[edge3key]['elems'] = tuple((elem for elem in EdgeTable[edge3key]['elems'] if elem not in oldset)) + tuple([elem1key])

                EdgeTable[edge4key]['elems'] = tuple((elem for elem in EdgeTable[edge4key]['elems'] if elem not in oldset)) + tuple([elem2key])
                EdgeTable[edge5key]['elems'] = tuple((elem for elem in EdgeTable[edge5key]['elems'] if elem not in oldset)) + tuple([elem2key])
                EdgeTable[edge6key]['elems'] = tuple((elem for elem in EdgeTable[edge6key]['elems'] if elem not in oldset)) + tuple([elem2key])

                EdgeTable[edge7key]['elems'] = tuple((elem for elem in EdgeTable[edge7key]['elems'] if elem not in oldset)) + (elem1key,elem2key)
                EdgeTable[edge8key]['elems'] = tuple((elem for elem in EdgeTable[edge8key]['elems'] if elem not in oldset)) + (elem1key,elem2key)
                EdgeTable[edge9key]['elems'] = tuple((elem for elem in EdgeTable[edge9key]['elems'] if elem not in oldset)) + (elem1key,elem2key)

                success = True
                break

    return success

def _Tet23Flip(elemkey, NodeCoords, ElemTable, FaceTable, EdgeTable, qualfunc, target='min'):

    success = False

    key = elemkey 
    keyset = set(key)
    for i,face in enumerate(ElemTable[key]['faces']):
        valid = True
        adjacent_tets = FaceTable[face]['elems']
        
        if len(adjacent_tets) == 1:
            # Surface face, not flippable
            continue
        other_tet = adjacent_tets[0] if key == adjacent_tets[1] else adjacent_tets[1]
        otherset = set(other_tet)

        current_quality = [ElemTable[key]['quality'], ElemTable[other_tet]['quality']]

        a = keyset.difference(otherset).pop()   # Node from key element
        b,c,d = face                            # Nodes of shared face
        e = otherset.difference(keyset).pop()   # Node from neighboring element

        # Outer faces
        face1key = tuple(sorted((a,c,d)))
        face2key = tuple(sorted((c,d,e)))
        face3key = tuple(sorted((a,b,c)))
        face4key = tuple(sorted((b,c,e)))
        face5key = tuple(sorted((a,b,d)))
        face6key = tuple(sorted((b,d,e)))

        # Convexity test
        normals = np.array([FaceTable[face1key]['normal'],
                            FaceTable[face1key]['normal'],
                            FaceTable[face2key]['normal'],
                            FaceTable[face2key]['normal'],
                            FaceTable[face3key]['normal'],
                            FaceTable[face3key]['normal'],
                            FaceTable[face4key]['normal'],
                            FaceTable[face4key]['normal'],
                            FaceTable[face5key]['normal'],
                            FaceTable[face5key]['normal'],
                            FaceTable[face6key]['normal'],
                            FaceTable[face6key]['normal']
                        ])
        pts = NodeCoords[np.array([b, e,
                                    a, b,
                                    d, e,
                                    a, d,
                                    c, e,
                                    a, c
                                    ])]
        ds = -np.sum(normals * NodeCoords[np.array([a, a, c, c, a, a, b, b, a, a, b, b])], axis=1)
        dist_signs = np.sign(np.sum(normals * pts, axis=1) + ds).reshape(6,2)
        if not np.all(np.all((dist_signs < 0),axis=1) | np.all((dist_signs > 0), axis=1)):
            # Not convex (includes coplanar faces as nonconvex as such a flip would produce 0 volume elements)
            break

        # Test which way the face is facing to allow for appropriate tet construction
        if (np.dot(NodeCoords[a], FaceTable[face]['normal']) - np.dot(NodeCoords[b], FaceTable[face]['normal'])) > 0:
            # face is facing `a`
            elem1 = (e, c, d, a)
            elem2 = (e, b, c, a)
            elem3 = (e, d, b, a)
        else:
            # face is facing `e`
            elem1 = (a, c, d, e)
            elem2 = (a, b, c, e)
            elem3 = (a, d, b, e)

        elem1key = tuple(sorted(elem1)) 
        elem2key = tuple(sorted(elem2)) 
        elem3key = tuple(sorted(elem3))
        
        elems = (elem1, elem2, elem3)
        elemkeys = (elem1key, elem2key, elem3key)

        for elem,elemkey in zip(elems, elemkeys):
            if elemkey not in ElemTable.keys():
                vol = quality.Volume(NodeCoords, [elem], ElemType='tet')[0]
                ElemTable[elemkey] = {
                    'status'  : False,
                    'elem'    : elem,
                    'volume'  : vol
                    }
                if vol > 0:
                    ElemTable[elemkey]['quality'] = qualfunc(NodeCoords, [elem], [vol])[0]
                    ElemTable[elemkey]['faces'] = [
                        (elemkey[0], elemkey[1], elemkey[2]), 
                        (elemkey[0], elemkey[1], elemkey[3]), 
                        (elemkey[1], elemkey[2], elemkey[3]), 
                        (elemkey[0], elemkey[2], elemkey[3])
                        ]
                    ElemTable[elemkey]['edges'] = [
                        (elemkey[0], elemkey[1]), 
                        (elemkey[1], elemkey[2]), 
                        (elemkey[0], elemkey[2]), 
                        (elemkey[0], elemkey[3]), 
                        (elemkey[1], elemkey[3]), 
                        (elemkey[2], elemkey[3])
                        ]
                else:
                    # Invalid flip, no need to keep processing this configuration
                    valid = False
                    break
            elif ElemTable[elemkey]['volume'] <= 0:
                valid = False
                break

        if valid:
            proposed_quality = [ElemTable[elemkey]['quality'] for elemkey in elemkeys]

            if ((target == 'min') & (min(proposed_quality) > min(current_quality))) | ((target == 'mean') & (np.mean(proposed_quality) > np.mean(current_quality))):
                # Flip leads to a quality improvement - accept the flip and update edges/faces
                # a, b, c, d, e are still defined from checking step
                # key is still the reference to the current element and other_tet is the reference to its flipping neighbor
                ElemTable[elem1key]['status']  = True
                ElemTable[elem2key]['status']  = True
                ElemTable[elem3key]['status']  = True
                ElemTable[key]['status']       = False
                ElemTable[other_tet]['status'] = False

                oldset = {key, other_tet}

                ### Update Face Table ###
                # Inner faces
                face7key = tuple(sorted((a,c,e)))
                face8key = tuple(sorted((a,d,e)))
                face9key = tuple(sorted((a,b,e)))

                newfacekeys = (face7key, face8key, face9key)
                for facekey in newfacekeys:
                    if facekey not in FaceTable.keys():
                        FaceTable[facekey] = {'elems'  : (None, None),
                                                'normal' : utils.CalcFaceNormal(NodeCoords, [facekey])[0]
                                            }

                FaceTable[face1key]['elems'] = tuple((elem for elem in FaceTable[face1key]['elems'] if elem not in oldset)) + tuple([elem1key])
                FaceTable[face2key]['elems'] = tuple((elem for elem in FaceTable[face2key]['elems'] if elem not in oldset)) + tuple([elem1key])
                FaceTable[face3key]['elems'] = tuple((elem for elem in FaceTable[face3key]['elems'] if elem not in oldset)) + tuple([elem2key])
                FaceTable[face4key]['elems'] = tuple((elem for elem in FaceTable[face4key]['elems'] if elem not in oldset)) + tuple([elem2key])
                FaceTable[face5key]['elems'] = tuple((elem for elem in FaceTable[face5key]['elems'] if elem not in oldset)) + tuple([elem3key])
                FaceTable[face6key]['elems'] = tuple((elem for elem in FaceTable[face6key]['elems'] if elem not in oldset)) + tuple([elem3key])
                FaceTable[face7key]['elems'] = (elem1key, elem2key)
                FaceTable[face8key]['elems'] = (elem1key, elem3key)
                FaceTable[face9key]['elems'] = (elem2key, elem3key)

                ### Update Edge Table ###
                edge1key = tuple(sorted((a,b)))
                edge2key = tuple(sorted((a,c)))
                edge3key = tuple(sorted((a,d)))
                edge4key = tuple(sorted((e,b)))
                edge5key = tuple(sorted((e,c)))
                edge6key = tuple(sorted((e,d)))
                edge7key = tuple(sorted((b,c)))
                edge8key = tuple(sorted((c,d)))
                edge9key = tuple(sorted((d,b)))
                edge10key = tuple(sorted((a,e)))

                # All edges already guaranteed to exist except edge 10
                EdgeTable[edge1key]['elems'] = tuple((elem for elem in EdgeTable[edge1key]['elems'] if elem not in oldset)) + (elem2key,elem3key)
                EdgeTable[edge2key]['elems'] = tuple((elem for elem in EdgeTable[edge2key]['elems'] if elem not in oldset)) + (elem1key,elem2key)
                EdgeTable[edge3key]['elems'] = tuple((elem for elem in EdgeTable[edge3key]['elems'] if elem not in oldset)) + (elem1key,elem3key)

                EdgeTable[edge4key]['elems'] = tuple((elem for elem in EdgeTable[edge4key]['elems'] if elem not in oldset)) + (elem2key,elem3key)
                EdgeTable[edge5key]['elems'] = tuple((elem for elem in EdgeTable[edge5key]['elems'] if elem not in oldset)) + (elem1key,elem2key)
                EdgeTable[edge6key]['elems'] = tuple((elem for elem in EdgeTable[edge6key]['elems'] if elem not in oldset)) + (elem1key,elem3key)

                EdgeTable[edge7key]['elems'] = tuple((elem for elem in EdgeTable[edge7key]['elems'] if elem not in oldset)) + tuple([elem2key])
                EdgeTable[edge8key]['elems'] = tuple((elem for elem in EdgeTable[edge8key]['elems'] if elem not in oldset)) + tuple([elem1key])
                EdgeTable[edge9key]['elems'] = tuple((elem for elem in EdgeTable[edge9key]['elems'] if elem not in oldset)) + tuple([elem3key])

                EdgeTable[edge10key] = {'elems' : (elem1key, elem2key, elem3key)}

                success = True
                break

    return success

def _Tet44Flip(elemkey, NodeCoords, ElemTable, FaceTable, EdgeTable, qualfunc, target='min'):
    success = False

    key = elemkey 
    keyset = set(key)
    for i,edge in enumerate(ElemTable[key]['edges']):
        
        adjacent_tets = EdgeTable[edge]['elems']
        
        if len(adjacent_tets) != 4:
            # Not flippable
            continue

        tet1, tet2, tet3 = (tet for tet in adjacent_tets if tet != key)

        pts = keyset.union(tet1).union(tet2).union(tet3)
        if len(pts) != 6:
            # the four tets must form a hull of 6 points
            continue

        current_quality = [ElemTable[key]['quality'], ElemTable[tet1]['quality'], ElemTable[tet2]['quality'], ElemTable[tet3]['quality']]

        c, e = edge # Nodes of shared edge
        # This could probably be more elegant
        pts.difference_update(edge)
        a = pts.pop() # 
        connected_to_a = set()
        for tet in adjacent_tets:
            if a in tet:
                connected_to_a.update(tet)
        f = pts.difference(connected_to_a).pop()
        b, d = pts.difference({f})

        # Outer faces (sorting might not be needed?)
        face1key = tuple(sorted((a,b,c)))
        face2key = tuple(sorted((a,c,d)))
        face3key = tuple(sorted((a,d,e)))
        face4key = tuple(sorted((a,b,e)))
        face5key = tuple(sorted((b,c,f)))
        face6key = tuple(sorted((b,e,f)))
        face7key = tuple(sorted((c,d,f)))
        face8key = tuple(sorted((d,e,f)))

        # Convexity test
        normals = np.repeat([FaceTable[face1key]['normal'],
                            FaceTable[face2key]['normal'],
                            FaceTable[face3key]['normal'],
                            FaceTable[face4key]['normal'],
                            FaceTable[face5key]['normal'],
                            FaceTable[face6key]['normal'],
                            FaceTable[face7key]['normal'],
                            FaceTable[face8key]['normal']
                        ], 3, axis=0)

        pts = NodeCoords[np.array([
                                    d, e, f,    # Other nodes for face 1
                                    b, e, f,    # Other nodes for face 2
                                    b, c, f,    # Other nodes for face 3
                                    c, d, f,    # Other nodes for face 4
                                    a, e, d,    # Other nodes for face 5
                                    a, c, d,    # Other nodes for face 6
                                    a, b, e,    # Other nodes for face 7
                                    a, b, c     # Other nodes for face 8
                                    ])]
        ds = -np.sum(normals * NodeCoords[np.array([a, a, a, a, a, a, a, a, a, a, a, a, b, b, b, b, b, b, c, c, c, d, d, d])], axis=1) 
        dist_signs = np.sign(np.sum(normals * pts, axis=1) + ds).reshape(12,2)
        if not np.all(np.all((dist_signs < 0),axis=1) | np.all((dist_signs > 0), axis=1)):
            # Not convex (includes coplanar faces as nonconvex as such a flip would produce 0 volume elements)
            break
        
        # Config 1
        newface1_1 = tuple(sorted((b,c,d)))
        newface1_2 = tuple(sorted((b,e,d)))
        newface1_3 = tuple(sorted((b,a,d)))
        newface1_4 = tuple(sorted((b,d,f)))
        # Config 2
        newface2_1 = tuple(sorted((a,c,f)))
        newface2_2 = tuple(sorted((a,e,f)))
        newface2_3 = tuple(sorted((b,a,f)))
        newface2_4 = tuple(sorted((d,a,f)))

        normals = [FaceTable[newface]['normal'] if newface in FaceTable.keys() else utils.CalcFaceNormal(NodeCoords, [newface])[0] for newface in (newface1_1, newface1_2, newface2_1, newface2_2)]


        # Test which way the face is facing to allow for appropriate tet construction
        if (np.dot(NodeCoords[a], normals[0]) - np.dot(NodeCoords[b], normals[0])) > 0:
            # face1 is facing `a`
            elem1_1 = (*newface1_1, a)
            elem1_2 = (*newface1_1[::-1], f)
        else:
            # face1 is facing `f`
            elem1_1 = (*newface1_1[::-1], a)
            elem1_2 = (*newface1_1, f)

        if (np.dot(NodeCoords[a], normals[1]) - np.dot(NodeCoords[d], normals[1])) > 0:
            # face2 is facing `a`
            elem1_3 = (*newface1_2, a)
            elem1_4 = (*newface1_2[::-1], f)
        else:
            # face2 is facing `f`
            elem1_3 = (*newface1_2[::-1], a)
            elem1_4 = (*newface1_2, f)

        if (np.dot(NodeCoords[b], normals[2]) - np.dot(NodeCoords[a], normals[2])) > 0:
            # face2_1 is facing `b`
            elem2_1 = (*newface2_1, b)
            elem2_2 = (*newface2_1[::-1], d)
        else:
            # face2_1 is facing `d`
            elem2_1 = (*newface2_1[::-1], b)
            elem2_2 = (*newface2_1, d)

        if (np.dot(NodeCoords[b], normals[3]) - np.dot(NodeCoords[f], normals[3])) > 0:
            # face2_2 is facing `b`
            elem2_3 = (*newface2_2, b)
            elem2_4 = (*newface2_2[::-1], d)
        else:
            # face2_2 is facing `d`
            elem2_3 = (*newface2_2[::-1], b)
            elem2_4 = (*newface2_2, d)

        elem1_1key = tuple(sorted(elem1_1)) 
        elem1_2key = tuple(sorted(elem1_2)) 
        elem1_3key = tuple(sorted(elem1_3)) 
        elem1_4key = tuple(sorted(elem1_4)) 
        elem2_1key = tuple(sorted(elem2_1)) 
        elem2_2key = tuple(sorted(elem2_2)) 
        elem2_3key = tuple(sorted(elem2_3)) 
        elem2_4key = tuple(sorted(elem2_4)) 
        
        elems = (elem1_1, elem1_2, elem1_3, elem1_4, elem2_1, elem2_2, elem2_3, elem2_4)
        elemkeys = (elem1_1key, elem1_2key, elem1_3key, elem1_4key, elem2_1key, elem2_2key, elem2_3key, elem2_4key)

        # Check validity of constructed elements and fill out their table entries (regardless of whether or not they will be activated)
        valid = np.repeat(True, 8)
        for i,(elem,elemkey) in enumerate(zip(elems, elemkeys)):
            if elemkey not in ElemTable.keys():
                vol = quality.Volume(NodeCoords, [elem], ElemType='tet')[0]
                ElemTable[elemkey] = {
                    'status'  : False,
                    'elem'    : elem,
                    'volume'  : vol
                    }
                if vol > 0:
                    ElemTable[elemkey]['quality'] = qualfunc(NodeCoords, [elem], [vol])[0]
                    ElemTable[elemkey]['faces'] = [
                        (elemkey[0], elemkey[1], elemkey[2]), 
                        (elemkey[0], elemkey[1], elemkey[3]), 
                        (elemkey[1], elemkey[2], elemkey[3]), 
                        (elemkey[0], elemkey[2], elemkey[3])
                        ]
                    ElemTable[elemkey]['edges'] = [
                        (elemkey[0], elemkey[1]), 
                        (elemkey[1], elemkey[2]), 
                        (elemkey[0], elemkey[2]), 
                        (elemkey[0], elemkey[3]), 
                        (elemkey[1], elemkey[3]), 
                        (elemkey[2], elemkey[3])
                        ]
                else:
                    # Invalid flip, no need to keep processing this configuration
                    valid[i] = False
            elif ElemTable[elemkey]['volume'] <= 0:
                valid[i] = False

        q1 = [ElemTable[elemkey]['quality'] if valid[i] else -1 for i,elemkey in enumerate(elemkeys[:4])]
        q2 = [ElemTable[elemkey]['quality'] if valid[i+4] else -1 for i,elemkey in enumerate(elemkeys[4:])]

        if np.all(valid[:4]) and (min(q1) > min(q2)):
            # Use configuration 1
            elem1key, elem2key, elem3key, elem4key = elemkeys[:4]
            proposed_quality = q1
            normal1, normal2 = normals[:2]
            newedge = tuple(sorted((b,d)))
            newface1, newface2, newface3, newface4 = newface1_1, newface1_2, newface1_3, newface1_4
        elif np.all(valid[4:]):
            # Use configuration 2
            elem1key, elem2key, elem3key, elem4key = elemkeys[4:]
            proposed_quality = q2
            normal1, normal2 = normals[2:]
            newedge = tuple(sorted((a,f)))
            newface1, newface2, newface3, newface4 = newface2_1, newface2_2, newface2_3, newface2_4
        else:
            # Neither configuration is valid
            success = False
            return success

        # Modify tables with new tets
        if ((target == 'min') & (min(proposed_quality) > min(current_quality))) | ((target == 'mean') & (np.mean(proposed_quality) > np.mean(current_quality))):
            elemkeys = (elem1key, elem2key, elem3key, elem4key)
            oldset = {key, tet1, tet2, tet3}
            # Flip leads to a quality improvement - accept the flip and update edges/faces
            # a, b, c, d, e, f are still defined from checking step
            # key is still the reference to the current element and other_tet is the reference to its flipping neighbor
            ElemTable[elem1key]['status']  = True
            ElemTable[elem2key]['status']  = True
            ElemTable[elem3key]['status']  = True
            ElemTable[elem4key]['status']  = True
            ElemTable[key]['status']       = False
            ElemTable[tet1]['status']      = False
            ElemTable[tet2]['status']      = False
            ElemTable[tet3]['status']      = False

            ### Update Face Table ###
            # New face
            if newface1 not in FaceTable.keys():
                FaceTable[newface1] = {'elems'  : (elem1key, elem2key),
                                        'normal' : normal1
                                    }
            else:
                FaceTable[newface1]['elems'] = (elem1key, elem2key)

            if newface2 not in FaceTable.keys():
                FaceTable[newface2] = {'elems'  : (elem3key, elem4key),
                                        'normal' : normal2
                                    }
            else:
                FaceTable[newface2]['elems']  = (elem3key, elem4key)

            if newface3 not in FaceTable.keys():
                FaceTable[newface3] = {'elems'  : (elem1key, elem3key),
                                        'normal' : utils.CalcFaceNormal(NodeCoords, [newface3])[0]
                                    }
            else:
                FaceTable[newface3]['elems']  = (elem1key, elem3key)

            if newface4 not in FaceTable.keys():
                FaceTable[newface4] = {'elems'  : (elem2key, elem4key),
                                        'normal' : utils.CalcFaceNormal(NodeCoords, [newface4])[0]
                                    }
            else:
                FaceTable[newface4]['elems']  = (elem2key, elem4key)

            
            for facekey in (face1key, face2key, face3key, face4key, face5key, face6key, face7key, face8key):
                FaceTable[facekey]['elems'] = tuple((elem for elem in FaceTable[facekey]['elems'] if elem not in oldset)) + tuple(elemkey for elemkey in elemkeys if np.all(np.isin(facekey, elemkey)))

            ### Update Edge Table ###
            edge1key = tuple(sorted((a,b)))
            edge2key = tuple(sorted((a,c)))
            edge3key = tuple(sorted((a,d)))
            edge4key = tuple(sorted((a,e)))

            edge5key = tuple(sorted((b,f)))
            edge6key = tuple(sorted((c,f)))
            edge7key = tuple(sorted((d,f)))
            edge8key = tuple(sorted((e,f)))

            edge9key = tuple(sorted((b,c)))
            edge10key = tuple(sorted((c,d)))
            edge11key = tuple(sorted((d,e)))
            edge12key = tuple(sorted((e,b)))

            edge13key = newedge

            # All outer edges already exist
            for edgekey in (edge1key, edge2key, edge3key, edge4key, edge5key, edge6key, edge7key, edge8key, edge9key, edge10key, edge11key, edge12key):
                EdgeTable[edgekey]['elems'] = tuple((elem for elem in EdgeTable[edgekey]['elems'] if elem not in oldset)) + tuple(elemkey for elemkey in elemkeys if np.all(np.isin(edgekey, elemkey)))

            EdgeTable[edge13key] = {'elems' : (elem1key,elem2key,elem3key,elem4key)}

            success = True
            break

    return  success