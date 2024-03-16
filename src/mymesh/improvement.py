# -*- coding: utf-8 -*-
"""
Mesh quality improvement.

Created on Wed Jan 26 09:27:53 2022

@author: toj
"""

import numpy as np
import sys, warnings, time, random, copy, tqdm
from . import converter, utils, quality, rays, octree
from scipy import sparse, spatial
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize

def CollapseSlivers(NodeCoords, NodeConn, skewThreshold=0.9, FixedNodes=set(), verbose=False):
    """
    Collapse sliver elements above a skewness threshold (default 0.9)

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
    
def SegmentSpringSmoothing(NodeCoords,NodeConn,NodeNeighbors=None,ElemConn=None,
    StiffnessFactor=1,FixedNodes=set(),Forces=None,L0Override='min',
    CellCentered=True,FaceCentered=True,return_KF=False):
    
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

    if NodeNeighbors is None or ElemConn is None:
        NodeNeighbors,ElemConn = utils.getNodeNeighbors(NodeCoords,NodeConn)
    if Forces is None or len(Forces) == 0:
        Forces = np.zeros((len(NodeCoords),3))
    else:
        assert len(Forces) == len(NodeCoords), 'Forces must be assigned for every node'
    
    TempCoords = np.array(NodeCoords+[[np.nan,np.nan,np.nan]])
    NodeCoords = np.array(NodeCoords)
    RNeighbors = utils.PadRagged(NodeNeighbors+[[-1,-1,-1]])
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

    FixedArray = np.array(list(FixedNodes))
    Forces = np.array(Forces)
    Forces[FixedArray] = [0,0,0]
    
    Krows_diag = np.arange(len(NodeCoords))
    Kcols_diag = np.arange(len(NodeCoords))
    Kvals_diag = np.nansum(k[:-1],axis=1) 
    if CellCentered:
        centroids = utils.Centroids(NodeCoords,NodeConn)
        centroids = np.append(centroids,[[np.nan,np.nan,np.nan]],axis=0)
        RElemConn = utils.PadRagged(ElemConn)
        ElemConnCentroids = centroids[RElemConn]
        ElemConnCenterDist = np.sqrt((NodeCoords[:,0,None]-ElemConnCentroids[:,:,0])**2 + (NodeCoords[:,1,None]-ElemConnCentroids[:,:,1])**2 + (NodeCoords[:,2,None]-ElemConnCentroids[:,:,2])**2)
        kcenters = StiffnessFactor/ElemConnCenterDist
        Kvals_diag += np.nansum(kcenters,axis=1)
    if FaceCentered:
        faces = converter.solid2faces(NodeCoords,NodeConn)
        Faces = converter.faces2unique(faces)
        fcentroids = utils.Centroids(NodeCoords,Faces)
        fcentroids = np.append(fcentroids,[[np.nan,np.nan,np.nan]],axis=0)
        FConn = utils.getElemConnectivity(NodeCoords,Faces)
        RFConn = utils.PadRagged(FConn)
        FConnCentroids = fcentroids[RFConn]
        FConnCenterDist = np.sqrt((NodeCoords[:,0,None]-FConnCentroids[:,:,0])**2 + (NodeCoords[:,1,None]-FConnCentroids[:,:,1])**2 + (NodeCoords[:,2,None]-FConnCentroids[:,:,2])**2)
        fkcenters = StiffnessFactor/FConnCenterDist
        Kvals_diag += np.nansum(fkcenters,axis=1)


    Kvals_diag[FixedArray] = 1
    UnfixedNodes = np.array(list(set(range(len(NodeCoords))).difference(FixedNodes)))
    
    template = (RNeighbors[:-1]>=0)[UnfixedNodes]
    flattemplate = template.flatten()
    Krows_off = (template.astype(int)*UnfixedNodes[:,None]).flatten()[flattemplate]
    Kcols_off = RNeighbors[UnfixedNodes].flatten()[flattemplate]
    Kvals_off = -k[UnfixedNodes].flatten()[flattemplate]
    
    Krows = np.concatenate((Krows_diag,Krows_off))
    Kcols = np.concatenate((Kcols_diag,Kcols_off))
    Kvals = np.concatenate((Kvals_diag,Kvals_off))

    if CellCentered:
        RNodeConn = utils.PadRagged(NodeConn+[[-1,-1,-1]],fillval=-1)
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
        RFaces = utils.PadRagged(Faces+[[-1,-1,-1]],fillval=-1)
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
    
    Xnew = np.array(NodeCoords) + dXnew
    Xnew[list(FixedNodes)] = np.array(NodeCoords)[list(FixedNodes)] # Enforce fixed nodes
    if return_KF:
        return Xnew.tolist(), dXnew.tolist(), (K,F)
    return Xnew.tolist(), dXnew.tolist()

def NodeSpringSmoothing(NodeCoords,NodeConn,NodeNeighbors,Stiffness=1,FixedNodes=[],Forces=None,maxIter=np.inf,converge=1e-4):
    # Blom, F.J., 2000. Considerations on the spring analogy. International journal for numerical methods in fluids, 32(6), pp.647-668.
    
    if Forces == None or len(Forces) == 0:
        Forces = np.zeros(len(NodeCoords))
    else:
        assert len(Forces) == len(NodeCoords), 'Forces must be assigned for every node'
    
    if type(FixedNodes) == list:
        FixedNodes = set(FixedNodes)
    
    # k = [[Stiffness for n in NodeNeighbors[i]] for i in range(len(NodeCoords))]   
    k = Stiffness
    
    X = np.array(NodeCoords)
    thinking = True
    iteration = 0
    while thinking:
        Xnew = np.zeros(X.shape)
        # print(iteration)
        for i in range(len(X)):
            if i in FixedNodes:
                Xnew[i,:] = X[i,:]
                continue
            Xnew[i,:] = (np.sum([k*X[n,:] for j,n in enumerate(NodeNeighbors[i])],axis=0) + Forces[i])/sum(k for j,n in enumerate(NodeNeighbors[i]))
        iteration += 1
        print(np.linalg.norm(X-Xnew))
        if iteration > maxIter or np.linalg.norm(X-Xnew) < converge:
            thinking = False
        else:
            X = copy.copy(Xnew)
                   
    return Xnew.tolist()
    
def LocalLaplacianSmoothing(NodeCoords,NodeConn,iterate,NodeNeighbors=None,ElemConn=None,FixedNodes=set(),FixFeatures=False):
    """
    Performs iterative Laplacian smoothing, repositioning each node to the center of its adjacent nodes.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal coordinates.
    iterate : int
        Number of iterations to perform.
    NodeNeighbors : list/None, optional
        List of node neighbors for each node in the mesh, as calculated by :func:``utils.getNodeNeighbors``.
        If provided, will avoid the need to recalculate the element node neighbors, by default None
    ElemConn : list/None, optional
        List of elements connected to each node in the mesh, as calculated by :func:``utils.getElemConnectivity``. 
         If provided, will avoid the need to recalculate the element node neighbors, by default None
    FixedNodes : set, optional
        Set of nodes to hold fixed throughout the Laplacian smoothing process, by default set().

    Returns
    -------
    NewCoords : np.ndarray
        List of updated nodal coordinates.
    NodeConn : list/np.ndarray
        Original node connectivity passed through

    """    
    if type(FixedNodes) is list: FixedNodes = set(FixedNodes)
    if FixFeatures:
        edges,corners = utils.DetectFeatures(NodeCoords,NodeConn)
        FixedNodes.update(edges)
        FixedNodes.update(corners)
    if NodeNeighbors is None:
        NodeNeighbors = utils.getNodeNeighbors(NodeCoords,NodeConn)
    if ElemConn is None:
        ElemConn = utils.getElemConnectivity(NodeCoords,NodeConn)
    lens = np.array([len(n) for n in NodeNeighbors])
    r = utils.PadRagged(NodeNeighbors,fillval=-1)
    idx = np.unique(NodeConn)
    FreeNodes = list(set(idx).difference(FixedNodes))
    ArrayCoords = np.vstack([NodeCoords,[np.nan,np.nan,np.nan]])
    
    for i in range(iterate):
        Q = ArrayCoords[r]
        U = (1/lens)[:,None] * np.nansum(Q - ArrayCoords[:-1,None,:],axis=1)
        ArrayCoords[FreeNodes] += U[FreeNodes]

    NewCoords = ArrayCoords[:-1]
    
    return NewCoords, NodeConn

def TangentialLaplacianSmoothing(NodeCoords,NodeConn,iterate,NodeNeighbors=None,ElemConn=None,FixedNodes=set(),FixFeatures=False):
        
    """
    TangentialLaplacianSmoothing Performs iterative Laplacian smoothing, repositioning each node to the center of its adjacent nodes. Primarily for use on surface meshes, not well defined for volume meshes.
    Ohtake, Y., Belyaev, A., & Pasko, A. (2003). Dynamic mesh optimization for polygonized implicit surfaces with sharp features. Visual Computer, 19(2-3), 115-126. https://doi.org/10.1007/s00371-002-0181-z

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal coordinates.
    iterate : int
        Number of iterations to perform.
    NodeNeighbors : list/None, optional
        List of node neighbors for each node in the mesh, as calculated by :func:``utils.getNodeNeighbors``.
        If provided, will avoid the need to recalculate the element node neighbors, by default None
    ElemConn : list/None, optional
        List of elements connected to each node in the mesh, as calculated by :func:``utils.getElemConnectivity``. 
         If provided, will avoid the need to recalculate the element node neighbors, by default None
    FixedNodes : set, optional
        Set of nodes to hold fixed throughout the Laplacian smoothing process, by default set().

    Returns
    -------
    NewCoords : list
        List of updated nodal coordinates.
    NodeConn : list/np.ndarray
        Original node connectivity passed through
    """    

    if type(FixedNodes) is list: FixedNodes = set(FixedNodes)
    if FixFeatures:
        edges,corners = utils.DetectFeatures(NodeCoords,NodeConn)
        FixedNodes.update(edges)
        FixedNodes.update(corners)
    if NodeNeighbors is None:
        NodeNeighbors = utils.getNodeNeighbors(NodeCoords,NodeConn)
    if ElemConn is None:
        ElemConn = utils.getElemConnectivity(NodeCoords,NodeConn)
    lens = np.array([len(n) for n in NodeNeighbors])
    r = utils.PadRagged(NodeNeighbors,fillval=-1)
    idx = np.unique(NodeConn)
    FreeNodes = list(set(idx).difference(FixedNodes))

    ArrayCoords = np.vstack([NodeCoords,[np.nan,np.nan,np.nan]])
    
    ElemNormals = utils.CalcFaceNormal(ArrayCoords[:-1],NodeConn)
    NodeNormals = utils.Face2NodeNormal(ArrayCoords[:-1],NodeConn,ElemConn,ElemNormals)
    
    for i in range(iterate):
        Q = ArrayCoords[r]
        U = (1/lens)[:,None] * np.nansum(Q - ArrayCoords[:-1,None,:],axis=1)
        R = 1*(U - np.sum(U*NodeNormals,axis=1)[:,None]*NodeNormals)
        ArrayCoords[FreeNodes] += R[FreeNodes]

    NewCoords = np.copy(NodeCoords)
    NewCoords[idx] = ArrayCoords[idx]

    return NewCoords, NodeConn

def GlobalLaplacianSmoothing(NodeCoords,NodeConn,FeatureNodes=[],FixedNodes=set(),FeatureWeight=1,BaryWeight=1/3):
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

def ResolveSurfSelfIntersections(NodeCoords,SurfConn,FixedNodes=set(),octree='generate',maxIter=10):

    if type(FixedNodes) != set:
        FixedNodes = set(FixedNodes)

    NewCoords = np.array(NodeCoords)
    SurfConn = np.asarray(SurfConn)
    if octree == 'generate':
        octree = octree.Surf2Octree(NewCoords,SurfConn)
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
        Edges, EdgeConn, EdgeElem = converter.solid2edges(NewCoords,NewConn,return_EdgeConn=True,return_EdgeElem=True,ReturnType=np.ndarray)
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
        Number of passes if the free nodes in the mesh, by default 1.
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
        f, jac = func(NewCoords,NewConn, nodeid)
        print(np.nanmean(q(NewCoords, NewConn)))

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
            print(nodeid)
            x0 = NewCoords[nodeid]
            out = minimize(obj, x0, jac=True, args=(nodeid), method='L-BFGS-B', options=dict(maxiter=10))
            NewCoords[nodeid] = out.x

    return NewCoords, NodeConn

def Tet23Flip(NodeCoords,NodeConn,Faces,FaceElemConn,FaceConn,FaceID,quality,QualityMetric='Skewness',Validate=True):
    
    # NodeConn should be an array for the input to avoid a lot of overhead
    
    if np.any(np.isnan(FaceElemConn[FaceID])):
        return NodeConn,Faces,FaceElemConn,FaceConn,quality
    
    NewConn = np.asarray(NodeConn)
    FaceNodes = Faces[FaceID]
    NonFaceNodes = [np.setdiff1d(NodeConn[FaceElemConn[FaceID][0]],FaceNodes)[0],np.setdiff1d(NodeConn[FaceElemConn[FaceID][1]],FaceNodes)[0]]
    # Elem1 = np.append(FaceNodes,NonFaceNodes[0])
    # Elem2 = np.append(np.flip(FaceNodes),NonFaceNodes[1])
    a = NonFaceNodes[1]
    e = NonFaceNodes[0]
    b,c,d = FaceNodes
    
    # Make New Elements
    NewElem1 = [a,e,d,b]
    NewElem2 = [a,b,c,e]
    NewElem3 = [a,e,c,d]
    
    # Check New Elements
    if QualityMetric == 'Skewness':
        OldWorstQuality = np.max([quality[FaceElemConn[FaceID][0]], quality[FaceElemConn[FaceID][1]]])
        NewQuality = quality.Skewness(NodeCoords,[NewElem1,NewElem2,NewElem3])
        NewWorstQuality = np.max(NewQuality)
        
        Improved = NewWorstQuality < OldWorstQuality

    else:
        raise Exception('quality metric {:s} unknown or not yet implemented.'.format(QualityMetric))
    if Validate:
        if not Improved:
            # Flip didn't improve quality or created invalid elements, keep old mesh
            return NodeConn,Faces,FaceElemConn,FaceConn,quality
        
        NewVolume = quality.Volume(NodeCoords,[NewElem1,NewElem2,NewElem3])
        Valid = np.min(NewVolume) >= 0 and min(NewQuality) >= 0 and max(NewQuality) <= 1
        if not Valid:
            # Flip didn't improve quality or created invalid elements, keep old mesh
            return NodeConn,Faces,FaceElemConn,FaceConn,quality
    
    # Flip validly improves quality, modify mesh
    print('flip')
    # Delete old elements
    maxelem = max(FaceElemConn[FaceID])
    minelem = min(FaceElemConn[FaceID])
    
    AdjacentFaces = np.unique([FaceConn[maxelem],FaceConn[minelem]])
    AdjacentFaces = AdjacentFaces[AdjacentFaces!=FaceID]    
    OldFaceSets = [set(Faces[fid]) for fid in AdjacentFaces]
    
    NewConn = np.delete(NewConn,[maxelem,minelem],axis=0)
    quality = np.delete(quality, [maxelem,minelem])
    del FaceConn[maxelem], FaceConn[minelem]
    
    # Add new elems
    NewElemIds = [len(NewConn),len(NewConn)+1,len(NewConn)+2]
    NewConn = np.append(NewConn, [NewElem1, NewElem2, NewElem3],axis=0)
    quality = np.append(quality, [NewQuality[0], NewQuality[1], NewQuality[2]])
    
    
    # Delete old face
    del Faces[FaceID]
    del FaceElemConn[FaceID]
    
    # Add new faces
    NewFaceIds = [len(Faces),len(Faces)+1,len(Faces)+2]
    Faces += [[a,e,d],[a,b,e],[a,c,e]]
    FaceElemConn += [[1,3],[2,1],[3,2]]
    
    # Adjust FaceElemConn
    NewFaceSets = [{a,d,c},{c,d,e},{a,b,d},{b,e,d},{a,c,b},{b,c,e}] 
    NewFaceElemConnID = [0,0,1,1,2,2]
    MatchedFaces = [AdjacentFaces[i] for FaceSet in NewFaceSets for i,OldSet in enumerate(OldFaceSets) if FaceSet==OldSet]
    for i,fid in enumerate(MatchedFaces):
        FaceElemConn[fid] = [NewElemIds[NewFaceElemConnID[i]] if (x == maxelem or x == minelem) else x for x in FaceElemConn[fid]]
    
    FaceConn += [[MatchedFaces[0],MatchedFaces[1],NewFaceIds[0],NewFaceIds[1]],
                [MatchedFaces[2],MatchedFaces[3],NewFaceIds[1],NewFaceIds[2]],
                [MatchedFaces[3],MatchedFaces[4],NewFaceIds[2],NewFaceIds[0]]]
    
    return NewConn,Faces,FaceElemConn,FaceConn,quality
    
def Tet32Flip(NodeCoords, NodeConn, Edges, EdgeElemConn, EdgeConn, EdgeID, quality, QualityMetric='Skewness',Validate=True):
    
    # NodeConn should be an array for the input to avoid a lot of overhead
    
    if len(EdgeElemConn[EdgeID]) != 3:
        # Not valid for a 3-2 flip
        return NodeConn, Edges, EdgeElemConn, EdgeConn, quality
    
    NewConn = np.asarray(NodeConn)
    EdgeNodes = Edges[EdgeID]
    Elements = EdgeElemConn[EdgeID]
    Face = np.setdiff1d(np.unique(NewConn[Elements]),EdgeNodes)
    
    if len(Face) != 3:
        # Not valid for a 3-2 flip
        return NodeConn, Edges, EdgeElemConn, EdgeConn, quality
    
    # Elem1 = np.append(FaceNodes,NonFaceNodes[0])
    # Elem2 = np.append(np.flip(FaceNodes),NonFaceNodes[1])
    a = EdgeNodes[1]
    e = EdgeNodes[0]
    b,c,d = Face

    Normal = np.cross(np.subtract(NodeCoords[b],NodeCoords[c]),np.subtract(NodeCoords[d],NodeCoords[c]))
    if np.dot(Normal,NodeCoords[a])-np.dot(Normal,NodeCoords[c]) < 0:
        # Reorder a, e for consistent tet formation
        a,e = e,a
    
    # Make New Elements
    NewElem1 = [b,c,d,e]
    NewElem2 = [b,d,c,a]
    
    # Check New Elements
    if QualityMetric == 'Skewness':
        OldWorstQuality = np.max([quality[Elements[0]], quality[Elements[1]], quality[Elements[2]]])
        NewQuality = quality.Skewness(NodeCoords,[NewElem1,NewElem2])
        NewWorstQuality = np.max(NewQuality)
        
        Improved = NewWorstQuality < OldWorstQuality

    else:
        raise Exception('quality metric {:s} unknown or not yet implemented.'.format(QualityMetric))
    if Validate:
        if not Improved:
            # Flip didn't improve quality or created invalid elements, keep old mesh
            return NodeConn, Edges, EdgeElemConn, EdgeConn, quality
        
        NewVolume = quality.Volume(NodeCoords,[NewElem1,NewElem2])
        Valid = np.min(NewVolume) >= 0 and min(NewQuality) >= 0 and max(NewQuality) <= 1
        if not Valid:
            # Flip didn't improve quality or created invalid elements, keep old mesh
            return NodeConn, Edges, EdgeElemConn, EdgeConn, quality
    
    # Delete old elements
    NewConn = np.delete(NewConn, Elements,axis=0)
    quality = np.delete(quality, Elements)
    
    
    # Remove edge and adjust EdgeConn, EdgeElemConn
    