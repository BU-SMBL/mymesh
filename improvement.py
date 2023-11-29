# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:27:53 2022

@author: toj
"""

import numpy as np
import sys, warnings, time, random, copy
from . import converter, utils, quality, rays, octree
from scipy import sparse, spatial
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize

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
    
def LocalLaplacianSmoothing(NodeCoords,NodeConn,iterate,NodeNeighbors=None,FixedNodes=set(),FixFeatures=False):
    """
    LocalLaplacianSmoothing Performs iterative Laplacian smoothing, repositioning each node
    to the center of its adjacent nodes.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal coordinates.
    iterate : int
        Number of iterations to perform.
    NodeNeighbors : list (or None), optional
        List of node neighbors for each node in the mesh
        If provided, will avoid the need to recalculate the element node neighbors, by default None
    FixedNodes : set, optional
        Set of nodes to hold fixed throughout the Laplacian smoothing process, by default set().

    Returns
    -------
    NewCoords : list
        List of updated nodal coordinates.
    """    
    if type(FixedNodes) is list: FixedNodes = set(FixedNodes)
    if FixFeatures:
        edges,corners = utils.DetectFeatures(NodeCoords,NodeConn)
        FixedNodes.update(edges)
        FixedNodes.update(corners)
    NodeNeighbors = utils.getNodeNeighbors(NodeCoords,NodeConn)
    ElemConn = utils.getElemConnectivity(NodeCoords,NodeConn)
    lens = np.array([len(n) for n in NodeNeighbors])
    r = utils.PadRagged(NodeNeighbors,fillval=-1)
    FreeNodes = list(set(range(len(NodeCoords))).difference(FixedNodes))
    ArrayCoords = np.vstack([NodeCoords,[np.nan,np.nan,np.nan]])
    
    ElemNormals = utils.CalcFaceNormal(ArrayCoords[:-1],NodeConn)
    NodeNormals = utils.Face2NodeNormal(ArrayCoords[:-1],NodeConn,ElemConn,ElemNormals)
    
    for i in range(iterate):
        Q = ArrayCoords[r]
        U = (1/lens)[:,None] * np.nansum(Q - ArrayCoords[:-1,None,:],axis=1)
        ArrayCoords[FreeNodes] += U[FreeNodes]

    NewCoords = ArrayCoords[:-1]
    
    return NewCoords

def TangentialLaplacianSmoothing(NodeCoords,NodeConn,iterate,FixedNodes=set(),FixFeatures=False):
        
        """
        TangentialLaplacianSmoothing Performs iterative Laplacian smoothing, repositioning each node
        to the center of its adjacent nodes. Primarily for use on surface meshes, not well defined for 
        volume meshes.
        Ohtake, Y., Belyaev, A., & Pasko, A. (2003). Dynamic mesh optimization for polygonized implicit surfaces with sharp features. Visual Computer, 19(2–3), 115–126. https://doi.org/10.1007/s00371-002-0181-z

        Parameters
        ----------
        NodeCoords : list
            List of nodal coordinates.
        NodeConn : list
            List of nodal coordinates.
        iterate : int
            Number of iterations to perform.
        FixedNodes : set, optional
            Set of nodes to hold fixed throughout the Laplacian smoothing process, by default set().

        Returns
        -------
        NewCoords : list
            List of updated nodal coordinates.
        """    

        if type(FixedNodes) is list: FixedNodes = set(FixedNodes)
        if FixFeatures:
            edges,corners = utils.DetectFeatures(NodeCoords,NodeConn)
            FixedNodes.update(edges)
            FixedNodes.update(corners)
        NodeNeighbors = utils.getNodeNeighbors(NodeCoords,NodeConn)
        ElemConn = utils.getElemConnectivity(NodeCoords,NodeConn)
        lens = np.array([len(n) for n in NodeNeighbors])
        r = utils.PadRagged(NodeNeighbors,fillval=-1)
        FreeNodes = list(set(range(len(NodeCoords))).difference(FixedNodes))
        ArrayCoords = np.vstack([NodeCoords,[np.nan,np.nan,np.nan]])
        
        ElemNormals = utils.CalcFaceNormal(ArrayCoords[:-1],NodeConn)
        NodeNormals = utils.Face2NodeNormal(ArrayCoords[:-1],NodeConn,ElemConn,ElemNormals)
        
        for i in range(iterate):
            Q = ArrayCoords[r]
            U = (1/lens)[:,None] * np.nansum(Q - ArrayCoords[:-1,None,:],axis=1)
            R = 1*(U - np.sum(U*NodeNormals,axis=1)[:,None]*NodeNormals)
            ArrayCoords[FreeNodes] += R[FreeNodes]

        NewCoords = ArrayCoords[:-1]
        return NewCoords

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
