# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:27:53 2022

@author: toj
"""

import numpy as np
import sys, warnings, time, random, copy
from . import converter, MeshUtils, Quality, Rays, Octree
from joblib import Parallel, delayed
from scipy import sparse, spatial
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize

def CollapseSlivers_old(NodeCoords, NodeConn, skewThreshold=0.9, FixedNodes=[], verbose=False, pool=Parallel(n_jobs=1)):
    
    if type(NodeCoords) is np.array: NodeCoords = NodeCoords.tolist()
    if type(NodeConn) is np.array: NodeConn = NodeConn.tolist()
    NewCoords = np.array(NodeCoords)
    NewConn = copy.copy(NodeConn)
    fixed = set(FixedNodes)
    if verbose:
        print('Initial Skewness:')
    skew = Quality.Skewness(NewCoords,NewConn,verbose=verbose,pool=pool)
    if verbose:
        print(str(sum([1 for i in skew if i > skewThreshold])) + ' slivers')
    ignore = []
    while max([s for i,s in enumerate(skew) if i not in ignore]) > skewThreshold:
        NewCoords = np.array(NewCoords)
        ignore = []
        for i,elem in enumerate(NewConn):
            # s = Qualit1y.get_skewness([NewCoords[n] for n in elem])
            if skew[i] <= skewThreshold:
            # if s <= skewThreshold:
                continue
            if len(elem) == 3:
                # Triangluar Mesh
                A = NewCoords[elem[0]]
                B = NewCoords[elem[1]]
                C = NewCoords[elem[2]]
                if all(A == B) or all(B == C) or all(A == C):
                    # Already collapsed
                    continue
                a2 = (B[0]-C[0])**2 + (B[1]-C[1])**2 + (B[2]-C[2])**2
                a = np.sqrt(a2)
                b2 = (C[0]-A[0])**2 + (C[1]-A[1])**2 + (C[2]-A[2])**2
                b = np.sqrt(b2)
                c2 = (A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2
                c = np.sqrt(c2)
                
                minside = min([a,b,c])
                if minside == a:
                    # Collapse vertices B and C
                    if elem[2] not in fixed:
                        NewCoords[elem[2]] = B
                    elif elem[1] not in fixed:
                        NewCoords[elem[1]] = C
                    else:
                        ignore.append(i)
                elif minside == b:
                    # Collapse vertices A and C
                    if elem[0] not in fixed:
                        NewCoords[elem[0]] = C
                    elif elem[2] not in fixed:
                        NewCoords[elem[2]] = A
                    else:
                        ignore.append(i)
                elif minside == c:
                    # Collapse vertices A and B
                    if elem[1] not in fixed:
                        NewCoords[elem[1]] = A
                    elif elem[0] not in fixed:
                        NewCoords[elem[0]] = B
                    else:
                        ignore.append(i)
            elif len(elem) == 4:
                # Tetrahedral Mesh
                A = NewCoords[elem[0]]
                B = NewCoords[elem[1]]
                C = NewCoords[elem[2]]
                D = NewCoords[elem[3]]
                if A == B or B == C or A == C or A == D or D == B or D == C:
                    # Already collapsed
                    continue
                CB2 = (B[0]-C[0])**2 + (B[1]-C[1])**2 + (B[2]-C[2])**2
                CB = np.sqrt(CB2)
                AC2 = (C[0]-A[0])**2 + (C[1]-A[1])**2 + (C[2]-A[2])**2
                AC = np.sqrt(AC2)
                AB2 = (A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2
                AB = np.sqrt(AB2)
                AD2 = (A[0]-D[0])**2 + (A[1]-D[1])**2 + (A[2]-D[2])**2
                AD = np.sqrt(AD2)
                BD2 = (B[0]-D[0])**2 + (B[1]-D[1])**2 + (B[2]-D[2])**2
                BD = np.sqrt(BD2)
                CD2 = (C[0]-D[0])**2 + (C[1]-D[1])**2 + (C[2]-D[2])**2
                CD = np.sqrt(CD2)
                
                sides = [CB,AC,AB,AD,BD,CD]
                thinking = True
                while thinking:
                    if len(sides) == 0:
                        ignore.append(i)
                        thinking = False
                        break
                    minside = min(sides)
                    if minside == CB:
                        # Collapse vertices B and C
                        if elem[2] not in fixed:
                            NewCoords[elem[2]] = B
                            thinking = False
                        elif elem[1] not in fixed:
                            NewCoords[elem[1]] = C
                            thinking = False
                        else:
                            sides.remove(CB)
                    elif minside == AC:
                        # Collapse vertices A and C
                        if elem[0] not in fixed:
                            NewCoords[elem[0]] = C
                            thinking = False
                        elif elem[2] not in fixed:
                            NewCoords[elem[2]] = A
                            thinking = False
                        else:
                            sides.remove(AC)
                    elif minside == AB:
                        # Collapse vertices A and B
                        if elem[1] not in fixed:
                            NewCoords[elem[1]] = A 
                            thinking = False
                        elif elem[0] not in fixed:
                            NewCoords[elem[0]] = B
                            thinking = False
                        else:
                            sides.remove(AB)
                    elif minside == AD:
                        # Collapse vertices A and D
                        if elem[3] not in fixed:
                            NewCoords[elem[3]] = A 
                            thinking = False
                        elif elem[0] not in fixed:
                            NewCoords[elem[0]] = D
                            thinking = False
                        else:
                            sides.remove(AD)
                    elif minside == BD:
                        # Collapse vertices B and D
                        if elem[3] not in fixed:
                            NewCoords[elem[3]] = B 
                            thinking = False
                        elif elem[1] not in fixed:
                            NewCoords[elem[1]] = D
                            thinking = False
                        else:
                            sides.remove(BD)
                    elif minside == CD:
                        # Collapse vertices C and D
                        if elem[3] not in fixed:
                            NewCoords[elem[3]] = C
                            thinking = False 
                        elif elem[2] not in fixed:
                            NewCoords[elem[2]] = D
                            thinking = False
                        else:
                            sides.remove(CD)
                
                
        NewCoords,NewConn,_ = MeshUtils.DeleteDuplicateNodes(NewCoords,NewConn)
        NewCoords,NewConn = MeshUtils.DeleteDegenerateElements(NewCoords,NewConn,strict=True)
        NewCoords,NewConn,_ = converter.removeNodes(NewCoords,NewConn)
            
        if verbose:
            print('Improved Skewness:')
        skew = Quality.Skewness(NewCoords,NewConn,verbose=verbose,pool=pool)
        if verbose:
            print(str(sum([1 for i in skew if i > skewThreshold])) + ' slivers remaining')
    
    return NewCoords, NewConn

def CollapseSlivers(NodeCoords, NodeConn, skewThreshold=0.9, FixedNodes=set(), verbose=False):
    
    if type(FixedNodes) is list: FixedNodes = set(FixedNodes)
    ArrayCoords = np.asarray(NodeCoords)
    skew = Quality.Skewness(ArrayCoords,NodeConn,verbose=verbose)
    Slivers = np.where(skew>skewThreshold)[0]
    if len(Slivers) > 0:

        NewConn = copy.copy(NodeConn)
        RNodeConn = MeshUtils.PadRagged(NodeConn)
        
        Edges, EdgeConn, EdgeElem = converter.solid2edges(ArrayCoords,NodeConn,return_EdgeConn=True,return_EdgeElem=True)
        Edges = np.asarray(Edges)
        REdgeConn = MeshUtils.PadRagged(EdgeConn)
        
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
        NewCoords,NewConn = MeshUtils.DeleteDegenerateElements(ArrayCoords,NewConn,strict=True)
        NewCoords = NewCoords.tolist()
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
    skew = Quality.Skewness(NodeCoords,NodeConn)
    Faces,FaceConn,FaceElem = converter.solid2faces(NodeCoords,NodeConn,return_FaceConn=True,return_FaceElem=True)
    FaceElemConn,UFaces, UFaceConn, UFaceElem, idx, inv = converter.faces2faceelemconn(Faces,FaceConn,FaceElem,return_UniqueFaceInfo=True)
    UFaceConn = MeshUtils.PadRagged(UFaceConn)
    ElemNormals = np.array(MeshUtils.CalcFaceNormal(NodeCoords,Faces))

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
    SurfFacePairs = MeshUtils.PadRagged(FaceConn)[Face01][np.any(X,axis=2)[Face01]].reshape((sum(Face01),2))
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
    V = Quality.Volume(NodeCoords, NodeConn)
    if min(V) > 0:
        return NodeCoords
    
    InversionElems = np.where(np.asarray(V) < 0)[0]
    InversionConn = [NodeConn[i] for i in InversionElems]
    InversionNodes = np.unique([n for elem in InversionConn for n in elem])
    ProblemNodes = list(set(InversionNodes).difference(FixedNodes))
    if len(ProblemNodes) == 0:
        warnings.warn('Fixed nodes prevent any repositioning.')
        return NodeCoords
    _,ElemConn = MeshUtils.getNodeNeighbors(NodeCoords, NodeConn)
    NeighborhoodElems = np.unique([e for i in ProblemNodes for e in ElemConn[i]])
    NeighborhoodConn = [NodeConn[i] for i in NeighborhoodElems]

    ArrayCoords = np.array(NodeCoords)

    def fun(x):
        ArrayCoords[ProblemNodes] = x.reshape((int(len(x)/3),3))
        v = Quality.Volume(ArrayCoords,NeighborhoodConn)
        # print(sum(v<0))
        return -min(v)

    x0 = ArrayCoords[ProblemNodes].flatten()

    out = minimize(fun,x0,method='Nelder-Mead',options=dict(adaptive=True,xatol=.01,fatol=.01))#,maxfev=maxfev))
    if not out.success:
        warnings.warn('Unable to eliminate all element inversions.')
    ArrayCoords[ProblemNodes] = out.x.reshape((int(len(out.x)/3),3))

    NewCoords = ArrayCoords.tolist()
    return NewCoords

def SegmentSpringSmoothing_old(NodeCoords,NodeConn,NodeNeighbors,StiffnessFactor=1,FixedNodes=set(),Displacements=None,Forces=None):
    # Blom, F.J., 2000. Considerations on the spring analogy. International journal for numerical methods in fluids, 32(6), pp.647-668.
    
    if Forces == None or len(Forces) == 0:
        Forces = np.zeros((len(NodeCoords),3))
    else:
        assert len(Forces) == len(NodeCoords), 'Forces must be assigned for every node'
        
    if Displacements == None or len(Displacements) == 0:
        Displacements = np.zeros(np.shape(NodeCoords))
    else:
        assert len(Displacements) == len(NodeCoords), 'Forces must be assigned for every node'
    
    
    def dist(p1,p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    lengths = [[dist(NodeCoords[i],NodeCoords[n]) for n in NodeNeighbors[i]] for i in range(len(NodeCoords))]
    minL = min([l for L in lengths for l in L if l!=0])
    k = [[StiffnessFactor/max(li,minL) for li in lengths[i]] for i in range(len(NodeCoords))]
   
    Krows = []
    Kcols = []
    Kvals = []
    for i in range(len(NodeCoords)):
        Krows.append(i)
        Kcols.append(i)
        
        if i not in FixedNodes:
            Kvals.append(sum(k[i]))
            for j,col in enumerate(NodeNeighbors[i]):
                Krows.append(i)
                Kcols.append(col)
                Kvals.append(-1*k[i][j])
        else:
            Kvals.append(1)
            Forces[i] = [0,0,0]
    K = sparse.coo_matrix((Kvals,(Krows,Kcols)))
    
    dXnew = spsolve(K.tocsc(), sparse.csc_matrix(Forces)).toarray()
    
    Xnew = np.array(NodeCoords) + dXnew
    
    return Xnew.tolist(), dXnew.tolist()
    
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
        NodeNeighbors,ElemConn = MeshUtils.getNodeNeighbors(NodeCoords,NodeConn)
    if Forces is None or len(Forces) == 0:
        Forces = np.zeros((len(NodeCoords),3))
    else:
        assert len(Forces) == len(NodeCoords), 'Forces must be assigned for every node'
    
    TempCoords = np.array(NodeCoords+[[np.nan,np.nan,np.nan]])
    NodeCoords = np.array(NodeCoords)
    RNeighbors = MeshUtils.PadRagged(NodeNeighbors+[[-1,-1,-1]])
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
        centroids = MeshUtils.Centroids(NodeCoords,NodeConn)
        centroids = np.append(centroids,[[np.nan,np.nan,np.nan]],axis=0)
        RElemConn = MeshUtils.PadRagged(ElemConn)
        ElemConnCentroids = centroids[RElemConn]
        ElemConnCenterDist = np.sqrt((NodeCoords[:,0,None]-ElemConnCentroids[:,:,0])**2 + (NodeCoords[:,1,None]-ElemConnCentroids[:,:,1])**2 + (NodeCoords[:,2,None]-ElemConnCentroids[:,:,2])**2)
        kcenters = StiffnessFactor/ElemConnCenterDist
        Kvals_diag += np.nansum(kcenters,axis=1)
    if FaceCentered:
        faces = converter.solid2faces(NodeCoords,NodeConn)
        Faces = converter.faces2unique(faces)
        fcentroids = MeshUtils.Centroids(NodeCoords,Faces)
        fcentroids = np.append(fcentroids,[[np.nan,np.nan,np.nan]],axis=0)
        _,FConn = MeshUtils.getNodeNeighbors(NodeCoords,Faces)
        RFConn = MeshUtils.PadRagged(FConn)
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
        RNodeConn = MeshUtils.PadRagged(NodeConn+[[-1,-1,-1]],fillval=-1)
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
        RFaces = MeshUtils.PadRagged(Faces+[[-1,-1,-1]],fillval=-1)
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
        edges,corners = MeshUtils.DetectFeatures(NodeCoords,NodeConn)
        FixedNodes.update(edges)
        FixedNodes.update(corners)


    Nodes = set(range(len(NodeCoords))).difference(FixedNodes)
    if not NodeNeighbors: NodeNeighbors = MeshUtils.getNodeNeighbors(NodeCoords, NodeConn)
    NewCoords = copy.copy(NodeCoords)#[node for node in NodeCoords]
    OldCoords = copy.copy(NodeCoords)
    for it in range(iterate):        
        for i in Nodes:
            Ni = len(NodeNeighbors[i])
            if Ni > 0:
                NewCoords[i] = (1/Ni * np.sum([OldCoords[NodeNeighbors[i][j]] for j in range(len(NodeNeighbors[i]))],axis=0)).tolist() 
        OldCoords = copy.copy(NewCoords)
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
            edges,corners = MeshUtils.DetectFeatures(NodeCoords,NodeConn)
            FixedNodes.update(edges)
            FixedNodes.update(corners)
        NodeNeighbors,ElemConn = MeshUtils.getNodeNeighbors(NodeCoords,NodeConn)
        lens = np.array([len(n) for n in NodeNeighbors])
        r = MeshUtils.PadRagged(NodeNeighbors,fillval=-1)
        FreeNodes = list(set(range(len(NodeCoords))).difference(FixedNodes))
        ArrayCoords = np.vstack([NodeCoords,[np.nan,np.nan,np.nan]])
        
        for i in range(iterate):
            ElemNormals = MeshUtils.CalcFaceNormal(ArrayCoords[:-1],NodeConn)
            NodeNormals = MeshUtils.Face2NodeNormal(ArrayCoords[:-1],NodeConn,ElemConn,ElemNormals)
            Q = ArrayCoords[r]
            U = (1/lens)[:,None] * np.nansum(Q - ArrayCoords[:-1,None,:],axis=1)
            R = 1*(U - np.sum(U*NodeNormals,axis=1)[:,None]*NodeNormals)
            ArrayCoords[FreeNodes] += R[FreeNodes]

        NewCoords = ArrayCoords[:-1].tolist()
        return NewCoords

def GlobalLaplacianSmoothing(NodeCoords,NodeConn,FeatureNodes=[],FixedNodes=set(),FeatureWeight=1,BaryWeight=1/3):
    # Ji, Z., Liu, L. and Wang, G., 2005, December. A global laplacian 
    # smoothing approach with feature preservation. In Ninth International 
    # Conference on Computer Aided Design and Computer Graphics
    
    NodeNeighbors,ElemConn = MeshUtils.getNodeNeighbors(NodeCoords,NodeConn)
    
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
        octree = Octree.Surf2Octree(NewCoords,SurfConn)
    IntersectionPairs = Rays.SurfSelfIntersection(NewCoords,SurfConn,octree=octree)
    Intersected = np.unique(IntersectionPairs).tolist()
    
    count = 0
    while len(Intersected) > 0 and count < maxIter:
        print(count)
        _,ElemConn = MeshUtils.getNodeNeighbors(NewCoords, SurfConn)
        NeighborhoodElems = np.unique([e for i in (SurfConn[Intersected]).flatten() for e in ElemConn[i]])
        PatchConn = SurfConn[NeighborhoodElems]
        BoundaryEdges = converter.surf2edges(NewCoords,PatchConn) 

        NewCoords = np.asarray(LocalLaplacianSmoothing(NewCoords,PatchConn,2,FixedNodes=FixedNodes.union(set([n for edge in BoundaryEdges for n in edge]))))

        IntersectionPairs = Rays.SurfSelfIntersection(NewCoords,SurfConn,octree=octree)
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

def DelaunayFlips(NodeCoords,NodeConn,ElemNeighbors=None):
    NewConn = copy.copy(NodeConn)
    NodeCoords = np.array(NodeCoords)
    if ElemNeighbors==None: ElemNeighbors = MeshUtils.getElemNeighbors(NodeCoords,NodeConn)

    eps = 1e-12
    # Circumcenters
    Points = NodeCoords[np.array(NodeConn)]
    a = np.linalg.norm(Points[:,1]-Points[:,2],axis=1)
    b = np.linalg.norm(Points[:,2]-Points[:,0],axis=1)
    c = np.linalg.norm(Points[:,1]-Points[:,0],axis=1)
    wA = a**2 * (b**2 + c**2 - a**2)
    wB = b**2 * (c**2 + a**2 - b**2)
    wC = c**2 * (a**2 + b**2 - c**2)
    O = (wA[:,None]*Points[:,0] + wB[:,None]*Points[:,1] + wC[:,None]*Points[:,2])/(wA + wB + wC)[:,None]
    # Circumradii
    R = np.linalg.norm(O-Points[:,0,:],axis=1)
    # NeighborPoints = Points[ElemNeighbors]
    # NeighborDists = np.linalg.norm(NeighborPoints-O[:,None,None,:],axis=3) + eps
    # Flips = np.any(NeighborDists < R[:,None,None],axis=2)

    flips = 1
    # while flips > 0:
    # print(flips)
    flips = 0
    for i in range(len(NewConn)):
        for j in ElemNeighbors[i]:
            if len(set(ElemNeighbors[i]).intersection(ElemNeighbors[j])) > 0:
                # This condition checks if the flip will be legal
                continue
            if np.any(np.linalg.norm(Points[j]-O[i],axis=1) + eps < R[i]):
                flips+=1
                oldi = copy.copy(NewConn[i]); oldj = copy.copy(NewConn[j])
                Newi,Newj = FlipEdge(NodeCoords,NewConn,i,j)
                NewConn[i] = Newi; NewConn[j] = Newj
                ENi = []; ENj = []
                Si = set(Newi); Sj = set(Newj)
                for k in np.unique([ElemNeighbors[i] + ElemNeighbors[j]]):
                    if len(Si.intersection(NewConn[k])) == 2:
                        ENi.append(k)
                    if len(Sj.intersection(NewConn[k])) == 2:
                        ENj.append(k)
                if len(ENi) != 3 or len(ENj) != 3:
                    NewConn[i] = oldi; NewConn[j] = oldj
                    continue
                for k in np.unique([ElemNeighbors[i] + ElemNeighbors[j]]):
                    if i in ElemNeighbors[k]: ElemNeighbors[k].remove(i)
                    if j in ElemNeighbors[k]: ElemNeighbors[k].remove(j)
                    if len(Si.intersection(NewConn[k])) == 2:
                        ElemNeighbors[k].append(i)
                    if len(Sj.intersection(NewConn[k])) == 2:
                        ElemNeighbors[k].append(j)  

                Points[i] = NodeCoords[Newi]
                Points[j] = NodeCoords[Newj]
                a = np.linalg.norm(Points[i,1]-Points[i,2])
                b = np.linalg.norm(Points[i,2]-Points[i,0])
                c = np.linalg.norm(Points[i,1]-Points[i,0])
                wA = a**2 * (b**2 + c**2 - a**2)
                wB = b**2 * (c**2 + a**2 - b**2)
                wC = c**2 * (a**2 + b**2 - c**2)
                O[i] = (wA*Points[i,0] + wB*Points[i,1] + wC*Points[i,2])/(wA + wB + wC)
                R[i] = np.linalg.norm(O[i]-Points[i,0])

                a = np.linalg.norm(Points[j,1]-Points[j,2])
                b = np.linalg.norm(Points[j,2]-Points[j,0])
                c = np.linalg.norm(Points[j,1]-Points[j,0])
                wA = a**2 * (b**2 + c**2 - a**2)
                wB = b**2 * (c**2 + a**2 - b**2)
                wC = c**2 * (a**2 + b**2 - c**2)
                O[j] = (wA*Points[j,0] + wB*Points[j,1] + wC*Points[j,2])/(wA + wB + wC)
                R[j] = np.linalg.norm(O[j]-Points[j,0])

                # NewConn[i] = Newi; NewConn[j] = Newj
                ElemNeighbors[i] = ENi; ElemNeighbors[j] = ENj
                break

    return NewConn, ElemNeighbors

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

                    if not Rays.SegmentSegmentIntersection(Array[shared],Array[Newshared]):
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
    if not NodeNeighbors: NodeNeighbors = MeshUtils.getNodeNeighbors(NodeCoords,NodeConn)
    thinking = True
    iter = 0
    while thinking:
        iter += 1
        flips = 0
        # ElemEdges = np.array(converter.EdgesByElement(NewCoords,NewConn))
        # SortElemEdges = np.sort(ElemEdges,axis=2)

        # ElemEdgeVectors = NewCoords[ElemEdges[:,:,1]] - NewCoords[ElemEdges[:,:,0]]
        # ElemLengths = np.linalg.norm(ElemEdgeVectors,axis=2)
        # # Opposite angles corresponding to each edge
        # OppositeAngles = np.zeros(ElemLengths.shape)
        # OppositeAngles[:,0] = np.arccos(np.sum(ElemEdgeVectors[:,2]*-ElemEdgeVectors[:,1],axis=1)/(ElemLengths[:,1]*ElemLengths[:,2]))
        # OppositeAngles[:,1] = np.arccos(np.sum(ElemEdgeVectors[:,0]*-ElemEdgeVectors[:,2],axis=1)/(ElemLengths[:,0]*ElemLengths[:,2]))
        # OppositeAngles[:,2] = np.arccos(np.sum(ElemEdgeVectors[:,1]*-ElemEdgeVectors[:,0],axis=1)/(ElemLengths[:,1]*ElemLengths[:,0]))
        
        # AllEdges = SortElemEdges.reshape((SortElemEdges.shape[0]*SortElemEdges.shape[1],SortElemEdges.shape[2]))
        # AllAngles = OppositeAngles.reshape(SortElemEdges.shape[0]*SortElemEdges.shape[1])
        # AllLengths = ElemLengths.reshape(SortElemEdges.shape[0]*SortElemEdges.shape[1])

        # ElemIdxs = np.arange(len(NewConn))
        # ElemRef = np.vstack([ElemIdxs,ElemIdxs,ElemIdxs]).T.reshape(SortElemEdges.shape[0]*SortElemEdges.shape[1])
        # # Unique set of edges:
        # Edges,Idx,Inv = np.unique(AllEdges,axis=0,return_index=True,return_inverse=True)
        # Lengths = AllLengths[Idx]   # Length of edge in 'Edges'
        # # For each edge, contains the indices for the 2 adjacent elements
        # ConnectedElements = [[] for i in range(len(Edges))]
        # for i,I in enumerate(Inv): ConnectedElements[I].append(ElemRef[i])      # TODO: This might be a bottleneck
        # ConnectedElements = np.array(ConnectedElements)
        # Angles = [[] for i in range(len(Edges))]
        # for i,I in enumerate(Inv): Angles[I].append(AllAngles[i])     # Opposite angle corresponding to edge in 'Edges'

        
        Edges, EdgeConn, EdgeElem = converter.solid2edges(NewCoords,NewConn,return_EdgeConn=True,return_EdgeElem=True)
        UEdges, UIdx, UInv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
        UEdgeElem = np.asarray(EdgeElem)[UIdx]
        UEdgeConn = UInv[MeshUtils.PadRagged(EdgeConn)]
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

            if not Rays.SegmentSegmentIntersection(NewCoords[Edge],NewCoords[NewEdge]):
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

            # Update angles 
            # Note this could flip the relationship between the angle and the connected elements
            # but this doesn't actually matter here
            # v1 = NewCoords[NewEdge[0]] - NewCoords[Edge[0]]
            # v2 = NewCoords[NewEdge[1]] - NewCoords[Edge[0]]
            # v3 = NewCoords[NewEdge[0]] - NewCoords[Edge[1]]
            # v4 = NewCoords[NewEdge[1]] - NewCoords[Edge[1]]
            # Angles[k][0] = np.arccos(np.dot(v1,v2)/(np.prod([np.linalg.norm([v1,v2],axis=1)])))
            # Angles[k][1] = np.arccos(np.dot(v3,v4)/(np.prod([np.linalg.norm([v3,v4],axis=1)])))

            flips += 1
        # print(flips)
        if flips < len(NonDelaunay)-nskipped:
            thinking = True
            NodeNeighbors = MeshUtils.getNodeNeighbors(NewCoords, NewConn)    
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
        UEdgeConn = UInv[MeshUtils.PadRagged(EdgeConn)]
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
    edges,corners = MeshUtils.DetectFeatures(NewCoords,NewConn)
    if FixFeatures:
        FixedNodes.update(edges)
        FixedNodes.update(corners)
    FeatureRank = [2 if i in corners else 1 if i in edges else 0 for i in range(len(NewCoords))]
    
    while iter < iterate:
        iter += 1
        NodeNeighbors = MeshUtils.getNodeNeighbors(NewCoords,NewConn)
        ElemConn = MeshUtils.getElemConnectivity(NewCoords,NewConn)
        Edges, EdgeConn, EdgeElem = converter.solid2edges(NewCoords,NewConn,return_EdgeConn=True,return_EdgeElem=True,ReturnType=np.ndarray)
        UEdges, UIdx, UInv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
        UEdgeElem = np.asarray(EdgeElem)[UIdx]
        UEdgeConn = UInv[MeshUtils.PadRagged(EdgeConn)]
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
            OldNormals = MeshUtils.CalcFaceNormal(NewCoords,[NewConn[e] for e in (ElemConn[edge[0]]+ElemConn[edge[1]])])
            
            NewCoords[edge[0]] = newpoint
            NewCoords[edge[1]] = newpoint
            NewNormals = MeshUtils.CalcFaceNormal(NewCoords,[NewConn[e] for e in (ElemConn[edge[0]]+ElemConn[edge[1]])])
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
        NewCoords,NewConn = MeshUtils.DeleteDegenerateElements(NewCoords,NewConn,strict=True)
        if k == 0:
            break

    NewCoords,NewConn,_ = MeshUtils.DeleteDuplicateNodes(NewCoords,NewConn)

    if type(NewCoords) is np.ndarray: NewCoords = NewCoords.tolist()
    return NewCoords, NewConn

def TetOpt(NodeCoords, NodeConn, ElemConn=None, objective='eta', method='BFGS', p=2, FreeNodes='all', FixedNodes=set(), iterate=1):
    # Escobar, et al. 2003. “Simultaneous untangling and smoothing of tetrahedral meshes.”

    if FreeNodes == 'all': FreeNodes = set(range(len(NodeCoords)))
    if type(FreeNodes) is list: FreeNodes = set(FreeNodes)
    FreeNodes = FreeNodes.difference(FixedNodes)

    # if ElemConn is None: _,ElemConn = MeshUtils.getNodeNeighbors(NodeCoords,NodeConn)
    if ElemConn is None: ElemConn = MeshUtils.getElemConnectivity(NodeCoords,NodeConn)
    ArrayCoords = np.array(NodeCoords); ArrayConn = np.asarray(NodeConn)
    assert ArrayConn.dtype != 'O', 'Input mesh must be purely tetrahedral.'

    eps = 100*np.finfo(float).eps
    # LooseNodes = set([i for i in range(len(NodeCoords)) if len(ElemConn[i]) == 0])
    # FreeNodes = set(range(len(NodeCoords))).difference(FixedNodes).difference(LooseNodes)
    
    # W = np.array([[1, 1/2, 1/2],[0, np.sqrt(3)/2, np.sqrt(3)/6],[0, 0, np.sqrt(2)/np.sqrt(3)]])
    Winv = np.array([
                [ 1.        , -0.57735027, -0.40824829],
                [ 0.        ,  1.15470054, -0.40824829],
                [ 0.        ,  0.        ,  1.22474487]])
    def tet_jacobians(ArrayCoords,LocalConn):
        # Escobar, et al. 2003. “Simultaneous untangling and smoothing of tetrahedral meshes.”
        Points = ArrayCoords[LocalConn]
        x = Points[:,:,0]; y = Points[:,:,1]; z = Points[:,:,2]

        A = np.swapaxes(np.swapaxes(np.array([
            [x[:,1]-x[:,0], x[:,2]-x[:,0], x[:,3]-x[:,0]],
            [y[:,1]-y[:,0], y[:,2]-y[:,0], y[:,3]-y[:,0]],
            [z[:,1]-z[:,0], z[:,2]-z[:,0], z[:,3]-z[:,0]],
            ]),0,2),1,2)

        # Affine map that takes an equilateral tetrahedron Ti(v0,v1,v2,v3) to a given 
        # tetrahedron T(x0,x1,x2,x3) is: x=Aw^-v1+x0. It's jacobian is S=AW^-1

        S = np.matmul(A,Winv)
        return S
    def dS(i,var,ArrayCoords,LocalConn):

        idx = np.where(LocalConn == i)[1]
        tetidx = np.arange(len(LocalConn))
        x = np.zeros((len(LocalConn),4)); y = np.zeros((len(LocalConn),4)); z = np.zeros((len(LocalConn),4))
        if var == 'x':
            x[tetidx,idx] = 1
        elif var == 'y':
            y[tetidx,idx] = 1
        elif var == 'z':
            z[tetidx,idx] = 1
        else:
            raise Exception('Invalid var.')

        dA = np.swapaxes(np.swapaxes(np.array([
            [x[:,1]-x[:,0], x[:,2]-x[:,0], x[:,3]-x[:,0]],
            [y[:,1]-y[:,0], y[:,2]-y[:,0], y[:,3]-y[:,0]],
            [z[:,1]-z[:,0], z[:,2]-z[:,0], z[:,3]-z[:,0]],
            ]),0,2),1,2)
        
        dS = np.matmul(dA,Winv)
        return dS

    
    if objective == 'eta':
        def eta(x,i,LocalConn):
            # x is the new coordinate of the node with index i
            ArrayCoords[i] = x
            S = tet_jacobians(ArrayCoords,LocalConn)
            sigma = np.linalg.det(S)
            if min(sigma) >= eps:
                delta = 0
                h = sigma
            else:
                delta = np.sqrt(eps*(eps-min(sigma)))
                h = 1/2 * (sigma + np.sqrt(sigma**2 + 4*delta**2))
            e = np.linalg.norm(S,'fro',axis=(1,2))**2 / (3*h**(2/3))
            return e, delta
        def K_eta(x,i,LocalConn): 
            K = np.linalg.norm(eta(x,i,LocalConn)[0],ord=p)
            return K
        def grad_K_eta(x,i,LocalConn):
            
            e,delta = eta(x,i,LocalConn)
            S = tet_jacobians(ArrayCoords,LocalConn)
            sigma = np.linalg.det(S)
            da_S = np.array([dS(i,a,ArrayCoords,LocalConn) for a in ['x','y','z']])
            da_sigma = np.array([np.linalg.det(S) * np.trace(np.matmul(np.linalg.inv(S), da_S[j]),axis1=1,axis2=2) for j in range(3)])

            da_eta = (2*e)*np.array([
                np.trace(np.matmul(np.swapaxes(da_S[j],1,2),S),axis1=1,axis2=2)/np.linalg.norm(S,'fro',axis=(1,2))**2 - 
                da_sigma[j]/(3*np.sqrt(sigma**2 + 4*delta**2))
                for j in range(3)])

            # grad_K = np.linalg.norm(da_eta,axis=1,ord=p)
            grad_K = np.array([1/p * sum(e**p)**(1/p-1) * p*sum(e**(p-1)*da_eta[j]) for j in range(3)])

            return grad_K
        
        ifun = lambda x,i,LocalConn : K_eta(x,i,LocalConn)
        igrad = lambda x,i,LocalConn : grad_K_eta(x,i,LocalConn)
    elif objective == 'kappa':
        def kappa(x,i,LocalConn):
            # x is the new coordinate of the node with index i
            ArrayCoords[i] = x
            S = tet_jacobians(ArrayCoords,LocalConn)
            sigma = np.linalg.det(S)
            Sigma = sigma[:,None,None]*np.linalg.inv(S)
            if min(sigma) >= eps:
                delta = 0
                h = sigma
            else:
                delta = np.sqrt(eps*(eps-min(sigma)))
                h = 1/2 * (sigma + np.sqrt(sigma**2 + 4*delta**2))

            k = np.linalg.norm(S,'fro',axis=(1,2)) * np.linalg.norm(Sigma,'fro',axis=(1,2)) / (3*h)
            return k, delta
        def K_kappa(x,i,LocalConn): return np.linalg.norm(kappa(x,i,LocalConn)[0],ord=p)
        def grad_K_kappa(x,i,LocalConn):
            k,delta = kappa(x,i,LocalConn)
            S = tet_jacobians(ArrayCoords,LocalConn)
            sigma = np.linalg.det(S)
            Sigma = sigma[:,None,None]*np.linalg.inv(S)

            da_S = np.array([dS(i,a,ArrayCoords,LocalConn) for a in ['x','y','z']])
            da_sigma = np.array([np.linalg.det(da_S[j]) for j in range(3)])
            da_Sigma = np.array([sigma[:,None,None]*np.linalg.inv(da_S[j]) + da_sigma[j,:,None,None]*np.linalg.inv(S) for j in range(3)])
            da_kappa = (k)*np.array([
                np.trace(np.matmul(np.swapaxes(da_S[j],1,2), S),axis1=1,axis2=2)/np.linalg.norm(S,'fro',axis=(1,2))**2 +
                np.trace(np.matmul(np.swapaxes(da_Sigma[j],1,2), Sigma),axis1=1,axis2=2)/np.linalg.norm(S,'fro',axis=(1,2))**2 - 
                da_sigma[j]/np.sqrt(sigma**2 + 4*delta**2)
                for j in range(3)])

            grad_K = np.linalg.norm(da_kappa,axis=1,ord=p)
            return grad_K

        ifun = lambda x,i,LocalConn : K_kappa(x,i,LocalConn)
        igrad = lambda x,i,LocalConn : grad_K_kappa(x,i,LocalConn)

    else:
        raise Exception('Invalid objective. Must be "eta" or "kappa".')
    for iter in range(iterate):
        for i in FreeNodes:
            LocalConn = ArrayConn[ElemConn[i]]
            if len(LocalConn) == 0:
                continue
            fun = lambda x : ifun(x,i,LocalConn)
            grad = lambda x : igrad(x,i,LocalConn)
            x0 = copy.copy(ArrayCoords[i])
            # out = minimize(fun,x0,jac=grad,method='BFGS')
            out = minimize(fun,x0,jac=grad,method=method)
            ArrayCoords[i] = out.x
    
    NewCoords = ArrayCoords.tolist()
    return NewCoords

def PatchHoles(NodeCoords, SurfConn):
    ## Work in Progress
    edges = np.array(converter.surf2edges(NodeCoords, SurfConn))
    ignore = set()
    for i,edge in edges:
        if i in ignore:
            continue
        ignore.add(i)
        where0 = np.where(edge[0]==edges)[0]
        where1 = np.where(edge[1]==edges)[0]
        if len(where0) != 2 or len(where1) != 2:
            # Can only patch edges if both nodes in the edge are only shared by one other edge
            continue

        


# %%
