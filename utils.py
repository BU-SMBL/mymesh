# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 18:31:03 2021

@author: toj
"""

import numpy as np
import sys, warnings, copy, time, itertools
from . import converter, rays, octree, improvement, quality

def getNodeNeighbors(NodeCoords,NodeConn,ElemType='auto'):
    """
    getNodeNeighbors Gives the connected nodes for each node in the mesh
    TODO: This should probably split into getNodeNeighbors and getElemConn -  the only shared
    operation is solid2edges (edges could be an optional input argument to each)

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.

    Returns
    -------
    NodeNeighbors : list
        List of neighboring nodes for each node in NodeCoords.
    ElemConn : list
        List of elements connected to each node for each node in NodeCoords.
    """

    # NodeNeighbors = [set() for i in range(len(NodeCoords))]    # Neighboring nodes for each vertex
    # ElemConn = [set() for i in range(len(NodeCoords))]         # Connected elements for each vertex 
    # for i in range(len(NodeConn)):
    #     edges = converter.solid2edges(NodeCoords, [NodeConn[i]], ElemType=ElemType)
    #     for edge in edges:
    #         for j in edge:
    #             NodeNeighbors[j].add(edge[edge.index(j)-1])
    #             ElemConn[j].add(i)
    #### 
    Edges,EdgeElem = converter.solid2edges(NodeCoords,NodeConn,return_EdgeElem=True, ElemType=ElemType, ReturnType=np.ndarray)
    # NodeNeighbors = [set() for i in range(len(NodeCoords))] 
    # ElemConn = [set() for i in range(len(NodeCoords))]         # Connected elements for each vertex 
    # for i in range(len(Edges)):
    #     NodeNeighbors[Edges[i][0]].add(Edges[i][1])
    #     NodeNeighbors[Edges[i][1]].add(Edges[i][0])
    #     ElemConn[Edges[i][0]].add(EdgeElem[i])
    #     ElemConn[Edges[i][1]].add(EdgeElem[i])
    # NodeNeighbors = [list(s) for s in NodeNeighbors]  
    # ElemConn = [list(s) for s in ElemConn] 
    ####
    
    UEdges,idx,inv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
    NotInMesh = set(range(len(NodeCoords))).difference(np.unique(UEdges))
    Neighbors = np.append(UEdges.flatten(order='F'),np.repeat(-1,len(NotInMesh)))
    Idx = np.append(np.fliplr(UEdges).flatten(order='F'),list(NotInMesh))
    arg = Idx.argsort()

    key_func = lambda x : x[0]
    NodeNeighbors = [[z for y,z in x[1] if z != -1] for x in itertools.groupby(zip(Idx[arg],Neighbors[arg]), key_func)]
    

    return NodeNeighbors#,ElemConn               

def getElemConnectivity(NodeCoords,NodeConn,ElemType='auto'):
    Edges,EdgeElem = converter.solid2edges(NodeCoords,NodeConn,return_EdgeElem=True, ElemType=ElemType, ReturnType=np.ndarray)
    NodeNeighbors = [set() for i in range(len(NodeCoords))] 
    ElemConn = [set() for i in range(len(NodeCoords))]         # Connected elements for each vertex 
    for i in range(len(Edges)):
        ElemConn[Edges[i][0]].add(EdgeElem[i])
        ElemConn[Edges[i][1]].add(EdgeElem[i])
    ElemConn = [list(s) for s in ElemConn] 
    return ElemConn

def getNodeNeighborhood(NodeCoords,NodeConn,nRings):
    """
    getNodeNeighborhood Gives the connected nodes in an n ring neighborhood 
    for each node in the mesh

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.
    nRings : int
        Number of rings to include.

    Returns
    -------
    NodeNeighborhoods : list
        List of neighboring nodes in an n ring neighborhood around each node in 
        NodeCoords.
    """
    
    NodeNeighbors = getNodeNeighbors(NodeCoords,NodeConn)
    NodeNeighborhoods = [[j for j in NodeNeighbors[i]] for i in range(len(NodeNeighbors))]
    if nRings == 1:
        return NodeNeighborhoods
    else:
        for n in range(nRings-1):
            # For each ring, loop through and add the neighbors of the nodes in the neighborhood to the neighborhood
            for i in range(len(NodeNeighborhoods)):
                temp = [j for j in NodeNeighborhoods[i]]
                for j in temp:
                    for k in range(len(NodeNeighbors[j])):
                        if (NodeNeighbors[j][k] not in NodeNeighborhoods[i]) and (NodeNeighbors[j][k] != i):
                            NodeNeighborhoods[i].append(NodeNeighbors[j][k])
    return NodeNeighborhoods
            
def getNodeNeighborhoodByRadius(NodeCoords,NodeConn,Radius):
    """
    getNodeNeighborhoodByRadius Gives the connected nodes in a neighborhood 
    with a specified radius for each node in the mesh.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.
    nRings : int
        Number of rings to include.

    Returns
    -------
    NodeNeighborhoods : list
        List of neighboring nodes in an neighborhood around each node in 
        NodeCoords with the neighborhoods specified by a radius.
    """
    
    NodeNeighbors = getNodeNeighbors(NodeCoords,NodeConn)
    NodeNeighborhoods = [[] for i in range(len(NodeNeighbors))]
    for i in range(len(NodeNeighborhoods)):
        thisNode = NodeCoords[i]
        thinking = True
        NodeNeighborhoods[i] = [j for j in NodeNeighbors[i] if \
                    np.sqrt((thisNode[0]-NodeCoords[j][0])**2 + \
                            (thisNode[1]-NodeCoords[j][1])**2 + \
                                (thisNode[2]-NodeCoords[j][2])**2) <= Radius]
        while thinking:
            thinking = False
            temp = [j for j in NodeNeighborhoods[i]]
            for j in temp:
                for k in range(len(NodeNeighbors[j])):                    
                    if (NodeNeighbors[j][k] not in NodeNeighborhoods[i]) and (NodeNeighbors[j][k] != i):
                        otherNode = NodeCoords[NodeNeighbors[j][k]]
                        if np.sqrt((thisNode[0]-otherNode[0])**2 + (thisNode[1]-otherNode[1])**2 + (thisNode[2]-otherNode[2])**2) <= Radius:
                            thinking = True
                            NodeNeighborhoods[i].append(NodeNeighbors[j][k])
    return NodeNeighborhoods   

def getElemNeighbors(NodeCoords,NodeConn,mode='face',ElemConn=None):
    """
    getElemNeighbors Get list of neighboring elements for each element in the mesh.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.
    mode : str, optional
        Neighbor mode, will determine what type of connectivity constitutes an element
        neighbor, by default 'face'.
        'node' : Any elements that share at least one node are considered neighbors. TODO: Not currently implemeneted.
        'edge' : Any elements that share an edge are considered neighbors.
        'face' : Any elements that share a face are considered neighbors. NOTE that in surface meshes, no elements share faces.
    ElemConn : list, optional
        Node-Element connectivity of the mesh as obtained by getNodeNeighbors.
        If supplied, won't require an additional call to getNodeNeighbors.
        Only relevant if mode = 'node', by default None.

    Returns
    -------
    ElemNeighbors : list
        List of element neighbors. For each element, there is a list of the
        indices of the neighboring elements.
    """
    # Get Element neighbors 
    ElemNeighbors = [set() for i in range(len(NodeConn))]
    # if mode == 'node':
        # SetConn = [set(elem) for elem in NodeConn]
        # for i,elem in enumerate(SetConn):
        #     ElemNeighbors[i] = [j for j,s in enumerate(SetConn) if len(elem.intersection(s)) == 2]
        
        # if not ElemConn: _,ElemConn = getNodeNeighbors(NodeCoords,NodeConn)
        # SetElemConn = [set(E) for E in ElemConn]

        # for i,e in enumerate(NodeConn):
        #     ElemNeighbors[i] = 
    if mode=='node':
        ElemConn = getElemConnectivity(NodeCoords,NodeConn)
        for i,elem in enumerate(NodeConn):
            for n in elem:
                ElemNeighbors[i].update(ElemConn[n])
        ElemNeighbors = [list(s) for s in ElemNeighbors] 
    elif mode=='edge':
        
        Edges,EdgeConn,EdgeElem = converter.solid2edges(NodeCoords,NodeConn,return_EdgeElem=True,return_EdgeConn=True,ReturnType=np.ndarray)

        UEdges,idx,inv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
        inv = np.append(inv,-1)
        UEdgeConn = inv[PadRagged(EdgeConn)]
        UEdgeElem = EdgeElem[idx]

        EdgeElemConn = np.nan*(np.ones((len(UEdges),2))) # Elements attached to each edge
        r = np.repeat(np.arange(len(UEdgeConn))[:,None],UEdgeConn.shape[1],axis=1)
        EECidx = (UEdgeElem[UEdgeConn] == r).astype(int)
        EdgeElemConn[UEdgeConn,EECidx] = r
        # EdgeElemConn = EdgeElemConn.astype(int)

        for i in range(len(EdgeElemConn)):
            if not any(np.isnan(EdgeElemConn[i])):
                ElemNeighbors[int(EdgeElemConn[i][0])].add(int(EdgeElemConn[i][1]))
                ElemNeighbors[int(EdgeElemConn[i][1])].add(int(EdgeElemConn[i][0]))
        ElemNeighbors = [list(s) for s in ElemNeighbors] 

    elif mode=='face':
        faces,faceconn,faceelem = converter.solid2faces(NodeCoords,NodeConn,return_FaceConn=True,return_FaceElem=True)
        # Pad Ragged arrays in case of mixed-element meshes
        Rfaces = PadRagged(faces)
        Rfaceconn = PadRagged(faceconn)
        # Get all unique element faces (accounting for flipped versions of faces)
        _,idx,inv = np.unique(np.sort(Rfaces,axis=1),axis=0,return_index=True,return_inverse=True)
        RFaces = Rfaces[idx]
        FaceElem = faceelem[idx]
        RFaces = np.append(RFaces, np.repeat(-1,RFaces.shape[1])[None,:],axis=0)
        inv = np.append(inv,-1)
        RFaceConn = inv[Rfaceconn] # Faces attached to each element
        # Face-Element Connectivity
        FaceElemConn = np.nan*(np.ones((len(RFaces),2)))


        FECidx = (FaceElem[RFaceConn] == np.repeat(np.arange(len(NodeConn))[:,None],RFaceConn.shape[1],axis=1)).astype(int)
        FaceElemConn[RFaceConn,FECidx] = np.repeat(np.arange(len(NodeConn))[:,None],RFaceConn.shape[1],axis=1)
        FaceElemConn = [[int(x) if not np.isnan(x) else x for x in y] for y in FaceElemConn[:-1]]

        for i in range(len(FaceElemConn)):
            if np.any(np.isnan(FaceElemConn[i])): continue
            ElemNeighbors[FaceElemConn[i][0]].add(FaceElemConn[i][1])
            ElemNeighbors[FaceElemConn[i][1]].add(FaceElemConn[i][0])
        ElemNeighbors = [list(s) for s in ElemNeighbors] 
    else:
        raise Exception('Invalid mode. Must be "edge" or "face".')

    return ElemNeighbors

def getConnectedNodes(NodeCoords,NodeConn,NodeNeighbors=None,BarrierNodes=set()):
    """
    getConnectedNodes Identifies groups of connected nodes. For a fully 
    connected mesh, a single region will be identified

    Parameters
    ----------
    NodeCoords : list of lists
        List of nodal coordinates.
    NodeConn : list of lists
        Nodal connectivity list.
    NodeNeighbors : list, optional
        List of neighboring nodes for each node in NodeCoords. The defau lt is 
        None. If no value is provided, it will be computed with getNodeNeighbors

    Returns
    -------
    NodeRegions : list of sets
        Each set in the list contains a region of connected nodes.
    """
    
    NodeRegions = []
    if not NodeNeighbors: NodeNeighbors = getNodeNeighbors(NodeCoords,NodeConn)
    if len(BarrierNodes) > 0:
        NodeNeighbors = [[] if i in BarrierNodes else n for i,n in enumerate(NodeNeighbors)]
    NeighborSets = [set(n) for n in NodeNeighbors]
    AllNodes = set(range(len(NodeCoords)))
    DetachedNodes = AllNodes.difference(set(np.unique(NodeConn)))
    todo = AllNodes.difference(DetachedNodes).difference(BarrierNodes)
    while len(todo) > 0:
        seed = todo.pop()
        region = {seed}
        new = {seed}
        nOld = 0
        nCurrent = len(region)
        k = 0
        while nOld != nCurrent:
            k += 1
            nOld = nCurrent
            old = copy.copy(new)
            new = set()
            for i in old:
                new.update(NeighborSets[i])
            new.difference_update(region)
            region.update(new)
            nCurrent = len(region)
        todo.difference_update(region)
        NodeRegions.append(region)
        
    return NodeRegions  

def getConnectedElements(NodeCoords,NodeConn,ElemNeighbors=None,mode='edge',BarrierElems=set()):
    """
    getConnectedElements Identifies groups of connected nodes. For a fully 
    connected mesh, a single region will be identified

    Parameters
    ----------
    NodeCoords : list of lists
        List of nodal coordinates.
    NodeConn : list of lists
        Nodal connectivity list.
    ElemNeighbors : list, optional
        List of neighboring elements for each element in NodeConn. The default is 
        None. If no value is provided, it will be computed with getNodeNeighbors
    mode : str, optional
        Connectivity method to be used for getElemNeighbors. The default is 'edge'.
    BarrierElems : set, optional
        Set of barrier elements that the connected region cannot move past. 
        They can be included in a region, but will not connect to their neighbors
        

    Returns
    -------
    ElemRegions : list of sets
        Each set in the list contains a region of connected nodes.
    """
    # warnings.warn('getConnectedElements is still under development.')
    ElemRegions = []
    if not ElemNeighbors: ElemNeighbors = getElemNeighbors(NodeCoords,NodeConn,mode=mode)
    if len(BarrierElems) > 0:
        ElemNeighbors = [[] if i in BarrierElems else e for i,e in enumerate(ElemNeighbors)]
    NeighborSets = [set(n) for n in ElemNeighbors]

    todo = set(range(len(NodeConn))).difference(BarrierElems)
    while len(todo) > 0:
        seed = todo.pop()
        region = {seed}
        new = {seed}
        nOld = 0
        nCurrent = len(region)
        k = 0
        while nOld != nCurrent:
            k += 1
            nOld = nCurrent
            old = copy.copy(new)
            new = set()
            for i in old:
                new.update(NeighborSets[i])
            new.difference_update(region)
            region.update(new)
            nCurrent = len(region)
        todo.difference_update(region)
        ElemRegions.append(region)
        
    
    return ElemRegions  

def SurfElemNeighbors(NodeCoords,SurfConn):
    warnings.warn('Deprecation Warning - Use getElemNeighbors instead')
    ElemNeighbors = [[] for i in range(len(SurfConn))]
    NodeConnList = np.reshape(SurfConn,np.array(SurfConn).size)
    NodeConnArray = np.array(SurfConn)
    dims = NodeConnArray.shape
    
    # An element is neighboring another element when two nodes are shared
    for i in range(len(SurfConn)):        
        bool0 = (NodeConnArray[i,0] == NodeConnList)*1
        bool1 = (NodeConnArray[i,1] == NodeConnList)*1
        bool2 = (NodeConnArray[i,2] == NodeConnList)*1
        
        ElemNeighbors[i] = np.where(np.sum(np.reshape(bool0+bool1+bool2,dims),axis=1)==2)[0].tolist()
    return ElemNeighbors

def Centroids(NodeCoords,NodeConn):
    """
    Centroids calculate element centroids.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.

    Returns
    -------
    centroids : list
        list of element centroids.
    """
    if len(NodeConn) == 0:
        return []
    ArrayCoords = np.vstack([NodeCoords,[np.nan,np.nan,np.nan]])
    R = PadRagged(NodeConn,fillval=-1)
    Points = ArrayCoords[R]
    centroids = np.nanmean(Points,axis=1)
    return centroids
    
def CalcFaceNormal(NodeCoords,SurfConn):
    """
    CalcFaceNormal Calculates normal vectors on the faces of a triangular 
    surface mesh. Asumes triangles are in counter-clockwise when viewed from the outside

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    SurfConn : list
        Nodal connectivity list of a triangular surface mesh.

    Returns
    -------
    ElemNormals list
        List of element surface normals .

    """
    # def func(elem):
    #     p0 = NodeCoords[elem[0]]
    #     p1 = NodeCoords[elem[1]]
    #     p2 = NodeCoords[elem[2]]
        
    #     U = np.subtract(p1,p0)
    #     V = np.subtract(p2,p0)
        
    #     Nx = U[1]*V[2] - U[2]*V[1]
    #     Ny = U[2]*V[0] - U[0]*V[2]
    #     Nz = U[0]*V[1] - U[1]*V[0]
    #     d = np.sqrt(Nx**2+Ny**2+Nz**2)
    #     return [Nx/d,Ny/d,Nz/d]
    # ElemNormals = [func(elem) for elem in SurfConn]

    ArrayCoords = np.append(NodeCoords,[[np.nan,np.nan,np.nan]],axis=0)
    points = ArrayCoords[PadRagged(SurfConn)]

    U = points[:,1,:]-points[:,0,:]
    V = points[:,2,:]-points[:,0,:]
    Nx = U[:,1]*V[:,2] - U[:,2]*V[:,1]
    Ny = U[:,2]*V[:,0] - U[:,0]*V[:,2]
    Nz = U[:,0]*V[:,1] - U[:,1]*V[:,0]
    N = np.vstack([Nx,Ny,Nz]).T
    d = np.linalg.norm(N,axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ElemNormals = (N/d[:,None]).tolist()

    return ElemNormals

def Face2NodeNormal(NodeCoords,NodeConn,ElemConn,ElemNormals,method='Angle'):
    """
    Face2NodeNormal Calculate node normal vectors based on the element face normals

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.
    ElemConn : list
        List of elements connected to each node.
    ElemNormals : list
        List of element normal vectors.
    method : str, optional
        Method used to determine node normals. The default is 'Angle'.
        Angle - performs an angle weighted average of connected element normals
        Average - performs a simple averaging of connected element normals
        MostVisible - Determines the most visible normal - Aubry et al. 2007
        MostVisible_Loop - Non-vectorized version of MostVisible, slower but more readable
        MostVisible_Iter - Iterative method for determining the most visible normal - Aubry et al. 2007

    Returns
    -------
    NodeNormals : list
        Unit normal vectors for each node.

    """
    
    if (method == 'Angle') or (method == 'angle'):
        # Based on: Grit Thürrner & Charles A. Wüthrich (1998)
        # Perform angle weighted average to compute vertex normals
        # Calculate the angles to use as weight

        # Cast ElemConn into a rectangular matrix
        # Warning: This code is very vectorized - it might be difficult to debug
        NodeSet = np.unique(PadRagged(NodeConn,fillval=-1))
        if -1 in NodeSet:
            NodeSet = np.delete(NodeSet,0)
        ArrayCoords = np.vstack([NodeCoords,[np.nan,np.nan,np.nan]])
        R = PadRagged(ElemConn,fillval=-1)[NodeSet]
        Mask0 = (R>=0).astype(int)
        Masknan = Mask0.astype(float)
        Masknan[Mask0 == 0] = np.nan 
        Ns = np.vstack([ElemNormals,[np.nan,np.nan,np.nan]])[R]
        # elemlens = [len(e) for e in NodeConn]
        # RNodeConn = -1*np.ones([len(NodeConn),max(elemlens)],dtype=int)
        # for i,elem in enumerate(NodeConn): RNodeConn[i,:elemlens[i]] = elem
        RNodeConn = PadRagged(NodeConn,fillval=-1)
        ArrayConn = np.vstack([RNodeConn,-1*np.ones((1,RNodeConn.shape[1]),dtype=int)])
        IncidentNodes = ArrayConn[R]
        x = (ArrayCoords[IncidentNodes]-ArrayCoords[NodeSet,None,None,:])
        x[np.all(x==[0,0,0],axis=-1)] = np.nan
        # For each node and for each incident element on that node, dot product of the two edges of the element that meet at the node
        dots = np.sum((np.nanprod(x,axis=2)*Mask0[:,:,None]),axis=2)
        # For each node and for each incident element on that node, the product of the norms of the two edges of the element that meet at the node
        norms = np.nanprod(np.linalg.norm(x,axis=3),axis=2)
        # cos(alpha) = dot(u,v)/(norm(u)*norm(v))
        cosAlpha = dots/norms
        alpha = np.arccos(cosAlpha)*Masknan

        sumAlphaN = np.nansum(alpha[:,:,None]*Ns,axis=1)
        NodeNormals = np.nan*np.ones_like(NodeCoords)
        NodeNormals[NodeSet] = sumAlphaN/np.linalg.norm(sumAlphaN,axis=1)[:,None]

    elif (method == 'Average') or (method == 'average') or (method == 'none') or (method == None):
        # Cast ElemConn into a rectangular matrix
        NodeSet = np.unique(PadRagged(NodeConn,fillval=-1))
        R = PadRagged(ElemConn,fillval=-1)[NodeSet]
        Ns = np.array(ElemNormals+[[np.nan,np.nan,np.nan]])[R]
        NodeNormals = np.nan*np.ones_like(NodeCoords)
        NodeNormals[NodeSet] = np.nanmean(Ns,axis=1)
        NodeNormals[NodeSet] = (NodeNormals[NodeSet]/np.linalg.norm(NodeNormals[NodeSet],axis=1)[:,None]).tolist()
        
    elif method == 'MostVisible':
        
        # Note: this code uses dot(Ni,Nj) as a surrogate for radius; since Ni,Nj are both unit vectors
        # cos(theta) = dot(Ni,Nj) -> theta = arccos(dot(Ni,Nj)). Since arccos is a monotonically 
        # decreasing function, if dot(Ni,Nj) < dot(Ni,Nk), then rij > rik
        eps = -1e-8
        NodeSet = np.unique(PadRagged(NodeConn,fillval=-1))
        if -1 in NodeSet:
            NodeSet = np.delete(NodeSet,0)

        R = PadRagged(ElemConn,fillval=-1)
        Ns = np.vstack([ElemNormals,[np.nan,np.nan,np.nan]])[R]

        # 2 Point Circles
        scalmin = -1
        Combos2 = Ns[NodeSet][:,np.array(list(itertools.combinations(range(Ns.shape[1]),2)))]
        Nb = np.sum(Combos2,axis=2)
        Nb = Nb/np.linalg.norm(Nb,axis=2)[:,:,None]
        scal2 = np.sum(Nb * Combos2[:,:,0,:],axis=2)

        # 3 Point Circles
        Combos3 = Ns[NodeSet][:,np.array(list(itertools.combinations(range(Ns.shape[1]),3)))]
        Ni = Combos3[:,:,0,:]
        Nj = Combos3[:,:,1,:]
        Nk = Combos3[:,:,2,:]
        denom = 2*np.linalg.norm(np.cross(Ni-Nk,Nj-Nk),axis=2)**2
        with np.errstate(divide='ignore', invalid='ignore'):
            Nc = np.cross(((np.linalg.norm(Ni-Nk,axis=2)**2)[:,:,None] * (Nj-Nk)) - (np.linalg.norm(Nj-Nk,axis=2)**2)[:,:,None] * (Ni-Nk),
            np.cross(Ni-Nk,Nj-Nk))/denom[:,:,None] + Nk
            Nc = Nc/np.linalg.norm(Nc,axis=2)[:,:,None]
            scal3 = np.sum(Nc*Ni,axis=2)

            Nc[scal3<0] = -Nc[scal3<0]
            scal3[scal3<0] = -scal3[scal3<0]

        scal23 = np.hstack([scal2,scal3])
        Nbc = np.hstack([Nb,Nc])
        check = np.any((np.einsum('lij,ljk->lik', Nbc, np.swapaxes(Ns[NodeSet],1,2)) - scal23[:,:,None]) < eps,axis=2)
        scal23[check] = scalmin

        # Indices of the smallest radius that contains all points
        Idx = scal23 == np.nanmax(scal23,axis=1)[:,None]
        # In case of duplicates, only taking the first one
        newIdx = np.zeros_like(Idx)
        newIdx[np.arange(len(Idx)), Idx.argmax(axis=1)] = Idx[np.arange(len(Idx)), Idx.argmax(axis=1)]

        NodeNormals = np.nan*np.ones_like(NodeCoords)
        NodeNormals[NodeSet] = Nbc[newIdx]

    else:
        NodeNormals = [[] for i in range(len(NodeCoords))]      # Normal vectors for each vertex
        NodeSet = {n for elem in NodeConn for n in elem}
        for i in range(len(NodeCoords)):
            if i not in NodeSet:
                NodeNormals[i] = [np.nan,np.nan,np.nan]
                continue
            angles = [0 for j in range(len(ElemConn[i]))]
            elemnormals = [np.array(ElemNormals[elem]) for elem in ElemConn[i]]
            if method == 'angle_old':
                # Based on: Grit Thürrner & Charles A. Wüthrich (1998)
                # Perform angle weighted average to compute vertex normals
                # Calculate the angles to use as weights
                for j in range(len(ElemConn[i])):  
                    # Loop through the neighboring elements for this node
                    n1 = NodeCoords[i]  # Coordinates of this node
                    n2n3 = []           
                    for k in range(len(NodeConn[ElemConn[i][j]])):  
                         # Loop through each node in that neighboring element
                        if NodeConn[ElemConn[i][j]][k] != i:
                            n2n3.append(NodeCoords[NodeConn[ElemConn[i][j]][k]])
                    n2 = n2n3[0]
                    n3 = n2n3[1]
                    # Using the dot product to calculate the angle of the neighboring element at the current node (Node i)
                    a = np.subtract(n3,n1)   # Vector a
                    b = np.subtract(n2,n1)   # Vector b
                    angles[j] = np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                top = np.sum([np.multiply(angles[j],elemnormals[j]) for j in range(len(angles))],0)
                bottom = np.linalg.norm(top)
                if bottom == 0:
                    NodeNormals[i] = [np.nan,np.nan,np.nan]
                else:
                    NodeNormals[i] = (top/bottom).tolist()

            elif method == 'MostVisible_Loop':
                
                # This is kept for readability; 'MostVisible' is a vectorized equivalent that performs significantly faster
                
                # Note: this code uses dot(Ni,Nj) as a surrogate for radius; since Ni,Nj are both unit vectors
                # cos(theta) = dot(Ni,Nj) -> theta = arccos(dot(Ni,Nj)). Since arccos is a monotonically 
                # decreasing function, if dot(Ni,Nj) < dot(Ni,Nk), then rij > rik
                eps = -1e-8
                scalmin = -1
                C = [np.nan,np.nan,np.nan]
                for ii in range(len(elemnormals)-1):
                    # Check the 2 point circles
                    Ni = np.array(elemnormals[ii])
                    for j in range(ii+1,len(elemnormals)):
                        Nj = np.array(elemnormals[j])
                        Nb = Ni+Nj
                        Nb = Nb/np.linalg.norm(Nb)
                        scal = np.dot(Nb,Ni)
                        if scal < scalmin:      
                            pass
                        elif any((np.dot(Nl,Nb) - scal) < eps for Nl in elemnormals):
                            pass
                        else:
                            C = Nb.tolist()
                            scalmin = scal
                for ii in range(len(elemnormals)-2): 
                    # Check the 3 point circles
                    Ni = elemnormals[ii]
                    for j in range(ii+1,len(elemnormals)-1):
                        Nj = elemnormals[j]
                        for k in range(j+1,len(elemnormals)):
                            Nk = elemnormals[k]
                            # denom = (Ni[0]-Nj[0])*(Ni[1]-Nk[1]) - (Ni[0]-Nk[0])*(Ni[1]-Nj[1])
                            # if denom == 0:
                            #     continue
                            # Ncx = ((Ni[2]-Nk[2])*(Ni[1]-Nj[1]) - (Ni[2]-Nj[2])*(Ni[1]*Nk[1]))/denom
                            # Ncy = ((Ni[0]-Nk[0])*(Ni[2]-Nj[2]) - (Ni[0]-Nj[0])*(Ni[2]-Nk[2]))/denom
                            # Ncz = 1/np.sqrt(1 + Ncx*Ncx + Ncy*Ncy)
                            # Ncx = Ncx*Ncz
                            # Ncy = Ncy*Ncz
                            # Nc = [Ncx, Ncy, Ncz]

                            denom = (2*np.linalg.norm(np.cross(Ni-Nk,Nj-Nk))**2) 
                            if denom == 0:
                                continue
                            Nc = np.cross(np.linalg.norm(Ni-Nk)**2 * (Nj-Nk) - np.linalg.norm(Nj-Nk)**2 * (Ni-Nk), np.cross(Ni-Nk,Nj-Nk))/denom + Nk
                            nNc = np.linalg.norm(Nc)
                            if nNc == 0:
                                continue
                            Nc = Nc/nNc
                            

                            scal = np.dot(Nc, Ni)
                            if scal < 0:
                                Nc = [-1*n for n in Nc]
                                scal = -scal
                            # Nb = Ni+Nj
                            # Nb = Nb/np.linalg.norm(Nb)
                            # scal = np.dot(Nc,Ni)
                            if scal < scalmin:
                                pass
                            elif any((np.dot(Nl,Nc) - scal) < eps for Nl in elemnormals):
                                pass   
                            else:
                                C = Nc
                                scalmin = scal
                NodeNormals[i] = C    
                if np.any(np.isnan(C)):
                    print(i)
                
            elif method == 'MostVisible_Iter':
                
                conv = 1e-3
                beta = 0.5
                
                # Initial weights]
                ws = [1/len(elemnormals) for i in range(len(elemnormals))]
                # Compute initial guess normal
                Sp = sum([w*n for w,n in zip(ws,elemnormals)])
                Np = Sp/np.linalg.norm(Sp)
                
                k = 0
                thinking = True
                while thinking:
                    k+=1
                    alphas = [np.arccos(np.clip(np.dot(Np,Ni),-1,1)) for Ni in elemnormals]
                    Salpha = sum(alphas)
                    if Salpha == 0:
                        thinking = False
                    else:
                        ws = [w*alpha/Salpha for w,alpha in zip(ws,alphas)]
                        Sw = sum(ws)
                        ws = [w/Sw for w in ws]
                        Spnew = sum([w*n for w,n in zip(ws,elemnormals)])
                        if np.linalg.norm(Spnew) == 0:
                            print('merp3')
                        Npnew = Spnew/np.linalg.norm(Spnew)
                        
                        # Relax
                        Nprel = beta*Npnew + (1-beta)*Np
                        if np.linalg.norm(Np-Nprel) < conv or k > 100:
                            thinking = False
                        Np = Nprel
                if any(np.isnan(Np)) and len(elemnormals)>0:
                    merp = 2
                NodeNormals[i] = Np.tolist()
    return NodeNormals

def BaryTri(Nodes, Pt):
    """
    BaryTri returns the bary centric coordinates of a point (Pt) relative to 
    a triangle (Nodes)

    Parameters
    ----------
    Nodes : list
        List of coordinates of the triangle vertices.
    Pt : list
        Coordinates of the point.

    Returns
    -------
    alpha : float
        First barycentric coordinate.
    beta : float
        Second barycentric coordinate.
    gamma : float
        Third barycentric coordinate.

    """
    
    A = Nodes[0]
    B = Nodes[1]
    C = Nodes[2]
    BA = np.subtract(B,A)
    # CB = np.subtract(C,B)
    # AC = np.subtract(A,C)
    CA = np.subtract(C,A)    
    BABA = np.dot(BA, BA)
    BACA = np.dot(BA, CA)
    CACA = np.dot(CA, CA)
    PABA = np.dot(np.subtract(Pt,A), BA)
    PACA = np.dot(np.subtract(Pt,A), CA)
    denom = 1/(BABA * CACA - BACA * BACA)
    beta = (CACA * PABA - BACA * PACA) * denom
    gamma = (BABA * PACA - BACA * PABA) * denom
    alpha = 1 - gamma - beta
    
    return alpha, beta, gamma

def BaryTris(Tris, Pt):
    """
    BaryTri returns the bary centric coordinates of a point (Pt) relative to 
    a triangle (Nodes)

    Parameters
    ----------
    Nodes : list
        List of coordinates of the triangle vertices.
    Pt : list
        Coordinates of the point.

    Returns
    -------
    alpha : float
        First barycentric coordinate.
    beta : float
        Second barycentric coordinate.
    gamma : float
        Third barycentric coordinate.

    """
    
    A = Tris[:,0]
    B = Tris[:,1]
    C = Tris[:,2]
    BA = np.subtract(B,A)
    # CB = np.subtract(C,B)
    # AC = np.subtract(A,C)
    CA = np.subtract(C,A)    
    BABA = np.sum(BA*BA,axis=1)
    BACA = np.sum(BA*CA,axis=1)
    CACA = np.sum(CA*CA,axis=1)
    PABA = np.sum(np.subtract(Pt,A)*BA,axis=1)
    PACA = np.sum(np.subtract(Pt,A)*CA,axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = 1/(BABA * CACA - BACA *BACA)
        beta = (CACA * PABA - BACA * PACA) * denom
        gamma = (BABA * PACA - BACA * PABA) * denom
        alpha = 1 - gamma - beta;    
    
    return alpha, beta, gamma

def BaryTet(Nodes, Pt):
    """
    BaryTet returns the bary centric coordinates of a point (Pt) relative to 
    a tetrahedron (Nodes)

    Parameters
    ----------
    Nodes : list
        List of coordinates of the tetrahedral vertices.
    Pt : list
        Coordinates of the point.

    Returns
    -------
    alpha : float
        First barycentric coordinate.
    beta : float
        Second barycentric coordinate.
    gamma : float
        Third barycentric coordinate.
    delta : float
        Fourth barycentric coordinate.
    """
    
    A = Nodes[0]
    B = Nodes[1]
    C = Nodes[2]
    D = Nodes[3]
    
    T = [[A[0]-D[0], B[0]-D[0], C[0]-D[0]],
         [A[1]-D[1], B[1]-D[1], C[1]-D[1]],
         [A[2]-D[2], B[2]-D[2], C[2]-D[2]]
         ]
    
    [alpha,beta,gamma] = np.linalg.solve(T,np.subtract(Pt,D))
    delta = 1 - (alpha + beta + gamma)
    
    return alpha, beta, gamma, delta

def Project2Surface(Point,Normal,NodeCoords,SurfConn,tol=np.inf,octree='generate'):
    """
    Project2Surface Projects a node (NodeCoord) along its normal vector (NodeNormal) onto the 
    surface defined by NodeCoord and SurfConn, returns the index of the 
    element (elemID) that contains the projected node and the barycentric 
    coordinates (alpha, beta, gamma) of that projection within that element

    Parameters
    ----------
    Point : list or np.ndarray
        Coordinates of the point to be projected on to the surface.
    Normal : list or np.ndarray
        Vector along which the point will be projected.
    NodeCoords : list or np.ndarray
        Node coordinates list of the mesh that the point is being projected to.
    SurfConn : list or np.ndarray
        Nodal connectivity of the surface mesh that the point is being projected to.
    tol : float, optional
        Tolerance value, if the projection distance is greater than tol, the projection will be exculded, default is np.inf
        octree : str (or octree.OctreeNode), optional
        octree options. An octree representation of the surface can significantly
        improve mapping speeds, by default 'generate'.
        'generate' - Will generate an octree for use in surface mapping.
        'none' or None - Won't generate an octree and will use a brute force approach.
        octree.OctreeNode - Provide a precompute octree structure corresponding to the surface mesh. Should be created by octree.Surf2Octree(NodeCoords,SurfConn)
    Returns
    -------
    elemID : int
        Index of the element in SurfConn that the point gets projected onto.
    alpha : float
        First barycentric coordinate of the projected point relative to the triangle identified by elemId
    beta : float
        Second barycentric coordinate of the projected point relative to the triangle identified by elemId
    gamma : float
        Third barycentric coordinate of the projected point relative to the triangle identified by elemId
    """
    if type(NodeCoords) is list: NodeCoords = np.array(NodeCoords)
    if type(SurfConn) is list: SurfConn = np.array(SurfConn)

    intersections, distances, intersectionPts = rays.RaySurfIntersection(Point, Normal, NodeCoords, SurfConn, octree=octree)
    if len(intersections) == 0:
        elemID = alpha = beta = gamma = -1
    elif min(np.abs(distances)) > tol:
        elemID = alpha = beta = gamma = -1
    else:
        try:
            idx = np.where(np.abs(distances) == min(np.abs(distances)))[0][0]
            p = intersectionPts[idx]
            elemID = intersections[idx]
            pts = NodeCoords[SurfConn[elemID]]
            alpha,beta,gamma = BaryTri(pts,p)
        except:
            print(distances)
            elemID = alpha = beta = gamma = -1

    return elemID,alpha,beta,gamma

def Project2Surface2(Points,Normals,NodeCoords,SurfConn,tol=np.inf,octree='generate'):
    """
    Project2Surface Projects a node (NodeCoord) along its normal vector (NodeNormal) onto the 
    surface defined by NodeCoord and SurfConn, returns the index of the 
    element (elemID) that contains the projected node and the barycentric 
    coordinates (alpha, beta, gamma) of that projection within that element

    Parameters
    ----------
    Point : list or np.ndarray
        Coordinates of the point to be projected on to the surface.
    Normal : list or np.ndarray
        Vector along which the point will be projected.
    NodeCoords : list or np.ndarray
        Node coordinates list of the mesh that the point is being projected to.
    SurfConn : list or np.ndarray
        Nodal connectivity of the surface mesh that the point is being projected to.
    tol : float, optional
        Tolerance value, if the projection distance is greater than tol, the projection will be exculded, default is np.inf
        octree : str (or octree.OctreeNode), optional
        octree options. An octree representation of the surface can significantly
        improve mapping speeds, by default 'generate'.
        'generate' - Will generate an octree for use in surface mapping.
        'none' or None - Won't generate an octree and will use a brute force approach.
        octree.OctreeNode - Provide a precompute octree structure corresponding to the surface mesh. Should be created by octree.Surf2Octree(NodeCoords,SurfConn)
    Returns
    -------
    elemID : int
        Index of the element in SurfConn that the point gets projected onto.
    alpha : float
        First barycentric coordinate of the projected point relative to the triangle identified by elemId
    beta : float
        Second barycentric coordinate of the projected point relative to the triangle identified by elemId
    gamma : float
        Third barycentric coordinate of the projected point relative to the triangle identified by elemId
    """
    if type(NodeCoords) is list: NodeCoords = np.array(NodeCoords)
    if type(SurfConn) is list: SurfConn = np.array(SurfConn)

    intersections, distances, intersectionPts = rays.RaysSurfIntersection(Points, Normals, NodeCoords, SurfConn, octree=octree)

    argmindist = [np.argmin(np.abs(x)) if len(x) > 0 else -1 for x in distances]
    mindist = np.array([x[argmindist[i]] if len(x) > 0 else np.inf for i,x in enumerate(distances)])
    
    elemID = np.array([intersections[i][argmindist[i]] if len(x) > 0 else -1 for i,x in enumerate(distances)])
    ps = np.array([intersectionPts[i][argmindist[i]] if len(x) > 0 else [np.nan,np.nan,np.nan] for i,x in enumerate(distances)])

    mappedbool = (elemID >= 0) & (mindist <= tol)
    alphas, betas, gammas = BaryTris(NodeCoords[SurfConn[elemID[mappedbool]]], ps[mappedbool,:])

    alpha = -1*np.ones(len(Points))
    beta = -1*np.ones(len(Points))
    gamma = -1*np.ones(len(Points))

    alpha[mappedbool] = alphas
    beta[mappedbool] = betas
    gamma[mappedbool] = gammas

    MappingMatrix = np.vstack([elemID, alpha, beta, gamma]).T


    # if len(intersections) == 0:
    #     elemID = alpha = beta = gamma = -1
    # elif min(np.abs(distances)) > tol:
    #     elemID = alpha = beta = gamma = -1
    # else:
    #     try:
    #         idx = np.where(np.abs(distances) == min(np.abs(distances)))[0][0]
    #         p = intersectionPts[idx]
    #         elemID = intersections[idx]
    #         pts = NodeCoords[SurfConn[elemID]]
    #         alpha,beta,gamma = BaryTri(pts,p)
    #     except:
    #         print(distances)
    #         elemID = alpha = beta = gamma = -1

    return MappingMatrix

def SurfMapping(NodeCoords1, SurfConn1, NodeCoords2, SurfConn2, tol=np.inf, verbose=False, octree='generate', return_octree=False):
    """
    SurfMapping Generate a mapping matrix from surface 1 (NodeCoords1, SurfConn1) to surface 2 (NodeCoords2, SurfConn2)
    Each row of the mapping matrix contains an element ID followed by barycentric coordinates alpha, beta, gamma
    that define the position of the nodes of surface 1 (NodeCoords1) relative to the specified surface element of 
    surface 2 (SurfConn2). An element ID of -1 indicates a failed mapping.
    NOTE: Only triangular surface meshes are supported.

    Parameters
    ----------
    NodeCoords1 : list
        List of nodal coordinates.
    SurfConn1 : list
        List of nodal connectivities.
    NodeCoords2 : list
        List of nodal coordinates.
    SurfConn2 : list
        List of nodal connectivities.
    tol : float, optional
        Tolerance value, if the projection distance is greater than tol, the projection will be exculded, default is np.inf
    verbose : bool, optional
        If true, will print mapping statistics, by default False.
    octree : str (or octree.OctreeNode), optional
        octree options. An octree representation of surface 2 can significantly
        improve mapping speeds, by default 'generate'.
        'generate' - Will generate an octree for use in surface mapping.
        'none' or None - Won't generate an octree and will use a brute force approach.
        octree.OctreeNode - Provide a precompute octree structure corresponding to surface 2. Should be created by octree.Surf2Octree(NodeCoords2,SurfConn2)
    return_octree : bool, optional
        If true, will return the generated or provided octree, by default False.

    Returns
    -------
    MappingMatrix : list
        len(NodeCoords1)x4 matrix of of barycentric coordinates, defining NodeCoords1 in terms
        of the triangular surface elements of Surface 2.
    octree : octree.OctreeNode, optional
        The generated or provided octree structure corresponding to Surface 2.

    """
    if type(NodeCoords1) is list: NodeCoords1 = np.array(NodeCoords1)
    if type(NodeCoords2) is list: NodeCoords2 = np.array(NodeCoords2)
    if type(SurfConn1) is list: SurfConn1 = np.array(SurfConn1)
    if type(SurfConn2) is list: SurfConn2 = np.array(SurfConn2)

    ElemConn1 = getElemConnectivity(NodeCoords1, SurfConn1)
    ElemNormals1 = CalcFaceNormal(NodeCoords1, SurfConn1)
    NodeNormals1 = Face2NodeNormal(NodeCoords1, SurfConn1, ElemConn1, ElemNormals1, method='angle')

    Surf1Nodes = np.unique(SurfConn1.flatten())

    
    if octree == 'generate': octree = octree.Surf2Octree(NodeCoords2,SurfConn2)
    # tic = time.time()
    MappingMatrix = -1*np.ones((len(NodeCoords1),4))
    MappingMatrix[Surf1Nodes,:] = Project2Surface2(NodeCoords1[Surf1Nodes,:], NodeNormals1[Surf1Nodes,:], NodeCoords2, SurfConn2, tol=tol, octree=octree)
    # print(time.time()-tic)
    # tic = time.time()
    # MappingMatrix = -1*np.ones((len(NodeCoords1),4))
    # for i in Surf1Nodes:
    #     MappingMatrix[i] = Project2Surface(NodeCoords1[i], NodeNormals1[i], NodeCoords2, SurfConn2, tol=tol, octree=octree)
    # print(time.time()-tic)
    if verbose: 
        failcount = np.sum(MappingMatrix[list(Surf1Nodes),0] == -1)
        print('{:.3f}% of nodes mapped'.format((len(Surf1Nodes)-failcount)/len(Surf1Nodes)*100))
    if return_octree:
        return MappingMatrix, octree
    return MappingMatrix

def ValueMapping(NodeCoords1, SurfConn1, NodeVals1, NodeCoords2, SurfConn2, tol=np.inf, 
octree='generate', MappingMatrix=None, verbose=False, return_MappingMatrix=False, return_octree=False):
    """
    ValueMapping Maps nodal values <NodeVals1> from surface 1 to surface 2
    - Currently only supports triangluar surface meshes
    TODO: Multi-value mapping may produce errors - need to better verify.
    
    Parameters
    ----------
    NodeCoords1 : List of lists
        Contains coordinates for each node in surface 1. Ex. [[x1,y1,z1],...]
    SurfConn1 : List of lists
        Contains the nodal connectivity defining the surface elements.
    NodeVals1 : List or List of lists
        Scalar nodal values associated with surface 1. For multiple values: [[x1,x2,x3,...],[y1,y2,y3,...],[z1,z2,z3,...],...]
    NodeCoords2 : List of lists
        Contains coordinates for each node in surface 2. Ex. [[x1,y1,z1],...].
    SurfConn2 : List of lists
        Contains the nodal connectivity defining the surface elements.
    tol : float, optional
        Tolerance value, if the projection distance is greater than tol, the projection will be exculded, default is np.inf 
    octree : str (or octree.OctreeNode), optional
        octree options. An octree representation of surface 1 can significantly
        improve mapping speeds, by default 'generate'.
        'generate' - Will generate an octree for use in surface mapping.
        'none' or None - Won't generate an octree and will use a brute force approach.
        octree.OctreeNode - Provide a precompute octree structure corresponding to surface 1. Should be created by octree.Surf2Octree(NodeCoords1,SurfConn1)
    MappingMatrix : list
        len(NodeCoords2)x4 matrix of of barycentric coordinates, defining NodeCoords2 in terms
        of the triangular surface elements of Surface 1.
    verbose : bool, optional
        If true, will print mapping statistics, by default False.
    return_MappingMatrix : bool, optional
        If true, will return MappingMatrix, by default False.
    return_octree : bool, optional
        If true, will return generated or provided octree, by defualt False.
        NOTE if MappingMatrix is provided, the octree structure won't be generated.
        In this cases, if octree='generate' and return_octree=True, the returned value
        for octree will simply be the string 'generate'.

    Returns
    -------
    NodeVals2 : List
        Scalar nodal values associated with surface 2, mapped from surface 1.

    """
    
    # if type(NodeVals1[0]) is list or type(NodeVals1[0]) is np.ndarray:
    #     singleVal = False
    #     # NodeVals2 = [[0 for j in range(len(NodeCoords2))] for i in range(len(NodeVals1))]
    # else:
    #     singleVal = True
        # NodeVals2 = [0 for i in range(len(NodeCoords2))]
    # Map the coordinates from surface 2 to surface 1
    if MappingMatrix is None:
        MappingMatrix,octree = SurfMapping(NodeCoords2, SurfConn2,  NodeCoords1, SurfConn1, octree=octree, tol=tol, verbose=verbose, return_octree=True)

    # if singleVal:
    if len(np.shape(NodeVals1)) == 1:
        # 1D data
        _NodeVals1 = np.append(NodeVals1, np.nan)
        alpha = MappingMatrix[:,1]
        beta = MappingMatrix[:,2]
        gamma = MappingMatrix[:,3]
    else:
        # ND data
        _NodeVals1 = np.append(NodeVals1,[np.repeat(np.nan,np.shape(NodeVals1)[1])],axis=0)
        alpha = MappingMatrix[:,1][:,None]
        beta = MappingMatrix[:,2][:,None]
        gamma = MappingMatrix[:,3][:,None]
    # NodeVals2 = np.nan*np.ones(np.shape(NodeVals1))
    elemID = MappingMatrix[:,0].astype(int)
    ArrayConn = np.append(SurfConn1,[[-1,-1,-1]],axis=0)
    NodeVals2 = alpha*_NodeVals1[ArrayConn[elemID][:,0]] + \
            beta*_NodeVals1[ArrayConn[elemID][:,1]] + \
            gamma*_NodeVals1[ArrayConn[elemID][:,2]]
        
            
    if return_MappingMatrix and return_octree:
        return NodeVals2, MappingMatrix, octree
    elif return_MappingMatrix:
        return NodeVals2, MappingMatrix
    elif return_octree:
        return NodeVals2, octree
    return NodeVals2

def DeleteDuplicateNodes(NodeCoords,NodeConn,tol=1e-12,return_idx=False):
    """
    DeleteDuplicateNodes Remove nodes that are duplicated in the mesh, either at exactly the same location as another 
    node or a distance < tol apart. Nodes are renumbered and elements reconnected such that the geometry and structure
    of the mesh remains unchanged. 

    Parameters
    ----------
    NodeCoords : list
        Contains coordinates for each node. Ex. [[x1,y1,z1],...]
    NodeConn : list
        Nodal connectivity list.
    tol : float, optional
        Tolerance value to be used when determining if two nodes are the same. The default is 1e-14.

    Returns
    -------
    NodeCoords2 : list
        Updated node coordinates without duplicates.
    NodeConn2 : list 
        Updated node connectivity without duplicate nodes.
    newIds : list
        List of node ids used to relabel the node connectivity. The new node id for a node referenced in the original mesh can be obtained as newId = newIds[oldId]. This can be used to reorder data that was associated with the original mesh to be consistent with the new mesh.
    
    """

    # assert len(NodeCoords) > 0, 'No nodes in mesh.'
    if len(NodeCoords) == 0:
        if return_idx:
            return NodeCoords,NodeConn,[],[]
        return NodeCoords, NodeConn, []

    if tol > 0:
        arrayCoords = np.round(np.array(NodeCoords)/tol)*tol
    else:
        arrayCoords = np.array(NodeCoords)
    unq,idx,inv = np.unique(arrayCoords, return_inverse=True, return_index=True, axis=0)
    newIds = np.arange(len(unq))[inv]
    if type(NodeCoords) is list:
        NodeCoords2 = np.asarray(NodeCoords)[idx].tolist()
    else:
        NodeCoords2 = np.asarray(NodeCoords)[idx]
    if len(NodeConn) > 0:
        tempIds = np.append(newIds,-1)
        R = PadRagged(NodeConn,fillval=-1)
        NodeConn2 = ExtractRagged(tempIds[R],delval=-1)
    else:
        NodeConn2 = NodeConn
    if return_idx:
        return NodeCoords2,NodeConn2,newIds,idx
    return NodeCoords2,NodeConn2,newIds

def RelabelNodes(NodeCoords,NodeConn,newIds,faces=None):

    # newIds is a list of node ids where the new index is located at the old index
    # def func(newIds,elem):
    #     return [newIds[node] for node in elem]
    # NewConn = pool(delayed(func)(newIds,elem) for elem in NodeConn)
    NewConn = ExtractRagged(np.append(newIds,[-1])[PadRagged(NodeConn)],dtype=int)
    if faces != None: 
        if len(faces) > 0: 
            NewFaces = ExtractRagged(np.append(newIds,[-1])[PadRagged(faces)],dtype=int)
        else:
            NewFaces = faces
    NewCoords = np.nan*np.ones(np.shape(NodeCoords)) # [[] for i in range(len(NodeCoords))]
    NewCoords[newIds.astype(int)] = np.array(NodeCoords)
    # for i,node in enumerate(NodeCoords):
    #     NewCoords[newIds[i]] = node
    if faces != None:
        return NewCoords,NewConn,NewFaces
    else:
        return NewCoords, NewConn

def DeleteDegenerateElements(NodeCoords,NodeConn,tol=1e-12,angletol=1e-3,strict=False):
    """
    DeleteDegenerateElements Deletes degenerate elements from a mesh.
    TODO: Currently only valid for triangles.
    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.
    angletol : float, optional
        Tolerance value for determining what constitutes a degenerate element, by default 1e-3. Degenerate elements will be those who have an angle greater than or equal to 180-180*angletol (default 179.82 degrees)

    Returns
    -------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.
    """

    # Remove elements that have a collapsed edge - i.e. two collinear edges
    # TODO: when tol!=0, using CollapseSlivers this is an imperfect solution (at a minimum, tol parameter should be changed)
    if len(NodeConn) == 0:
        return NodeCoords,NodeConn
    if strict:
        NewCoords = NodeCoords
        NewConn = [elem for elem in NodeConn if len(elem) == len(set(elem))]
    else:
        NewCoords,NewConn = DeleteDegenerateElements(NodeCoords,NodeConn,strict=True)
        if len(NewConn) == 0:
            return NewCoords,NewConn
        if angletol == 0:
            warnings.warn("Change to strict=True")
        
        thetal = np.pi-np.pi*angletol # Maximum angle threshold 
        def do_split(NewCoords,NewConn,EdgeSort,ConnSort,AngleSort,i):
            
            elem0 = NewConn[ConnSort[i,0]]
            elem1 = NewConn[ConnSort[i,1]]
            NotShared0 = set(elem0).difference(EdgeSort[i]).pop()
            NotShared1 = set(elem1).difference(EdgeSort[i]).pop()
            # Get the node not belonging to the edge
            if (AngleSort[i,0] >= thetal and ConnSort[i,0] >= 0 and type(NewConn[ConnSort[i,0]][0]) != list) and (AngleSort[i,1] >= thetal and ConnSort[i,1] >= 0 and type(NewConn[ConnSort[i,1]][0]) != list):
                # Both connected elements are degenerate
                NewNode = NewCoords[NotShared0]
            elif (AngleSort[i,0] >= thetal and ConnSort[i,0] >= 0 and type(NewConn[ConnSort[i,0]][0]) != list):
                NewNode = NewCoords[NotShared0]
            elif (AngleSort[i,1] >= thetal and ConnSort[i,1] >= 0 and type(NewConn[ConnSort[i,1]][0]) != list):
                NewNode = NewCoords[NotShared1]
            else:
                return NewCoords,NewConn
            
            NewId = len(NewCoords)
            NewCoords = np.vstack([NewCoords,NewNode])
            if ConnSort[i,0] >= 0: 
                while elem0[0] != NotShared0: elem0 = [elem0[-1]]+elem0[0:-1] # cycle the element definition so that it starts with the non-shared node (Might be unnecessarily slow)
                NewConn[ConnSort[i,0]] = [[elem0[0],elem0[1],NewId],[elem0[0],NewId,elem0[2]]]
            if ConnSort[i,1] >= 0: 
                while elem1[0] != NotShared1: elem1 = [elem1[-1]]+elem1[0:-1]
                NewConn[ConnSort[i,1]] = [[elem1[0],elem1[1],NewId],[elem1[0],NewId,elem1[2]]]
            
            return NewCoords, NewConn

        if type(NewConn) is np.ndarray: NewConn = NewConn.tolist()
        Thinking = True
        k = 0; maxiter = 3
        while Thinking and k < 3:
            k += 1
            NewCoords = np.array(NewCoords)
            Edges, EdgeConn, EdgeElem = converter.solid2edges(NewCoords,NewConn,return_EdgeConn=True,return_EdgeElem=True)
            UEdges, UIdx, UInv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
            UEdgeElem = np.asarray(EdgeElem)[UIdx]
            UEdgeConn = UInv[PadRagged(EdgeConn)]
            EECidx = (UEdgeElem[UEdgeConn] == np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)).astype(int)
            EdgeElemConn = -1*(np.ones((len(UEdges),2),dtype=int))
            EdgeElemConn[UEdgeConn,EECidx] = np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)

            Edges = np.asarray(Edges); EdgeConn = np.asarray(EdgeConn)
            EdgeVectors = NewCoords[Edges[:,1]] - NewCoords[Edges[:,0]]
            EdgeLengths = np.linalg.norm(EdgeVectors,axis=1)

            ElemVectors = EdgeVectors[EdgeConn]
            ElemLengths = EdgeLengths[EdgeConn]

            OppositeAngles = -1*np.ones(ElemLengths.shape)
            with np.errstate(divide='ignore', invalid='ignore'):
                OppositeAngles[:,0] = np.clip(np.sum(ElemVectors[:,2]*-ElemVectors[:,1],axis=1)/(ElemLengths[:,1]*ElemLengths[:,2]),-1,1)
                OppositeAngles[:,1] = np.clip(np.sum(ElemVectors[:,0]*-ElemVectors[:,2],axis=1)/(ElemLengths[:,0]*ElemLengths[:,2]),-1,1)
                OppositeAngles[:,2] = np.clip(np.sum(ElemVectors[:,1]*-ElemVectors[:,0],axis=1)/(ElemLengths[:,1]*ElemLengths[:,0]),-1,1)
                OppositeAngles = np.arccos(OppositeAngles)

            EdgeOppositeAngles =  -1*np.ones((len(UEdges),2))
            EdgeOppositeAngles[UEdgeConn,EECidx] = OppositeAngles

            sortkey = np.argsort(EdgeLengths[UIdx])[::-1]
            LengthSort = EdgeLengths[UIdx][sortkey]
            AngleSort = EdgeOppositeAngles[sortkey]
            EdgeSort = np.asarray(UEdges)[sortkey]
            ConnSort = np.array(EdgeElemConn)[sortkey]

            AbsLargeAngle = np.any(AngleSort >= thetal,axis=1)

            todo = np.where(AbsLargeAngle)[0]
            # Splits
            repeat = False
            for i in todo:
                if type(NewConn[ConnSort[i,0]][0]) is list or type(NewConn[ConnSort[i,1]][0]) is list:
                    repeat = True
                    continue
                NewCoords,NewConn = do_split(NewCoords,NewConn,EdgeSort,ConnSort,AngleSort,i)

            NewConn = [elem if (type(elem[0]) != list) else elem[0] for elem in NewConn] + [elem[1] for elem in NewConn if (type(elem[0]) == list)]
            NewCoords = NewCoords.tolist()
            if repeat:
                Thinking = True
            else:
                Thinking = False

            NewCoords,NewConn,_ = DeleteDuplicateNodes(NewCoords,NewConn,tol=tol)
            NewCoords,NewConn = DeleteDegenerateElements(NewCoords,NewConn,strict=True)
            # if len(converter.surf2edges(NewCoords,NewConn)) > 0:
            #     print('merp')
                
    return NewCoords,NewConn

def MirrorMesh(NodeCoords,NodeConn,x=None,y=None,z=None):
    """
    MirrorMesh Creates a mirrored copy of a mesh by mirroring about the planes
    defined by X=x, Y=y, and Z=z

    Parameters
    ----------
    NodeCoords : list
        Nodal Coordinates.
    NodeConn : list
        Nodal Connectivity.
    x : Numeric, optional
        YZ plane at X = x. The default is None.
    y : TYPE, optional
        XZ plane at Y = y. The default is None.
    z : TYPE, optional
        XY plane at Z = z. The default is None.

    Returns
    -------
    MirroredCoords : list
        Mirrored Nodal Coordinates.
    MirroredConn : list
        Nodal Connectivity of Mirrored Elements.
    """
    
    MirroredCoords = [copy.copy(node) for node in NodeCoords]
    MirroredConn = [copy.copy(elem) for elem in NodeConn]
    if x != None:
        for i in range(len(MirroredCoords)):
            MirroredCoords[i][0] = -(MirroredCoords[i][0] - x) + x 
    if y != None:
        for i in range(len(MirroredCoords)):
            MirroredCoords[i][1] = -(MirroredCoords[i][1] - y) + y
    if z != None:
        for i in range(len(MirroredCoords)):
            MirroredCoords[i][2] = -(MirroredCoords[i][2] - z) + z
    
    
    return MirroredCoords, MirroredConn
    
def MergeMesh(NodeCoords1, NodeConn1, NodeCoords2, NodeConn2, NodeVals1=[], NodeVals2=[], cleanup=True):
    """
    MergeMesh Merge two meshes together

    Parameters
    ----------
    NodeCoords1 : list
        List of nodal coordinates for mesh 1.
    NodeConn1 : list
        List of nodal connectivities for mesh 1.
    NodeCoords2 : list
        List of nodal coordinates for mesh 2.
    NodeConn2 : list
        List of nodal connectivities for mesh 2.
    NodeVals1 : list, optional
        List of node data associated with mesh 1, by default []
    NodeVals2 : list, optional
        List of node data associated with mesh 2, by default []
    cleanup : bool, optional
        If true, duplicate nodes will be deleted and renumbered accordingly, by default True.

    Returns
    -------
    MergedCoords : list
        List of nodal coordinates of the merged mesh.
        Nodes from mesh 1 appear first, followed by those of mesh 2.
    MergedConn : list
        List of nodal connectivities of the merged mesh.
    MergedVals : list, optional
        If provided, merged list of NodeVals.
    
    """
    if type(NodeCoords1) == np.ndarray:
        NodeCoords1 = NodeCoords1.tolist()
    if type(NodeConn1) == np.ndarray:
        NodeConn1 = NodeConn1.tolist()
    if type(NodeCoords2) == np.ndarray:
        NodeCoords2 = NodeCoords2.tolist()
    if type(NodeConn2) == np.ndarray:
        NodeConn2 = NodeConn2.tolist()
    if type(NodeVals1) == np.ndarray:
        NodeVals1 = NodeVals1.tolist()
    if type(NodeVals2) == np.ndarray:
        NodeVals2 = NodeVals2.tolist()
        
    MergeCoords = NodeCoords1 + NodeCoords2

    MergeConn = NodeConn1 + [[node+len(NodeCoords1) for node in elem] for elem in NodeConn2]
    
    if len(NodeVals1) > 0:
        assert len(NodeVals1) == len(NodeCoords1), 'NodeVals lists must contain the number of entries as nodes.'
        assert len(NodeVals2) == len(NodeCoords2), 'NodeVals lists must contain the number of entries as nodes.'

        MergeVals = [[] for i in range(len(NodeVals1))]
        for i in range(len(NodeVals1)):
            MergeVals[i] = NodeVals1[i] + NodeVals2[i]
        if cleanup:
            MergeCoords,MergeConn,newIds = DeleteDuplicateNodes(MergeCoords,MergeConn)
            for i in range(len(MergeVals)):
                MergeVals[i] = [MergeVals[i][j] for j in newIds]
                
            return MergeCoords, MergeConn, MergeVals
    
    elif cleanup:
        MergeCoords,MergeConn,_ = DeleteDuplicateNodes(MergeCoords,MergeConn)
    return MergeCoords, MergeConn
    
def DetectFeatures(NodeCoords,SurfConn,angle=25):
    """
    DetectFeatures Classifies nodes as edges or corners if the angle between adjacent
    surface elements is less than or equal to <angle> (deg)

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    SurfConn : list
        List of nodal connectivities of a surface mesh.
    angle : float, optional
        Dihedral angle threshold (in degrees) used to determine whether an edge
        exists between two adjacent faces, by default 140.

    Returns
    -------
    edges : list
        list of nodes identified to lie on an edge of the geometry.
    corners : list
        list of nodes identified to lie on a corner of the geometry.

    """
    ElemNormals = np.asarray(CalcFaceNormal(NodeCoords,SurfConn))
    Edges, EdgeConn, EdgeElem = converter.solid2edges(NodeCoords,SurfConn,return_EdgeConn=True,return_EdgeElem=True)
    UEdges, UIdx, UInv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
    UEdgeElem = np.asarray(EdgeElem)[UIdx]
    UEdgeConn = UInv[PadRagged(EdgeConn)]
    EECidx = (UEdgeElem[UEdgeConn] == np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)).astype(int)
    EdgeElemConn = -1*(np.ones((len(UEdges),2),dtype=int))
    EdgeElemConn[UEdgeConn,EECidx] = np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)
    
    ConnectedNormals = ElemNormals[EdgeElemConn]
    angles = quality.dihedralAngles(ConnectedNormals[:,0],ConnectedNormals[:,1],Abs=False)
    
    FeatureEdges = np.where(angles > angle*np.pi/180)[0]
    FeatureNodes = [n for edge in FeatureEdges for n in UEdges[edge]]
    unq,counts = np.unique(FeatureNodes,return_counts=True)
    corners = unq[counts>2].tolist()
    edges = unq[counts<=2].tolist()
    # angle = 140
    # ElemNormals = CalcFaceNormal(NodeCoords,SurfConn)
    # ElemNeighbors = getElemNeighbors(NodeCoords,SurfConn,mode='edge')
    # edges = []
    # corners = []
    # for i in range(len(SurfConn)):  # For each element:
    #     Ni = ElemNormals[i]
    #     sharedNodes = []
    #     for j in ElemNeighbors[i]:   # For each adjacent element:
    #         Nj = ElemNormals[j]
    #         theta = 180 - np.arccos(min([1,abs(np.dot(Ni,Nj)/(np.linalg.norm(Ni)*np.linalg.norm(Nj)))]))*180/np.pi
    #         if theta <= angle:
    #             sharedNodes.append(np.intersect1d(SurfConn[i],SurfConn[j]).tolist())
    #     unq,counts = np.unique([x for sublist in sharedNodes for x in sublist],return_counts=True)
    #     for j in range(len(unq)):
    #         if counts[j] == 1:
    #             edges.append(unq[j])
    #         else:   # counts >= 2
    #             corners.append(unq[j])
    # edges = np.unique(edges).tolist()
    # corners = np.unique(corners).tolist()
    # edges = [edge for edge in edges if edge not in corners]

    return edges,corners

def PeelHex(NodeCoords,NodeConn,nLayers=1):
    """
    PeelHex Removes the specified number of layers from a hexahedral mesh

    Parameters
    ----------
    NodeCoords : list of lists
        Contains coordinates for each node in a voxel mesh. Ex. [[x1,y1,z1],...].
        The mesh is assumed to consist of only hexahedral elements.
    NodeConn : List of lists
        Nodal connectivity list.
    nLayers : int, optional
        Number of layers to peel. The default is 1.

    Returns
    -------
    PeeledCoords : List
        Node coordinates for each node in the peeled mesh.
    PeeledConn : list
        Nodal connectivity for each element in the peeled mesh.
    PeelCoords : list
        Node coordinates for each node in the layers of the mesh that have
        been removed.
    PeelConn : list
        Nodal connectivity for each element in the layers of the mesh that have
        been removed.

    """
    
    NewCoords = copy.copy(NodeCoords)
    NewConn = copy.copy(NodeConn)   
    PeelConn = []
    for i in range(nLayers):
        HexSurfConn = converter.solid2surface(NewCoords,NewConn)
        SurfNodes = np.unique(HexSurfConn)
        SurfNodeSet = set(SurfNodes)
        PeelConn += [NewConn[i] for i in range(len(NewConn)) if (set(NewConn[i])&SurfNodeSet)]
        NewConn = [NewConn[i] for i in range(len(NewConn)) if not (set(NewConn[i])&SurfNodeSet)]
        
    
    PeelCoords,PeelConn,_ = converter.removeNodes(NewCoords,PeelConn)
    PeeledCoords,PeeledConn,_ = converter.removeNodes(NewCoords,NewConn)
    
    return PeeledCoords, PeeledConn, PeelCoords, PeelConn
    
def makePyramidLayer(VoxelCoords,VoxelConn,PyramidHeight=None):
    """
    makePyramidLayer For a given voxel mesh, will generate a set of pyramid 
    elements that cover the surface of the voxel mesh. To merge the pyramid 
    layer with the voxel mesh, use MergeMesh

    Parameters
    ----------
    VoxelCoords : list
        Contains coordinates for each node in a voxel mesh. Ex. [[x0,y0,z0],...].
    VoxelConn : List
        Nodal connectivity list.
        The voxel mesh is assumed to consist of a set of uniform cubic hexahedral 
        elements.
    PyramidHeight : float (or None), optional
        Height of the pyramids. The default is None.
        If no height as assigned, it will default to 1/2 of the voxel size

    Returns
    -------
    PyramidCoords : list
        List of nodal coordinates for the pyramid elements.
    PyramidConn : list
        List of nodal connectivities for the pyramid elements.

    """
    
    if PyramidHeight == None:
        PyramidHeight = abs(VoxelCoords[VoxelConn[0][0]][0] - VoxelCoords[VoxelConn[0][1]][0])/2
        
    SurfConn = converter.solid2surface(VoxelCoords,VoxelConn)
    SurfCoords, SurfConn, _ = converter.removeNodes(VoxelCoords,SurfConn)
    
    FaceNormals = CalcFaceNormal(SurfCoords,SurfConn)
    ArrayCoords = np.array(SurfCoords)
    PyramidConn = [[] for i in range(len(SurfConn))]
    PyramidCoords = SurfCoords
    for i,face in enumerate(SurfConn):
        nodes = ArrayCoords[face]
        centroid = np.mean(nodes,axis=0)
        tipCoord = centroid + PyramidHeight*np.array(FaceNormals[i])
        
        PyramidConn[i] = face + [len(PyramidCoords)]
        PyramidCoords.append(tipCoord.tolist())
    
    return PyramidCoords, PyramidConn

def makeVoxelLayer(VoxelCoords,VoxelConn):
    """
    makeVoxelLayer For a given voxel mesh, will generate a layer of voxels that
    wrap around the current voxel mesh. To merge the pyramid 
    layer with the voxel mesh, use MergeMesh

    Parameters
    ----------
    VoxelCoords : list of lists
        Contains coordinates for each node in a voxel mesh. Ex. [[x1,y1,z1],...].
        The voxel mesh is assumed to consist of a set of uniform cubic hexahedral 
        elements.
    VoxelConn : List of lists
        Nodal connectivity list.

    Returns
    -------
    LayerCoords : TYPE
        DESCRIPTION.
    LayerConn : TYPE
        DESCRIPTION.

    """
    # TODO: This has the potential to create overlapping voxels
    VoxelSize = abs(VoxelCoords[VoxelConn[0][0]][0] - VoxelCoords[VoxelConn[0][1]][0])
        
    SurfConn = converter.solid2surface(VoxelCoords,VoxelConn)
    SurfCoords, SurfConn, _ = converter.removeNodes(VoxelCoords,SurfConn)
    
    FaceNormals = CalcFaceNormal(SurfCoords,SurfConn)
    ArrayCoords = np.array(SurfCoords)
    LayerConn = [[] for i in range(len(SurfConn))]
    LayerCoords = SurfCoords
    for i,face in enumerate(SurfConn):
        nodes = ArrayCoords[face]
        coord0 = nodes[0] + VoxelSize*np.array(FaceNormals[i])
        coord1 = nodes[1] + VoxelSize*np.array(FaceNormals[i])
        coord2 = nodes[2] + VoxelSize*np.array(FaceNormals[i])
        coord3 = nodes[3] + VoxelSize*np.array(FaceNormals[i])
        
        LayerConn[i] = face + [len(LayerCoords), len(LayerCoords)+1, len(LayerCoords)+2, len(LayerCoords)+3]
        LayerCoords.append(coord0.tolist())
        LayerCoords.append(coord1.tolist())
        LayerCoords.append(coord2.tolist())
        LayerCoords.append(coord3.tolist())
        
    return LayerCoords, LayerConn
        
def TriSurfVol(NodeCoords, SurfConn):
    """
    TriSurfVol Calculates the volume contained within a surface mesh
    Based on 'Efficient feature extraction for 2D/3D objects in mesh 
    representation.' - Zhang, C. and Chen, T., 2001
    
    Parameters
    ----------
    NodeCoords : list of lists
        Contains coordinates for each node. Ex. [[x1,y1,z1],...].
    SurfConn : List of lists
        Nodal connectivity list for a triangular surface mesh.

    Returns
    -------
    V : float
        Volume contained within the surface mesh.

    """
    def TriSignedVolume(nodes):
        return 1/6*(-nodes[2][0]*nodes[1][1]*nodes[0][2] + 
                     nodes[1][0]*nodes[2][1]*nodes[0][2] + 
                     nodes[2][0]*nodes[0][1]*nodes[1][2] -
                     nodes[0][0]*nodes[2][1]*nodes[1][2] - 
                     nodes[1][0]*nodes[0][1]*nodes[2][2] + 
                     nodes[0][0]*nodes[1][1]*nodes[2][2])
    V = sum([TriSignedVolume([NodeCoords[node] for node in elem]) for elem in SurfConn])
    return V
    
def TetMeshVol(NodeCoords, NodeConn):
    """
    TetMeshVol Calculates the volume contained within a tetrahedral mesh
    
    Parameters
    ----------
    NodeCoords : list of lists
        Contains coordinates for each node. Ex. [[x1,y1,z1],...].
    NodeConn : List of lists
        Nodal connectivity list for a tetrahedral mesh.

    Returns
    -------
    V : float
        Volume contained within the tetrahedral mesh.

    """
    def TetVolume(nodes):
        return np.abs(np.dot(np.subtract(nodes[0],nodes[1]),
                      np.cross(np.subtract(nodes[1],nodes[3]),
                               np.subtract(nodes[2],nodes[3]))))/6
    V = sum([TetVolume([NodeCoords[node] for node in elem]) for elem in NodeConn])
    return V

def PadRagged(In,fillval=-1):
    """
    PadRagged Pads a 2d list of lists with variable length into a rectangular 
    numpy array with specified fill value.

    Parameters
    ----------
    In : list
        Input list of lists to be padded.
    fillval : int (or other), optional
        Value used to pad the ragged array, by default -1

    Returns
    -------
    Out : np.ndarray
        Padded array.
    """
    Out = np.array(list(itertools.zip_longest(*In,fillvalue=fillval))).T
    return Out

def ExtractRagged_old(In,delval=-1,dtype=None):
    """
    ExtractRagged Extracts a list of list from a 2d numpy array, removing the 
    specified value, generally creating a ragged array unless there is no padding

    Parameters
    ----------
    In : numpy.ndarray
        Input array.
    delval : int (or other), optional
        Padding value to be removed from the input array, by default -1.
    dtype : type, optional
        type to cast the output result to, by default None.
        If None, no casting will be performed.

    Returns
    -------
    Out : list
        Output list of lists with the specified value <delval> removed.
    """
    if dtype:
        if type(In) is list: In = np.array(In)
        In = In.astype(dtype)
        delval = np.array([delval]).astype(dtype)[0]
    if np.any(delval == In):
        if len(In.shape) == 2:
            Out = [[x for x in y if x != delval] for y in In]
        elif len(In.shape) == 3:
            Out = [[[x for x in y if x != delval] for y in z if all([x!= delval for x in y])] for z in In]
        else:
            raise Exception('Currently only supported for 2- or 3D matrices')
    else:
        Out = In.tolist()
    return Out

def ExtractRagged(In,delval=-1,dtype=None):
    if dtype:
        if type(In) is list: In = np.array(In)
        In = In.astype(dtype)
        delval = np.array([delval]).astype(dtype)[0]
    where = In != delval
    if not np.all(where):
        if len(In.shape) == 2:
            Out = np.split(In[where],np.cumsum(np.sum(where,axis=1)))[:-1]
            # Out = [x.tolist() for x in Out]
        elif len(In.shape) == 3:
            Out = [[[x for x in y if x != delval] for y in z if all([x!= delval for x in y])] for z in In]
        else:
            raise Exception('Currently only supported for 2- or 3D matrices')
    else:
        Out = In.tolist()
    return Out