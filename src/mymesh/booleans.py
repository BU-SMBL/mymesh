# -*- coding: utf-8 -*-
# Created on Wed Feb 16 11:28:28 2022
# @author: toj
"""
Boolean (union, intersection, difference) operations for meshes

Operations
==========
.. autosummary::
    :toctree: submodules/
    
    MeshBooleans
    SplitMesh

"""

import warnings
import numpy as np
from . import mesh, contour, converter, utils, tree, rays, improvement, delaunay, primitives
import itertools

def MeshBooleans(Surf1, Surf2, tol=1e-8):
    """
    Boolean (union, intersection, difference) operations for two surface meshes.
    Currently operations are only supported on purely triangular meshes.
    If a volume or non-triangular mesh is given, it will be converted
    and the output will be purely triangular.
    https://dl.acm.org/doi/pdf/10.1145/15922.15904

    Parameters
    ----------
    Surf1 : mesh.mesh
        Mesh object containing a surface mesh
    Surf2 : mesh.mesh
        Mesh object containing a surface mesh
    tol : type, optional
        Tolerance value, by default 1e-8

    Returns
    -------
    Union : mesh.mesh
        Mesh object containing the union of the two input surfaces
    Intersection : mesh.mesh
        Mesh object containing the intersection of the two input surfaces
    Difference : mesh.mesh
        Mesh object containing the difference of the two input surfaces
    """    
    if (Surf1.BoundaryNodes) > 0:
        warnings.warn('Surf1 is not a fully closed surface - boolean operations may produce unexpected results.')
    if (Surf2.BoundaryNodes) > 0:
        warnings.warn('Surf2 is not a fully closed surface - boolean operations may produce unexpected results.')

    eps = tol/100
    eps_final = tol*10    

    # Split Mesh
    Split1, Split2 = SplitMesh(Surf1, Surf2, eps=eps)

    Split1.cleanup(tol=eps)
    Split2.cleanup(tol=eps)

    # Get Shared Nodes
    Shared1,Shared2 = _GetSharedNodes(Split1.NodeCoords, Split2.NodeCoords, eps=tol)
    
    # Classify Tris
    AinB, AoutB, AsameB, AflipB, BinA, BoutA, BsameA, BflipA = _ClassifyTris(Split1, Shared1, Split2, Shared2, eps=eps)
    # Perform Boolean Operations
    # Union
    AUtris = AoutB.union(AsameB)
    BUtris = BoutA
    AUConn = Split1.NodeConn[list(AUtris)]
    BUConn = Split2.NodeConn[list(BUtris)]
    # Intersection
    AItris = AinB.union(AsameB)
    BItris = BinA
    AIConn = Split1.NodeConn[list(AItris)]
    BIConn = Split2.NodeConn[list(BItris)]
    # Difference
    ADtris = AoutB.union(AflipB)
    BDtris = BinA
    ADConn = Split1.NodeConn[list(ADtris)]
    BDConn = Split2.NodeConn[list(BDtris)]
    
    # Merge and Cleanup Mesh
    MergedUCoords, MergedUConn = utils.MergeMesh(Split1.NodeCoords, AUConn, Split2.NodeCoords, BUConn)
    MergedICoords, MergedIConn = utils.MergeMesh(Split1.NodeCoords, AIConn, Split2.NodeCoords, BIConn)
    MergedDCoords, MergedDConn = utils.MergeMesh(Split1.NodeCoords, ADConn, Split2.NodeCoords, np.fliplr(BDConn))

    if 'mesh' in dir(mesh):
        Union = mesh.mesh(MergedUCoords,MergedUConn)
        Intersection = mesh.mesh(MergedICoords,MergedIConn)
        Difference = mesh.mesh(MergedDCoords,MergedDConn)
    else:
        Union = mesh(MergedUCoords,MergedUConn)
        Intersection = mesh(MergedICoords,MergedIConn)
        Difference = mesh(MergedDCoords,MergedDConn)

    Union.cleanup(tol=eps_final)
    Intersection.cleanup(tol=eps_final)
    Difference.cleanup(tol=eps_final)

    return Union, Intersection, Difference
       
def SplitMesh(Surf1, Surf2, eps=1e-12):
    """
    Find intersections between two surfaces and split them. The resulting meshes
    will have nodes placed along their interfaces and retriangulated to contain 
    the lines of intersection. If a volume or non-triangular mesh is given, it 
    will be converted and the output will be purely triangular.

    Parameters
    ----------
    Surf1 : mymesh.mesh
        First surface mesh to split.
    Surf2 : mymesh.mesh
        Second surface mesh to split. 
    eps : float, optional
        Small tolerance parameter, by default 1e-12

    Returns
    -------
    Surf1 : mymesh.mesh
        First surface mesh split by the second.
    Surf2 : mymesh.mesh
        Second surface mesh split by the first.
    """    
    if Surf1.Type != 'surf':
        Surf1 = Surf1.Surface
    if Surf2.Type != 'surf':
        Surf2 = Surf2.Surface

    if len(Surf1.ElemType) != 1 or Surf1.ElemType[0] != 'tri':
        Surf1.NodeCoords, Surf1.NodeConn = converter.surf2tris(*Surf1)
    if len(Surf2.ElemType) != 1 or Surf2.ElemType[0] != 'tri':
        Surf2.NodeCoords, Surf2.NodeConn = converter.surf2tris(*Surf2)

    Surf1Intersections,Surf2Intersections,IntersectionPts = rays.SurfSurfIntersection(*Surf1,*Surf2,return_pts=True)
    
    SurfIntersections12 = [Surf1Intersections,Surf2Intersections]
    
    Surf12 = [Surf1.copy(), Surf2.copy()]
    for i,surf in enumerate(Surf12):
        # Group nodes for each triangle
        SurfIntersections = SurfIntersections12[i]
        ArrayCoords = np.array(surf.NodeCoords)
        SplitGroupNodes = ArrayCoords[surf.NodeConn].tolist()
        for j,elemid in enumerate(SurfIntersections):
            if len(IntersectionPts[j]) == 2:
                SplitGroupNodes[elemid] = np.append(SplitGroupNodes[elemid], IntersectionPts[j], axis=0)
            else:
                for k in range(len(IntersectionPts[j])):
                    SplitGroupNodes[elemid] = np.append(SplitGroupNodes[elemid], [IntersectionPts[j][k],IntersectionPts[j][(k+1)%len(IntersectionPts[j])]], axis=0)
        
        ElemNormals = utils.CalcFaceNormal(*surf)
        for j in range(surf.NElem):
            if len(SplitGroupNodes[j]) > 3:

                n = ElemNormals[j]
                # Set edge constraints
                Constraints = np.transpose([np.arange(3,len(SplitGroupNodes[j]),2), np.arange(4,len(SplitGroupNodes[j]),2)])

                # Reduce node list
                SplitGroupNodes[j],_,idx,newId = utils.DeleteDuplicateNodes(SplitGroupNodes[j],[],return_idx=True,return_inv=True,tol=eps)

                Constraints = newId[Constraints]
                Constraints = np.unique(np.sort(Constraints,axis=1),axis=0)

                # Transform to Local Coordinates
                # Rotation matrix from global z (k=[0,0,1]) to local z(n)
                k=np.array([0,0,1],dtype=np.float64)
                flatnodes,R = utils.RotateNormalToVector(SplitGroupNodes[j], n, np.array([0,0,1],dtype=np.float64))
                    
                coords, conn = delaunay.Triangulate(flatnodes[:,0:2],method='Triangle',Constraints=Constraints,tol=eps)  
                

                if len(coords) != len(SplitGroupNodes[j]):
                    # Accounts for possible constraint splitting
                    SplitGroupNodes[j] = np.matmul(np.linalg.inv(R),np.append(coords,flatnodes[0,2]*np.ones((len(coords),1)),axis=1).T).T
                conn = (conn+surf.NNode).tolist()
                surf.addElems(conn)
                surf.addNodes(SplitGroupNodes[j].tolist())

        # Collinear check
        Edges = converter.solid2edges(*surf,ElemType='tri')
        ArrayCoords = np.array(surf.NodeCoords)
        EdgePoints = ArrayCoords[Edges]
        ElemPoints = ArrayCoords[surf.NodeConn]
        A2 = np.linalg.norm(np.cross(ElemPoints[:,1]-ElemPoints[:,0],ElemPoints[:,2]-ElemPoints[:,0]),axis=1)
        EdgeLen = np.max(np.linalg.norm(EdgePoints[:,0]-EdgePoints[:,1],axis=1).reshape((int(len(Edges)/3),3)),axis=1)
        deviation = A2/EdgeLen # the double area divided by the longest side gives the deviation of the middle point from the line of the other two
        
        iset = set(SurfIntersections)
        colset = set(np.where(deviation<eps/2)[0])
        surf.NodeConn = [elem for i,elem in enumerate(surf.NodeConn) if i not in iset and i not in colset]
    Split1, Split2 = Surf12

    return Split1, Split2
    
def SurfaceBoundary(Surf, Boundary, mode='intersection', ElemType=None, eps=0):
    """
    Boolean (union, intersection, difference) operations for coplanar surface
    and boundary meshes.  

    Parameters
    ----------
    Surf : mymesh.mesh
        Surface mesh that will be cut by the boundary mesh. Surf.Type should
        be 'surf'.
    Boundary : mymesh.mesh
        Boundary mesh that will cut the surface mesh. Boundary.Type should
        be 'line'.
    mode : str, optional
        Boolean operation to perform, by default 'intersection'.

        - 'intersection': returns a mesh of surface elements inside the boundary mesh

        - 'difference': returns a mesh of surface elements outside the boundary mesh

        - 'union': returns the surface mesh split by the boundary mesh

    ElemType : str, NoneType, optional
        Element type of the produced mesh. Either 'tri' to return a purely
        triangular mesh or None for a mixed Element mesh, by default None.
    eps : int, optional
        Small tolerance value used by :func:`utils.DeleteDuplicateNodes`,
        :func:`rays.PointInBoundary`, and 
        :func:`rays.SegmentsSegmentsIntersection` by default 0.

    Returns
    -------
    mnew : mymesh.mesh
        New mesh

    """
    # Edge intersection
    e1 = Boundary.NodeCoords[Boundary.Edges]
    e2 = Surf.NodeCoords[Surf.Edges]
    pairwise = np.array(list(zip(*itertools.product(range(len(e1)), range(len(e2))))))

    edgeidx1 = pairwise[0]
    edgeidx2 = pairwise[1]
    s1 = e1[edgeidx1]
    s2 = e2[edgeidx2]

    ix, pts = rays.SegmentsSegmentsIntersection(s1, s2, return_intersection=True,eps=eps)

    # upts, idx1 = np.unique(pts[ix],return_index=True, axis=0)
    upts, _, idx1 = utils.DeleteDuplicateNodes(pts[ix], [[]], return_idx=True, tol=eps)
    # cleanup intersections - use only one point per edge
    Surf_edge_ix, idx2, c = np.unique(edgeidx2[ix][idx1], return_index=True, return_counts=True)
    
    ixpts = pts[ix][idx1][idx2]

    # Define new node coordinates from intersections
    NewCoords = ixpts
    NewCoordIds = Surf.NNode + np.arange(len(ixpts), dtype=int)
    NewCoords = np.vstack([Surf.NodeCoords, ixpts])
    # Node IDs for new nodes created along edges
    Surf_edge_ids = -1*np.ones(Surf.NEdge, dtype=int)
    Surf_edge_ids[Surf_edge_ix] = NewCoordIds

    # Node IDs for new nodes created along edges grouped by element
    elem_edge_ids = np.append(Surf_edge_ids,-1)[utils.PadRagged(Surf.EdgeConn, -1)] 

    In = np.array([rays.PointInBoundary(pt, Boundary.NodeCoords, Boundary.NodeConn, eps=eps, inclusive=False) for pt in Surf.NodeCoords])
    if mode.lower() == 'difference':
        In = ~In

    # Split surface NodeConn into groups by element type to process with appropriate method 
    SplitNodeConn, splitidx = utils.SplitRaggedByLength(Surf.NodeConn, return_idx=True)
    NewConn = []
    for group,NodeConn in enumerate(SplitNodeConn):

        if np.shape(NodeConn)[1] == 4:
            # marching squares lookup
            quad_ints = np.sum(In[NodeConn] * 2**np.arange(0,4)[::-1], axis=1)
            if ElemType is not None and ElemType.lower() == 'tri':
                if mode == 'union':
                    element_lists = contour.MSTriangleSplit_Lookup[quad_ints]
                else:
                    element_lists = contour.MSTriangle_Lookup[quad_ints]
            else:
                if mode == 'union':
                    element_lists = contour.MSMixedSplit_Lookup[quad_ints]
                else:
                    element_lists = contour.MSMixed_Lookup[quad_ints]
            element_point_lookup = np.hstack([NodeConn, elem_edge_ids[splitidx[group]]])

        elif np.shape(NodeConn)[1] == 3:
            # marching triangles lookup
            tri_ints = np.sum(In[NodeConn] * 2**np.arange(0,3)[::-1], axis=1)
            if ElemType is not None and ElemType.lower() == 'tri':
                if mode == 'union':
                    element_lists = contour.MTriSplit_Lookup[tri_ints]
                else:
                    element_lists = contour.MTri_Lookup[tri_ints]
            else:
                if mode == 'union':
                    element_lists = contour.MTriMixedSplit_Lookup[tri_ints]
                else:
                    element_lists = contour.MTriMixed_Lookup[tri_ints]
            element_point_lookup = np.hstack([elem_edge_ids[splitidx[group],:3], NodeConn])
        else:
            raise ValueError('Invalid element type.')
        
        NewConn += [element_point_lookup[i][element_lists[i][j]] for i in range(len(NodeConn)) for j in range(len(element_lists[i])) if (len(element_lists[i][j]) != 0) and not np.any(element_point_lookup[i][element_lists[i][j]] == -1)]

    if 'mesh' in dir(mesh):
        NewSurf = mesh.mesh(NewCoords, NewConn)
    else:
        NewSurf = mesh(NewCoords, NewConn)
    
    return NewSurf

def _GetSharedNodes(NodeCoordsA, NodeCoordsB, eps=1e-10):
    
    RoundCoordsA = np.round(np.asarray(NodeCoordsA)/eps)*eps
    RoundCoordsB = np.round(np.asarray(NodeCoordsB)/eps)*eps

    setA = set(tuple(coord) for coord in RoundCoordsA)
    setB = set(tuple(coord) for coord in RoundCoordsB)
    setI = setA.intersection(setB)
    SharedA = {i for i,coord in enumerate(RoundCoordsA) if tuple(coord) in setI}
    SharedB = {i for i,coord in enumerate(RoundCoordsB) if tuple(coord) in setI}

    return SharedA, SharedB
                           
def _ClassifyTris(SplitA, SharedA, SplitB, SharedB, eps=1e-10):
    """
    Classify triangles of meshes split by :func:`SplitMesh`.

    Parameters
    ----------
    SplitA : mymesh.mesh
        Triangular surface mesh, the result of :func:`SplitMesh` 
    SharedA : set
        Set of nodes of A that are shared with B, the result of :func:`_GetSharedNodes`
    SplitB : mymesh.mesh
        Triangular surface mesh, the result of :func:`SplitMesh` 
    SharedB : set
        Set of nodes of B that are shared with A, the result of :func:`_GetSharedNodes`
    eps : float, optional
        Small tolerance parameter, by default 1e-10

    Returns
    -------
    AinB : set
        Set of element IDs of A that are inside surface B
    AoutB : set 
        Set of element IDs of A that are outside surface B
    AsameB : set 
        Set of element IDs of A that are on the surface of surface B
    AflipB : set
        Set of element IDs of A that are on the surface of surface B, facing the opposite direction
    BinA : set 
        Set of element IDs of B that are inside surface A
    BoutA : set
        Set of element IDs of B that are outside surface A
    BsameA : set 
        Set of element IDs of B that are on the surface of surface A
    BflipA : set
        Set of element IDs of B that are on the surface of surface A, facing the opposite direction
    """
    # Classifies each Triangle in A as inside, outside, or on the surface facing the same or opposite direction as surface B
    
    octA = None# octree.Surface2Octree(*SplitA)
    octB = None# octree.Surface2Octree(*SplitB)
    
    AShared = np.isin(SplitA.NodeConn, list(SharedA))
    BShared = np.isin(SplitB.NodeConn, list(SharedB))
    AllBoundaryA = np.where(np.all(AShared, axis=1))[0]
    AllBoundaryB = np.where(np.all(BShared, axis=1))[0]
    NotSharedConnA = SplitA.NodeConn[~np.any(AShared, axis=1)]
    NotSharedConnB = SplitB.NodeConn[~np.any(BShared, axis=1)]
    
    if len(NotSharedConnA) > 0:
        RegionsA = utils.getConnectedNodes(SplitA.NodeCoords,NotSharedConnA)  # Node Sets
    else:
        RegionsA = []
    if len(NotSharedConnB) > 0:
        RegionsB = utils.getConnectedNodes(SplitB.NodeCoords,NotSharedConnB)  # Node Sets
    else:
        RegionsB = []
        
    ElemNormalsA = np.asarray(utils.CalcFaceNormal(*SplitA))
    ElemNormalsB = np.asarray(utils.CalcFaceNormal(*SplitB))

    AinB = set()    # Elem Set
    AoutB = set()   # Elem Set
    AsameB = set()  # Elem Set
    AflipB = set()  # Elem Set
    BinA = set()    # Elem Set
    BoutA = set()   # Elem Set
    BsameA = set()  # Elem Set
    BflipA = set()  # Elem Set
    
    if len(AllBoundaryA) > 0:
        AllBoundaryACentroids = utils.Centroids(SplitA.NodeCoords,[elem for i,elem in enumerate(SplitA.NodeConn) if i in AllBoundaryA])
        check = rays.PointsInSurf(AllBoundaryACentroids,SplitB.NodeCoords,SplitB.NodeConn,ElemNormalsB,Octree=octB,rays=ElemNormalsA[AllBoundaryA],eps=eps)
        AinB.update(AllBoundaryA[check == True])
        AoutB.update(AllBoundaryA[check == False])
        AsameB.update(AllBoundaryA[check > 0])
        AflipB.update(AllBoundaryA[check <= 0])
    
    if len(AllBoundaryB) > 0:
        AllBoundaryBCentroids = utils.Centroids(SplitB.NodeCoords,[elem for i,elem in enumerate(SplitB.NodeConn) if i in AllBoundaryB])
        check = rays.PointsInSurf(AllBoundaryBCentroids,SplitA.NodeCoords,SplitA.NodeConn,ElemNormalsA,Octree=octA,rays=ElemNormalsB[AllBoundaryB],eps=eps)
        BinA.update(AllBoundaryB[check == True])
        BoutA.update(AllBoundaryB[check == False])
        BsameA.update(AllBoundaryB[check > 0])
        BflipA.update(AllBoundaryB[check <= 0])

    for r in range(len(RegionsA)):
        RegionElems = np.where(np.all(np.isin(SplitA.NodeConn, list(RegionsA[r])),axis=1))[0] # Elements of SplitA in this region
        pt = SplitA.NodeCoords[RegionsA[r].pop()]
        if rays.PointInSurf(pt,SplitB.NodeCoords,SplitB.NodeConn,ElemNormalsB,Octree=octB,eps=eps):
            AinB.update(RegionElems)
        else:
            AoutB.update(RegionElems)
            
    #
    for r in range(len(RegionsB)):
        RegionElems = np.where(np.all(np.isin(SplitB.NodeConn, list(RegionsB[r])),axis=1))[0] # Elements of SplitB in this region
        pt = SplitB.NodeCoords[RegionsB[r].pop()]
        if rays.PointInSurf(pt,SplitA.NodeCoords,SplitA.NodeConn,ElemNormalsA,Octree=octA,eps=eps):
            BinA.update(RegionElems)
        else:
            BoutA.update(RegionElems)

    AinNodes = set(elem for e in AinB for elem in SplitA.NodeConn[e])      # Node Set
    AoutNodes = set(elem for e in AoutB for elem in SplitA.NodeConn[e])    # Node Set    
    AsameNodes = set(); AflipNodes = set()
    BinNodes = set(elem for e in BinA for elem in SplitB.NodeConn[e])      # Node Set
    BoutNodes = set(elem for e in BoutA for elem in SplitB.NodeConn[e])    # Node Set
    BsameNodes = set(); BflipNodes = set()
    UnknownA = set(range(SplitA.NElem)).difference(AinB).difference(AoutB).difference(AsameB).difference(AflipB)
    UnknownB = set(range(SplitB.NElem)).difference(BinA).difference(BoutA).difference(BsameA).difference(BflipA)

    UnknownNodesA = set(elem for e in UnknownA for elem in SplitA.NodeConn[e]).difference(AinNodes).difference(AoutNodes).difference(SharedA)
    UnknownNodesB = set(elem for e in UnknownB for elem in SplitB.NodeConn[e]).difference(BinNodes).difference(BoutNodes).difference(SharedB)
    for node in UnknownNodesA:
        check = rays.PointInSurf(SplitA.NodeCoords[node],SplitB.NodeCoords,SplitB.NodeConn,ElemNormalsB,Octree=octB,ray=SplitA.NodeNormals[node],eps=eps)
        if check is True:
            AinNodes.add(node)
        elif check is False:
            AoutNodes.add(node)
        elif check >= 0:
            AsameNodes.add(node)
        else:
            AflipNodes.add(node)
    for node in UnknownNodesB:
        check = rays.PointInSurf(SplitB.NodeCoords[node],SplitA.NodeCoords,SplitA.NodeConn,ElemNormalsA,Octree=octA,ray=SplitB.NodeNormals[node],eps=eps)
        if check is True:
            BinNodes.add(node)
        elif check is False:
            BoutNodes.add(node)
        elif check >= 0:
            BsameNodes.add(node)
        else:
            BflipNodes.add(node)

    ProblemsA = set()
    ProblemsB = set()
    for e in UnknownA:
        if np.all([n in AinNodes or n in SharedA for n in SplitA.NodeConn[e]]):
            AinB.add(e)
        elif np.all([n in AoutNodes or n in SharedA for n in SplitA.NodeConn[e]]):
            AoutB.add(e)
        elif np.all([n in AsameNodes or n in SharedA for n in SplitA.NodeConn[e]]):
            AsameB.add(e)
        elif np.all([n in AflipNodes or n in SharedA for n in SplitA.NodeConn[e]]):
            AflipB.add(e)
        else:
            ProblemsA.add(e)

    for e in UnknownB:
        if np.all([n in BinNodes or n in SharedB for n in SplitB.NodeConn[e]]):
            BinA.add(e)
        elif np.all([n in BoutNodes or n in SharedB for n in SplitB.NodeConn[e]]):
            BoutA.add(e)
        elif np.all([n in BsameNodes or n in SharedB for n in SplitB.NodeConn[e]]):
            BsameA.add(e)
        elif np.all([n in BflipNodes or n in SharedB for n in SplitB.NodeConn[e]]):
            BflipA.add(e)
        elif np.all([n in BinNodes or n in SharedB or n in BsameNodes or n in BflipNodes for n in SplitB.NodeConn[e]]) and np.any([n in BinNodes for n in SplitB.NodeConn[e]]):
            BinA.add(e)
        elif np.all([n in BoutNodes or n in SharedB or n in BsameNodes or n in BflipNodes for n in SplitB.NodeConn[e]]) and np.any([n in BoutNodes for n in SplitB.NodeConn[e]]):
            BoutA.add(e)
        else:
            ProblemsB.add(e)
    if len(ProblemsB) > 0 or len(ProblemsA) > 0:
        warnings.warn('Some triangles failed to be labeled.')
       
    return AinB, AoutB, AsameB, AflipB, BinA, BoutA, BsameA, BflipA
                        