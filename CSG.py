# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:28:28 2022

@author: toj
"""

from scipy import spatial
import numpy as np
from . import mesh, converter, MeshUtils, Octree, Rays, Improvement

def MeshBooleans(Surf1, Surf2, tol=1e-6):
    """
    MeshBooleans summary
    https://dl.acm.org/doi/pdf/10.1145/15922.15904

    Parameters
    ----------
    Surf1 : mesh.mesh
        Mesh object containing a surface mesh
    Surf2 : mesh.mesh
        Mesh object containing a surface mesh
    tol : type, optional
        Tolerance value, by default 1e-6

    Returns
    -------
    Union : mesh.mesh
        Mesh object containing the union of the two input surfaces
    Intersection : mesh.mesh
        Mesh object containing the intersection of the two input surfaces
    Difference : mesh.mesh
        Mesh object containing the difference of the two input surfaces
    """    
    
    # Split Mesh
    Split1, Split2 = SplitMesh(Surf1, Surf2, eps=1e-14)

    Split1.cleanup(tol=tol,strict=True)
    Split2.cleanup(tol=tol,strict=True)

    # Get Shared Nodes
    Shared1,Shared2 = GetSharedNodes(Split1.NodeCoords, Split2.NodeCoords, eps=tol)
    
    # Classify Tris
    AinB, AoutB, AsameB, AflipB, BinA, BoutA, BsameA, BflipA = ClassifyTris(Split1, Shared1, Split2, Shared2)
    # Perform Boolean Operations
    # Union
    AUtris = AoutB.union(AsameB)
    BUtris = BoutA
    AUConn = [elem for e,elem in enumerate(Split1.NodeConn) if e in AUtris]
    BUConn = [elem for e,elem in enumerate(Split2.NodeConn) if e in BUtris]
    # Intersection
    AItris = AinB.union(AsameB)
    BItris = BinA
    AIConn = [elem for e,elem in enumerate(Split1.NodeConn) if e in AItris]
    BIConn = [elem for e,elem in enumerate(Split2.NodeConn) if e in BItris]
    # Difference
    ADtris = AoutB.union(AflipB)
    BDtris = BinA
    ADConn = [elem for e,elem in enumerate(Split1.NodeConn) if e in ADtris]
    BDConn = [elem for e,elem in enumerate(Split2.NodeConn) if e in BDtris]
    
    # Merge and Cleanup Mesh
    MergedUCoords, MergedUConn = MeshUtils.MergeMesh(Split1.NodeCoords, AUConn, Split2.NodeCoords, BUConn)
    # MergedUCoords, MergedUConn, _ = converter.removeNodes(MergedUCoords, MergedUConn)
    # MergedUCoords, MergedUConn, _ = MeshUtils.DeleteDuplicateNodes(MergedUCoords, MergedUConn)
        
    MergedICoords, MergedIConn = MeshUtils.MergeMesh(Split1.NodeCoords, AIConn, Split2.NodeCoords, BIConn)
    # MergedICoords, MergedIConn, _ = converter.removeNodes(MergedICoords, MergedIConn)
    # MergedICoords, MergedIConn, _ = MeshUtils.DeleteDuplicateNodes(MergedICoords, MergedIConn)
    
    MergedDCoords, MergedDConn = MeshUtils.MergeMesh(Split1.NodeCoords, ADConn, Split2.NodeCoords, BDConn)
    # MergedDCoords, MergedDConn, _ = converter.removeNodes(MergedDCoords, MergedDConn)
    # MergedDCoords, MergedDConn, _ = MeshUtils.DeleteDuplicateNodes(MergedDCoords, MergedDConn)
    
    Union = mesh.mesh(MergedUCoords,MergedUConn)
    Intersection = mesh.mesh(MergedICoords,MergedIConn)
    Difference = mesh.mesh(MergedDCoords,MergedDConn)

    # Split elements by the Absolute large angle criteria, this will split degenerate collinear elements so that they can be easily cleaned up
    # Union.NodeCoords,Union.NodeConn = Improvement.Split(*Union,1,criteria='AbsLargeAngle',iterate=1,thetal=179)
    # Intersection.NodeCoords,Intersection.NodeConn = Improvement.Split(*Intersection,1,criteria='AbsLargeAngle',iterate=1,thetal=179)
    # Difference.NodeCoords,Difference.NodeConn = Improvement.Split(*Difference,1,criteria='AbsLargeAngle',iterate=1,thetal=179)

    Union.cleanup(tol=tol,angletol=5e-3)
    Intersection.cleanup(tol=tol,angletol=5e-3)
    Difference.cleanup(tol=tol,angletol=5e-3)


    return Union, Intersection, Difference
               
def VoxelIntersect(VoxelCoordsA, VoxelConnA, VoxelCoordsB, VoxelConnB):
    # Requires Voxel meshes that exsits within the same grid
    centroidsA = [np.mean([VoxelCoordsA[n] for n in elem],axis=0).tolist() for elem in VoxelConnA]
    centroidsB = set([tuple(np.mean([VoxelCoordsB[n] for n in elem],axis=0).tolist()) for elem in VoxelConnB])
    
    IConn = [elem for i,elem in enumerate(VoxelConnA) if tuple(centroidsA[i]) in centroidsB]
    ICoords,IConn,_ = converter.removeNodes(VoxelCoordsA, IConn)
    return ICoords, IConn
    
def VoxelDifference(VoxelCoordsA, VoxelConnA, VoxelCoordsB, VoxelConnB):
    # Requires Voxel meshes that exsits within the same grid
    centroidsA = [np.mean([VoxelCoordsA[n] for n in elem],axis=0).tolist() for elem in VoxelConnA]
    centroidsB = set([tuple(np.mean([VoxelCoordsB[n] for n in elem],axis=0).tolist()) for elem in VoxelConnB])
    
    DConn = [elem for i,elem in enumerate(VoxelConnA) if tuple(centroidsA[i]) not in centroidsB]
    DCoords,DConn,_ = converter.removeNodes(VoxelCoordsA, DConn)
    return DCoords, DConn

def SplitMesh(Surf1, Surf2, eps=1e-14):
    
    Surf1Intersections,Surf2Intersections,IntersectionPts = Rays.SurfSurfIntersection(*Surf1,*Surf2,return_pts=True)
    
    SurfIntersections12 = [Surf1Intersections,Surf2Intersections]
    
    Surf12 = [Surf1.copy(), Surf2.copy()]
    for i,surf in enumerate(Surf12):
        SurfIntersections = SurfIntersections12[i]
        ArrayCoords = np.array(surf.NodeCoords)
        SplitGroupNodes = [ArrayCoords[elem] for elem in surf.NodeConn]
        for j,elemid in enumerate(SurfIntersections):
            SplitGroupNodes[elemid] = np.append(SplitGroupNodes[elemid], IntersectionPts[j], axis=0)

        ElemNormals = MeshUtils.CalcFaceNormal(*surf)
        for j in range(surf.NElem):
            if len(SplitGroupNodes[j]) > 3:
                n = ElemNormals[j]

                # Transform to Local Coordinates
                # Rotation matrix from global z (k=[0,0,1]) to local z(n)
                k=[0,0,1]
                if n == k or n == [0,0,-1]:
                    # rotAxis = k
                    # angle = 0
                    flatnodes = SplitGroupNodes[j][:,0:2]

                else:
                    kxn = np.cross(k,n)
                    rotAxis = kxn/np.linalg.norm(kxn)
                    angle = -np.arccos(np.dot(k,n))
                    q = [np.cos(angle/2),               # Quaternion Rotation
                            rotAxis[0]*np.sin(angle/2),
                            rotAxis[1]*np.sin(angle/2),
                            rotAxis[2]*np.sin(angle/2)]
                
                    R = [[2*(q[0]**2+q[1]**2)-1,   2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
                            [2*(q[1]*q[2]+q[0]*q[3]), 2*(q[0]**2+q[2]**2)-1,   2*(q[2]*q[3]-q[0]*q[1])],
                            [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 2*(q[0]**2+q[3]**2)-1]
                            ]

                    # Delaunay Triangulation to retriangulate the split face
                    flatnodes = np.matmul(R,np.transpose(SplitGroupNodes[j])).T[:,0:2].tolist()

                conn = spatial.Delaunay(flatnodes,qhull_options="Qbb Qc Qz Q12").simplices
                flip = [np.dot(MeshUtils.CalcFaceNormal(SplitGroupNodes[j],[conn[i]]), n)[0] < 0 for i in range(len(conn))]
                conn = (conn+surf.NNode).tolist()
                surf.addElems([elem[::-1] if flip[i] else elem for i,elem in enumerate(conn)])
                surf.addNodes(SplitGroupNodes[j].tolist())
        iset = set(SurfIntersections)
        surf.NodeConn = [elem for i,elem in enumerate(surf.NodeConn) if i not in iset]
    Split1, Split2 = Surf12

    return Split1, Split2
    
def GetSharedNodes(NodeCoordsA, NodeCoordsB, eps=1e-10):
    
    RoundCoordsA = np.round(np.asarray(NodeCoordsA)/eps)*eps
    RoundCoordsB = np.round(np.asarray(NodeCoordsB)/eps)*eps

    setA = set(tuple(coord) for coord in RoundCoordsA)
    setB = set(tuple(coord) for coord in RoundCoordsB)
    setI = setA.intersection(setB)
    SharedA = {i for i,coord in enumerate(RoundCoordsA) if tuple(coord) in setI}
    SharedB = {i for i,coord in enumerate(RoundCoordsB) if tuple(coord) in setI}

    # setA = set((round(coord[0],tol), round(coord[1],tol), round(coord[2],tol)) for coord in NodeCoordsA)
    # setB = set((round(coord[0],tol), round(coord[1],tol), round(coord[2],tol)) for coord in NodeCoordsB)
    # setI = setA.intersection(setB)
    
    # SharedA = {i for i,coord in enumerate(NodeCoordsA) if (round(coord[0],tol), round(coord[1],tol), round(coord[2],tol)) in setI}
    # SharedB = {i for i,coord in enumerate(NodeCoordsB) if (round(coord[0],tol), round(coord[1],tol), round(coord[2],tol)) in setI}
    
    return SharedA, SharedB
                           
def ClassifyTris(SplitA, SharedA, SplitB, SharedB):
    # Classifies each Triangle in A as inside, outside, or on the surface facing the same or opposite direction as surface B
    
    
    octA = None# Octree.Surf2Octree(*SplitA)
    octB = None#Octree.Surf2Octree(*SplitB)
    AllBoundaryA = [i for i,elem in enumerate(SplitA.NodeConn) if all([n in SharedA for n in elem])]  
    AllBoundaryB = [i for i,elem in enumerate(SplitB.NodeConn) if all([n in SharedB for n in elem])]  
    NotSharedConnA = [elem for i,elem in enumerate(SplitA.NodeConn) if not any([n in SharedA for n in elem]) and i not in AllBoundaryA]  
    NotSharedConnB = [elem for i,elem in enumerate(SplitB.NodeConn) if not any([n in SharedB for n in elem]) and i not in AllBoundaryB]  
    
    RegionsA = MeshUtils.getConnectedNodes(SplitA.NodeCoords,NotSharedConnA)  # Node Sets
    RegionsB = MeshUtils.getConnectedNodes(SplitB.NodeCoords,NotSharedConnB)  # Node Sets

    ElemNormalsA = MeshUtils.CalcFaceNormal(*SplitA)
    ElemNormalsB = MeshUtils.CalcFaceNormal(*SplitB)

    AinB = set()    # Elem Set
    AoutB = set()   # Elem Set
    AsameB = set()  # Elem Set
    AflipB = set()  # Elem Set
    BinA = set()    # Elem Set
    BoutA = set()   # Elem Set
    BsameA = set()  # Elem Set
    BflipA = set()  # Elem Set
    
    AllBoundaryACentroids = MeshUtils.Centroids(SplitA.NodeCoords,[elem for i,elem in enumerate(SplitA.NodeConn) if i in AllBoundaryA])
    AllBoundaryBCentroids = MeshUtils.Centroids(SplitB.NodeCoords,[elem for i,elem in enumerate(SplitB.NodeConn) if i in AllBoundaryB])
    for i,centroid in enumerate(AllBoundaryACentroids):
        check = Rays.isInsideSurf(centroid,SplitB.NodeCoords,SplitB.NodeConn,ElemNormalsB,octree=octB,ray=ElemNormalsA[AllBoundaryA[i]]+np.random.rand(3)/1000)
        if check is True:
            AinB.add(AllBoundaryA[i])
        elif check is False:
            AoutB.add(AllBoundaryA[i])
        elif check > 0:
            AsameB.add(AllBoundaryA[i])
        else:
            AflipB.add(AllBoundaryA[i])
    for i,centroid in enumerate(AllBoundaryBCentroids):
        check = Rays.isInsideSurf(centroid,SplitA.NodeCoords,SplitA.NodeConn,ElemNormalsA,octree=octA,ray=ElemNormalsB[AllBoundaryB[i]]+np.random.rand(3)/1000)
        if check is True:
            BinA.add(AllBoundaryB[i])
        elif check is False:
            BoutA.add(AllBoundaryB[i])
        elif check > 0:
            BsameA.add(AllBoundaryB[i])
        else:
            BflipA.add(AllBoundaryB[i])

    for r in range(len(RegionsA)):
        RegionElems = [e for e in range(len(SplitA.NodeConn)) if all([n in RegionsA[r] for n in SplitA.NodeConn[e]])] # Elem Set
        # centroid = np.mean([SplitA.NodeCoords[n] for n in SplitA.NodeConn[RegionElems[0]]], axis=0)
        # normal = ElemNormalsA[RegionElems[0]]
        pt = SplitA.NodeCoords[RegionsA[r].pop()]
        if Rays.isInsideSurf(pt,SplitB.NodeCoords,SplitB.NodeConn,ElemNormalsB,octree=octB):
            # for e in RegionElems: AinB.add(e)
            AinB.update(RegionElems)
        else:
            # for e in RegionElems: AoutB.add(e)
            AoutB.update(RegionElems)
            
    #
    for r in range(len(RegionsB)):
        RegionElems = [e for e in range(len(SplitB.NodeConn)) if all([n in RegionsB[r] for n in SplitB.NodeConn[e]])] # Elem Set
        # centroid = np.mean([SplitB.NodeCoords[n] for n in SplitB.NodeConn[RegionElems[0]]], axis=0)
        pt = SplitB.NodeCoords[RegionsB[r].pop()]
        if Rays.isInsideSurf(pt,SplitA.NodeCoords,SplitA.NodeConn,ElemNormalsA,octree=octA):
            # for e in RegionElems: BinA.add(e)
            BinA.update(RegionElems)
        else:
            # for e in RegionElems: BoutA.add(e)
            BoutA.update(RegionElems)

    AinNodes = set(elem for e in AinB for elem in SplitA.NodeConn[e])      # Node Set
    AoutNodes = set(elem for e in AoutB for elem in SplitA.NodeConn[e])    # Node Set    

    BinNodes = set(elem for e in BinA for elem in SplitB.NodeConn[e])      # Node Set
    BoutNodes = set(elem for e in BoutA for elem in SplitB.NodeConn[e])    # Node Set

    UnknownA = set(range(SplitA.NElem)).difference(AinB).difference(AoutB).difference(AsameB).difference(AflipB)
    UnknownB = set(range(SplitB.NElem)).difference(BinA).difference(BoutA).difference(BsameA).difference(BflipA)

    UnknownNodesA = set(elem for e in UnknownA for elem in SplitA.NodeConn[e]).difference(AinNodes).difference(AoutNodes).difference(SharedA)
    UnknownNodesB = set(elem for e in UnknownB for elem in SplitB.NodeConn[e]).difference(BinNodes).difference(BoutNodes).difference(SharedB)
    for node in UnknownNodesA:
        if Rays.isInsideSurf(SplitA.NodeCoords[node],SplitB.NodeCoords,SplitB.NodeConn,ElemNormalsB,octree=octB):
            AinNodes.add(node)
        else:
            AoutNodes.add(node)
    for node in UnknownNodesB:
        if Rays.isInsideSurf(SplitB.NodeCoords[node],SplitA.NodeCoords,SplitA.NodeConn,ElemNormalsA,octree=octA):
            BinNodes.add(node)
        else:
            BoutNodes.add(node)

    ProblemsA = set()
    ProblemsB = set()
    for e in UnknownA:
        if np.all([n in AinNodes or n in SharedA for n in SplitA.NodeConn[e]]):
            AinB.add(e)
        elif np.all([n in AoutNodes or n in SharedA for n in SplitA.NodeConn[e]]):
            AoutB.add(e)
        else:
            ProblemsA.add(e)

    for e in UnknownB:
        if np.all([n in BinNodes or n in SharedB for n in SplitB.NodeConn[e]]):
            BinA.add(e)
        elif np.all([n in BoutNodes or n in SharedB for n in SplitB.NodeConn[e]]):
            BoutA.add(e)
        else:
            ProblemsB.add(e)
       
    return AinB, AoutB, AsameB, AflipB, BinA, BoutA, BsameA, BflipA
                        