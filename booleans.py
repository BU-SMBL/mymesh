# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:28:28 2022

@author: toj
"""

import warnings, itertools, copy
from scipy import spatial
import numpy as np
from . import mesh, converter, utils, octree, rays, improvement, delaunay, primitives

def MeshBooleans(Surf1, Surf2, tol=1e-8):
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
    eps = tol/100
    eps_final = tol*10
    # Split Mesh
    Split1, Split2 = SplitMesh(Surf1, Surf2, eps=eps)

    Split1.cleanup(tol=eps,strict=True)
    Split2.cleanup(tol=eps,strict=True)

    # Get Shared Nodes
    Shared1,Shared2 = GetSharedNodes(Split1.NodeCoords, Split2.NodeCoords, eps=tol)
    
    # Classify Tris
    AinB, AoutB, AsameB, AflipB, BinA, BoutA, BsameA, BflipA = ClassifyTris(Split1, Shared1, Split2, Shared2, eps=eps)
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
    MergedUCoords, MergedUConn = utils.MergeMesh(Split1.NodeCoords, AUConn, Split2.NodeCoords, BUConn)
    # MergedUCoords, MergedUConn, _ = converter.removeNodes(MergedUCoords, MergedUConn)
    # MergedUCoords, MergedUConn, _ = utils.DeleteDuplicateNodes(MergedUCoords, MergedUConn)
        
    MergedICoords, MergedIConn = utils.MergeMesh(Split1.NodeCoords, AIConn, Split2.NodeCoords, BIConn)
    # MergedICoords, MergedIConn, _ = converter.removeNodes(MergedICoords, MergedIConn)
    # MergedICoords, MergedIConn, _ = utils.DeleteDuplicateNodes(MergedICoords, MergedIConn)
    
    MergedDCoords, MergedDConn = utils.MergeMesh(Split1.NodeCoords, ADConn, Split2.NodeCoords, np.fliplr(BDConn).tolist())
    # MergedDCoords, MergedDConn, _ = converter.removeNodes(MergedDCoords, MergedDConn)
    # MergedDCoords, MergedDConn, _ = utils.DeleteDuplicateNodes(MergedDCoords, MergedDConn)
    if 'mesh' in dir(mesh):
        Union = mesh.mesh(MergedUCoords,MergedUConn)
        Intersection = mesh.mesh(MergedICoords,MergedIConn)
        Difference = mesh.mesh(MergedDCoords,MergedDConn)
    else:
        Union = mesh(MergedUCoords,MergedUConn)
        Intersection = mesh(MergedICoords,MergedIConn)
        Difference = mesh(MergedDCoords,MergedDConn)

    # Split elements by the Absolute large angle criteria, this will split degenerate collinear elements so that they can be easily cleaned up
    # Union.NodeCoords,Union.NodeConn = improvement.Split(*Union,1,criteria='AbsLargeAngle',iterate=1,thetal=179)
    # Intersection.NodeCoords,Intersection.NodeConn = improvement.Split(*Intersection,1,criteria='AbsLargeAngle',iterate=1,thetal=179)
    # Difference.NodeCoords,Difference.NodeConn = improvement.Split(*Difference,1,criteria='AbsLargeAngle',iterate=1,thetal=179)

    Union.cleanup(tol=eps_final,angletol=5e-3)
    Intersection.cleanup(tol=eps_final,angletol=5e-3)
    Difference.cleanup(tol=eps_final,angletol=5e-3)


    return Union, Intersection, Difference

def PlaneClip(pt, normal, Surf, fill=False, fill_h=None, tol=1e-8, flip=True, return_splitplane=False):
    
    
    Tris = np.asarray(Surf.NodeCoords)[Surf.NodeConn]
    pt = np.asarray(pt)
    normal = np.asarray(normal)/np.linalg.norm(normal)
    sd = np.sum(normal*Tris,axis=2) - np.dot(normal,pt)
    Intersections = ~(np.all((sd < -tol),axis=1) | np.all((sd > tol),axis=1))
    
    if flip:
        Clipped = mesh(Surf.NodeCoords,(np.asarray(Surf.NodeConn)[~Intersections & np.all(sd < 0,axis=1)]))
    else:
        Clipped = mesh(Surf.NodeCoords,(np.asarray(Surf.NodeConn)[~Intersections & np.all(sd > 0,axis=1)]))

    Intersected = mesh(Surf.NodeCoords, [elem for i,elem in enumerate(Surf.NodeConn) if Intersections[i]])
    
    mins = np.min(Surf.NodeCoords,axis=0)
    maxs = np.max(Surf.NodeCoords,axis=0)
    bounds = [mins[0]-tol,maxs[0]+tol,mins[1]-tol,maxs[1]+tol,mins[2]-tol,maxs[2]+tol]
    if fill_h is None:
        fill_h = np.linalg.norm(maxs-mins)/10
    Plane = primitives.Plane(pt, normal, bounds, fill_h, meshobj=True, exact_h=False, ElemType='tri')

    SplitSurf, SplitPlane = SplitMesh(Intersected,Plane) # TODO: This could be done more efficiently for planar case
    SplitSurf.cleanup()
    SplitPlane.cleanup()

    # if fill:
    #     # TODO: Efficiency and Robustness
    #     Shared1,Shared2 = GetSharedNodes(SplitSurf.NodeCoords, SplitPlane.NodeCoords, eps=tol)
    #     root = octree.Surf2Octree(*Surf)
    #     if np.any(np.abs(normal) != [0,0,1]):
    #         ray = np.cross(normal, [0,0,1])
    #     else:
    #         ray = np.cross(normal, [1,0,0])

    #     AllBoundary = [i for i,elem in enumerate(SplitPlane.NodeConn) if all([n in Shared2 for n in elem])]  
        
    #     NotSharedConn = [elem for i,elem in enumerate(SplitPlane.NodeConn) if not any([n in Shared2 for n in elem]) and i not in AllBoundary] 
    #     SharedConn = [elem for i,elem in enumerate(SplitPlane.NodeConn) if any([n in Shared2 for n in elem]) and i not in AllBoundary]  
        
    #     Regions = utils.getConnectedElements(SplitPlane.NodeCoords,NotSharedConn)  # Node Sets
    #     FillConn = []
    #     for region in Regions:
    #         RegionConn = [SplitPlane.NodeConn[i] for i in region]
    #         point = utils.Centroids(SplitPlane.NodeCoords,[RegionConn[0]])
    #         inside = rays.isInsideSurf(point, Surf.NodeCoords, Surf.NodeConn, Surf.ElemNormals, Octree=None, eps=1e-8, ray=ray)
    #         if inside:
    #             FillConn += RegionConn
    #     InsideNodes = set(np.unique(FillConn))
    #     OutsideNodes = set(range(SplitPlane.NNode)).difference(InsideNodes)

    #     centroids = utils.Centroids(SplitPlane.NodeCoords,SharedConn)
    #     for i,elem in enumerate(SharedConn):
    #         if all([n in Shared2 or n in InsideNodes for n in elem]):
    #             FillConn.append(elem)
    #         elif all([n in Shared2 or n in InsideNodes for n in elem]):
    #             continue
    #         else:
    #             centroid = centroids[i]
    #             if rays.isInsideSurf(centroid, Surf.NodeCoords, Surf.NodeConn, Surf.ElemNormals, Octree=None, eps=1e-8, ray=ray):
    #                 FillConn.append(elem)
    #             #     InsideNodes.update([n for n in elem if n not in Shared2])
    #             # else:
    #             #     OutsideNodes.update([n for n in elem if n not in Shared2])


    SplitTris = np.asarray(SplitSurf.NodeCoords)[SplitSurf.NodeConn]
    sd2 = np.sum(normal*SplitTris,axis=2) - np.dot(normal,pt)

    if flip:
        Clipped2 = mesh(SplitSurf.NodeCoords,(np.asarray(SplitSurf.NodeConn)[np.all(sd2 <= tol,axis=1)]))
        # if fill:
        #     Clipped2.addElems(FillConn)
    else:
        Clipped2 = mesh(SplitSurf.NodeCoords,(np.asarray(SplitSurf.NodeConn)[np.all(sd2 >= -tol,axis=1)]))
        # if fill:
        #     Clipped2.addElems(FillConn)
    Clipped.merge(Clipped2)
    Clipped.cleanup()
    if return_splitplane:
        return Clipped,SplitPlane
    return Clipped
       
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

def SplitMesh(Surf1, Surf2, eps=1e-12):
    
    Surf1Intersections,Surf2Intersections,IntersectionPts = rays.SurfSurfIntersection(*Surf1,*Surf2,return_pts=True)
    
    SurfIntersections12 = [Surf1Intersections,Surf2Intersections]
    
    Surf12 = [Surf1.copy(), Surf2.copy()]
    for i,surf in enumerate(Surf12):
        # Group nodes for each triangle
        SurfIntersections = SurfIntersections12[i]
        ArrayCoords = np.array(surf.NodeCoords)
        SplitGroupNodes = [ArrayCoords[elem] for elem in surf.NodeConn]
        for j,elemid in enumerate(SurfIntersections):
            if len(IntersectionPts[j]) == 2:
                SplitGroupNodes[elemid] = np.append(SplitGroupNodes[elemid], IntersectionPts[j], axis=0)
            else:
                for k in range(len(IntersectionPts[j])):
                    SplitGroupNodes[elemid] = np.append(SplitGroupNodes[elemid], [IntersectionPts[j][k],IntersectionPts[j][(k+1)%len(IntersectionPts[j])]], axis=0)
        
        ElemNormals = utils.CalcFaceNormal(*surf)
        for j in range(surf.NElem):
            if len(SplitGroupNodes[j]) > 3:
                # print(j)
                # if j == 3595:
                #     meep = 'morp'
                n = ElemNormals[j]
                # Set edge constraints
                Constraints = np.transpose([np.arange(3,len(SplitGroupNodes[j]),2), np.arange(4,len(SplitGroupNodes[j]),2)])
                Constraints = np.append([[0,1],[1,2],[2,0]],Constraints,axis=0)
                boundary = SplitGroupNodes[j][:3]
                # Reduce node list
                SplitGroupNodes[j],_,newId,idx = utils.DeleteDuplicateNodes(SplitGroupNodes[j],[],return_idx=True,tol=eps)
                # Tris = np.repeat([boundary],len(SplitGroupNodes[j]),axis=0)
                # In = rays.PointsInTris(Tris,SplitGroupNodes[j],eps=eps)

                Constraints = newId[Constraints]
                Constraints = np.unique(np.sort(Constraints,axis=1),axis=0)
                # # Constraints = np.array([c for c in Constraints if c[0] != c[1]])
                # bk = copy.copy(Constraints)
                # pbk = copy.copy(SplitGroupNodes[j])
                # # Check for intersections within the constraints
                # combinations = np.array(list(itertools.combinations(range(len(Constraints)),2)))
                # e1 = SplitGroupNodes[j][Constraints[combinations[:,0]]]
                # e2 = SplitGroupNodes[j][Constraints[combinations[:,1]]]
                # eIntersections,eIntersectionPts = rays.SegmentsSegmentsIntersection(e1,e2,return_intersection=True,endpt_inclusive=True,eps=eps)
                # NewConstraints = np.empty((0,2),dtype=int)
                # for ic,c in enumerate(Constraints):
                #     # ids of other constraints that intersect with this constraint
                #     ids = np.unique(np.array([combo for combo in combinations[eIntersections] if ic in combo]))
                #     # Check collinear lines - currently rays.SegmentsSegmentsIntersection doesn't do this properly
                #     coix = np.empty((0,3))
                #     for combo in combinations[~eIntersections]:
                #         if ic not in combo:
                #             continue
                #         segments = np.vstack(SplitGroupNodes[j][Constraints[combo]])
                #         if np.linalg.norm(SplitGroupNodes[j][Constraints[combo[0]][0]] - 
                #                                 SplitGroupNodes[j][Constraints[combo[0]][1]]) > eps:
                #             check1 = np.linalg.norm(np.cross(segments[1]-segments[0], segments[2]-segments[0])) < eps
                #             check2 = np.linalg.norm(np.cross(segments[1]-segments[0], segments[3]-segments[0])) < eps
                #         elif np.linalg.norm(SplitGroupNodes[j][Constraints[combo[1]][0]] - 
                #                                 SplitGroupNodes[j][Constraints[combo[1]][1]]) > eps:
                #             check1 = np.linalg.norm(np.cross(segments[3]-segments[2], segments[0]-segments[2])) < eps
                #             check2 = np.linalg.norm(np.cross(segments[3]-segments[2], segments[1]-segments[2])) < eps
                #         else:
                #             continue
                #         if check1 and check2:
                #             # For segments AB and CD if the (double) area of the triangles ABC and ABD are both (near) zero, the segments are collinear
                #             # segsort = np.lexsort(segments.T) # Lexographic sort of the segments
                #             segsort = np.lexsort((np.round(segments/eps)*eps).T) 
                #             if (0 in segsort[:2] and 1 in segsort[:2]) or (2 in segsort[:2] and 3 in segsort[:2]):
                #                 # if both points if a segment are on the same side of the other, they at most intersect at an endpt
                #                 if np.linalg.norm(np.diff(segments[segsort[1:3]],axis=0)) < eps: # Norm of the difference between interior points
                #                     # end point intersection
                #                     coix = np.append(coix,[segments[segsort[1]]],axis=0)
                #             else:
                #                 # overlapping segments, get the two interior points
                #                 coix = np.append(coix,segments[segsort[1:3]],axis=0)
                #         elif check1:
                #             segsort = np.lexsort(segments[[0,1,2]].T)
                #             if segsort[1] == 2:
                #                 coix = np.append(coix,[segments[2]],axis=0)
                #         elif check2:
                #             segsort = np.lexsort(segments[[0,1,3]].T)
                #             if segsort[1] == 2:
                #                 coix = np.append(coix,[segments[3]],axis=0)
                                
                #     ids = np.delete(ids,ids==ic)
                #     if len(ids) == 0:
                #         NewConstraints = np.append(NewConstraints,[c],axis=0)
                #     else:
                #         # corresponding intersection points
                #         ixs = np.array([eIntersectionPts[eIntersections][x] for x,combo in enumerate(combinations[eIntersections]) if ic in combo])
                #         ixs = np.append(ixs,coix,axis=0)
                #         ixsort = ixs[np.lexsort((np.round(ixs/eps)*eps).T)]

                #         NewConstraints = np.append(NewConstraints,np.vstack([np.arange(0,len(ixsort)-1),np.arange(1,len(ixsort))]).T+len(SplitGroupNodes[j]),axis=0)
                #         SplitGroupNodes[j] = np.append(SplitGroupNodes[j],ixsort,axis=0)

                # SplitGroupNodes[j], Constraints, _ = utils.DeleteDuplicateNodes(SplitGroupNodes[j],NewConstraints,tol=eps)
                # Constraints = np.unique([c for c in Constraints if c[0] != c[1]],axis=0)
                # SplitGroupNodes[j] = np.asarray(SplitGroupNodes[j])
                    # import plotly.graph_objects as go
                    # fig = go.Figure()
                    # for i in range(len(Constraints)):
                    #     fig.add_trace(go.Scatter(x=SplitGroupNodes[j][Constraints[i]][:,0], y=SplitGroupNodes[j][Constraints[i]][:,1],text=Constraints[i]))
                    # fig.show()

                # Transform to Local Coordinates
                # Rotation matrix from global z (k=[0,0,1]) to local z(n)
                k=[0,0,1]
                if n == k or n == [0,0,-1]:
                    # rotAxis = k
                    # angle = 0
                    R = np.eye(3)
                    flatnodes = SplitGroupNodes[j]#[:,0:2]

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
                    flatnodes = np.matmul(R,np.transpose(SplitGroupNodes[j])).T#[:,0:2]#.tolist()
                    
                # conn = spatial.Delaunay(flatnodes,qhull_options="Qbb Qc Qz Q12").simplices   
                ###
                
                
                # coords, conn = delaunay.Triangulate(flatnodes[:,0:2],method='scipy',tol=eps)  
                ## conn = delaunay.Triangulate(flatnodes,Constraints=idx[Constraints],method='Flips')     
                coords, conn = delaunay.Triangulate(flatnodes[:,0:2],method='Triangle',Constraints=Constraints,tol=eps)  
                ## coords, conn = delaunay.Triangulate(flatnodes[:,0:2],method='scipy',Constraints=None,tol=eps)        
                ## coords,conn = delaunay.Triangle(flatnodes[:,0:2],Constraints=Constraints)
                SplitGroupNodes[j] = np.matmul(np.linalg.inv(R),np.append(coords,flatnodes[0,2]*np.ones((len(coords),1)),axis=1).T).T
                ###
                flip = [np.dot(utils.CalcFaceNormal(SplitGroupNodes[j],[conn[i]]), n)[0] < 0 for i in range(len(conn))]
                conn = (conn+surf.NNode).tolist()
                surf.addElems([elem[::-1] if flip[i] else elem for i,elem in enumerate(conn)])
                surf.addNodes(SplitGroupNodes[j].tolist())
        # Collinear check
        Edges = converter.solid2edges(*surf,ElemType='tri')
        ArrayCoords = np.array(surf.NodeCoords)
        EdgePoints = ArrayCoords[Edges]
        ElemPoints = ArrayCoords[surf.NodeConn]
        A2 = np.linalg.norm(np.cross(ElemPoints[:,1]-ElemPoints[:,0],ElemPoints[:,2]-ElemPoints[:,0]),axis=1)
        EdgeLen = np.max(np.linalg.norm(EdgePoints[:,0]-EdgePoints[:,1],axis=1).reshape((int(len(Edges)/3),3)),axis=1)
        deviation = A2/EdgeLen # the double area devided by the longest side gives the deviation of the middple point from the line of the other two
        
        iset = set(SurfIntersections)
        colset = set(np.where(deviation<eps/2)[0])
        surf.NodeConn = [elem for i,elem in enumerate(surf.NodeConn) if i not in iset and i not in colset]
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
                           
def ClassifyTris(SplitA, SharedA, SplitB, SharedB, eps=1e-10):
    # Classifies each Triangle in A as inside, outside, or on the surface facing the same or opposite direction as surface B
    
    
    octA = None# octree.Surf2Octree(*SplitA)
    octB = None#octree.Surf2Octree(*SplitB)
    AllBoundaryA = [i for i,elem in enumerate(SplitA.NodeConn) if all([n in SharedA for n in elem])]  
    AllBoundaryB = [i for i,elem in enumerate(SplitB.NodeConn) if all([n in SharedB for n in elem])]  
    NotSharedConnA = [elem for i,elem in enumerate(SplitA.NodeConn) if not any([n in SharedA for n in elem]) and i not in AllBoundaryA]  
    NotSharedConnB = [elem for i,elem in enumerate(SplitB.NodeConn) if not any([n in SharedB for n in elem]) and i not in AllBoundaryB]  
    
    if len(NotSharedConnA) > 0:
        RegionsA = utils.getConnectedNodes(SplitA.NodeCoords,NotSharedConnA)  # Node Sets
    else:
        RegionsA = []
    if len(NotSharedConnB) > 0:
        RegionsB = utils.getConnectedNodes(SplitB.NodeCoords,NotSharedConnB)  # Node Sets
    else:
        RegionsB = []
        
    ElemNormalsA = utils.CalcFaceNormal(*SplitA)
    ElemNormalsB = utils.CalcFaceNormal(*SplitB)

    AinB = set()    # Elem Set
    AoutB = set()   # Elem Set
    AsameB = set()  # Elem Set
    AflipB = set()  # Elem Set
    BinA = set()    # Elem Set
    BoutA = set()   # Elem Set
    BsameA = set()  # Elem Set
    BflipA = set()  # Elem Set
    
    AllBoundaryACentroids = utils.Centroids(SplitA.NodeCoords,[elem for i,elem in enumerate(SplitA.NodeConn) if i in AllBoundaryA])
    AllBoundaryBCentroids = utils.Centroids(SplitB.NodeCoords,[elem for i,elem in enumerate(SplitB.NodeConn) if i in AllBoundaryB])
    for i,centroid in enumerate(AllBoundaryACentroids):
        check = rays.isInsideSurf(centroid,SplitB.NodeCoords,SplitB.NodeConn,ElemNormalsB,Octree=octB,ray=ElemNormalsA[AllBoundaryA[i]]+np.random.rand(3)/1000,eps=eps)
        if check is True:
            AinB.add(AllBoundaryA[i])
        elif check is False:
            AoutB.add(AllBoundaryA[i])
        elif check > 0:
            AsameB.add(AllBoundaryA[i])
        else:
            AflipB.add(AllBoundaryA[i])
    for i,centroid in enumerate(AllBoundaryBCentroids):
        check = rays.isInsideSurf(centroid,SplitA.NodeCoords,SplitA.NodeConn,ElemNormalsA,Octree=octA,ray=ElemNormalsB[AllBoundaryB[i]]+np.random.rand(3)/1000,eps=eps)
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
        if rays.isInsideSurf(pt,SplitB.NodeCoords,SplitB.NodeConn,ElemNormalsB,Octree=octB,eps=eps):
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
        if rays.isInsideSurf(pt,SplitA.NodeCoords,SplitA.NodeConn,ElemNormalsA,Octree=octA,eps=eps):
            # for e in RegionElems: BinA.add(e)
            BinA.update(RegionElems)
        else:
            # for e in RegionElems: BoutA.add(e)
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
        check = rays.isInsideSurf(SplitA.NodeCoords[node],SplitB.NodeCoords,SplitB.NodeConn,ElemNormalsB,Octree=octB,ray=SplitA.NodeNormals[node],eps=eps)
        if check is True:
            AinNodes.add(node)
        elif check is False:
            AoutNodes.add(node)
        elif check >= 0:
            AsameNodes.add(node)
        else:
            AflipNodes.add(node)
    for node in UnknownNodesB:
        check = rays.isInsideSurf(SplitB.NodeCoords[node],SplitA.NodeCoords,SplitA.NodeConn,ElemNormalsA,Octree=octA,ray=SplitB.NodeNormals[node],eps=eps)
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
                        