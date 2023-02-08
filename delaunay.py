# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 12:02:26 2022

@author: toj
"""
#%%
import numpy as np
import sys, copy, itertools
from . import *
from . import MeshUtils, Rays, converter
from scipy import spatial

try:
    import triangle # pip install triangle
    from plotly.offline import plot
    import plotly.graph_objects as go
except:
    warnings.warn('Optional dependencies not found - some functions may not work properly')

def Triangulate(NodeCoords,Constraints=None,method='Flips',tol=1e-8):
    """
    Triangulate _summary_

    Parameters
    ----------
    NodeCoords : _type_
        _description_
    Constraints : _type_, optional
        _description_, by default None
    method : str, optional
        _description_, by default 'Flips'
        'Flips' - Generate a Delaunay triangulation by a flipping algorithm
        'IncrementalFlips' - Generate a Delaunay triangulation by an incremental insertion flipping algorithm
        'NonDelaunay' - Generate a non-Delaunay triangulation by triangle splitting
        'scipy' - Use scipy.spatial.delaunay
        'Triangle' - Use Jonathon Shewchuk's Delaunay triangulator
    """    
    
    Points = np.asarray(NodeCoords)

    if Constraints is None or len(Constraints) == 0:
        Points,_,newId,idx = MeshUtils.DeleteDuplicateNodes(Points,[],return_idx=True,tol=tol)
        if method == 'NonDelauanay':
            Hull = ConvexHull_GiftWrapping(Points,IncludeCollinear=True)
            NodeConn = idx[TriangleSplittingTriangulation(Points,Hull=Hull)]
        
        elif method == 'Flips':
            Hull = ConvexHull_GiftWrapping(Points,IncludeCollinear=True)
            InitialConn = TriangleSplittingTriangulation(Points,Hull=Hull)
            NodeConn = idx[DelaunayFlips(Points,InitialConn)]
        
        elif method == 'IncrementalFlips':
            Hull = ConvexHull_GiftWrapping(Points,IncludeCollinear=True)
            NodeConn = ConvexHullFanTriangulation(Hull)
            interior = np.setdiff1d(np.arange(len(Points)),Hull,assume_unique=True)
            for i in interior:
                alpha,beta,gamma = MeshUtils.BaryTris(Points[NodeConn],Points[i])
            
                # currently not using special treatment for nodes on boundaries
                inside = (alpha >= 0-tol) & (beta >= 0-tol) & (gamma >= 0-tol)
                TriId = np.where(inside)[0][0]
                Elem = copy.copy(NodeConn[TriId])
                NodeConn[TriId] = [Elem[0],Elem[1],i]
                NodeConn = np.append(NodeConn,[[Elem[1],Elem[2],i],[Elem[2],Elem[0],i]],axis=0)
                NodeConn = DelaunayFlips(Points,NodeConn)
            NodeConn = idx[NodeConn]
        elif method == 'scipy':
            NodeConn = idx[SciPy(Points)]
        elif method == 'Triangle':
            NodeConn = idx[Triangle(Points)]
        else:
            raise Exception('Invalid method.')
    else: 
        Points,Constraints = SplitConstraints_2d(Points,Constraints)#,tol=tol)
        # Constrained Deluanay Traingulation - Sloan (1993)
        # Generate initial triangulation
        if method == 'Triangle':
            Points, NodeConn = Triangle(Points,Constraints=Constraints)
        ###
        else:
            bk = copy.copy(Constraints)
            pbk = copy.copy(NodeCoords)
            ###
            # Check constraints for intersections (Shouldn't be necessary here):
            # if len(Constraints) > 1:
            #     combinations = np.array(list(itertools.combinations(range(len(Constraints)),2)))
            #     e1 = NodeCoords[Constraints[combinations[:,0]]]
            #     e2 = NodeCoords[Constraints[combinations[:,1]]]
            #     eIntersections,eIntersectionPts = Rays.SegmentsSegmentsIntersection(np.append(e1,np.zeros((e1.shape[0],e1.shape[1],1)),axis=2),np.append(e2,np.zeros((e2.shape[0],e2.shape[1],1)),axis=2),return_intersection=True, endpt_inclusive=False,eps=1e-14)
            #     if np.any(eIntersections):
            #         print('aaaa')
                    # raise Exception('Invalid constraints - the following constraint pairs intersect each other: {}'.format(combinations[eIntersections]))

            ###
            NodeCoords3d = np.append(Points,np.zeros(len(Points))[None,:].T,axis=1)
            NodeConn = Triangulate(Points,Constraints=None,method='Triangle')
            # NodeConn = Triangle(Points,Constraints=NewConstraints)
            ### This is to handle cases where scipy only returns a single triangle, possibly due to the way it handles 'collinear' triangles
            # Not an ideal work around
            if len(NodeConn) == 1:
                return NodeConn
            ###
            # Get Edges
            # Edges = converter.solid2edges(Points,NodeConn)
            # Edges = np.asarray(Edges)

            # Filter out degenerate triangles
            # EdgePoints = NodeCoords3d[Edges]
            # ElemPoints = NodeCoords3d[NodeConn]
            # A2 = np.linalg.norm(np.cross(ElemPoints[:,1]-ElemPoints[:,0],ElemPoints[:,2]-ElemPoints[:,0]),axis=1)
            # EdgeLen = np.max(np.linalg.norm(EdgePoints[:,0]-EdgePoints[:,1],axis=1).reshape((int(len(Edges)/3),3)),axis=1)
            # deviation = A2/EdgeLen
            # if sum(deviation>tol*10) > 1:
            #     NodeConn = NodeConn[deviation>tol]

            Edges = converter.solid2edges(Points,NodeConn)
            UEdges = converter.edges2unique(Edges)
            ElemConn = MeshUtils.getElemConnectivity(Points, NodeConn)
            
            if np.all([np.any(np.all(Constraint==UEdges,axis=1)) or np.any(np.all(Constraint[::-1]==UEdges,axis=1))for Constraint in NewConstraints]):
                # All constraints are present
                NodeConn = idx[NodeConn]
                return NodeConn
            for constraint in NewConstraints:
                if np.any(np.all(constraint == UEdges,axis=1)) or np.any(np.all(constraint[::-1] == UEdges,axis=1)):
                    # Constrained edge already exists, no need to do anything
                    continue
                # For each constraint edge, find intersecting edges
                s1 = np.tile(NodeCoords3d[constraint],(len(UEdges),1,1))
                s2 = NodeCoords3d[UEdges]

                intersections = Rays.SegmentsSegmentsIntersection(s1,s2,endpt_inclusive=True) # Having endpt_inclusive=True is important here

                Iedges = set(np.where(intersections)[0])
                ww = 0 # Not working sometimes for some reason;fuck
                while len(Iedges) > 0 and ww < 100:
                    ww += 1
                    if ww == 99:
                        a = 2
                    k = Iedges.pop()
                    if UEdges[k][0] in constraint or UEdges[k][1] in constraint:
                        # Intersection is at an endpoint, no need to do anything
                        continue
                    elif np.any(np.all(UEdges[k] == NewConstraints,axis=1)) or np.any(np.all(UEdges[k][::-1] == NewConstraints,axis=1)):
                        # Edge is a constraint, cant be flipped
                        a = 2
                        continue
                    intersect = np.intersect1d(ElemConn[UEdges[k,0]],ElemConn[UEdges[k,1]])
                    if len(intersect) < 2:
                        # Boundary edge, skip
                        continue

                    # [i,j] = EdgeElemConn[k]
                    [i,j] = intersect

                    
                    nodes = list(set(NodeConn[i]).union(NodeConn[j]))
                    if len(ConvexHull_GiftWrapping(Points[nodes],IncludeCollinear=True)) != 4:
                        # If the 4 nodes of the two triangles connected at the edge are non-convex, put the edge back in the queue
                        # if len(Iedges) != 1:
                        Iedges.add(k)
                    else:
                        # Flip edge
                        Newi,Newj = FlipEdge(NodeConn,i,j)
                        NodeConn[i] = Newi; NodeConn[j] = Newj
                        NewEdge = np.setdiff1d(nodes,UEdges[k])
                        if np.any(np.all(NewEdge==UEdges,axis=1)) or np.any(np.all(NewEdge[::-1]==UEdges,axis=1)):
                            # If the new edge is already an edge, it needs to be flipped again
                            Iedges.add(k)
                        UEdges[k] = NewEdge
                        ElemConn = MeshUtils.getElemConnectivity(Points, NodeConn)
                        if Rays.SegmentSegmentIntersection(NodeCoords3d[constraint],NodeCoords3d[NewEdge],endpt_inclusive=True):
                            Iedges.add(k)
                            if len(Iedges) == 1:
                                a = 2
                    ## This shouldn't be needed,
                    brk = True
                    for C in NewConstraints:
                        if not (np.any(np.all(C == UEdges, axis=1)) or np.any(np.all(C[::-1] == UEdges, axis=1))):
                            brk = False
                        if brk: 
                            break
                    

            
            ### Double Checking
            for constraint in NewConstraints:
                if not (np.any(np.all(constraint == UEdges, axis=1)) or np.any(np.all(constraint[::-1] == UEdges, axis=1))):
                    print('merp')

            ###
            # Reinforce delaunay
            # try:
            #     NodeConn = DelaunayFlips(Points,NodeConn,Constraints=NewConstraints)
            # except:
            #     a = 2
        

        # #
        # import plotly.graph_objects as go
        # fig = go.Figure()
        # for i in range(len(UEdges)):
        #     fig.add_trace(go.Scatter(x=NodeCoords[idx[UEdges[i]]][:,0], y=NodeCoords[idx[UEdges[i]]][:,1],text=UEdges[i],line=dict(dash='dash')))
        # for i in range(len(NewConstraints)):
        #     fig.add_trace(go.Scatter(x=NodeCoords[idx[NewConstraints[i]]][:,0], y=NodeCoords[idx[NewConstraints[i]]][:,1],text=NewConstraints[i]))
        # fig.show()

    NodeCoords = Points
    return NodeCoords, NodeConn

def SplitConstraints_2d(NodeCoords,Constraints,tol=1e-12):
    
    # import plotly.graph_objects as go
    # fig = go.Figure()
    # for i in range(len(Constraints)):
    #     fig.add_trace(go.Scatter(x=NodeCoords[Constraints[i]][:,0], y=NodeCoords[Constraints[i]][:,1],text=Constraints[i]))
    # fig.show()

    # Check for intersections within the constraints
    pbk = copy.copy(NodeCoords)
    cbk = copy.copy(Constraints)
    combinations = np.array(list(itertools.combinations(range(len(Constraints)),2)))
    NodeCoords3 = np.append(NodeCoords,np.zeros((len(NodeCoords),1)),axis=1)
    e1 = NodeCoords3[Constraints[combinations[:,0]]]
    e2 = NodeCoords3[Constraints[combinations[:,1]]]
    eIntersections,eIntersectionPts = Rays.SegmentsSegmentsIntersection(e1,e2,return_intersection=True,endpt_inclusive=True,eps=tol)
    eIntersectionPts = eIntersectionPts[:,:2]
    NewConstraints = np.empty((0,2),dtype=int)
    for ic,c in enumerate(Constraints):
        # ids of other constraints that intersect with this constraint
        ids = np.unique(np.array([combo for combo in combinations[eIntersections] if ic in combo]))
        # Check collinear lines - currently Rays.SegmentsSegmentsIntersection doesn't do this properly
        coix = np.empty((0,2))
        for combo in combinations[~eIntersections]:
            if ic not in combo:
                continue
            segments = np.vstack(NodeCoords[Constraints[combo]])

            # Check area of triangles formed by 3 of the 4 nodes of the segment to assess collinearity
            if np.linalg.norm(NodeCoords[Constraints[combo[0]][0]] - 
                                    NodeCoords[Constraints[combo[0]][1]]) > tol:
                check1 = np.linalg.norm(np.cross(segments[1]-segments[0], segments[2]-segments[0])) < tol
                check2 = np.linalg.norm(np.cross(segments[1]-segments[0], segments[3]-segments[0])) < tol
            elif np.linalg.norm(NodeCoords[Constraints[combo[1]][0]] - 
                                    NodeCoords[Constraints[combo[1]][1]]) > tol:
                check1 = np.linalg.norm(np.cross(segments[3]-segments[2], segments[0]-segments[2])) < tol
                check2 = np.linalg.norm(np.cross(segments[3]-segments[2], segments[1]-segments[2])) < tol
            else:
                continue
            v1 = segments[1]-segments[0]
            v2 = segments[3]-segments[2]
            with np.errstate(divide='ignore', invalid='ignore'):
                if np.abs(np.cross(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2))) > 0.1:
                    # vectors aren't parallel (cross product of parallel vectors is [0,0,0])
                    # This is needed in addition to the previous area checks because a perpindicular segment
                    # in line with the end point of the other segment could also give area ~ 0
                    continue
            if check1 and check2:
                # For segments AB and CD if the (double) area of the triangles ABC and ABD are both (near) zero, the segments are collinear
                # segsort = np.lexsort(segments.T) # Lexographic sort of the segments
                segsort = np.lexsort((np.round(segments/tol)*tol).T) 
                if (0 in segsort[:2] and 1 in segsort[:2]) or (2 in segsort[:2] and 3 in segsort[:2]):
                    # if both points if a segment are on the same side of the other, they at most intersect at an endpt
                    if np.linalg.norm(np.diff(segments[segsort[1:3]],axis=0)) < tol: # Norm of the difference between interior points
                        # end point intersection
                        coix = np.append(coix,[segments[segsort[1]]],axis=0)
                else:
                    # overlapping segments, get the two interior points
                    coix = np.append(coix,segments[segsort[1:3]],axis=0)
            elif check1:
                segsort = np.lexsort(segments[[0,1,2]].T)
                if segsort[1] == 2:
                    coix = np.append(coix,[segments[2]],axis=0)
            elif check2:
                segsort = np.lexsort(segments[[0,1,3]].T)
                if segsort[1] == 2:
                    coix = np.append(coix,[segments[3]],axis=0)
                    
        ids = np.delete(ids,ids==ic)
        if len(ids) == 0:
            NewConstraints = np.append(NewConstraints,[c],axis=0)
        else:
            # corresponding intersection points
            ixs = np.array([eIntersectionPts[eIntersections][x] for x,combo in enumerate(combinations[eIntersections]) if ic in combo])
            ixs = np.append(ixs,coix,axis=0)
            ixsort = ixs[np.lexsort((np.round(ixs/tol)*tol).T)]

            NewConstraints = np.append(NewConstraints,np.vstack([np.arange(0,len(ixsort)-1),np.arange(1,len(ixsort))]).T+len(NodeCoords),axis=0)
            NodeCoords = np.append(NodeCoords,ixsort,axis=0)

    NodeCoords, Constraints, _ = MeshUtils.DeleteDuplicateNodes(NodeCoords,NewConstraints,tol=10*tol)
    Constraints = np.unique([c for c in Constraints if c[0] != c[1]],axis=0)
    NodeCoords = np.asarray(NodeCoords)

    # import plotly.graph_objects as go
    # fig = go.Figure()
    # for i in range(len(Constraints)):
    #     fig.add_trace(go.Scatter(x=NodeCoords[Constraints[i]][:,0], y=NodeCoords[Constraints[i]][:,1],text=Constraints[i]))
    # fig.show()

    return NodeCoords, Constraints

def ConvexHull(NodeCoords,IncludeCollinear=True,method='GiftWrapping'):
    if method == 'GiftWrapping':
        Hull = ConvexHull_GiftWrapping(NodeCoords,IncludeCollinear=IncludeCollinear)
    else:
        raise Exception('Invalid method')
    return Hull

def SciPy(NodeCoords):
    """
    SciPy _summary_

    Parameters
    ----------
    NodeCoords : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    out = spatial.Delaunay(NodeCoords,qhull_options='Qbb Qc Qz Q12 Qt')
    NodeConn = out.simplices
    return NodeConn

def Triangle(NodeCoords,Constraints=None):
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

    return NodeCoords, NodeConn


def BowyerWatson3d(NodeCoords):
    # Someting isn't working right
    NodeConn = []
    TempConn = []
    
    # Bounding Box
    minx = min([NodeCoords[i][0] for i in range(len(NodeCoords))])
    maxx = max([NodeCoords[i][0] for i in range(len(NodeCoords))])
    miny = min([NodeCoords[i][1] for i in range(len(NodeCoords))])
    maxy = max([NodeCoords[i][1] for i in range(len(NodeCoords))])
    minz = min([NodeCoords[i][2] for i in range(len(NodeCoords))])
    maxz = max([NodeCoords[i][2] for i in range(len(NodeCoords))])
    
    # Bounding Tetrahedron (with slight buffer)
    minX = minx+minx/10-10*np.finfo(float).eps
    maxX = maxx+maxx/10+10*np.finfo(float).eps
    minY = miny+miny/10-10*np.finfo(float).eps
    maxY = maxy+maxy/10+10*np.finfo(float).eps
    minZ = minz+minz/10-10*np.finfo(float).eps
    maxZ = maxz+maxz/10+10*np.finfo(float).eps
    tetPts = [[minX,maxY,minZ],
              [minX,minY-max(maxX-minX,maxZ-minZ),minZ],
              [maxX+max(maxY-minY,maxZ-minz),maxY,minZ],
              [minX,maxY,maxZ+max(maxX-minX,maxY-minY)]]
    tetPts = [[tetPts[i][j] + np.random.rand()*np.finfo(float).eps for j in range(len(tetPts[i]))] for i in range(len(tetPts))]
    tempCoords = NodeCoords + tetPts
    tempIdx = [len(NodeCoords),len(NodeCoords)+1,len(NodeCoords)+2,len(NodeCoords)+3]
    TempConn.append(tempIdx)
    
    for i in range(len(NodeCoords)):
        # For Each Node
        badElems = []
        Pt = NodeCoords[i]
        for j in range(len(TempConn)):
            # Check all elements
            Nodes = np.array(tempCoords)[TempConn[j]]
            C, R = TetCircumsphere(Nodes)
            if dist(C, Pt) < R:
                # If new point is inside the circumsphere of an existing element
                badElems.append(j)
            
        # Identify the polyhedral hole
        poly = []        
        badFaces = Converter.tetGetFaces(tempCoords,[TempConn[k] for k in badElems])
        sortedBadFaces = [np.sort(badFaces[j]) for j in range(len(badFaces))]
        for j in range(len(badElems)):
            theseFaces = Converter.tetGetFaces(tempCoords,[TempConn[badElems[j]]])
            for k in range(len(theseFaces)):
                if sum(np.all(np.array(np.sort(theseFaces[k])) == sortedBadFaces,axis=1)) == 1:
                    poly.append(theseFaces[k])
                    
        # Remove old elements  
        sortBad = np.sort(badElems)         
        for j in range(len(sortBad)-1,-1,-1):
            del TempConn[sortBad[j]]
        
        # Retriangulate the polyhedral hole          
        for face in poly:            
            newTet = face + [i] # create a new tet between the face and the new node
            TempConn.append(newTet)
            
        print(len(TempConn))
        fig = go.Figure()
        for elem in TempConn:
            fig = plotTet(fig,tempCoords,elem)
        plot(fig)        
        sciCoords = [tempCoords[j] for j in range(len(tempCoords)) if j in np.unique(TempConn)]
        sciConn = spatial.Delaunay(sciCoords)
        print(len(sciConn.simplices))
        fig = go.Figure()
        for elem in sciConn.simplices:
            fig = plotTet(fig,sciCoords,elem)       
        plot(fig)

    NodeConn = []
    for i in range(len(TempConn)):
        add = True
        for j in tempIdx:
            if j in TempConn[i]:
                add = False
        if add:
            NodeConn.append(TempConn[i])
    
    return NodeConn
    
def ConvexHull_GiftWrapping(NodeCoords,IncludeCollinear=True):
    """
    ConvexHull_GiftWrapping Gift wrapping algorithm for computing the convex hull of a set of 2D points.

    Jarvis, R. A. (1973). On the identification of the convex hull of a finite set of points in the plane. Information Processing Letters, 2(1), 18–21. https://doi.org/10.1016/0020-0190(73)90020-3

    Parameters
    ----------
    NodeCoords : list or np.ndarray
        List of 2D point coordinates
     Returns
    -------
    Hull : list
        List of point indices that form the convex hull, in counterclockwise order
    """    

    assert len(NodeCoords) > 2, 'At least three points are required.'
    assert len(NodeCoords[0]) == 2, 'The gift wrapping algorithm is only valid for points on a plane, coordinates must be two dimensional.'
    Points = np.asarray(NodeCoords)
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
    return Hull

def ConvexHullFanTriangulation(Hull):
    """
    ConvexHullFanTriangulation Generate a fan triangulation of a convex hull

    Parameters
    ----------
    Hull : list or np.ndarray
        List of point indices that form the convex hull. Points should be ordered in 
        either clockwise or counterclockwise order. The ordering of the triangles will
        follow the ordering of the hull.

    Returns
    -------
    NodeConn np.ndarray
        Nodal connectivity of the triangulated hull.
    """
    assert len(Hull) >= 3
    Hull = np.asarray(Hull)
    NodeConn = np.array([
                    np.repeat(Hull[0], len(Hull)-2),
                    Hull[np.arange(1, len(Hull)-1, dtype=int)],
                    Hull[np.arange(2, len(Hull), dtype=int)]
                ]).T
    return NodeConn
    
def TriangleSplittingTriangulation(NodeCoords, Hull=None, return_Hull=False):

    assert len(NodeCoords) > 2, 'At least three points are required.'
    assert len(NodeCoords[0]) == 2, 'Only supported for points on a plane, coordinates must be two dimensional.'

    Points = np.asarray(NodeCoords)
    if Hull is None: Hull = ConvexHull_GiftWrapping(NodeCoords)
    NodeConn = ConvexHullFanTriangulation(Hull)

    interior = np.setdiff1d(np.arange(len(NodeCoords)),Hull,assume_unique=True)
    for i in interior:
        alpha,beta,gamma = MeshUtils.BaryTris(Points[NodeConn],Points[i])
        
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
    if return_Hull:
        return NodeConn, Hull
    return NodeConn
        
def DelaunayFlips_(NodeCoords,NodeConn,Constraints=None):
    """
    DelaunayFlips _summary_

    Parameters
    ----------
    NodeCoords : _type_
        _description_
    NodeConn : _type_
        _description_
    Constrained : list, optional
        list of , by default None

    Returns
    -------
    _type_
        _description_
    """
    NewCoords = np.array(NodeCoords)
    NewConn = np.array(NodeConn)

    # Get Edges
    Edges, EdgeConn, EdgeElem = converter.solid2edges(NewCoords,NewConn,return_EdgeConn=True,return_EdgeElem=True)
    Edges = np.asarray(Edges); EdgeConn = np.asarray(EdgeConn)
    UEdges, UIdx, UInv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
    L = len(UEdges)
    UEdgeElem = np.asarray(EdgeElem)[UIdx]
    UEdgeConn = UInv[EdgeConn]
    EECidx = (UEdgeElem[UEdgeConn] == np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)).astype(int)
    EdgeElemConn = -1*(np.ones((len(UEdges),2),dtype=int))
    EdgeElemConn[UEdgeConn,EECidx] = np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)
    # Get Constraints
    if Constraints is not None:
        Constrained = []
        for constraint in Constraints:
            where = np.where(np.all(UEdges==constraint,axis=1) | np.all(UEdges==constraint[::-1],axis=1))[0]
            if len(where) > 0:
                Constrained.append(where[0])
            else:
                # TODO: Not sure why this happens
                continue

    # Get Opposite Angles
    EdgeVectors = NewCoords[Edges[:,0]] - NewCoords[Edges[:,1]]
    EdgeLengths = np.linalg.norm(EdgeVectors,axis=1)
    ElemVectors = EdgeVectors[EdgeConn]
    ElemLengths = EdgeLengths[EdgeConn]
    with np.errstate(divide='ignore', invalid='ignore'):
        OppositeAngles = np.zeros(ElemLengths.shape)
        OppositeAngles[:,0] = np.arccos(np.sum(ElemVectors[:,2]*-ElemVectors[:,1],axis=1)/(ElemLengths[:,2]*ElemLengths[:,1]))
        OppositeAngles[:,1] = np.arccos(np.sum(ElemVectors[:,0]*-ElemVectors[:,2],axis=1)/(ElemLengths[:,0]*ElemLengths[:,2]))
        OppositeAngles[:,2] = np.arccos(np.sum(ElemVectors[:,1]*-ElemVectors[:,0],axis=1)/(ElemLengths[:,1]*ElemLengths[:,0]))
    EdgeOppositeAngles = np.empty((len(UEdges),2))
    EdgeOppositeAngles[UEdgeConn,EECidx] = OppositeAngles
    EdgeOppositeAngles[EdgeElemConn==-1]=0
    if Constraints is not None: EdgeOppositeAngles[Constrained] = 0
    # Set of non-Delaunay triangles (i.e. ones that have at least one edge where the sum of the opposite angles is greater than pi)
    NonDelaunay = list(EdgeElemConn[np.where((np.sum(EdgeOppositeAngles,axis=1) > np.pi))[0]].flatten())
    iter = 0
    while len(NonDelaunay) > 0:
        
        iter += 1
        if -1 in NonDelaunay:
            NonDelaunay = [x for x in NonDelaunay if x != -1]
        # elem = random.choice(NonDelaunay)#.pop()
        elem = NonDelaunay.pop()
        flippable = []; angles = []
        for k in UEdgeConn[elem]:
            if -1 in EdgeElemConn[k]:
                continue
            [i,j] = EdgeElemConn[k]
            nodes = list(set(NewConn[i]).union(NewConn[j]))
            if len(ConvexHull_GiftWrapping(NewCoords[nodes],IncludeCollinear=True)) == 4:
                angles.append(np.sum(EdgeOppositeAngles[k]))
                flippable.append(k)
        if len(flippable) == 0:
            continue
        else:
            k = np.array(flippable)[np.argsort(angles)][-1]
            # k = random.choice(flippable)
            [i,j] = EdgeElemConn[k]

        if i == -1 or j == -1:
            continue
        Newi,Newj = FlipEdge(NewConn,i,j)
        NewConn[i] = Newi; NewConn[j] = Newj; 

        # Update EdgeConn
        OldEdges = Edges.copy()
        Edges[EdgeConn[i]] = [[NewConn[i,0],NewConn[i,1]], [NewConn[i,1],NewConn[i,2]], [NewConn[i,2],NewConn[i,0]]]
        Edges[EdgeConn[j]] = [[NewConn[j,0],NewConn[j,1]], [NewConn[j,1],NewConn[j,2]], [NewConn[j,2],NewConn[j,0]]]

        ###
        UEdges, UIdx, UInv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
        if len(UEdges) != L:
            print('merp')
            Edges = OldEdges
            continue
        UEdgeElem = np.asarray(EdgeElem)[UIdx]
        UEdgeConn = UInv[EdgeConn]
        EECidx = (UEdgeElem[UEdgeConn] == np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)).astype(int)
        EdgeElemConn = -1*(np.ones((len(UEdges),2),dtype=int))
        EdgeElemConn[UEdgeConn,EECidx] = np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)
        ###

        v11 = NewCoords[NewConn[i,0]] - NewCoords[NewConn[i,1]]
        v12 = NewCoords[NewConn[i,1]] - NewCoords[NewConn[i,2]]
        v13 = NewCoords[NewConn[i,2]] - NewCoords[NewConn[i,0]]
        v21 = NewCoords[NewConn[j,0]] - NewCoords[NewConn[j,1]]
        v22 = NewCoords[NewConn[j,1]] - NewCoords[NewConn[j,2]]
        v23 = NewCoords[NewConn[j,2]] - NewCoords[NewConn[j,0]]
        
        OppositeAngles[i] = [
            np.arccos(np.sum(v12*-v13)/(np.linalg.norm(v12)*np.linalg.norm(v13))),
            np.arccos(np.sum(v13*-v11)/(np.linalg.norm(v13)*np.linalg.norm(v11))),
            np.arccos(np.sum(v11*-v12)/(np.linalg.norm(v11)*np.linalg.norm(v12)))
            ]
        OppositeAngles[j] = [
            np.arccos(np.sum(v22*-v23)/(np.linalg.norm(v22)*np.linalg.norm(v23))),
            np.arccos(np.sum(v23*-v21)/(np.linalg.norm(v23)*np.linalg.norm(v21))),
            np.arccos(np.sum(v21*-v22)/(np.linalg.norm(v21)*np.linalg.norm(v22)))
            ]
        EdgeOppositeAngles[UEdgeConn,EECidx] = OppositeAngles
        EdgeOppositeAngles[EdgeElemConn==-1] = 0
        if Constraints is not None: EdgeOppositeAngles[Constrained] = 0 # Prevents flipping of constrained edges
        NonDelaunay = list(EdgeElemConn[np.where((np.sum(EdgeOppositeAngles,axis=1) > np.pi))[0]].flatten())
    return NewConn

def DelaunayFlips(NodeCoords,NodeConn,Constraints=None):
    """
    DelaunayFlips _summary_

    Parameters
    ----------
    NodeCoords : _type_
        _description_
    NodeConn : _type_
        _description_
    Constrained : list, optional
        list of , by default None

    Returns
    -------
    _type_
        _description_
    """
    NewCoords = np.array(NodeCoords)
    NewConn = np.array(NodeConn)

    # Get Edges
    Edges, EdgeConn, EdgeElem = converter.solid2edges(NewCoords,NewConn,return_EdgeConn=True,return_EdgeElem=True)
    Edges = np.asarray(Edges); EdgeConn = np.asarray(EdgeConn)
    UEdges, UIdx, UInv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
    L = len(UEdges)
    UEdgeElem = np.asarray(EdgeElem)[UIdx]
    UEdgeConn = UInv[EdgeConn]
    EECidx = (UEdgeElem[UEdgeConn] == np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)).astype(int)
    EdgeElemConn = -1*(np.ones((len(UEdges),2),dtype=int))
    EdgeElemConn[UEdgeConn,EECidx] = np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)
    # Get Constraints
    if Constraints is not None:
        Constrained = []
        for constraint in Constraints:
            where = np.where(np.all(UEdges==constraint,axis=1) | np.all(UEdges==constraint[::-1],axis=1))[0]
            if len(where) > 0:
                Constrained.append(where[0])
            else:
                # TODO: Not sure why this happens
                continue

    # Get Opposite Angles
    EdgeVectors = NewCoords[Edges[:,0]] - NewCoords[Edges[:,1]]
    EdgeLengths = np.linalg.norm(EdgeVectors,axis=1)
    ElemVectors = EdgeVectors[EdgeConn]
    ElemLengths = EdgeLengths[EdgeConn]
    with np.errstate(divide='ignore', invalid='ignore'):
        OppositeAngles = np.zeros(ElemLengths.shape)
        OppositeAngles[:,0] = np.arccos(np.sum(ElemVectors[:,2]*-ElemVectors[:,1],axis=1)/(ElemLengths[:,2]*ElemLengths[:,1]))
        OppositeAngles[:,1] = np.arccos(np.sum(ElemVectors[:,0]*-ElemVectors[:,2],axis=1)/(ElemLengths[:,0]*ElemLengths[:,2]))
        OppositeAngles[:,2] = np.arccos(np.sum(ElemVectors[:,1]*-ElemVectors[:,0],axis=1)/(ElemLengths[:,1]*ElemLengths[:,0]))
    EdgeOppositeAngles = np.empty((len(UEdges),2))
    EdgeOppositeAngles[UEdgeConn,EECidx] = OppositeAngles
    EdgeOppositeAngles[EdgeElemConn==-1]=0
    if Constraints is not None: EdgeOppositeAngles[Constrained] = 0
    # Set of non-Delaunay triangles (i.e. ones that have at least one edge where the sum of the opposite angles is greater than pi)
    NonDelaunay = list(EdgeElemConn[np.where((np.sum(EdgeOppositeAngles,axis=1) > np.pi))[0]].flatten())
    iter = 0
    while len(NonDelaunay) > 0:
        
        iter += 1
        if iter > 100:
            print('morp')
        if -1 in NonDelaunay:
            NonDelaunay = [x for x in NonDelaunay if x != -1]
        # elem = random.choice(NonDelaunay)#.pop()
        elem = NonDelaunay.pop()
        flippable = []; angles = []
        for k in UEdgeConn[elem]:
            if -1 in EdgeElemConn[k]:
                continue
            [i,j] = EdgeElemConn[k]
            nodes = list(set(NewConn[i]).union(NewConn[j]))
            if len(ConvexHull_GiftWrapping(NewCoords[nodes],IncludeCollinear=True)) == 4:
                angles.append(np.sum(EdgeOppositeAngles[k]))
                flippable.append(k)
        if len(flippable) == 0:
            continue
        else:
            k = np.array(flippable)[np.argsort(angles)][-1]
            [i,j] = EdgeElemConn[k]

        if i == -1 or j == -1:
            continue
        Newi,Newj = FlipEdge(NewConn,i,j)
        NewEdge = np.intersect1d(Newi,Newj)
        if np.any(np.all(NewEdge==UEdges,axis=1)) or np.any(np.all(NewEdge[::-1]==UEdges,axis=1)):
            # Invalid flip - edge already exists
            continue
        NewConn[i] = Newi; NewConn[j] = Newj; 

        # Update EdgeConn
        Edges[EdgeConn[i]] = [[NewConn[i,0],NewConn[i,1]], [NewConn[i,1],NewConn[i,2]], [NewConn[i,2],NewConn[i,0]]]
        Edges[EdgeConn[j]] = [[NewConn[j,0],NewConn[j,1]], [NewConn[j,1],NewConn[j,2]], [NewConn[j,2],NewConn[j,0]]]

        ###
        UEdges, UIdx, UInv = converter.edges2unique(Edges,return_idx=True,return_inv=True)
        if len(UEdges) != L:
            print('merp')
        UEdgeElem = np.asarray(EdgeElem)[UIdx]
        UEdgeConn = UInv[EdgeConn]
        EECidx = (UEdgeElem[UEdgeConn] == np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)).astype(int)
        EdgeElemConn = -1*(np.ones((len(UEdges),2),dtype=int))
        EdgeElemConn[UEdgeConn,EECidx] = np.repeat(np.arange(len(EdgeConn))[:,None],UEdgeConn.shape[1],axis=1)
        ###

        v11 = NewCoords[NewConn[i,0]] - NewCoords[NewConn[i,1]]
        v12 = NewCoords[NewConn[i,1]] - NewCoords[NewConn[i,2]]
        v13 = NewCoords[NewConn[i,2]] - NewCoords[NewConn[i,0]]
        v21 = NewCoords[NewConn[j,0]] - NewCoords[NewConn[j,1]]
        v22 = NewCoords[NewConn[j,1]] - NewCoords[NewConn[j,2]]
        v23 = NewCoords[NewConn[j,2]] - NewCoords[NewConn[j,0]]
        
        OppositeAngles[i] = [
            np.arccos(np.sum(v12*-v13)/(np.linalg.norm(v12)*np.linalg.norm(v13))),
            np.arccos(np.sum(v13*-v11)/(np.linalg.norm(v13)*np.linalg.norm(v11))),
            np.arccos(np.sum(v11*-v12)/(np.linalg.norm(v11)*np.linalg.norm(v12)))
            ]
        OppositeAngles[j] = [
            np.arccos(np.sum(v22*-v23)/(np.linalg.norm(v22)*np.linalg.norm(v23))),
            np.arccos(np.sum(v23*-v21)/(np.linalg.norm(v23)*np.linalg.norm(v21))),
            np.arccos(np.sum(v21*-v22)/(np.linalg.norm(v21)*np.linalg.norm(v22)))
            ]
        EdgeOppositeAngles[UEdgeConn,EECidx] = OppositeAngles
        EdgeOppositeAngles[EdgeElemConn==-1] = 0
        if Constraints is not None: EdgeOppositeAngles[Constrained] = 0 # Prevents flipping of constrained edges
        NonDelaunay = list(EdgeElemConn[np.where((np.sum(EdgeOppositeAngles,axis=1) > np.pi))[0]].flatten())
    return NewConn
    
def plotTet(fig,NodeCoords,elem):
    x = [NodeCoords[elem[0]][0], NodeCoords[elem[1]][0], NodeCoords[elem[2]][0],
         NodeCoords[elem[3]][0], NodeCoords[elem[0]][0], NodeCoords[elem[2]][0],
         NodeCoords[elem[1]][0], NodeCoords[elem[3]][0]]
    y = [NodeCoords[elem[0]][1], NodeCoords[elem[1]][1], NodeCoords[elem[2]][1],
         NodeCoords[elem[3]][1], NodeCoords[elem[0]][1], NodeCoords[elem[2]][1],
         NodeCoords[elem[1]][1], NodeCoords[elem[3]][1]]
    z = [NodeCoords[elem[0]][2], NodeCoords[elem[1]][2], NodeCoords[elem[2]][2],
         NodeCoords[elem[3]][2], NodeCoords[elem[0]][2], NodeCoords[elem[2]][2],
         NodeCoords[elem[1]][2], NodeCoords[elem[3]][2]]
         
    fig.add_trace(go.Scatter3d(x=x,y=y,z=z))
    return fig
   
## Utils ##
def dist(pt1,pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)

def TetCircumsphere(Nodes):
    
    A = np.array([np.subtract(Nodes[1],Nodes[0]),
         np.subtract(Nodes[2],Nodes[0]),
         np.subtract(Nodes[3],Nodes[0])])
    if np.linalg.det(A) == 0:
        Nodes = [[Nodes[i][j] + np.random.rand()*np.finfo(float).eps for j in range(len(Nodes[i]))] for i in range(len(Nodes))]
        A = np.array([np.subtract(Nodes[1],Nodes[0]),
             np.subtract(Nodes[2],Nodes[0]),
             np.subtract(Nodes[3],Nodes[0])])
    B = 0.5 * np.array([[np.linalg.norm(Nodes[1])**2 - np.linalg.norm(Nodes[0])**2],
                        [np.linalg.norm(Nodes[2])**2 - np.linalg.norm(Nodes[0])**2],
                        [np.linalg.norm(Nodes[3])**2 - np.linalg.norm(Nodes[0])**2],
                        ])
    C = np.transpose(np.linalg.solve(A,B)).tolist()[0]
    
    
    a1 = dist(Nodes[0],Nodes[1])
    b1 = dist(Nodes[0],Nodes[2])
    c1 = dist(Nodes[0],Nodes[3])
    a2 = dist(Nodes[2],Nodes[3])
    b2 = dist(Nodes[1],Nodes[3])
    c2 = dist(Nodes[1],Nodes[2])    
    
    V = np.linalg.norm(np.dot(np.subtract(Nodes[0],Nodes[3]),
                              (np.cross(np.subtract(Nodes[1],Nodes[3]),
                                        np.subtract(Nodes[2],Nodes[3]))
                                  )))/6
    R = np.sqrt((a1*a2 + b1*b2 + c1*c2) *
                (a1*a2 + b1*b2 - c1*c2) *
                (a1*a2 - b1*b2 + c1*c2) *
                (-a1*a2 + b1*b2 + c1*c2))/(24*V)
    
    return C, R

def FlipEdge(NodeConn,i,j):
    Si = set(NodeConn[i])
    Sj = set(NodeConn[j])
    shared = list(Si.intersection(Sj))
    assert len(shared)==2, 'Elements {:d} & {:d} are not properly connected for an edge flip.'.format(i,j)
    Losti = Sj.difference(Si).pop()
    Lostj = Si.difference(Sj).pop()
    NewEdge = [Losti,Lostj]
    Newi = [n if n != shared[0] else Losti for n in NodeConn[i]]
    Newj = [n if n != shared[1] else Lostj for n in NodeConn[j]]
    return Newi,Newj
    
