# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:23:07 2022

@author: toj
"""
#%%
from . import MeshUtils, Octree, delaunay
import numpy as np
import itertools, random, sys

## Intersection Tests:
def RayTriangleIntersection(pt, ray, TriCoords, bidirectional=False):
    # Möller-Trumbore Intersection Algorithm
    eps = 0.000001
    edge1 = np.subtract(TriCoords[1],TriCoords[0])
    edge2 = np.subtract(TriCoords[2], TriCoords[0])
    
    p = np.cross(ray, edge2)
    det = np.dot(edge1, p)
    if (det > -eps) and (det < eps):
        return []
    invdet = 1/det
    tvec = np.subtract(pt, TriCoords[0])
    u = np.dot(tvec,p) * invdet
    if (u < 0) or (u > 1):
        return []
    
    q = np.cross(tvec, edge1)
    v = np.dot(ray, q) * invdet
    if (v < 0) or (u+v > 1):
        return []
    
    t = np.dot(edge2, q) * invdet
    if (abs(t) > eps) or bidirectional:
        intersectionPt = np.array(pt) + np.array(ray)*t
    else:
        return []
    
    return intersectionPt

def RayTrianglesIntersection(pt, ray, Tris, bidirectional=False, eps=1e-14):
    # Vectorized Möller-Trumbore Intersection Algorithm
    # Single ray, n triangles
    # eps = 0.000001
    with np.errstate(divide='ignore', invalid='ignore'):
        edge1 = Tris[:,1] - Tris[:,0]
        edge2 = Tris[:,2] - Tris[:,0]
        
        p = np.cross(ray, edge2)
        det = np.sum(edge1*p,axis=1)
        
        invdet = 1/det
        tvec = pt - Tris[:,0]
        u = np.sum(tvec*p,axis=1) * invdet
        
        q = np.cross(tvec,edge1)
        v = np.sum(ray*q,axis=1) * invdet
        
        t = np.sum(edge2*q,axis=1) * invdet
        
        checks = (
            ((det > -eps) & (det < eps)) |
            ((u < 0) | (u > 1)) |
            ((v < 0) | (u+v > 1)) |
            ((abs(t) <= eps) & (not bidirectional))
            )
        intersections = np.where(~checks)[0]
        intersectionPts = pt + ray*t[intersections,None]
        
    return intersections, intersectionPts
    
def RayBoxIntersection(pt, ray, xlim, ylim, zlim):
    # Williams, Barrus, Morley and Shirley (2002)
    
    if ray[0] > 0:
        divx = 1/ray[0]
        tmin = (xlim[0] - pt[0]) * divx
        tmax = (xlim[1] - pt[0]) * divx
    elif ray[0] < 0:
        divx = 1/ray[0]
        tmin = (xlim[1] - pt[0]) * divx
        tmax = (xlim[0] - pt[0]) * divx
    else:
        tmin = tmax = np.inf
    
    
    if ray[1] > 0:
        divy = 1/ray[1]
        tymin = (ylim[0] - pt[1]) * divy
        tymax = (ylim[1] - pt[1]) * divy
    elif ray[1] < 0:
        divy = 1/ray[1]
        tymin = (ylim[1] - pt[1]) * divy
        tymax = (ylim[0] - pt[1]) * divy
    else:
        tymin = tymax = np.inf
    
    if (tmin > tymax) or (tymin > tmax):
        return False
    if (tymin > tmin):
        tmin = tymin
    if (tymax < tmax):
        tmax = tymax
    
    
    if ray[2] > 0:
        divz = 1/ray[2]
        tzmin = (zlim[0] - pt[2]) * divz
        tzmax = (zlim[1] - pt[2]) * divz
    elif ray[2] < 0:
        divz = 1/ray[2]
        tzmin = (zlim[1] - pt[2]) * divz
        tzmax = (zlim[0] - pt[2]) * divz
    else:
        tzmin = tzmax = np.inf
        
    if (tmin > tzmax) or (tzmin > tmax):
        return False
    
    return True

def PlaneBoxIntersection(pt, Normal, xlim, ylim, zlim):
    
    BoxCoords = [
        [xlim[0],ylim[0],zlim[0]],
        [xlim[1],ylim[0],zlim[0]],
        [xlim[0],ylim[1],zlim[0]],
        [xlim[1],ylim[1],zlim[0]],
        [xlim[0],ylim[0],zlim[1]],
        [xlim[1],ylim[0],zlim[1]],
        [xlim[0],ylim[1],zlim[1]],
        [xlim[1],ylim[1],zlim[1]],
        ]
    # Signed Distances from the vertices of the box to the plane
    sd = [np.dot(Normal,p)-np.dot(Normal,pt) for p in BoxCoords]
    signs = [np.sign(x) for x in sd]
    if all(signs) == 1 or all(signs) == -1:
        # No Intersection, all points on same side of plane
        return False
    else:
        # Intersection, points on different sides of the plane
        return True
    
def PlaneTriangleIntersection(pt, Normal, TriCoords):
    # Signed Distances from the vertices of the box to the plane
    sd = [np.dot(Normal,p)-np.dot(Normal,pt) for p in TriCoords]
    signs = [np.sign(x) for x in sd]
    if all(signs) == 1 or all(signs) == -1:
        # No Intersection, all points on same side of plane
        return False
    else:
        # Intersection, points on different sides of the plane
        return True
    
def TriangleTriangleIntersection(Tri1,Tri2,eps=1e-14,edgeedge=False):
    
    # If <edgeedge> is true, two triangles that meet exactly at the edges will be counted as an intersection
    #   this inclues two adjacent triangles that share an edge, but also cases where two points of Tri1 lie exactly on the edges of Tri2

    if type(Tri1) is list: Tri1 = np.array(Tri1)
    if type(Tri2) is list: Tri2 = np.array(Tri2)

    # Moller 1997
    # Plane2 (N2.X+d2):
    N2 = np.cross(np.subtract(Tri2[1,:],Tri2[0,:]),np.subtract(Tri2[2,:],Tri2[0,:]))
    d2 = -np.dot(N2,Tri2[0,:])
    
    # Signed distances from vertices in Tri1 to Plane2:
    sd1 = np.round([np.dot(N2,v)+d2 for v in Tri1],16)
    signs1 = np.sign(sd1)
    if all(signs1 == 1) or all(signs1 == -1):
        # All vertices of Tri1 are on the same side of Plane2
        return False
    elif all(np.abs(sd1) < eps):
        # Coplanar
        # Perform Edge Intersection
        edges = np.array([[0,1],[1,2],[2,0]])
        edges1idx = np.array([edges[0],edges[1],edges[2],edges[0],edges[1],edges[2],edges[0],edges[1],edges[2]])
        edges2idx = np.array([edges[0],edges[1],edges[2],edges[1],edges[2],edges[0],edges[2],edges[0],edges[1]])
        edges1 = Tri1[edges1idx]
        edges2 = Tri2[edges2idx]

        intersections = SegmentsSegmentsIntersection(edges1,edges2,return_intersection=False,eps=eps)
        if any(intersections):
            return True
        else:
            # Peform point-in-tri test
            alpha,beta,gamma = MeshUtils.BaryTri(Tri1, Tri2[0])
            if all([alpha>=0,beta>=0,gamma>=0]):
                return True
            else:
                alpha,beta,gamma = MeshUtils.BaryTri(Tri2, Tri1[0])
                if all([alpha>=0,beta>=0,gamma>=0]):
                    return True
                else:
                    return False
    
    # Plane1 (N1.X+d1): 
    N1 = np.cross(np.subtract(Tri1[1,:],Tri1[0,:]),np.subtract(Tri1[2,:],Tri1[0,:]))
    d1 = -np.dot(N1,Tri1[0,:])
    
    # Signed distances from vertices in Tri1 to Plane2:
    # sd2 = np.round([np.dot(N1,v)+d1 for v in Tri2],16)
    sd2 = np.array([np.dot(N1,v)+d1 for v in Tri2])
    signs2 = np.sign(sd2)
    if all(signs2 == 1) or all(signs2 == -1):
        # All vertices of Tri2 are on the same side of Plane1
        return False

    # Intersection line of Tri1 & Tri2: L = O+tD
    D = np.cross(N1,N2).tolist()
  
    Dmax = max(D)
    # Projections of Tri1 to L
    Pv1 = np.array([v[D.index(Dmax)] for v in Tri1])

    if signs1[0] == signs1[2] :
        t11 = Pv1[0] + (Pv1[1]-Pv1[0])*sd1[0]/(sd1[0]-sd1[1])
        t12 = Pv1[2] + (Pv1[1]-Pv1[2])*sd1[2]/(sd1[2]-sd1[1])
    elif signs1[0] == signs1[1]:
        t11 = Pv1[0] + (Pv1[2]-Pv1[0])*sd1[0]/(sd1[0]-sd1[2])
        t12 = Pv1[1] + (Pv1[2]-Pv1[1])*sd1[1]/(sd1[1]-sd1[2])
    elif signs1[1] == signs1[2]:
        t11 = Pv1[2] + (Pv1[0]-Pv1[2])*sd1[2]/(sd1[2]-sd1[0])
        t12 = Pv1[1] + (Pv1[0]-Pv1[1])*sd1[1]/(sd1[1]-sd1[0])
    elif signs1[1] != 0:
        t11 = Pv1[0] + (Pv1[1]-Pv1[0])*sd1[0]/(sd1[0]-sd1[1])
        t12 = Pv1[2] + (Pv1[1]-Pv1[2])*sd1[2]/(sd1[2]-sd1[1])
    elif signs1[2] != 0:
        t11 = Pv1[0] + (Pv1[2]-Pv1[0])*sd1[0]/(sd1[0]-sd1[2])
        t12 = Pv1[1] + (Pv1[2]-Pv1[1])*sd1[1]/(sd1[1]-sd1[2])
    else:
        t11 = Pv1[2] + (Pv1[0]-Pv1[2])*sd1[2]/(sd1[2]-sd1[0])
        t12 = Pv1[1] + (Pv1[0]-Pv1[1])*sd1[1]/(sd1[1]-sd1[0])
    # Projections of Tri2 to L
    Pv2 = np.array([v[D.index(Dmax)] for v in Tri2])

    # sumzero = np.sum(signs2==0)
    if signs2[0] == signs2[2]:
        t21 = Pv2[0] + (Pv2[1]-Pv2[0])*sd2[0]/(sd2[0]-sd2[1])
        t22 = Pv2[2] + (Pv2[1]-Pv2[2])*sd2[2]/(sd2[2]-sd2[1])
    elif signs2[0] == signs2[1]:
        t21 = Pv2[0] + (Pv2[2]-Pv2[0])*sd2[0]/(sd2[0]-sd2[2])
        t22 = Pv2[1] + (Pv2[2]-Pv2[1])*sd2[1]/(sd2[1]-sd2[2])
    elif signs2[1] == signs2[2]:
        t21 = Pv2[2] + (Pv2[0]-Pv2[2])*sd2[2]/(sd2[2]-sd2[0])
        t22 = Pv2[1] + (Pv2[0]-Pv2[1])*sd2[1]/(sd2[1]-sd2[0])
    elif signs2[1] != 0:
        t21 = Pv2[0] + (Pv2[1]-Pv2[0])*sd2[0]/(sd2[0]-sd2[1])
        t22 = Pv2[2] + (Pv2[1]-Pv2[2])*sd2[2]/(sd2[2]-sd2[1])
    elif signs2[2] != 0:
        t21 = Pv2[0] + (Pv2[2]-Pv2[0])*sd2[0]/(sd2[0]-sd2[2])
        t22 = Pv2[1] + (Pv2[2]-Pv2[1])*sd2[1]/(sd2[1]-sd2[2])
    else:
        t21 = Pv2[2] + (Pv2[0]-Pv2[2])*sd2[2]/(sd2[2]-sd2[0])
        t22 = Pv2[1] + (Pv2[0]-Pv2[1])*sd2[1]/(sd2[1]-sd2[0])

   
    t11,t12 = min([t11,t12]),max([t11,t12])
    t21,t22 = min([t21,t22]),max([t21,t22])
    
    # if (t12 <= t21 or t22 <= t11) or (t11 == t21 and t12 == t22):
    if (t12-t21 <= eps or t22-t11 <= eps) or ((not edgeedge) and abs(t11-t21) < eps and abs(t12-t22) < eps):
        return False
    return True

def TriangleTriangleIntersectionPt(Tri1,Tri2,eps=1e-14, edgeedge=False):
    # Moller 1997
    
    # Plane2 (N2.X+d2):
    N2 = np.cross(np.subtract(Tri2[1],Tri2[0]),np.subtract(Tri2[2],Tri2[0]))
    d2 = -np.dot(N2,Tri2[0])
    
    # Signed distances from vertices in Tri1 to Plane2:
    sd1 = np.round([np.dot(N2,v)+d2 for v in Tri1],16)
    signs1 = np.sign(sd1)
    if all(signs1 == 1) or all(signs1 == -1):
        # All vertices of Tri1 are on the same side of Plane2
        return []
    elif all(np.abs(sd1) < eps):
        # Coplanar
        # Perform Edge Intersection
        edges = np.array([[0,1],[1,2],[2,0]])
        edges1idx = np.array([edges[0],edges[1],edges[2],edges[0],edges[1],edges[2],edges[0],edges[1],edges[2]])
        edges2idx = np.array([edges[0],edges[1],edges[2],edges[1],edges[2],edges[0],edges[2],edges[0],edges[1]])
        edges1 = Tri1[edges1idx]
        edges2 = Tri2[edges2idx]

        intersections,pts = SegmentsSegmentsIntersection(edges1,edges2,return_intersection=True,eps=eps)
        if any(intersections):
            points = pts[intersections]
            # Check if there are any verticies within the triangles
            for i in range(3):
                alpha,beta,gamma = MeshUtils.BaryTri(Tri1, Tri2[i])
                if all([alpha>=0,beta>=0,gamma>=0]):
                    points = np.vstack([points,Tri2[i]])
                alpha,beta,gamma = MeshUtils.BaryTri(Tri2, Tri1[i])
                if all([alpha>=0,beta>=0,gamma>=0]):
                    points = np.vstack([points,Tri1[i]])
            return points
        else:
            # Peform point-in-tri test
            alpha,beta,gamma = MeshUtils.BaryTri(Tri1, Tri2[0])
            if all([alpha>=0,beta>=0,gamma>=0]):
                return Tri2
            else:
                alpha,beta,gamma = MeshUtils.BaryTri(Tri2, Tri1[0])
                if all([alpha>=0,beta>=0,gamma>=0]):
                    return Tri1
                else:
                    return []
    
    # Plane1 (N1.X+d1): 
    N1 = np.cross(np.subtract(Tri1[1],Tri1[0]),np.subtract(Tri1[2],Tri1[0]))
    d1 = -np.dot(N1,Tri1[0])
    
    # Signed distances from vertices in Tri1 to Plane2:
    sd2 = np.round([np.dot(N1,v)+d1 for v in Tri2],16)
    signs2 = np.sign(sd2)
    if all(signs2 == 1) or all(signs2 == -1):
        # All vertices of Tri2 are on the same side of Plane1
        return []

    # Intersection line of Tri1 & Tri2: L = O+tD
    D = np.cross(N1,N2)
    D = D/np.linalg.norm(D)
    if abs(D[0]) == max(np.abs(D)):
        Ox = 0
        Oy = -(d1*N2[2]-d2*N1[2])/(N1[1]*N2[2] - N2[1]*N1[2])
        Oz = -(d2*N1[1]-d1*N2[1])/(N1[1]*N2[2] - N2[1]*N1[2])
    elif abs(D[1]) == max(np.abs(D)):
        Ox = -(d1*N2[2]-d2*N1[2])/(N1[0]*N2[2] - N2[0]*N1[2])
        Oy = 0
        Oz = -(d2*N1[0]-d1*N2[0])/(N1[0]*N2[2] - N2[0]*N1[2])
    else: #elif abs(D[2]) == max(np.abs(D)):
        Ox = -(d1*N2[1]-d2*N1[1])/(N1[0]*N2[1] - N2[0]*N1[1])
        Oy = -(d2*N1[0]-d1*N2[0])/(N1[0]*N2[1] - N2[0]*N1[1])
        Oz = 0
    O = [Ox,Oy,Oz]

    # Dmax = max(D)
    # Projections of Tri1 to L
    # Pv1 = [v[D.index(Dmax)] for v in Tri1]
    Pv1 = [np.dot(D,(v-O)) for v in Tri1]
    

    if signs1[0] == signs1[2] :
        t11 = Pv1[0] + (Pv1[1]-Pv1[0])*sd1[0]/(sd1[0]-sd1[1])
        t12 = Pv1[2] + (Pv1[1]-Pv1[2])*sd1[2]/(sd1[2]-sd1[1])
    elif signs1[0] == signs1[1]:
        t11 = Pv1[0] + (Pv1[2]-Pv1[0])*sd1[0]/(sd1[0]-sd1[2])
        t12 = Pv1[1] + (Pv1[2]-Pv1[1])*sd1[1]/(sd1[1]-sd1[2])
    elif signs1[1] == signs1[2]:
        t11 = Pv1[2] + (Pv1[0]-Pv1[2])*sd1[2]/(sd1[2]-sd1[0])
        t12 = Pv1[1] + (Pv1[0]-Pv1[1])*sd1[1]/(sd1[1]-sd1[0])
    elif signs1[1] != 0:
        t11 = Pv1[0] + (Pv1[1]-Pv1[0])*sd1[0]/(sd1[0]-sd1[1])
        t12 = Pv1[2] + (Pv1[1]-Pv1[2])*sd1[2]/(sd1[2]-sd1[1])
    elif signs1[2] != 0:
        t11 = Pv1[0] + (Pv1[2]-Pv1[0])*sd1[0]/(sd1[0]-sd1[2])
        t12 = Pv1[1] + (Pv1[2]-Pv1[1])*sd1[1]/(sd1[1]-sd1[2])
    else:
        t11 = Pv1[2] + (Pv1[0]-Pv1[2])*sd1[2]/(sd1[2]-sd1[0])
        t12 = Pv1[1] + (Pv1[0]-Pv1[1])*sd1[1]/(sd1[1]-sd1[0])
    # Projections of Tri2 to L
    # Pv2 = [v[D.index(Dmax)] for v in Tri2]
    Pv2 = [np.dot(D,(v-O)) for v in Tri2]
    # sumzero = np.sum(signs2==0)
    if signs2[0] == signs2[2]:
        t21 = Pv2[0] + (Pv2[1]-Pv2[0])*sd2[0]/(sd2[0]-sd2[1])
        t22 = Pv2[2] + (Pv2[1]-Pv2[2])*sd2[2]/(sd2[2]-sd2[1])
    elif signs2[0] == signs2[1]:
        t21 = Pv2[0] + (Pv2[2]-Pv2[0])*sd2[0]/(sd2[0]-sd2[2])
        t22 = Pv2[1] + (Pv2[2]-Pv2[1])*sd2[1]/(sd2[1]-sd2[2])
    elif signs2[1] == signs2[2]:
        t21 = Pv2[2] + (Pv2[0]-Pv2[2])*sd2[2]/(sd2[2]-sd2[0])
        t22 = Pv2[1] + (Pv2[0]-Pv2[1])*sd2[1]/(sd2[1]-sd2[0])
    elif signs2[1] != 0:
        t21 = Pv2[0] + (Pv2[1]-Pv2[0])*sd2[0]/(sd2[0]-sd2[1])
        t22 = Pv2[2] + (Pv2[1]-Pv2[2])*sd2[2]/(sd2[2]-sd2[1])
    elif signs2[2] != 0:
        t21 = Pv2[0] + (Pv2[2]-Pv2[0])*sd2[0]/(sd2[0]-sd2[2])
        t22 = Pv2[1] + (Pv2[2]-Pv2[1])*sd2[1]/(sd2[1]-sd2[2])
    else:
        t21 = Pv2[2] + (Pv2[0]-Pv2[2])*sd2[2]/(sd2[2]-sd2[0])
        t22 = Pv2[1] + (Pv2[0]-Pv2[1])*sd2[1]/(sd2[1]-sd2[0])

   
    t11,t12 = min([t11,t12]),max([t11,t12])
    t21,t22 = min([t21,t22]),max([t21,t22])
    
    # if (t12 <= t21 or t22 <= t11) or (t11 == t21 and t12 == t22):
    if (t12-t21 <= eps or t22-t11 <= eps) or ((not edgeedge) and abs(t11-t21) < eps and abs(t12-t22) < eps):
        return []

    t1,t2 = np.sort([t11,t12,t21,t22])[1:3]
    edge = np.array([O + t1*D, O + t2*D])
    
    return edge
    
def TrianglesTrianglesIntersection(Tri1s,Tri2s,eps=1e-14,edgeedge=False):
    
    # Vectorized version of TriangleTriangleIntersection to perform simultaneous comparisons between the triangles in Tri1s and Tri2s
    # TODO: Currently considering coplanar as a non-intersection, need to implement a separate coplanar test
    
    # Plane2 (N2.X+d2):
    N2s = np.cross(np.subtract(Tri2s[:,1],Tri2s[:,0]),np.subtract(Tri2s[:,2],Tri2s[:,0]))
    d2s = -np.sum(N2s*Tri2s[:,0,:],axis=1)

    # Signed distances from vertices in Tri1 to Plane2:
    sd1s = np.round([np.sum(N2s*Tri1s[:,i],axis=1)+d2s for i in range(3)],16).T
    signs1 = np.sign(sd1s)
    
    # Plane1 (N1.X+d1): 
    N1s = np.cross(np.subtract(Tri1s[:,1],Tri1s[:,0]),np.subtract(Tri1s[:,2],Tri1s[:,0]))
    d1s = -np.sum(N1s*Tri1s[:,0,:],axis=1)
    
    # Signed distances from vertices in Tri1 to Plane2:
    sd2s = np.round([np.sum(N1s*Tri2s[:,i],axis=1)+d1s for i in range(3)],16).T
    signs2 = np.sign(sd2s)
    
    # Intersection line of Tri1 & Tri2: L = O+tD
    Ds = np.cross(N1s,N2s)
    Dmaxs = np.max(Ds,axis=1)
    
    # Projections of Tri1 to L
    Pv1s = Tri1s.transpose(0,2,1)[(Ds==Dmaxs[:,None]) & ((Ds==Dmaxs[:,None])*[3,2,1] == np.max((Ds==Dmaxs[:,None])*[3,2,1],axis=1)[:,None])]
    # Projections of Tri2 to L
    Pv2s = Tri2s.transpose(0,2,1)[(Ds==Dmaxs[:,None]) & ((Ds==Dmaxs[:,None])*[3,2,1] == np.max((Ds==Dmaxs[:,None])*[3,2,1],axis=1)[:,None])]

    t11s = np.zeros(len(Tri1s)); t12s = np.zeros(len(Tri1s))
    t21s = np.zeros(len(Tri1s)); t22s = np.zeros(len(Tri1s))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # if signs[:,0] == signs[:,2]:

        a = (signs1[:,0] == signs1[:,2]) 
        b = (signs1[:,0] == signs1[:,1]) 
        c = (signs1[:,1] == signs1[:,2]) 

        A = ((signs1[:,1]!=0)) & ~(a|b|c)
        B = ((signs1[:,2]!=0)) & ~(a|b|c)
        C = ((signs1[:,0]!=0)) & ~(a|b|c)

        Aa = (A | a)
        Bb = (B | b) & ~Aa
        Cc = (C | c) & ~(Aa|Bb)
        
        t11s[Aa] = Pv1s[Aa,0] + (Pv1s[Aa,1]-Pv1s[Aa,0])*sd1s[Aa,0]/(sd1s[Aa,0]-sd1s[Aa,1])
        t12s[Aa] = Pv1s[Aa,2] + (Pv1s[Aa,1]-Pv1s[Aa,2])*sd1s[Aa,2]/(sd1s[Aa,2]-sd1s[Aa,1])
        
        t11s[Bb] = Pv1s[Bb,0] + (Pv1s[Bb,2]-Pv1s[Bb,0])*sd1s[Bb,0]/(sd1s[Bb,0]-sd1s[Bb,2])
        t12s[Bb] = Pv1s[Bb,1] + (Pv1s[Bb,2]-Pv1s[Bb,1])*sd1s[Bb,1]/(sd1s[Bb,1]-sd1s[Bb,2])
        
        t11s[Cc] = Pv1s[Cc,2] + (Pv1s[Cc,0]-Pv1s[Cc,2])*sd1s[Cc,2]/(sd1s[Cc,2]-sd1s[Cc,0])
        t12s[Cc] = Pv1s[Cc,1] + (Pv1s[Cc,0]-Pv1s[Cc,1])*sd1s[Cc,1]/(sd1s[Cc,1]-sd1s[Cc,0])
    
    
        a = (signs2[:,0] == signs2[:,2]) 
        b = (signs2[:,0] == signs2[:,1]) 
        c = (signs2[:,1] == signs2[:,2]) 

        A = ((signs2[:,1]!=0)) & ~(a|b|c)
        B = ((signs2[:,2]!=0)) & ~(a|b|c)
        C = ((signs2[:,0]!=0)) & ~(a|b|c)

        Aa = (A | a)
        Bb = (B | b) & ~Aa
        Cc = (C | c) & ~(Aa|Bb)

        t21s[Aa] = Pv2s[Aa,0] + (Pv2s[Aa,1]-Pv2s[Aa,0])*sd2s[Aa,0]/(sd2s[Aa,0]-sd2s[Aa,1])
        t22s[Aa] = Pv2s[Aa,2] + (Pv2s[Aa,1]-Pv2s[Aa,2])*sd2s[Aa,2]/(sd2s[Aa,2]-sd2s[Aa,1])
        
        t21s[Bb] = Pv2s[Bb,0] + (Pv2s[Bb,2]-Pv2s[Bb,0])*sd2s[Bb,0]/(sd2s[Bb,0]-sd2s[Bb,2])
        t22s[Bb] = Pv2s[Bb,1] + (Pv2s[Bb,2]-Pv2s[Bb,1])*sd2s[Bb,1]/(sd2s[Bb,1]-sd2s[Bb,2])
        
        t21s[Cc] = Pv2s[Cc,2] + (Pv2s[Cc,0]-Pv2s[Cc,2])*sd2s[Cc,2]/(sd2s[Cc,2]-sd2s[Cc,0])
        t22s[Cc] = Pv2s[Cc,1] + (Pv2s[Cc,0]-Pv2s[Cc,1])*sd2s[Cc,1]/(sd2s[Cc,1]-sd2s[Cc,0])
    
        t11s,t12s = np.fmin(t11s,t12s),np.fmax(t11s,t12s)
        t21s,t22s = np.fmin(t21s,t22s),np.fmax(t21s,t22s)
        
        # Initialize Intersections Array
        Intersections = np.repeat(True,len(Tri1s))
        
        # Perform Checks
        edgeedgebool = np.repeat(edgeedge,len(signs1))
        coplanar = np.all(np.abs(sd1s) < eps, axis=1)
        checks = (np.all(signs1==1,axis=1) | np.all(signs1==-1,axis=1) |
                 np.all(signs2==1,axis=1) | np.all(signs2==-1,axis=1) |
                 (t12s-t21s <= eps) | (t22s-t11s <= eps) | 
                 (~edgeedgebool & (np.abs(t11s-t21s) < eps) & (np.abs(t12s-t22s) < eps)))

        Intersections[checks | coplanar] = False
        
        CoTri1s = Tri1s[coplanar]; CoTri2s = Tri2s[coplanar]
        edges = np.array([[0,1],[1,2],[2,0]])
        edges1idx = np.array([edges[0],edges[1],edges[2],edges[0],edges[1],edges[2],edges[0],edges[1],edges[2]])
        edges2idx = np.array([edges[0],edges[1],edges[2],edges[1],edges[2],edges[0],edges[2],edges[0],edges[1]])
        edges1 = CoTri1s[:,edges1idx]
        edges2 = CoTri2s[:,edges2idx]

        coplanar_where = np.where(coplanar)[0] 

        edges1r = edges1.reshape(edges1.shape[0]*edges1.shape[1],edges1.shape[2],edges1.shape[3])
        edges2r = edges2.reshape(edges2.shape[0]*edges2.shape[1],edges2.shape[2],edges2.shape[3])
        intersectionsr = SegmentsSegmentsIntersection(edges1r,edges2r,return_intersection=False,eps=eps)
        
        intersections = intersectionsr.reshape(edges1.shape[0],edges1.shape[1])

        Intersections[coplanar_where] = np.any(intersections,axis=1)

        PtInTriChecks = coplanar_where[~Intersections[coplanar_where]]
        for i in PtInTriChecks:
            # Peform point-in-tri test
            alpha,beta,gamma = MeshUtils.BaryTri(Tri1s[i], Tri2s[i][0])
            if all([alpha>=0,beta>=0,gamma>=0]):
                Intersections[i]  = True
        ###
        # coplanar_intersections = np.repeat(False,len(coplanar))
        # for i in range(len(edges1)):
        #     intersections = SegmentsSegmentsIntersection(edges1,edges2,return_intersection=False)
        #     if any(intersections):
        #         coplanar_intersections[coplanar_where[i]] = True                
        #     else:
        #         # Peform point-in-tri test
        #         alpha,beta,gamma = MeshUtils.BaryTri(Tri1s[coplanar_where[i]], Tri2s[coplanar_where[i]][0])
        #         if all([alpha>=0,beta>=0,gamma>=0]):
        #             coplanar_intersections[coplanar_where[i]]  = True
        #         else:
        #             alpha,beta,gamma = MeshUtils.BaryTri(Tri2s[coplanar_where[i]], Tri1s[coplanar_where[i]][0])
        #             if all([alpha>=0,beta>=0,gamma>=0]):
        #                 coplanar_intersections[coplanar_where[i]]  = True
        
    return Intersections
    
def TrianglesTrianglesIntersectionPts(Tri1s,Tri2s,eps=1e-14,edgeedge=False):
    
    # Vectorized version of TriangleTriangleIntersection to perform simultaneous comparisons between the triangles in Tri1s and Tri2s
    # TODO: Currently considering coplanar as a non-intersection, need to implement a separate coplanar test
    
    # Plane2 (N2.X+d2):
    N2s = np.cross(np.subtract(Tri2s[:,1],Tri2s[:,0]),np.subtract(Tri2s[:,2],Tri2s[:,0]))
    d2s = -np.sum(N2s*Tri2s[:,0,:],axis=1)

    # Signed distances from vertices in Tri1 to Plane2:
    sd1s = np.round([np.sum(N2s*Tri1s[:,i],axis=1)+d2s for i in range(3)],16).T
    signs1 = np.sign(sd1s)
    
    # Plane1 (N1.X+d1): 
    N1s = np.cross(np.subtract(Tri1s[:,1],Tri1s[:,0]),np.subtract(Tri1s[:,2],Tri1s[:,0]))
    d1s = -np.sum(N1s*Tri1s[:,0,:],axis=1)
    
    # Signed distances from vertices in Tri1 to Plane2:
    sd2s = np.round([np.sum(N1s*Tri2s[:,i],axis=1)+d1s for i in range(3)],16).T
    signs2 = np.sign(sd2s)
    
    # Intersection line of Tri1 & Tri2: L = O+tD
    Ds = np.cross(N1s,N2s)
    norm = np.linalg.norm(Ds,axis=1)
    Ds = np.divide(Ds,norm[:,None],where=(norm>0)[:,None],out=Ds)
    # Dmaxs = np.max(Ds,axis=1)
    absDs = np.abs(Ds)

    O = np.nan*np.zeros((len(Tri1s),3))
    o1 = absDs[:,0] == np.max(absDs,axis=1) 
    o2 = absDs[:,1] == np.max(absDs,axis=1)
    o3 = absDs[:,2] == np.max(absDs,axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        O[o1] = np.array([
                np.zeros(sum(o1)),
                -(d1s[o1]*N2s[o1,2]-d2s[o1]*N1s[o1,2])/(N1s[o1,1]*N2s[o1,2] - N2s[o1,1]*N1s[o1,2]),
                -(d2s[o1]*N1s[o1,1]-d1s[o1]*N2s[o1,1])/(N1s[o1,1]*N2s[o1,2] - N2s[o1,1]*N1s[o1,2])
                ]).T
        O[o2] = np.array([
                -(d1s[o2]*N2s[o2,2]-d2s[o2]*N1s[o2,2])/(N1s[o2,0]*N2s[o2,2] - N2s[o2,0]*N1s[o2,2]),
                np.zeros(sum(o2)),
                -(d2s[o2]*N1s[o2,0]-d1s[o2]*N2s[o2,0])/(N1s[o2,0]*N2s[o2,2] - N2s[o2,0]*N1s[o2,2])
                ]).T
        O[o3] = np.array([
                -(d1s[o3]*N2s[o3,1]-d2s[o3]*N1s[o3,1])/(N1s[o3,0]*N2s[o3,1] - N2s[o3,0]*N1s[o3,1]),
                -(d2s[o3]*N1s[o3,0]-d1s[o3]*N2s[o3,0])/(N1s[o3,0]*N2s[o3,1] - N2s[o3,0]*N1s[o3,1]),
                np.zeros(sum(o3))
                ]).T
    
        # Projections of Tri1 to L
        Pv1s = np.array([np.sum(Ds*(Tri1s[:,i]-O),axis=1) for i in range(3)]).T
        # Projections of Tri2 to L
        Pv2s = np.array([np.sum(Ds*(Tri2s[:,i]-O),axis=1) for i in range(3)]).T

        t11s = np.zeros(len(Tri1s)); t12s = np.zeros(len(Tri1s))
        t21s = np.zeros(len(Tri1s)); t22s = np.zeros(len(Tri1s))
    
        a = (signs1[:,0] == signs1[:,2]) 
        b = (signs1[:,0] == signs1[:,1]) 
        c = (signs1[:,1] == signs1[:,2]) 

        A = ((signs1[:,1]!=0)) & ~(a|b|c)
        B = ((signs1[:,2]!=0)) & ~(a|b|c)
        C = ((signs1[:,0]!=0)) & ~(a|b|c)

        Aa = (A | a)
        Bb = (B | b) & ~Aa
        Cc = (C | c) & ~(Aa|Bb)
        
        t11s[Aa] = Pv1s[Aa,0] + (Pv1s[Aa,1]-Pv1s[Aa,0])*sd1s[Aa,0]/(sd1s[Aa,0]-sd1s[Aa,1])
        t12s[Aa] = Pv1s[Aa,2] + (Pv1s[Aa,1]-Pv1s[Aa,2])*sd1s[Aa,2]/(sd1s[Aa,2]-sd1s[Aa,1])
        
        t11s[Bb] = Pv1s[Bb,0] + (Pv1s[Bb,2]-Pv1s[Bb,0])*sd1s[Bb,0]/(sd1s[Bb,0]-sd1s[Bb,2])
        t12s[Bb] = Pv1s[Bb,1] + (Pv1s[Bb,2]-Pv1s[Bb,1])*sd1s[Bb,1]/(sd1s[Bb,1]-sd1s[Bb,2])
        
        t11s[Cc] = Pv1s[Cc,2] + (Pv1s[Cc,0]-Pv1s[Cc,2])*sd1s[Cc,2]/(sd1s[Cc,2]-sd1s[Cc,0])
        t12s[Cc] = Pv1s[Cc,1] + (Pv1s[Cc,0]-Pv1s[Cc,1])*sd1s[Cc,1]/(sd1s[Cc,1]-sd1s[Cc,0])
    
    
        a = (signs2[:,0] == signs2[:,2]) 
        b = (signs2[:,0] == signs2[:,1]) 
        c = (signs2[:,1] == signs2[:,2]) 

        A = ((signs2[:,1]!=0)) & ~(a|b|c)
        B = ((signs2[:,2]!=0)) & ~(a|b|c)
        C = ((signs2[:,0]!=0)) & ~(a|b|c)

        Aa = (A | a)
        Bb = (B | b) & ~Aa
        Cc = (C | c) & ~(Aa|Bb)

        t21s[Aa] = Pv2s[Aa,0] + (Pv2s[Aa,1]-Pv2s[Aa,0])*sd2s[Aa,0]/(sd2s[Aa,0]-sd2s[Aa,1])
        t22s[Aa] = Pv2s[Aa,2] + (Pv2s[Aa,1]-Pv2s[Aa,2])*sd2s[Aa,2]/(sd2s[Aa,2]-sd2s[Aa,1])
        
        t21s[Bb] = Pv2s[Bb,0] + (Pv2s[Bb,2]-Pv2s[Bb,0])*sd2s[Bb,0]/(sd2s[Bb,0]-sd2s[Bb,2])
        t22s[Bb] = Pv2s[Bb,1] + (Pv2s[Bb,2]-Pv2s[Bb,1])*sd2s[Bb,1]/(sd2s[Bb,1]-sd2s[Bb,2])
        
        t21s[Cc] = Pv2s[Cc,2] + (Pv2s[Cc,0]-Pv2s[Cc,2])*sd2s[Cc,2]/(sd2s[Cc,2]-sd2s[Cc,0])
        t22s[Cc] = Pv2s[Cc,1] + (Pv2s[Cc,0]-Pv2s[Cc,1])*sd2s[Cc,1]/(sd2s[Cc,1]-sd2s[Cc,0])
    
        t11s,t12s = np.fmin(t11s,t12s),np.fmax(t11s,t12s)
        t21s,t22s = np.fmin(t21s,t22s),np.fmax(t21s,t22s)
        
        # Initialize Intersections Array
        Intersections = np.repeat(True,len(Tri1s))
        
        # Perform Checks
        coplanar = np.all(np.abs(sd1s) < eps, axis=1)
        checks = (np.all(signs1==1,axis=1) | np.all(signs1==-1,axis=1) |
                 np.all(signs2==1,axis=1) | np.all(signs2==-1,axis=1) |
                 (t12s-t21s <= eps) | (t22s-t11s <= eps)) 
        if not edgeedge:
            edgeedgebool = (np.abs(t11s-t21s) < eps) & (np.abs(t12s-t22s) < eps)
            Intersections[checks | edgeedgebool | coplanar] = False
        else:
            Intersections[checks | coplanar] = False

        IntersectionPts = np.nan*np.ones((len(Intersections),18,3))
        # IntersectionPts = [[] for i in range(len(Intersections))]
        t1,t2 = np.sort([t11s,t12s,t21s,t22s],axis=0)[1:3]
        IntersectionPts[Intersections,0,:] = O[Intersections] + t1[Intersections,None]*Ds[Intersections]
        IntersectionPts[Intersections,1,:] = O[Intersections] + t2[Intersections,None]*Ds[Intersections]
        # Coplanar checks
        CoTri1s = Tri1s[coplanar]; CoTri2s = Tri2s[coplanar]
        edges = np.array([[0,1],[1,2],[2,0]])
        edges1idx = np.array([edges[0],edges[1],edges[2],edges[0],edges[1],edges[2],edges[0],edges[1],edges[2]])
        edges2idx = np.array([edges[0],edges[1],edges[2],edges[1],edges[2],edges[0],edges[2],edges[0],edges[1]])
        edges1 = CoTri1s[:,edges1idx]
        edges2 = CoTri2s[:,edges2idx]

        coplanar_where = np.where(coplanar)[0]

        ###
        edges1r = edges1.reshape(edges1.shape[0]*edges1.shape[1],edges1.shape[2],edges1.shape[3])
        edges2r = edges2.reshape(edges2.shape[0]*edges2.shape[1],edges2.shape[2],edges2.shape[3])
        intersectionsr,ptsr = SegmentsSegmentsIntersection(edges1r,edges2r,return_intersection=True,eps=eps)
        
        intersections = intersectionsr.reshape(edges1.shape[0],edges1.shape[1])
        pts = ptsr.reshape(edges1.shape[0],edges1.shape[1],ptsr.shape[1])

        ###
        edge1_ix = [[],[],[]]; edge1_ixpts = [[],[],[]]; edge1_ixcount = [[],[],[]]
        edge1_ix[0] = intersections[:,0::3]
        edge1_ix[1] = intersections[:,1::3]
        edge1_ix[2] = intersections[:,2::3]
        edge1_ixpts[0] = pts[:,0::3]#[edge1_ix[0]]
        edge1_ixpts[1] = pts[:,1::3]#[edge1_ix[1]]
        edge1_ixpts[2] = pts[:,2::3]#[edge1_ix[2]]
        edge1_ixcount[0] = np.sum(edge1_ix[0],axis=1)
        edge1_ixcount[1] = np.sum(edge1_ix[1],axis=1)
        edge1_ixcount[2] = np.sum(edge1_ix[2],axis=1)

        edge2_ix = [[],[],[]]; edge2_ixpts = [[],[],[]]; edge2_ixcount = [[],[],[]]
        edge2_ix[0] = intersections[:,[0,5,7]]
        edge2_ix[1] = intersections[:,[1,3,8]]
        edge2_ix[2] = intersections[:,[2,4,6]]
        edge2_ixpts[0] = pts[:,[0,5,7]]#[edge2_ix[0]]
        edge2_ixpts[1] = pts[:,[1,3,8]]#[edge2_ix[1]]
        edge2_ixpts[2] = pts[:,[2,4,6]]#[edge2_ix[2]]
        edge2_ixcount[0] = np.sum(edge2_ix[0],axis=1)
        edge2_ixcount[1] = np.sum(edge2_ix[1],axis=1)
        edge2_ixcount[2] = np.sum(edge2_ix[2],axis=1)

        # For edges that only interesect at one point, they way have a point inside the other triangle that needs to be accounted for
        # for i in range(len(edge1_ixcount)):
        #     where = np.where(edge1_ixcount[i]==1)[0]
        #     for j,edge in enumerate(edges1[edge1_ixcount[i]==1]):
        #         idx = where[j]
        #         for pt in edge[i]:
        #             if PointInTri(Tri2s[coplanar_where[idx]],pt,method='BaryArea') and not np.any(np.all(np.abs(pt - edge1_ixpts[i][idx])<eps,axis=1)):
        #                 # IntersectionPts[coplanar_where[idx],np.where(np.all(np.isnan(IntersectionPts[coplanar_where[idx]]),axis=1))[0][0],:] = pt
        #                 edge1_ixpts[i][idx][np.where(np.all(np.isnan(edge1_ixpts[i][idx]),axis=1))[0][0]] = pt
        #         # If there's still only one point, it probably means the endpoint of on edge is on the edge of the other triangle, duplicate the point
        #         if np.sum(np.all(np.isnan(edge1_ixpts[i][idx]),axis=1)) == 2:
        #             edge1_ixpts[i][idx][np.where(np.all(np.isnan(edge1_ixpts[i][idx]),axis=1))[0][0]] = edge1_ixpts[i][idx][~np.all(np.isnan(edge1_ixpts[i][idx]),axis=1)]
        #     where = np.where(edge2_ixcount[i]==1)[0]
        #     for j,edge in enumerate(edges2[edge2_ixcount[i]==1]):
        #         idx = where[j]
        #         for pt in edge[i]:
        #             if PointInTri(Tri1s[coplanar_where[idx]],pt,method='BaryArea') and not np.any(np.all(np.abs(pt - edge2_ixpts[i][idx])<eps,axis=1)):
        #                 # IntersectionPts[coplanar_where[idx],np.where(np.all(np.isnan(IntersectionPts[coplanar_where[idx]]),axis=1))[0][0],:] = pt
        #                 edge2_ixpts[i][idx][np.where(np.all(np.isnan(edge2_ixpts[i][idx]),axis=1))[0][0]] = pt
        #         if np.sum(np.all(np.isnan(edge2_ixpts[i][idx]),axis=1)) == 2:
        #             edge2_ixpts[i][idx][np.where(np.all(np.isnan(edge2_ixpts[i][idx]),axis=1))[0][0]] = edge2_ixpts[i][idx][~np.all(np.isnan(edge2_ixpts[i][idx]),axis=1)]
        
        # For edges that only interesect at one point, they way have a point inside the other triangle that needs to be accounted for
        for i in range(len(edge1_ixcount)):
            where1 = np.where(edge1_ixcount[i]==1)[0]
            es1 = edges1[where1][:,i]    # list of the first edges
            p11s = es1[:,0]   # First point in the edges
            p12s = es1[:,1]   # Second point in the edges
            tris1 = Tri2s[coplanar_where[where1]]    # Corresponding triangle
            In11 = PointsInTris(tris1,p11s,method='BaryArea') & (~np.any(np.all(np.abs(p11s[:,:,None] - edge1_ixpts[i][where1])<eps,axis=2),axis=1))
            In12 = PointsInTris(tris1,p12s,method='BaryArea') & (~np.any(np.all(np.abs(p12s[:,:,None] - edge1_ixpts[i][where1])<eps,axis=2),axis=1))

            nanidx1 = np.where(np.all(np.isnan(edge1_ixpts[i][where1]),axis=2).cumsum(axis=1).cumsum(axis=1) == 1)
            edge1_ixpts[i][where1[nanidx1[0][In11]],nanidx1[1][In11]] = p11s[In11]
            edge1_ixpts[i][where1[nanidx1[0][In12&~In11]],nanidx1[1][In12&~In11]] = p12s[In12&~In11]
            edge1_ixpts[i][where1[nanidx1[0][~In12&~In11]],nanidx1[1][~In12&~In11]] = edge1_ixpts[i][where1[~In12&~In11]][np.where(np.all(~np.isnan(edge1_ixpts[i][where1[~In12&~In11]]),axis=2).cumsum(axis=1).cumsum(axis=1) == 1)]
            #
            where2 = np.where(edge2_ixcount[i]==1)[0]
            es2 = edges2[where2][:,i]    # list of the first edges
            p21s = es2[:,0]   # First point in the edges
            p22s = es2[:,1]   # Second point in the edges
            tris2 = Tri1s[coplanar_where[where2]]    # Corresponding triangle
            In21 = PointsInTris(tris2,p21s,method='BaryArea') & (~np.any(np.all(np.abs(p21s[:,:,None] - edge2_ixpts[i][where2])<eps,axis=2),axis=1))
            In22 = PointsInTris(tris2,p22s,method='BaryArea') & (~np.any(np.all(np.abs(p22s[:,:,None] - edge2_ixpts[i][where2])<eps,axis=2),axis=1))

            nanidx2 = np.where(np.all(np.isnan(edge2_ixpts[i][where2]),axis=2).cumsum(axis=1).cumsum(axis=1) == 1)
            edge2_ixpts[i][where2[nanidx2[0][In21]],nanidx2[1][In21]] = p21s[In21]
            edge2_ixpts[i][where2[nanidx2[0][In22&~In21]],nanidx2[1][In22&~In21]] = p22s[In22&~In21]
            edge2_ixpts[i][where2[nanidx2[0][~In22&~In21]],nanidx2[1][~In22&~In21]] = edge2_ixpts[i][where2[~In22&~In21]][np.where(np.all(~np.isnan(edge2_ixpts[i][where2[~In22&~In21]]),axis=2).cumsum(axis=1).cumsum(axis=1) == 1)]


        # Collect the points so that the intersection edges 
        Edge1Stack = np.concatenate([edge1_ixpts[0],edge1_ixpts[1],edge1_ixpts[2]],axis=1)
        Edge2Stack = np.concatenate([edge2_ixpts[0],edge2_ixpts[1],edge2_ixpts[2]],axis=1)
        DoubleStack = np.concatenate([Edge1Stack,Edge2Stack],axis=1)

        NewDoubleStack = np.nan*np.ones(DoubleStack.shape)
        NewDoubleStack[np.flip(np.sort(np.all(~np.isnan(DoubleStack),axis=2),axis=1),axis=1)] = DoubleStack[np.all(~np.isnan(DoubleStack),axis=2)]

        Intersections[coplanar_where] = np.any(intersections,axis=1)
        IntersectionPts[coplanar_where] = NewDoubleStack

        # Check triangles with no edge intersections to see if one is completely within the other
        PtInTriChecks = coplanar_where[~Intersections[coplanar_where]]
        TwoInOne = PointsInTris(Tri1s[PtInTriChecks],Tri2s[PtInTriChecks,0],method='BaryArea',eps=eps,inclusive=False)
        OneInTwo = PointsInTris(Tri2s[PtInTriChecks],Tri1s[PtInTriChecks,0],method='BaryArea',eps=eps,inclusive=False)

        Intersections[PtInTriChecks] = OneInTwo | TwoInOne
        IntersectionPts[PtInTriChecks[TwoInOne],:6,:] = Tri2s[PtInTriChecks[TwoInOne]][:,[0,1,1,2,2,0]]
        IntersectionPts[PtInTriChecks[OneInTwo],:6,:] = Tri1s[PtInTriChecks[OneInTwo]][:,[0,1,1,2,2,0]]

        # PtInTriChecks = coplanar_where[~Intersections[coplanar_where]]
        # for i in PtInTriChecks:
        #     # Peform point-in-tri test
        #     if PointInTri(Tri1s[i],Tri2s[i][0],method='BaryArea',inclusive=False):
        #         Intersections[i]  = True
        #         IntersectionPts[i,:6,:] = Tri2s[i][[0,1,1,2,2,0]]
        #     elif PointInTri(Tri2s[i],Tri1s[i][0],method='BaryArea',inclusive=False):
        #         Intersections[i]  = True
        #         IntersectionPts[i,:6,:] = Tri1s[i][[0,1,1,2,2,0]]


        # import plotly.graph_objects as go
        # fig = go.Figure()
        # Idx = 0#coplanar_where[0]
        # t1 = np.append(Tri1s[Idx],[Tri1s[Idx,0,:]],axis=0)
        # t2 = np.append(Tri2s[Idx],[Tri2s[Idx,0,:]],axis=0)
        # fig.add_trace(go.Scatter(x=t1[:,1],y=t1[:,2]))
        # fig.add_trace(go.Scatter(x=t2[:,1],y=t2[:,2]))
        # fig.add_trace(go.Scatter(x=IntersectionPts[:,:,1][Idx],y=IntersectionPts[:,:,2][Idx],mode='markers'))
        # for i in range(0,len(IntersectionPts[Idx]),2):
        #     fig.add_trace(go.Scatter(x=[IntersectionPts[Idx][i,1],IntersectionPts[Idx][i+1,1]],y=[IntersectionPts[Idx][i,2],IntersectionPts[Idx][i+1,2]]))
        # fig.show()
        # a = 2

    return Intersections, IntersectionPts

def TriangleBoxIntersection(TriCoords, xlim, ylim, zlim, TriNormal=None, BoxCenter=None):
    # Akenine-Moller (2001) Fast 3D Triangle-Box Overlap Test
    if not BoxCenter: BoxCenter = np.mean([xlim,ylim,zlim],axis=1)
    f0 = np.subtract(TriCoords[1],TriCoords[0])
    f1 = np.subtract(TriCoords[2],TriCoords[1])
    f2 = np.subtract(TriCoords[0],TriCoords[2])
    hx = (xlim[1]-xlim[0])/2
    hy = (ylim[1]-ylim[0])/2
    hz = (zlim[1]-zlim[0])/2
    # Move triangle so that the box is centered around the origin 
    [v0,v1,v2] = np.subtract(TriCoords,BoxCenter)
    
    # Test the box against the minimal Axis Aligned Bounding Box (AABB) of the tri
    if max(v0[0],v1[0],v2[0]) < -hx or min(v0[0],v1[0],v2[0]) > hx:
        return False
    if max(v0[1],v1[1],v2[1]) < -hy or min(v0[1],v1[1],v2[1]) > hy:
        return False
    if max(v0[2],v1[2],v2[2]) < -hz or min(v0[2],v1[2],v2[2]) > hz:
        return False
    
    # Test the normal of the triangle
    if TriNormal is None: 
        TriNormal = np.cross(f0,f1)
    elif type(TriNormal) is list:
        TriNormal = np.array(TriNormal)
    dist = np.dot(TriNormal,v0)
    r = hx*abs(TriNormal[0]) + hy*abs(TriNormal[1]) + hz*abs(TriNormal[2])
    if dist > r:
        return False
    
    # Test Axes
    # a00
    a00 = np.array([0,-f0[2],f0[1]])
    p0 = np.dot(v0,a00)
    # p1 = np.dot(v1,a00)
    p2 = np.dot(v2,a00)
    r = hy*abs(a00[1]) + hz*abs(a00[2])
    if min(p0,p2) > r or max(p0,p2) < -r:
        return False
    # a01
    a01 = np.array([0,-f1[2],f1[1]])
    p0 = np.dot(v0,a01)
    p1 = np.dot(v1,a01)
    # p2 = np.dot(v2,a01)
    r = hy*abs(a01[1]) + hz*abs(a01[2])
    if min(p0,p1) > r or max(p0,p1) < -r:
        return False
    # a02
    a02 = np.array([0,-f2[2],f2[1]])
    p0 = np.dot(v0,a02)
    p1 = np.dot(v1,a02)
    # p2 = np.dot(v2,a02)
    r = hy*abs(a02[1]) + hz*abs(a02[2])
    if min(p0,p1) > r or max(p0,p1) < -r:
        return False
    # a10
    a10 = np.array([f0[2],0,-f0[0]])
    p0 = np.dot(v0,a10)
    # p1 = np.dot(v1,a10)
    p2 = np.dot(v2,a10)
    r = hx*abs(a10[0]) + hz*abs(a10[2])
    if min(p0,p2) > r or max(p0,p2) < -r:
        return False
    # a11
    a11 = np.array([f1[2],0,-f1[0]])
    p0 = np.dot(v0,a11)
    p1 = np.dot(v1,a11)
    # p2 = np.dot(v2,a11)
    r = hx*abs(a11[0]) + hz*abs(a11[2])
    if min(p0,p1) > r or max(p0,p1) < -r:
        return False
    # a12
    a12 = np.array([f2[2],0,-f2[0]])
    p0 = np.dot(v0,a12)
    p1 = np.dot(v1,a12)
    # p2 = np.dot(v2,a10)
    r = hx*abs(a12[0]) + hz*abs(a12[2])
    if min(p0,p1) > r or max(p0,p1) < -r:
        return False
    # a20
    a20 = np.array([-f0[1],f0[0],0])
    p0 = np.dot(v0,a20)
    # p1 = np.dot(v1,a20)
    p2 = np.dot(v2,a20)
    r = hx*abs(a20[0]) + hy*abs(a20[1])
    if min(p0,p2) > r or max(p0,p2) < -r:
        return False
    # a21
    a21 = np.array([-f1[1],f1[0],0])
    p0 = np.dot(v0,a21)
    p1 = np.dot(v1,a21)
    # p2 = np.dot(v2,a21)
    r = hx*abs(a21[0]) + hy*abs(a21[1])
    if min(p0,p1) > r or max(p0,p1) < -r:
        return False
    # a22
    a22 = np.array([-f2[1],f2[0],0])
    p0 = np.dot(v0,a22)
    p1 = np.dot(v1,a22)
    # p2 = np.dot(v2,a22)
    r = hx*abs(a22[0]) + hy*abs(a22[1])
    if min(p0,p1) > r or max(p0,p1) < -r:
        return False
    
    return True

def BoxTrianglesIntersection(Tris, xlim, ylim, zlim, TriNormals=None, BoxCenter=None):
    # Akenine-Moller (2001) Fast 3D Triangle-Box Overlap Test
    # Vectorized version of TriangleBoxIntersection to test multiple triangles against a single box
    if not BoxCenter: BoxCenter = np.mean([xlim,ylim,zlim],axis=1)
    
    if type(Tris) is list: Tris = np.array(Tris)
        
    f0 = Tris[:,1]-Tris[:,0]
    f1 = Tris[:,2]-Tris[:,1]
    f2 = Tris[:,0]-Tris[:,2]
    hx = (xlim[1]-xlim[0])/2
    hy = (ylim[1]-ylim[0])/2
    hz = (zlim[1]-zlim[0])/2
    
    # Move triangles so that the box is centered around the origin 
    diff = Tris - BoxCenter
    v0 = diff[:,0]; v1 = diff[:,1]; v2 = diff[:,2]
    
    if TriNormals is None: 
        TriNormals = np.cross(f0,f1)
    elif type(TriNormals) is list:
        TriNormals = np.array(TriNormals)
    
    dist = np.sum(TriNormals*v0,axis=1)
    r0 = hx*np.abs(TriNormals[:,0]) + hy*np.abs(TriNormals[:,1]) + hz*np.abs(TriNormals[:,2])
    
    # Test Axes
    # a00
    a00 = np.vstack([np.zeros(len(f0)), -f0[:,2], f0[:,1]]).T
    p0 = np.sum(v0*a00,axis=1)
    p2 = np.sum(v2*a00,axis=1)
    r1 = hy*np.abs(a00[:,1]) + hz*np.abs(a00[:,2])
    ps1 = (p0,p2)
    
    # a01
    a01 = np.vstack([np.zeros(len(f1)), -f1[:,2], f1[:,1]]).T
    p0 = np.sum(v0*a01,axis=1)
    p1 = np.sum(v1*a01,axis=1)
    r2 = hy*np.abs(a01[:,1]) + hz*np.abs(a01[:,2])
    ps2 = (p0,p1)
    
    # a02
    a02 = np.vstack([np.zeros(len(f2)), -f2[:,2], f2[:,1]]).T
    p0 = np.sum(v0*a02,axis=1)
    p1 = np.sum(v1*a02,axis=1)
    r3 = hy*np.abs(a02[:,1]) + hz*np.abs(a02[:,2])
    ps3 = (p0,p1)
    
    # a10
    a10 = np.vstack([f0[:,2], np.zeros(len(f0)), -f0[:,0]]).T
    p0 = np.sum(v0*a10,axis=1)
    p2 = np.sum(v2*a10,axis=1)
    r4 = hx*np.abs(a10[:,0]) + hz*np.abs(a10[:,2])
    ps4 = (p0,p2)
    
    # a11
    a11 = np.vstack([f1[:,2], np.zeros(len(f1)), -f1[:,0]]).T
    p0 = np.sum(v0*a11,axis=1)
    p1 = np.sum(v1*a11,axis=1)
    r5 = hx*np.abs(a11[:,0]) + hz*np.abs(a11[:,2])
    ps5 = (p0,p1)
    
    # a12
    a12 = np.vstack([f2[:,2], np.zeros(len(f2)), -f2[:,0]]).T
    p0 = np.sum(v0*a12,axis=1)
    p1 = np.sum(v1*a12,axis=1)
    r6 = hx*np.abs(a12[:,0]) + hz*np.abs(a12[:,2])
    ps6 = (p0,p1)
    # a20
    a20 = np.vstack([-f0[:,1], f0[:,0], np.zeros(len(f0))]).T
    p0 = np.sum(v0*a20,axis=1)
    p2 = np.sum(v2*a20,axis=1)
    r7 = hx*np.abs(a20[:,0]) + hy*np.abs(a20[:,1])
    ps7 = (p0,p2)
    
    # a21
    a21 = np.vstack([-f1[:,1], f1[:,0], np.zeros(len(f1))]).T
    p0 = np.sum(v0*a21,axis=1)
    p1 = np.sum(v1*a21,axis=1)
    r8 = hx*np.abs(a21[:,0]) + hy*np.abs(a21[:,1])
    ps8 = (p0,p1)
    
    # a22
    a22 = np.vstack([-f2[:,1], f2[:,0], np.zeros(len(f2))]).T
    p0 = np.sum(v0*a22,axis=1)
    p1 = np.sum(v1*a22,axis=1)
    r9 = hx*np.abs(a22[:,0]) + hy*np.abs(a22[:,1])
    ps9 = (p0,p1)
    
    
    Intersections = np.repeat(True,len(Tris))
    
    checks = (
        # Test the box against the minimal Axis Aligned Bounding Box (AABB) of the tri
        (np.amax([v0[:,0],v1[:,0],v2[:,0]],axis=0) < -hx) | 
        (np.amin([v0[:,0],v1[:,0],v2[:,0]],axis=0) >  hx) |
        (np.amax([v0[:,1],v1[:,1],v2[:,1]],axis=0) < -hy) | 
        (np.amin([v0[:,1],v1[:,1],v2[:,1]],axis=0) >  hy) |
        (np.amax([v0[:,2],v1[:,2],v2[:,2]],axis=0) < -hz) | 
        (np.amin([v0[:,2],v1[:,2],v2[:,2]],axis=0) >  hz) |
        # Test normal of the triangle
        (dist > r0) |
        # Test Axes
        (np.minimum(*ps1) > r1) | (np.maximum(*ps1) < -r1) |
        (np.minimum(*ps2) > r2) | (np.maximum(*ps2) < -r2) |
        (np.minimum(*ps3) > r3) | (np.maximum(*ps3) < -r3) |
        (np.minimum(*ps4) > r4) | (np.maximum(*ps4) < -r4) |
        (np.minimum(*ps5) > r5) | (np.maximum(*ps5) < -r5) |
        (np.minimum(*ps6) > r6) | (np.maximum(*ps6) < -r6) |
        (np.minimum(*ps7) > r7) | (np.maximum(*ps7) < -r7) |
        (np.minimum(*ps8) > r8) | (np.maximum(*ps8) < -r8) |
        (np.minimum(*ps9) > r9) | (np.maximum(*ps9) < -r9)
        )
    
    
    Intersections[checks] = False
    return Intersections
    
def SegmentSegmentIntersection(s1,s2,return_intersection=False,endpt_inclusive=True,eps=0):
    # https://mathworld.wolfram.com/Line-LineIntersection.html
    # Goldman (1990)
    [p1,p2] = np.array(s1)
    [p3,p4] = np.array(s2)
    
    a = p2-p1; b = p4-p3; c = p3-p1
    axb = np.cross(a,b)
    axbnorm2 = (np.linalg.norm(axb))**2
    s = np.dot(np.cross(c,b),axb)/axbnorm2
    t = np.dot(np.cross(c,a),axb)/axbnorm2
    
    if endpt_inclusive:
        Intersection = (0 <= s <= 1) and (0 <= t <= 1) & (axbnorm2 > eps)
    else:
        Intersection = (0+eps < s < 1-eps) and (0+eps < t < 1-eps) & (axbnorm2 > eps)

    if return_intersection:
        pt = p1 + a*s
        return Intersection, pt

    ###
    # [x1,x2] = np.array(s1)
    # [x3,x4] = np.array(s2)
    # p1 = x1; V1 = x2-x1
    # p2 = x3; V2 = x4-x3
    # V1xV2 = np.cross(V1,V2)
    # denom = np.linalg.norm(V1xV2)**2
    # t = np.linalg.det([p2-p1,V2,V1xV2])/denom
    # s = np.linalg.det([p2-p1,V1,V1xV2])/denom
    # s = np.clip(s,0,1)
    # t = np.clip(t,0,1)
    # x1 = p1+V1*t
    # x2 = p2+V2*s
    # if endpt_inclusive:
    #     if np.linalg.norm(x2-x1) <= eps:
    #         Intersection = True
    #     else:
    #         Intersection = False
    # else:
    #     if np.linalg.norm(x2-x1,axis=1) <= eps and (0+eps < s) and (s < 1-eps) and (0+eps < t) and (t < 1-eps):
    #         Intersection = True
    #     else:
    #         Intersection = False
    # if return_intersection:
    #     if Intersection:
    #         return Intersection, x1
    #     else:
    #         return Intersection, np.repeat(np.nan,3)
    ###
    return Intersection

def SegmentsSegmentsIntersection(s1,s2,return_intersection=False,endpt_inclusive=True,eps=0):
    # https://mathworld.wolfram.com/Line-LineIntersection.html
    # Goldman (1990)
    if type(s1) is list: s1 = np.array(s1)
    if type(s2) is list: s2 = np.array(s2)
    p1 = s1[:,0]; p2 = s1[:,1]
    p3 = s2[:,0]; p4 = s2[:,1]
    
    a = p2-p1; b = p4-p3; c = p3-p1
    axb = np.cross(a,b,axis=1)
    cxb = np.cross(c,b,axis=1)
    cxa = np.cross(c,a,axis=1)
    axbnorm2 = np.sum(axb**2,axis=1) #+ 1e-32
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.sum(cxb*axb,axis=1)/axbnorm2
        t = np.sum(cxa*axb,axis=1)/axbnorm2
    # Collinear: Currently not getting intersection points for perfectly collinear lines
    if endpt_inclusive:
        Intersections = (0-eps <= s) & (s <= 1+eps) & (0-eps <= t) & (t <= 1+eps) & (axbnorm2 > eps**2) ## DON'T GET RID OF THE LAST CHECK
        # Intersections = (0 <= s) & (s <= 1) & (0 <= t) & (t <= 1) & (axbnorm2 > eps**2)
        ###

        # Collinear = (axbnorm2 <= eps**2)
        # np.linalg.norm(axb/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))[:,None],axis=1)
        
    else:
        Intersections = (0+eps < s) & (s < 1-eps) & (0+eps < t) & (t < 1-eps) & (axbnorm2 > eps**2)
    if return_intersection:
        ### Without collinear:
        pts = np.nan*np.ones((len(Intersections),3))
        pts[Intersections] = p1[Intersections] + a[Intersections]*s[Intersections,None]
        ###

        ### With collinear: (TBD)
        # pts = np.nan*np.ones((len(Intersections),2,3))
        ###
        return Intersections, pts
    return Intersections

def SegmentsSegmentsIntersection2(s1,s2,return_intersection=False,endpt_inclusive=True,eps=1e-14):
    # https://mathworld.wolfram.com/Line-LineIntersection.html
    # Goldman (1990)
    if type(s1) is list: s1 = np.array(s1)
    if type(s2) is list: s2 = np.array(s2)
    
    x1,x2 = s1[:,0],s1[:,1]
    x3,x4 = s2[:,0],s2[:,1]
    p1 = x1; V1 = x2-x1
    p2 = x3; V2 = x4-x3
    V1xV2 = np.cross(V1,V2,axis=1)
    denom = np.linalg.norm(V1xV2,axis=1)**2

    t = np.linalg.det(np.array([p2-p1,V2,V1xV2]).swapaxes(0,1))/denom
    s = np.linalg.det(np.array([p2-p1,V1,V1xV2]).swapaxes(0,1))/denom
    s = np.clip(s,0,1)
    t = np.clip(t,0,1)
    
    x1 = p1+V1*t[:,None]
    x2 = p2+V2*s[:,None]
    # print((np.linalg.norm(x1-s1[:,0],axis=1)))
    # print((np.linalg.norm(x1-s1[:,1],axis=1)))
    if endpt_inclusive:
        Intersections = (np.linalg.norm(x2-x1,axis=1) == 0)
    else:
        Intersections = (np.linalg.norm(x2-x1,axis=1) == 0) & ~((np.linalg.norm(x1-s1[:,0],axis=1) <= eps) | (np.linalg.norm(x1-s1[:,1],axis=1) <= eps))
    if return_intersection:
        pts = np.nan*np.ones((len(Intersections),3))
        pts[Intersections] = x1[Intersections]
        return Intersections, pts
    return Intersections
    
def RaySurfIntersection(pt, ray, NodeCoords, SurfConn, eps=1e-14, octree='generate'):
    
    def test(pt, ray, nodes):
        iPt = RayTriangleIntersection(pt, ray, nodes, bidirectional=True)
        return iPt
    ArrayCoords = np.array(NodeCoords)
    if type(pt) is list: pt = np.array(pt)
    if type(ray) is list: ray = np.array(ray)
    if octree == None or octree == 'None' or octree == 'none':
        # Won't use any octree structure to accelerate intersection tests
        intersections,intersectionPts = RayTrianglesIntersection(pt, ray, ArrayCoords[SurfConn], bidirectional=True, eps=eps)
        distances = np.sum(ray*(intersectionPts-pt),axis=1)
    elif octree == 'generate' or type(octree) == Octree.OctreeNode:
        if octree == 'generate':
            # Create an octree structure based on the provided structure
            root = Octree.Surf2Octree(NodeCoords,SurfConn)
        else:
            # Using an already generated octree structure
            # If this is the case, it should conform to the same structure and labeling as one generated with Octree.Surf2Octree
            root = octree
        # Proceeding with octree-accelerated intersection test
        def octreeTest(pt, ray, node, TriIds):
            [xlim,ylim,zlim] = node.getLimits()
            if RayBoxIntersection(pt, ray, xlim, ylim, zlim):
                if node.state == 'leaf':
                    TriIds += node.data
                elif node.state == 'root' or node.state == 'branch':
                    for child in node.children:
                        TriIds = octreeTest(pt, ray, child, TriIds)
            return TriIds
        
        TriIds = octreeTest(pt, ray, root, [])
        
        Tris = ArrayCoords[np.asarray(SurfConn)[TriIds]]
        intersections,intersectionPts = RayTrianglesIntersection(pt, ray, Tris, bidirectional=True, eps=eps)
        intersections = np.array(TriIds)[intersections]
        distances = np.sum(ray*(intersectionPts-pt),axis=1)
    else:
        raise Exception('Invalid octree argument given')
        
    
    
    return intersections, distances, intersectionPts
        
def SurfSelfIntersection(NodeCoords, SurfConn, octree='generate', eps=1e-14, return_pts=False):
    if octree == None or octree == 'None' or octree == 'none':
        # Won't use any octree structure to accelerate intersection tests
        root = None
    elif octree == 'generate':
        # Create an octree structure based on the provided structure
        root = Octree.Surf2Octree(NodeCoords,SurfConn)
    elif type(octree) == Octree.OctreeNode:
        # Using an already generated octree structure
        # If this is the case, it should conform to the same structure and labeling as one generated with Octree.Surf2Octree
        root = octree
    else:
        raise Exception('Invalid octree argument given: '+str(octree))
    
    Points = np.array(NodeCoords)[np.array(SurfConn)]   
    if root == None:
        combinations = list(itertools.combinations(range(len(SurfConn)),2))
        idx1,idx2 = zip(*combinations)
        Tri1s = Points[np.array(idx1)]; Tri2s = Points[np.array(idx2)]
    else:
        leaves = Octree.getAllLeaf(root)
        combinations = []
        for leaf in leaves:
            combinations += list(itertools.combinations(leaf.data,2))
        
        idx1,idx2 = zip(*combinations)
        Tri1s = Points[np.array(idx1)]; Tri2s = Points[np.array(idx2)]

    if return_pts:
        intersections,intersectionPts = TrianglesTrianglesIntersectionPts(Tri1s,Tri2s,eps=eps)
        IntersectionPairs = np.array(combinations)[intersections].tolist()
        IntersectionPoints = intersectionPts[intersections]
        return IntersectionPairs, IntersectionPoints
        
    else:
        intersections = TrianglesTrianglesIntersection(Tri1s,Tri2s,eps=eps)
        IntersectionPairs = np.array(combinations)[intersections].tolist()
            
    return IntersectionPairs
    
def SurfSurfIntersection(NodeCoords1, SurfConn1, NodeCoords2, SurfConn2, eps=1e-14, return_pts=False):
    # if octree == None or octree == 'None' or octree == 'none':
    #     # Won't use any octree structure to accelerate intersection tests
    #     root = None
    # elif octree == 'generate':
    #     # Create an octree structure based on the provided structure
    #     root = Octree.Surf2Octree(NodeCoords,SurfConn)
    # elif type(octree) == Octree.OctreeNode:
    #     # Using an already generated octree structure
    #     # If this is the case, it should conform to the same structure and labeling as one generated with Octree.Surf2Octree
    #     root = octree
    # else:
    #     raise Exception('Invalid octree argument given: '+str(octree))

    MergeCoords,MergeConn = MeshUtils.MergeMesh(NodeCoords1, SurfConn1, NodeCoords2, SurfConn2, cleanup=False)
    root = Octree.Surf2Octree(MergeCoords,MergeConn)
    
    Points = np.array(MergeCoords)[np.array(MergeConn)]   
    if root == None:
        combinations = list(itertools.combinations(range(len(MergeConn)),2))
        idx1,idx2 = zip(*combinations)
        Tri1s = Points[np.array(idx1)]; Tri2s = Points[np.array(idx2)]
    else:
        leaves = Octree.getAllLeaf(root)
        combinations = []
        for leaf in leaves:
            combinations += list(itertools.combinations(leaf.data,2))
        
        idx1,idx2 = zip(*combinations)
        Tri1s = Points[np.array(idx1)]; Tri2s = Points[np.array(idx2)]

    if return_pts:
        intersections,intersectionPts = TrianglesTrianglesIntersectionPts(Tri1s,Tri2s,eps=eps,edgeedge=True)
        IntersectionPairs = np.array(combinations)[intersections].tolist()
        IPoints = intersectionPts[intersections]
        IPoints[np.isnan(IPoints)]=-1
        IPoints = MeshUtils.ExtractRagged(IPoints)

        # TODO: I'm being lazy here
        Surf1Intersections = []; Surf2Intersections = []; IntersectionPoints = []
        for i in range(len(IntersectionPairs)):
            if IntersectionPairs[i][0] < len(SurfConn1) and IntersectionPairs[i][1] >= len(SurfConn1):
                Surf1Intersections.append(IntersectionPairs[i][0])
                Surf2Intersections.append(IntersectionPairs[i][1]-len(SurfConn1))
                IntersectionPoints.append(IPoints[i])
            elif IntersectionPairs[i][1] < len(SurfConn1) and IntersectionPairs[i][0] >= len(SurfConn1):
                Surf1Intersections.append(IntersectionPairs[i][1])
                Surf2Intersections.append(IntersectionPairs[i][0]-len(SurfConn1))
                IntersectionPoints.append(IPoints[i])
            # Ignoring self intersections
        return Surf1Intersections, Surf2Intersections, IntersectionPoints
        
    else:
        intersections = TrianglesTrianglesIntersection(Tri1s,Tri2s,eps=eps,edgeedge=True)
        IntersectionPairs = np.array(combinations)[intersections].tolist()
        # TODO: I'm being lazy here
        Surf1Intersections = []; Surf2Intersections = []
        for i in range(len(IntersectionPairs)):
            if IntersectionPairs[i][0] < len(SurfConn1) and IntersectionPairs[i][1] >= len(SurfConn1):
                Surf1Intersections.append(IntersectionPairs[i][0])
                Surf2Intersections.append(IntersectionPairs[i][1]-len(SurfConn1))
            elif IntersectionPairs[i][1] < len(SurfConn1) and IntersectionPairs[i][0] >= len(SurfConn1):
                Surf1Intersections.append(IntersectionPairs[i][1])
                Surf2Intersections.append(IntersectionPairs[i][0]-len(SurfConn1))
            
    return Surf1Intersections, Surf2Intersections

## Inside Tests
def isInsideSurf(pt, NodeCoords, SurfConn, ElemNormals, octree=None, eps=1e-8, ray=np.random.rand(3)):
    
    if octree == None or octree == 'None' or octree == 'none':
        # Won't use any octree structure to accelerate intersection tests
        root = None
    elif octree == 'generate':
        # Create an octree structure based on the provided structure
        root = Octree.Surf2Octree(NodeCoords,SurfConn)
    elif type(octree) == Octree.OctreeNode:
        # Using an already generated octree structure
        # If this is the case, it should conform to the same structure and labeling as one generated with Octree.Surf2Octree
        root = octree
    else:
        raise Exception('Invalid octree argument given: '+str(octree))
        
    intersections, distances,_ = RaySurfIntersection(pt,ray,NodeCoords,SurfConn,octree=root)
    posDistances = np.array([d for d in distances if d > eps])
    zero = np.any(np.abs(distances)<eps)
    # intersections2 = [intersections[i] for i,d in enumerate(distances) if d >= 0]
    
    # Checking unique to not double count instances where ray intersects an edge
    if len(np.unique(np.round(posDistances/eps)))%2 == 0 and not zero:
        # print(distances)
        # No intersection
        return False
    else:
        dist = min(np.abs(distances))
        if dist < eps:
            closest = np.array(intersections)[np.abs(distances)==dist][0]
            dot = np.dot(ray,ElemNormals[closest])
            if dot > 0:
                a = 'merp'
            return dot
        else:
            # Inside
            return True
            
def isInsideBox(pt, xlim, ylim, zlim):
    lims = [xlim,ylim,zlim]
    return all([lims[d][0] < pt[d] and lims[d][1] > pt[d] for d in range(3)])

def InsideVoxel(pts, VoxelCoords, VoxelConn, inclusive=True):    
    Root = Octree.Voxel2Octree(VoxelCoords, VoxelConn)
    inside = [False for i in range(len(pts))]
    for i,pt in enumerate(pts):
        inside[i] = Octree.isInsideOctree(pt, Root, inclusive=inclusive)    
    
    return inside
        
def PointInTri(Tri,pt,method='BaryArea',eps=1e-12,inclusive=True):

    if method == 'Normal':
        pts = np.vstack([Tri,pt])
        conn = [[0,1,3],[1,2,3],[2,0,3]]
        normals = MeshUtils.CalcFaceNormal(pts,conn)
        if np.dot(normals[0],normals[1]) < 0:
            In = False
        elif np.dot(normals[1],normals[2]) < 0:
            In = False
        else:
            In = True
    elif method == 'Bary':
        alpha,beta,gamma = MeshUtils.BaryTri(Tri,pt)
        In = all([alpha>=0,beta>=0,gamma>=0])
    elif method == 'BaryArea':
        A = Tri[0]
        B = Tri[1]
        C = Tri[2]
        AB = np.subtract(A,B)
        AC = np.subtract(A,C)
        PA = np.subtract(pt,A)
        PB = np.subtract(pt,B)
        PC = np.subtract(pt,C)

        Area2 = np.linalg.norm(np.cross(AB,AC))
        
        denom = 1/Area2
        alpha = np.linalg.norm(np.cross(PB,PC))*denom
        beta = np.linalg.norm(np.cross(PC,PA))*denom
        gamma = np.linalg.norm(np.cross(PA,PB))*denom
        if inclusive:
            In = all([alpha>=0,beta>=0,gamma>=0]) and np.abs(alpha+beta+gamma-1) < eps
        else:
            In = all([alpha>=eps,beta>=eps,gamma>=eps]) and np.abs(alpha+beta+gamma-1) < eps
    return In

def PointsInTris(Tris,pts,method='BaryArea',eps=1e-12,inclusive=True):
    # Pairwise comparisons between each triangle in tris and its corresponding point in pts
    if method == 'BaryArea':
        A = Tris[:,0]
        B = Tris[:,1]
        C = Tris[:,2]
        AB = np.subtract(A,B)
        AC = np.subtract(A,C)
        PA = np.subtract(pts,A)
        PB = np.subtract(pts,B)
        PC = np.subtract(pts,C)

        Area2 = np.linalg.norm(np.cross(AB,AC),axis=1)

        
        denom = 1/Area2
        alpha = np.linalg.norm(np.cross(PB,PC),axis=1)*denom
        beta = np.linalg.norm(np.cross(PC,PA),axis=1)*denom
        gamma = np.linalg.norm(np.cross(PA,PB),axis=1)*denom
        # print(alpha,beta,gamma)
        if inclusive:
            In = np.all([alpha>=0,beta>=0,gamma>=0],axis=0) & (np.abs(alpha+beta+gamma-1) <= eps)
        else:
            In = np.all([alpha>=eps,beta>=eps,gamma>=eps],axis=0) & (np.abs(alpha+beta+gamma-1) <= eps)
    return In