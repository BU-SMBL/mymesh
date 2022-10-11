# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:10:08 2021

@author: toj
"""
import numpy as np
import sympy as sp
from . import MeshUtils, converter
import warnings
  
def NormCurve(NodeCoords,SurfConn,NodeNeighbors,NodeNormals):
    #
    # Based on Goldfeather & Interrante (2004)
    
    
    SurfNodes = np.unique(SurfConn)
    
    MaxPrincipal = [0 for i in range(len(NodeCoords))]
    MinPrincipal = [0 for i in range(len(NodeCoords))]
    
    k = [0,0,1]
    for i in SurfNodes:
        p = NodeCoords[i]   # Coordinates of the current node
        n = NodeNormals[i]  # Unit normal vector of the current node
        
        # Rotation matrix from global z (k=[0,0,1]) to local z(n)
        if n == k:
            rotAxis = k
            angle = 0
        elif n == [-1*i for i in k]:
            rotAxis = [1,0,0]
            angle = np.pi
        else:
            rotAxis = np.cross(k,n)/np.linalg.norm(np.cross(k,n))
            angle = np.arccos(np.dot(k,n))
        q = [np.cos(angle/2),               # Quaternion Rotation
             rotAxis[0]*np.sin(angle/2),
             rotAxis[1]*np.sin(angle/2),
             rotAxis[2]*np.sin(angle/2)]
    
        R = [[2*(q[0]**2+q[1]**2)-1,   2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2]), 0],
             [2*(q[1]*q[2]+q[0]*q[3]), 2*(q[0]**2+q[2]**2)-1,   2*(q[2]*q[3]-q[0]*q[1]), 0],
             [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 2*(q[0]**2+q[3]**2)-1,   0],
             [0,                       0,                       0,                       1]
             ]
        # Translation to map p to (0,0,0)
        T = [[1,0,0,-p[0]],
             [0,1,0,-p[1]],
             [0,0,1,-p[2]],
             [0,0,0,1]]
        
        Amat = [[] for j in range(len(NodeNeighbors[i]))]
        Bmat = [0 for j in range(len(NodeNeighbors[i]))]
        for j in range(len(NodeNeighbors[i])):
            # Get adjacent nodes
            q = [x for x in NodeCoords[NodeNeighbors[i][j]]]
            kappa = 2 * np.dot(np.subtract(p,q),n)/(np.dot(np.subtract(p,q),np.subtract(p,q)))
            
            q.append(1)
            # Transform to local coordinate system
            [xj,yj,zj,one] = np.matmul(np.matmul(T,q),R)
            
            # Vector in local csys from p to qj
            y0 = [xj,yj,zj]
            # Projection onto the plane defined by p, n in local csys (n = [0,0,1])
            yproj = np.subtract(y0,np.multiply(np.dot(y0,[0,0,1]),[0,0,1]))
            uproj = yproj/np.linalg.norm(yproj)
            u = uproj[0]
            v = uproj[1]
            Amat[j] = [u**2, 2*u*v, v**2]
            Bmat[j] = kappa
        try:
            X = np.linalg.solve(np.matmul(np.transpose(Amat),Amat),np.matmul(np.transpose(Amat),Bmat))
        
            # Weingarten Matrix
            W = [[X[0],X[1]],
                 [X[1],X[2]]]
            [v,x] = np.linalg.eig(W)
        except:
            a = 'merp'
            v = [np.nan,np.nan]
        MaxPrincipal[i] = max(v)    # Max Principal Curvature
        MinPrincipal[i] = min(v)    # Min Principal Curvature
    return MaxPrincipal,MinPrincipal
                     
def QuadFit(NodeCoords,SurfConn,NodeNeighbors,NodeNormals):
    #QuadFit uses a local quadratic fitting method to determine the principal,
    # mean, and Gaussian curvatures at each node in a triangular surface mesh
    # Based on Goldfeather & Interrante (2004)
    
    
    SurfNodes = np.unique(SurfConn)
    
    MaxPrincipal = [0 for i in range(len(NodeCoords))]
    MinPrincipal = [0 for i in range(len(NodeCoords))]
    
    k = [0,0,1]
    for i in SurfNodes:
        p = NodeCoords[i]   # Coordinates of the current node
        n = np.multiply(-1,NodeNormals[i]).tolist()  # Unit normal vector of the current node
        
        # Rotation matrix from global z (k=[0,0,1]) to local z(n)
        if n == k:
            rotAxis = k
            angle = 0
        elif n == [-1*i for i in k]:
            rotAxis = [1,0,0]
            ange = np.pi
        else:
            rotAxis = np.cross(k,n)/np.linalg.norm(np.cross(k,n))
            angle = np.arccos(np.dot(k,n))
        q = [np.cos(angle/2),               # Quaternion Rotation
             rotAxis[0]*np.sin(angle/2),
             rotAxis[1]*np.sin(angle/2),
             rotAxis[2]*np.sin(angle/2)]
    
        R = [[2*(q[0]**2+q[1]**2)-1,   2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2]), 0],
             [2*(q[1]*q[2]+q[0]*q[3]), 2*(q[0]**2+q[2]**2)-1,   2*(q[2]*q[3]-q[0]*q[1]), 0],
             [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 2*(q[0]**2+q[3]**2)-1,   0],
             [0,                       0,                       0,                       1]
             ]
        # Translation to map p to (0,0,0)
        T = [[1,0,0,-p[0]],
             [0,1,0,-p[1]],
             [0,0,1,-p[2]],
             [0,0,0,1]]
        
        Amat = [[] for j in range(len(NodeNeighbors[i]))]
        Bmat = [0 for j in range(len(NodeNeighbors[i]))]
        for j in range(len(NodeNeighbors[i])):
            # Get adjacent nodes
            q = [x for x in NodeCoords[NodeNeighbors[i][j]]]
            q.append(1)
            # Transform to local coordinate system
            [xj,yj,zj,one] = np.matmul(np.matmul(T,q),R)
            # Scale z by 2/k^2, where k=sqrt(x^2+y^2)
            scale = 2/(xj**2+yj**2)   
            # Assemble matrices for least squares
            Amat[j] = [scale/2*xj**2, scale*xj*yj, scale/2*yj**2]
            Bmat[j] = scale*zj
        try:
            X = np.linalg.solve(np.matmul(np.transpose(Amat),Amat),np.matmul(np.transpose(Amat),Bmat))
        
            # Weingarten Matrix
            W = [[X[0],X[1]],
                 [X[1],X[2]]]
            [v,x] = np.linalg.eig(W)
        except:
            a = 'merp'
            v = [np.nan,np.nan]
        MaxPrincipal[i] = max(v)    # Max Principal Curvature
        MinPrincipal[i] = min(v)    # Min Principal Curvature
    return MaxPrincipal,MinPrincipal

def CubicFit(NodeCoords,SurfConn,NodeNeighborhoods,NodeNormals,Ignore=set(),IgnoreFeatures=False):
    #
    # Based on Goldfeather & Interrante (2004)    
    if IgnoreFeatures:
        edges,corners = MeshUtils.DetectFeatures(NodeCoords,SurfConn)
        FeatureNodes = set(edges).union(corners)
        Ignore.update(FeatureNodes)
    SurfNodes = np.unique(SurfConn)
    # EdgeConn = converter.surf2edges(NodeCoords,SurfConn)
    # EdgeNodes = set()#set(np.unique(EdgeConn))
    
    MaxPrincipal = [np.nan for i in range(len(NodeCoords))]
    MinPrincipal = [np.nan for i in range(len(NodeCoords))]
    
    k = [0,0,1]
    for i in SurfNodes:
        p = NodeCoords[i]   # Coordinates of the current node
        n = np.multiply(-1,NodeNormals[i]).tolist()  # Unit normal vector of the current node
        
        # Rotation matrix from global z (k=[0,0,1]) to local z (n)
        if n == k:
            rotAxis = k
            angle = 0
        elif n == [-1*i for i in k]:
            rotAxis = [1,0,0]
            angle = np.pi
        else:
            rotAxis = np.cross(k,n)/np.linalg.norm(np.cross(k,n))
            angle = np.arccos(np.dot(k,n))
        q = [np.cos(angle/2),               # Quaternion Rotation
             rotAxis[0]*np.sin(angle/2),
             rotAxis[1]*np.sin(angle/2),
             rotAxis[2]*np.sin(angle/2)]
    
        R = [[2*(q[0]**2+q[1]**2)-1,   2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2]), 0],
             [2*(q[1]*q[2]+q[0]*q[3]), 2*(q[0]**2+q[2]**2)-1,   2*(q[2]*q[3]-q[0]*q[1]), 0],
             [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 2*(q[0]**2+q[3]**2)-1,   0],
             [0,                       0,                       0,                       1]
             ]
        # Translation to map p to (0,0,0)
        T = [[1,0,0,-p[0]],
             [0,1,0,-p[1]],
             [0,0,1,-p[2]],
             [0,0,0,1]]
        
        Amat = []
        Bmat = []
        for j in range(len(NodeNeighborhoods[i])):
            # Get adjacent nodes
            if NodeNeighborhoods[i][j] in Ignore:
                # Skip ignored nodes
                continue
            cq = [x for x in NodeCoords[NodeNeighborhoods[i][j]]]
            nq = [x for x in NodeNormals[NodeNeighborhoods[i][j]]]
            cq.append(1)
            nq.append(1)
            # Transform to local coordinate system
            [xj,yj,zj,one] = np.matmul(np.matmul(T,cq),R)
            [aj,bj,cj,one] = np.matmul(nq,R)
            if cj == 0:
                cj = 1
            # Scale z by 2/k^2, where k=sqrt(x^2+y^2)
            scale = 2/(xj**2+yj**2)  
            # Assemble matrices for least squares
            Amat.append([scale/2*xj**2, scale*xj*yj, scale/2*yj**2, scale*xj**3, scale*xj**2*yj, scale*xj*yj**2, scale*yj**3])
            Bmat.append(scale*zj)
            Amat.append([scale*xj, scale*yj, 0, scale*3*xj**2, scale*2*xj*yj, scale*yj**2, 0])
            Bmat.append(-scale*aj/cj)
            Amat.append([0, scale*xj, scale*yj, 0, scale*xj**2, scale*2*xj*yj, scale*3*yj**2])
            Bmat.append(-scale*bj/cj)
            
        # add buffer if edge node
        # if i in EdgeNodes:
        #     for j in range(len(NodeNeighborhoods[i])):
        #         if NodeNeighborhoods[i][j] not in EdgeNodes:
        #             if NodeNeighborhoods[i][j] in Ignore:
        #                 # Skip ignored nodes
        #                 continue
        #             # For nodes in the neighborhood that aren't also edge nodes
        #             cq = [x for x in NodeCoords[NodeNeighborhoods[i][j]]]
        #             nq = [x for x in NodeNormals[NodeNeighborhoods[i][j]]]
        #             cq.append(1)
        #             nq.append(1)
        #             # Transform to local coordinate system
        #             [xj,yj,zj,one] = np.matmul(np.matmul(T,cq),R)
        #             [aj,bj,cj,one] = np.matmul(nq,R)
                    
        #             # Rotate 180 about the central node/local z axis
        #             q2 = [np.cos(np.pi/2),               # Quaternion Rotation
        #                  rotAxis[0]*np.sin(np.pi/2),
        #                  rotAxis[1]*np.sin(np.pi/2),
        #                  rotAxis[2]*np.sin(np.pi/2)]
                
        #             R2 = [[2*(q2[0]**2+q2[1]**2)-1,   2*(q2[1]*q2[2]-q2[0]*q2[3]), 2*(q2[1]*q2[3]+q2[0]*q2[2]), 0],
        #                  [2*(q2[1]*q2[2]+q2[0]*q2[3]), 2*(q2[0]**2+q2[2]**2)-1,   2*(q2[2]*q2[3]-q2[0]*q2[1]), 0],
        #                  [2*(q2[1]*q2[3]-q2[0]*q2[2]), 2*(q2[2]*q2[3]+q2[0]*q2[1]), 2*(q2[0]**2+q2[3]**2)-1,   0],
        #                  [0,                       0,                       0,                       1]
        #                  ]
        #             [xj,yj,zj,one] = np.matmul([xj,yj,zj,one],R2)
        #             [aj,bj,cj,one] = np.matmul([aj,bj,cj,one],R2)
        #             # Scale z by 2/k^2, where k=sqrt(x^2+y^2)
        #             scale = 2/(xj**2+yj**2)  
        #             # Assemble matrices for least squares
        #             Amat.append([scale/2*xj**2, scale*xj*yj, scale/2*yj**2, scale*xj**3, scale*xj**2*yj, scale*xj*yj**2, scale*yj**3])
        #             Bmat.append(scale*zj)
        #             Amat.append([scale*xj, scale*yj, 0, scale*3*xj**2, scale*2*xj*yj, scale*yj**2, 0])
        #             Bmat.append(-scale*aj/cj)
        #             Amat.append([0, scale*xj, scale*yj, 0, scale*xj**2, scale*2*xj*yj, scale*3*yj**2])
        #             Bmat.append(-scale*bj/cj)
        
        if len(Bmat) > 6:
            try:   
                A = np.matmul(np.transpose(Amat),Amat)
                B = np.matmul(np.transpose(Amat),Bmat)
                X = np.linalg.solve(A,B)
                W = [[X[0],X[1]],
                    [X[1],X[2]]]
                [v,x] = np.linalg.eig(W)
            except:
                warnings.warn('curvature problem.')
                v = [np.nan,np.nan]
            # Weingarten Matrix
            
        else:
            v = [np.nan,np.nan]
        MaxPrincipal[i] = max(v)    # Max Principal Curvature
        MinPrincipal[i] = min(v)    # Min Principal Curvature
    return MaxPrincipal,MinPrincipal    
   
def MeanCurvature(MaxPrincipal,MinPrincipal):
    
    if type(MaxPrincipal) == np.ndarray:
        MaxPrincipal = MaxPrincipal.tolist()
    if type(MinPrincipal) == np.ndarray:
        MinPrincipal = MinPrincipal.tolist()
    if type(MaxPrincipal) == list and type(MinPrincipal) == list and len(MaxPrincipal) == len(MinPrincipal):
        mean = [(MaxPrincipal[i] + MinPrincipal[i])/2 for i in range(len(MaxPrincipal))]
    elif (type(MaxPrincipal) == int or type(MaxPrincipal) == float) and (type(MinPrincipal) == int or type(MinPrincipal) == float):
        mean = (MaxPrincipal + MinPrincipal)/2
    return mean

def GaussianCurvature(MaxPrincipal,MinPrincipal):
    
    if type(MaxPrincipal) == np.ndarray:
        MaxPrincipal = MaxPrincipal.tolist()
    if type(MinPrincipal) == np.ndarray:
        MinPrincipal = MinPrincipal.tolist()
    if type(MaxPrincipal) == list and type(MinPrincipal) == list and len(MaxPrincipal) == len(MinPrincipal):
        gaussian = [MaxPrincipal[i] * MinPrincipal[i] for i in range(len(MaxPrincipal))]
    elif (type(MaxPrincipal) == int or type(MaxPrincipal) == float) and (type(MinPrincipal) == int or type(MinPrincipal) == float):
        gaussian = MaxPrincipal * MinPrincipal
    return gaussian

def Curvedness(MaxPrincipal,MinPrincipal):
    # Ref: Koenderink, J.J. and Van Doorn, A.J., 1992. Surface shape and curvature scales. Image and vision computing, 10(8), pp.557-564.
    if type(MaxPrincipal) == np.ndarray:
        MaxPrincipal = MaxPrincipal.tolist()
    if type(MinPrincipal) == np.ndarray:
        MinPrincipal = MinPrincipal.tolist()
    if type(MaxPrincipal) == list and type(MinPrincipal) == list and len(MaxPrincipal) == len(MinPrincipal):
        curvedness = [np.sqrt((MaxPrincipal[i]**2 + MinPrincipal[i]**2)/2) for i in range(len(MaxPrincipal))]
    elif (type(MaxPrincipal) == int or type(MaxPrincipal) == float) and (type(MinPrincipal) == int or type(MinPrincipal) == float):
        curvedness = np.sqrt((MaxPrincipal**2 + MinPrincipal**2)/2)
    
    return curvedness
 
def ShapeIndex(MaxPrincipal,MinPrincipal):
    # Ref: Koenderink, J.J. and Van Doorn, A.J., 1992. Surface shape and curvature scales. Image and vision computing, 10(8), pp.557-564.
    # Note: the equation from Koenderink & van Doorn has the equation: pi/2*arctan((min+max)/(min-max)), but this doesn't
    # seem to give values consistent with what are described as cups/caps - instead using pi/2*arctan((max+min)/(max-min))
    if type(MaxPrincipal) == np.ndarray:
        MaxPrincipal = MaxPrincipal.tolist()
    if type(MinPrincipal) == np.ndarray:
        MinPrincipal = MinPrincipal.tolist()
    if type(MaxPrincipal) == list and type(MinPrincipal) == list and len(MaxPrincipal) == len(MinPrincipal):
        shape = [(2/np.pi) * np.arctan((MaxPrincipal[i] + MinPrincipal[i])/(MaxPrincipal[i] - MinPrincipal[i])) if (MaxPrincipal[i] != MinPrincipal[i]) else 1 if MaxPrincipal[i]>0 else -1 for i in range(len(MaxPrincipal))]
    elif (type(MaxPrincipal) == int or type(MaxPrincipal) == float) and (type(MinPrincipal) == int or type(MinPrincipal) == float):
        if MaxPrincipal != MinPrincipal:
            shape = (2/np.pi) * np.arctan((MinPrincipal + MaxPrincipal)/(MinPrincipal - MaxPrincipal))
        else:
            shape = 0
    return shape

def ShapeCategory(shapeindex):
    # Ref: Koenderink, J.J. and Van Doorn, A.J., 1992. Surface shape and curvature scales. Image and vision computing, 10(8), pp.557-564.
    # Classifies the shape index into shape types
    # 0 = Spherical Cup
    # 1 = Trough
    # 2 = Rut
    # 3 = Saddle Rut
    # 4 = Saddle
    # 5 = Saddle Ridge
    # 6 = Ridge
    # 7 = Dome
    # 8 = Spherical Cap
    shape = [-1 for i in range(len(shapeindex))]
    for i in range(len(shapeindex)):
        if shapeindex[i] < -7/8:
            shape[i] = 0
        elif shapeindex[i] < -5/8:
            shape[i] = 1
        elif shapeindex[i] < -3/8:
            shape[i] = 2
        elif shapeindex[i] < -1/8:
            shape[i] = 3
        elif shapeindex[i] < 1/8:
            shape[i] = 4
        elif shapeindex[i] < 3/8:
            shape[i] = 5
        elif shapeindex[i] < 5/8:
            shape[i] = 6
        elif shapeindex[i] < 7/8:
            shape[i] = 7
        elif shapeindex[i] <= 1:
            shape[i] = 8
    return shape
                       
def AnalyticalCurvature(F,NodeCoords):
    x, y, z = sp.symbols('x y z', real=True)
    if type(NodeCoords) is list: NodeCoords = np.asarray(NodeCoords)

    def DiracDelta(x):
        if type(x) is np.ndarray:
            return (x == 0).astype(float)
        else:
            return float(x==0)

    Fx = sp.diff(F,x)
    Fy = sp.diff(F,y)
    Fz = sp.diff(F,z)

    Fxx = sp.diff(Fx,x)
    Fxy = sp.diff(Fx,y)
    Fxz = sp.diff(Fx,z)

    Fyx = sp.diff(Fy,x)
    Fyy = sp.diff(Fy,y)
    Fyz = sp.diff(Fy,z)

    Fzx = sp.diff(Fz,x)
    Fzy = sp.diff(Fz,y)
    Fzz = sp.diff(Fz,z)

    Grad = sp.Matrix([Fx, Fy, Fz]).T
    Hess = sp.Matrix([[Fxx, Fxy, Fxz],
                    [Fyx, Fyy, Fyz],
                    [Fzx, Fzy, Fzz]
                ])

    Cof = sp.Matrix([[Fyy*Fzz-Fyz*Fzy, Fyz*Fzx-Fyx*Fzz, Fyx*Fzy-Fyy*Fzx],
                    [Fxz*Fzy-Fxy*Fzz, Fxx*Fzz-Fxz*Fzx, Fxy*Fzx-Fxx*Fzy],
                    [Fxy*Fyz-Fxz*Fyy, Fyx*Fxz-Fxx*Fyz, Fxx*Fyy-Fxy*Fyx]
                ])

    grad = sp.lambdify((x,y,z),Grad,['numpy',{'DiracDelta':DiracDelta}])
    hess = sp.lambdify((x,y,z),Hess,['numpy',{'DiracDelta':DiracDelta}])
    cof = sp.lambdify((x,y,z),Cof,['numpy',{'DiracDelta':DiracDelta}])

    g = grad(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2]).swapaxes(0,2)
    h = hess(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2])
    if np.all(h[0][0] == 0):
        h[0][0] = np.zeros(len(NodeCoords))
    if np.all(h[0][1] == 0):
        h[0][1] = np.zeros(len(NodeCoords))
    if np.all(h[0][2] == 0):
        h[0][2] = np.zeros(len(NodeCoords))
    if np.all(h[1][0] == 0):
        h[1][0] = np.zeros(len(NodeCoords))
    if np.all(h[1][1] == 0):
        h[1][1] = np.zeros(len(NodeCoords))
    if np.all(h[1][2] == 0):
        h[1][2] = np.zeros(len(NodeCoords))
    if np.all(h[2][0] == 0):
        h[2][0] = np.zeros(len(NodeCoords))
    if np.all(h[2][1] == 0):
        h[2][1] = np.zeros(len(NodeCoords))
    if np.all(h[2][2] == 0):
        h[2][2] = np.zeros(len(NodeCoords))
    h = np.array(h.tolist()).swapaxes(0,2)
    c = cof(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2])
    if np.all(c[0][0] == 0):
        c[0][0] = np.zeros(len(NodeCoords))
    if np.all(c[0][1] == 0):
        c[0][1] = np.zeros(len(NodeCoords))
    if np.all(c[0][2] == 0):
        c[0][2] = np.zeros(len(NodeCoords))
    if np.all(c[1][0] == 0):
        c[1][0] = np.zeros(len(NodeCoords))
    if np.all(c[1][1] == 0):
        c[1][1] = np.zeros(len(NodeCoords))
    if np.all(c[1][2] == 0):
        c[1][2] = np.zeros(len(NodeCoords))
    if np.all(c[2][0] == 0):
        c[2][0] = np.zeros(len(NodeCoords))
    if np.all(c[2][1] == 0):
        c[2][1] = np.zeros(len(NodeCoords))
    if np.all(c[2][2] == 0):
        c[2][2] = np.zeros(len(NodeCoords))
    c = np.array(c.tolist()).swapaxes(0,2)

    gaussian = np.matmul(np.matmul(g.swapaxes(1,2),c),g)[:,0,0]/(np.linalg.norm(g,axis=1)[:,0]**4)
    mean = -(np.matmul(np.matmul(g.swapaxes(1,2),h),g)[:,0,0] - (np.linalg.norm(g,axis=1)[:,0]**2) * np.trace(h,axis1=1,axis2=2))/(2*np.linalg.norm(g,axis=1)[:,0]**3)

    MaxPrincipal = mean + np.sqrt(np.maximum(mean**2-gaussian,0))
    MinPrincipal = mean - np.sqrt(np.maximum(mean**2-gaussian,0))

    return MaxPrincipal, MinPrincipal, mean, gaussian