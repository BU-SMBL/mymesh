# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:10:08 2021

@author: toj
"""
import numpy as np
import sympy as sp
from . import MeshUtils, converter
import warnings
from scipy import ndimage
  
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

def CubicFit_iterative(NodeCoords,SurfConn,NodeNeighborhoods,NodeNormals,Ignore=set(),IgnoreFeatures=False):
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

def CubicFit(NodeCoords,SurfConn,NodeNeighborhoods,NodeNormals):
    
    # Based on Goldfeather & Interrante (2004)    
    SurfNodes = np.unique(SurfConn)
    
    MaxPrincipal = [np.nan for i in range(len(NodeCoords))]
    MinPrincipal = [np.nan for i in range(len(NodeCoords))]
    ### 
    ArrayCoords = np.append(NodeCoords,[[np.nan,np.nan,np.nan]],axis=0)
    N = np.append(NodeNormals,[[np.nan,np.nan,np.nan]],axis=0)

    RHoods = MeshUtils.PadRagged(NodeNeighborhoods)[SurfNodes]

    SurfCoords = np.append(ArrayCoords[SurfNodes],[[np.nan,np.nan,np.nan]],axis=0)
    SurfNormals = np.append(N[SurfNodes],[[np.nan,np.nan,np.nan]],axis=0)

    # Rotation Axes
    RotAxes = np.repeat([[0,0,1]],len(SurfCoords),axis=0)
    
    Bool = ((SurfNormals[:,0]!=0) | (SurfNormals[:,1]!=0)) & ~np.any(np.isnan(SurfNormals),axis=1)
    Cross = np.cross([0,0,1],-SurfNormals)
    RotAxes = Cross/np.linalg.norm(Cross,axis=1)[:,None]
    RotAxes[np.all(SurfNormals == [0,0,-1],axis=1)] = [1,0,0]
    RotAxes[np.all(SurfNormals == [0,0,1],axis=1)] = [0,0,1]
    # Rotation Angles
    Angles = np.zeros(len(SurfCoords))
    Angles[np.all(SurfNormals == [0,0,-1],axis=1)] = np.pi
    Angles = np.arccos(-1*np.sum([0,0,1]*SurfNormals,axis=1))

    # Quaternions
    Q = np.hstack([np.transpose([np.cos(Angles/2)]), RotAxes*np.sin(Angles/2)[:,None]])
    Zs = np.zeros(len(SurfCoords))
    Os = np.ones(len(SurfCoords))
    R = np.array([[2*(Q[:,0]**2+Q[:,1]**2)-1,   2*(Q[:,1]*Q[:,2]-Q[:,0]*Q[:,3]), 2*(Q[:,1]*Q[:,3]+Q[:,0]*Q[:,2]), Zs],
             [2*(Q[:,1]*Q[:,2]+Q[:,0]*Q[:,3]), 2*(Q[:,0]**2+Q[:,2]**2)-1,   2*(Q[:,2]*Q[:,3]-Q[:,0]*Q[:,1]), Zs],
             [2*(Q[:,1]*Q[:,3]-Q[:,0]*Q[:,2]), 2*(Q[:,2]*Q[:,3]+Q[:,0]*Q[:,1]), 2*(Q[:,0]**2+Q[:,3]**2)-1,   Zs],
             [Zs,                       Zs,                       Zs,                       Os]
             ])

    T = np.array([[Os,Zs,Zs,-SurfCoords[:,0]],
                [Zs,Os,Zs,-SurfCoords[:,1]],
                [Zs,Zs,Os,-SurfCoords[:,2]],
                [Zs,Zs,Zs,Os]])

    SurfNeighborCoords = np.append(ArrayCoords,np.transpose([np.ones(len(ArrayCoords))]),axis=1)[RHoods]
    SurfNeighborNormals = np.append(N,np.transpose([np.ones(len(ArrayCoords))]),axis=1)[RHoods]

    TRCoords = np.matmul(np.matmul(T[:,:,:-1].swapaxes(0,2).swapaxes(1,2),SurfNeighborCoords.swapaxes(1,2)).swapaxes(1,2),R[:,:,:-1].swapaxes(0,2).swapaxes(1,2))
    RNormals = np.matmul(SurfNeighborNormals,R[:,:,:-1].swapaxes(0,2).swapaxes(1,2))

    xjs = TRCoords[:,:,0]
    yjs = TRCoords[:,:,1]
    zjs = TRCoords[:,:,2]

    ajs = RNormals[:,:,0]
    bjs = RNormals[:,:,1]
    cjs = RNormals[:,:,2]

    scales = 2/(xjs**2+yjs**2) 

    nNeighbors = RHoods.shape[1]

    Amat = np.zeros((len(SurfNodes),nNeighbors*3,7))
    Amat[:,:nNeighbors] = np.array([scales/2*xjs**2, scales*xjs*yjs, scales/2*yjs**2, scales*xjs**3, scales*xjs**2*yjs, scales*xjs*yjs**2, scales*yjs**3]).T.swapaxes(0,1)
    Amat[:,nNeighbors:2*nNeighbors] = np.array([scales*xjs, scales*yjs, np.zeros(scales.shape), scales*3*xjs**2, scales*2*xjs*yjs, scales*yjs**2, np.zeros(scales.shape)]).T.swapaxes(0,1)
    Amat[:,2*nNeighbors:3*nNeighbors] = np.array([np.zeros(scales.shape), scales*xjs, scales*yjs, np.zeros(scales.shape), scales*xjs**2, scales*2*xjs*yjs, scales*3*yjs**2]).T.swapaxes(0,1)

    Bmat = np.zeros((len(SurfNodes),nNeighbors*3,1))
    Bmat[:,:nNeighbors,0] = scales*zjs
    Bmat[:,nNeighbors:2*nNeighbors,0] = -scales*ajs/cjs
    Bmat[:,2*nNeighbors:3*nNeighbors,0] = -scales*bjs/cjs

    MaxPrincipal = np.repeat(np.nan,len(NodeCoords))
    MinPrincipal = np.repeat(np.nan,len(NodeCoords))
    for i,idx in enumerate(SurfNodes):
        amat = Amat[i,~np.any(np.isnan(Amat[i]),axis=1) & ~np.any(np.isnan(Bmat[i]),axis=1)]
        bmat = Bmat[i,~np.any(np.isnan(Amat[i]),axis=1) & ~np.any(np.isnan(Bmat[i]),axis=1)]
        A = np.matmul(amat.T,amat)
        if np.linalg.det(A) == 0:
            MaxPrincipal[idx] = np.nan
            MinPrincipal[idx] = np.nan
        else:
            B = np.matmul(amat.T,bmat)
            X = np.linalg.solve(A,B).T[0]
            W = np.array([[X[0],X[1]],
                            [X[1],X[2]]])
            if np.any(np.isnan(W)):
                v = [np.nan, np.nan]
            else:
                [v,x] = np.linalg.eig(W)
            MaxPrincipal[idx] = max(v)
            MinPrincipal[idx] = min(v)
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
    h = hess(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2]).tolist()
    if not hasattr(h[0][0], "__len__"):
        h[0][0] = np.repeat(h[0][0],len(NodeCoords)).tolist()
    if not hasattr(h[0][1], "__len__"):
        h[0][1] = np.repeat(h[0][1],len(NodeCoords)).tolist()
    if not hasattr(h[0][2], "__len__"):
        h[0][2] = np.repeat(h[0][2],len(NodeCoords)).tolist()
    if not hasattr(h[1][0], "__len__"):
        h[1][0] = np.repeat(h[1][0],len(NodeCoords)).tolist()
    if not hasattr(h[1][1], "__len__"):
        h[1][1] = np.repeat(h[1][1],len(NodeCoords)).tolist()
    if not hasattr(h[1][2], "__len__"):
        h[1][2] = np.repeat(h[1][2],len(NodeCoords)).tolist()
    if not hasattr(h[2][0], "__len__"):
        h[2][0] = np.repeat(h[2][0],len(NodeCoords)).tolist()
    if not hasattr(h[2][1], "__len__"):
        h[2][1] = np.repeat(h[2][1],len(NodeCoords)).tolist()
    if not hasattr(h[2][2], "__len__"):
        h[2][2] = np.repeat(h[2][2],len(NodeCoords)).tolist()
    h = np.array(h).swapaxes(0,2)
    c = cof(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2]).tolist()
    if not hasattr(c[0][0], "__len__"):
        c[0][0] = np.repeat(c[0][0],len(NodeCoords)).tolist()
    if not hasattr(c[0][1], "__len__"):
        c[0][1] = np.repeat(c[0][1],len(NodeCoords)).tolist()
    if not hasattr(c[0][2], "__len__"):
        c[0][2] = np.repeat(c[0][2],len(NodeCoords)).tolist()
    if not hasattr(c[1][0], "__len__"):
        c[1][0] = np.repeat(c[1][0],len(NodeCoords)).tolist()
    if not hasattr(c[1][1], "__len__"):
        c[1][1] = np.repeat(c[1][1],len(NodeCoords)).tolist()
    if not hasattr(c[1][2], "__len__"):
        c[1][2] = np.repeat(c[1][2],len(NodeCoords)).tolist()
    if not hasattr(c[2][0], "__len__"):
        c[2][0] = np.repeat(c[2][0],len(NodeCoords)).tolist()
    if not hasattr(c[2][1], "__len__"):
        c[2][1] = np.repeat(c[2][1],len(NodeCoords)).tolist()
    if not hasattr(c[2][2], "__len__"):
        c[2][2] = np.repeat(c[2][2],len(NodeCoords)).tolist()
    c = np.array(c).swapaxes(0,2)

    gaussian = np.matmul(np.matmul(g.swapaxes(1,2),c),g)[:,0,0]/(np.linalg.norm(g,axis=1)[:,0]**4)
    mean = -(np.matmul(np.matmul(g.swapaxes(1,2),h),g)[:,0,0] - (np.linalg.norm(g,axis=1)[:,0]**2) * np.trace(h,axis1=1,axis2=2))/(2*np.linalg.norm(g,axis=1)[:,0]**3)

    MaxPrincipal = mean + np.sqrt(np.maximum(mean**2-gaussian,0))
    MinPrincipal = mean - np.sqrt(np.maximum(mean**2-gaussian,0))

    return MaxPrincipal, MinPrincipal, mean, gaussian

def ImageCurvature(I,gaussian_sigma=1,voxelsize=1,brightobject=True):
    # Note: Can lead to errors if surface is too close to the boundary of the image, consider building in padding based on gaussian_sigma
    # If the 'inside' of the imaged object is darker than background, the signs of the curvatures will be flipped, in this case use brightobject=False
    
    if not brightobject:
        I = -np.array(I)
    
    Fx = ndimage.gaussian_filter(I,gaussian_sigma,order=(1,0,0))
    Fy = ndimage.gaussian_filter(I,gaussian_sigma,order=(0,1,0))
    Fz = ndimage.gaussian_filter(I,gaussian_sigma,order=(0,0,1))

    Fxx = ndimage.gaussian_filter(Fx,gaussian_sigma,order=(1,0,0))
    Fxy = ndimage.gaussian_filter(Fx,gaussian_sigma,order=(0,1,0))
    Fxz = ndimage.gaussian_filter(Fx,gaussian_sigma,order=(0,0,1))
    
    Fyx = ndimage.gaussian_filter(Fy,gaussian_sigma,order=(1,0,0))
    Fyy = ndimage.gaussian_filter(Fy,gaussian_sigma,order=(0,1,0))
    Fyz = ndimage.gaussian_filter(Fy,gaussian_sigma,order=(0,0,1))
    
    Fzx = ndimage.gaussian_filter(Fz,gaussian_sigma,order=(1,0,0))
    Fzy = ndimage.gaussian_filter(Fz,gaussian_sigma,order=(0,1,0))
    Fzz = ndimage.gaussian_filter(Fz,gaussian_sigma,order=(0,0,1))


    Grad = np.transpose(np.array([Fx, Fy, Fz])[None,:,:,:,:],(2,3,4,0,1))
    Hess = np.transpose(np.array([[Fxx, Fxy, Fxz],
                    [Fyx, Fyy, Fyz],
                    [Fzx, Fzy, Fzz]
                ]),(2,3,4,0,1))

    Cof = np.transpose(np.array([[Fyy*Fzz-Fyz*Fzy, Fyz*Fzx-Fyx*Fzz, Fyx*Fzy-Fyy*Fzx],
                    [Fxz*Fzy-Fxy*Fzz, Fxx*Fzz-Fxz*Fzx, Fxy*Fzx-Fxx*Fzy],
                    [Fxy*Fyz-Fxz*Fyy, Fyx*Fxz-Fxx*Fyz, Fxx*Fyy-Fxy*Fyx]
                ]),(2,3,4,0,1))
    with np.errstate(divide='ignore', invalid='ignore'):
        gaussian = np.matmul(np.matmul(Grad,Cof),np.transpose(Grad,(0,1,2,4,3))).reshape(I.shape)/np.linalg.norm(Grad,axis=4).reshape(I.shape)**4
        mean = (np.matmul(np.matmul(Grad,Hess),np.transpose(Grad,(0,1,2,4,3))).reshape(I.shape)-np.linalg.norm(Grad,axis=4).reshape(I.shape)**2 * np.trace(Hess,axis1=3,axis2=4))/(2*np.linalg.norm(Grad,axis=4).reshape(I.shape)**3)

    gaussian = gaussian/voxelsize**2    
    mean = mean/voxelsize
    
    MaxPrincipal = mean + np.sqrt(np.maximum(mean**2-gaussian,0))
    MinPrincipal = mean - np.sqrt(np.maximum(mean**2-gaussian,0))

    return MaxPrincipal, MinPrincipal, mean, gaussian