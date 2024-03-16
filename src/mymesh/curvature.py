# -*- coding: utf-8 -*-
"""
Tools for calculating curvature

Created on Wed Sep 29 14:10:08 2021

@author: toj

.. currentmodule:: Mesh.curvature

Curvature Calculation
=====================
.. autosummary::
    :toctree: submodules/

    NormCurve
    QuadFit
    CubicFit
    AnalyticalCurvature
    ImageCurvature

Curvature Conversion
====================
.. autosummary::
    :toctree: submodules/

    MeanCurvature
    GaussianCurvature
    Curvedness
    ShapeIndex
    ShapeCategory

"""
import numpy as np
from . import utils, converter
import warnings
from scipy import ndimage, interpolate
  
def NormCurve(NodeCoords,SurfConn,NodeNeighbors,NodeNormals):
    """
    Mesh based curvatures by normal curvature approximation. Curvatures 
    calculated in this way are sensitive to triangulation, with highly 
    skewed triangles contributing error. In general, CubicFit provides 
    better results.
    From Goldfeather & Interrante (2004).
    :cite:p:`Goldfeather2004`

    Parameters
    ----------
    NodeCoords : list, np.ndarray
        List of nodal coordinates.
    SurfConn : list, np.ndarray
        Nodal connectivity list for a surface mesh.
    NodeNeighbors : list
        List of neighbors for each node. This can be obtained
        with utils.getNodeNeighbors or mesh.NodeNeighbors.
    NodeNormals : list, np.ndarray
        Unit normal vectors for each node in a nx3 array. This can be obtained
        with utils.CalcFaceNormal and utils.Face2NodeNormal, or mesh.NodeNormals.

    Returns
    -------
    MaxPrincipal : list
        List of maximum principal curvatures for each node.
    MinPrincipal : list
        List of minimum principal curvatures for each node.
    """  
    
    
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
    """
    Mesh based curvatures by quadratic surface fitting. Curvatures calculated
    in this way are sensitive to triangulation, with highly skewed triangles
    contributing error. In general, CubicFit provides better results.
    From Goldfeather & Interrante (2004).
    :cite:p:`Goldfeather2004`

    Parameters
    ----------
    NodeCoords : list, np.ndarray
        List of nodal coordinates.
    SurfConn : list, np.ndarray
        Nodal connectivity list for a surface mesh.
    NodeNeighbors : list
        List of neighbors for each node. This can be obtained
        with utils.getNodeNeighbors or mesh.NodeNeighbors.
    NodeNormals : list, np.ndarray
        Unit normal vectors for each node in a nx3 array. This can be obtained
        with utils.CalcFaceNormal and utils.Face2NodeNormal, or mesh.NodeNormals.

    Returns
    -------
    MaxPrincipal : list
        List of maximum principal curvatures for each node.
    MinPrincipal : list
        List of minimum principal curvatures for each node.
    """  
    
    
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

def CubicFit(NodeCoords,SurfConn,NodeNeighborhoods,NodeNormals):
    """
    Mesh based curvatures by cubic surface fitting. Curvatures calculated
    in this way are sensitive to triangulation, with highly skewed triangles
    contributing error.
    From Goldfeather & Interrante (2004).
    :cite:p:`Goldfeather2004`

    Parameters
    ----------
    NodeCoords : list, np.ndarray
        List of nodal coordinates.
    SurfConn : list, np.ndarray
        Nodal connectivity list for a surface mesh.
    NodeNeighbors : list
        List of neighbors for each node. This can be obtained
        with utils.getNodeNeighbors or mesh.NodeNeighbors.
    NodeNormals : list, np.ndarray
        Unit normal vectors for each node in a nx3 array. This can be obtained
        with utils.CalcFaceNormal and utils.Face2NodeNormal, or mesh.NodeNormals.

    Returns
    -------
    MaxPrincipal : list
        List of maximum principal curvatures for each node.
    MinPrincipal : list
        List of minimum principal curvatures for each node.
    """    

    SurfNodes = np.unique(SurfConn)
    
    MaxPrincipal = [np.nan for i in range(len(NodeCoords))]
    MinPrincipal = [np.nan for i in range(len(NodeCoords))]
    ### 
    ArrayCoords = np.append(NodeCoords,[[np.nan,np.nan,np.nan]],axis=0)
    N = np.append(NodeNormals,[[np.nan,np.nan,np.nan]],axis=0)

    RHoods = utils.PadRagged(NodeNeighborhoods)[SurfNodes]

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

def AnalyticalCurvature(F,NodeCoords):
    """
    Calculate curvature of an implicit function. Curvature is sampled at the provided list of node coordinates, which should lie on the surface of an isosurface of the function.
    Based on Curvature formulas for implicit curves and surfaces, Ron Goldman (2005). 
    :cite:p:`Goldman2005`

    Parameters
    ----------
    F : function
        Sympy symbolic function of three arguments (x,y,z)
    NodeCoords : list, np.ndarray
        List of node coordinates for evaluating the curvature.

    Returns
    -------
    MaxPrincipal : np.ndarray
        List of maximum principal curvatures for each node.
    MinPrincipal : np.ndarray
        List of minimum principal curvatures for each node.
    gaussian : np.ndarray
        List of Gaussian curvatures for each node.
    mean : np.ndarray
        List of mean curvatures.
    """
    try:
        import sympy as sp
    except:
        raise ImportError('AnalyticalCurvature requires sympy. Install with: pip install sympy.')
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

    if all([any([g.has(var) for g in Grad]) for var in [x,y,z]]):
        g = grad(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2]).swapaxes(0,2)
    else:
        g = np.array([grad(NodeCoords[i,0],NodeCoords[i,1],NodeCoords[i,2]) for i in range(len(NodeCoords))]).swapaxes(1,2)

    if all([any([h.has(var) for h in Hess]) for var in [x,y,z]]):
        h = hess(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2]).tolist()
    else:
        h = np.array([hess(NodeCoords[i,0],NodeCoords[i,1],NodeCoords[i,2]) for i in range(len(NodeCoords))]).T
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

    if all([any([c.has(var) for c in Cof]) for var in [x,y,z]]):
        c = cof(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2]).tolist()
    else:
        c = np.array([cof(NodeCoords[i,0],NodeCoords[i,1],NodeCoords[i,2]) for i in range(len(NodeCoords))]).T
    
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

def ImageCurvature(I,NodeCoords=None,gaussian_sigma=1,voxelsize=1,brightobject=True):
    """
    Calculate curvatures based on a 3D image. Curvature values are calculated for all voxels in the image,
    however, these curvature values are only meaningful at the surface of the imaged object(s). This can be used with 
    utils.grid2fun, converter.im2voxel, contours.MarchingCubesImage to evaluate the voxel curvatures at points on the surface
    of the imaged object(s).
    Based on Curvature formulas for implicit curves and surfaces, Ron Goldman (2005).
    :cite:p:`Goldman2005`

    NOTE: This can lead to errors if surface is too close to the boundary of the image, consider building in padding based on gaussian_sigma

    Parameters
    ----------
    I : np.ndarray
        3D array of grayscale voxel data.
    NodeCoords : array_like
        If provided, curvature from the grid will be evaluated at these points 
        and returned. If not, the returned values will be of the full image
    gaussian_sigma : int, optional
        Standard deviation used in calculating image gradients (in voxels), by default 1. 
        See scipy.ndimage.gaussian_filter.
    voxelsize : int, float, optional
        Voxel size of the image, by default 1.
        This is necessary to determine the correct magnitudes of 
        curvature. By default, the image coordinate system will 
        be used, so principal curvatures will be in units of
        (voxel)^-1
    brightobject : bool, optional
        Specifies whether the foreground of the image is bright or dark, by default True.
        If the imaged object is darker than the background, set brightobject=False. This 
        is important for determining the directionality of curvatures.

    Returns
    -------
    MaxPrincipal : np.ndarray
        Maximum principal curvatures for either each voxel or each node
        (if NodeCoords is provided).
    MinPrincipal : np.ndarray
        Minimum principal curvatures for either each voxel or each node
        (if NodeCoords is provided)
    gaussian : np.ndarray
        Gaussian curvatures for either each voxel or each node
        (if NodeCoords is provided)
    mean : np.ndarray
        Mean curvatures for either each voxel or each node
        (if NodeCoords is provided)
    """ 
    
    I = I.astype(float)

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

    if NodeCoords is not None:
        NodeCoords = np.asarray(NodeCoords)
        X = np.arange(I.shape[2])*voxelsize 
        Y = np.arange(I.shape[1])*voxelsize
        Z = np.arange(I.shape[0])*voxelsize
        
        points = (X,Y,Z)

        mean_fun = lambda x,y,z : interpolate.RegularGridInterpolator(points,mean.T,method='linear',bounds_error=False,fill_value=None)(np.vstack([x,y,z]).T)
        gauss_fun = lambda x,y,z : interpolate.RegularGridInterpolator(points,gaussian.T,method='linear',bounds_error=False,fill_value=None)(np.vstack([x,y,z]).T)

        mean = mean_fun(NodeCoords[:,0], NodeCoords[:,1], NodeCoords[:,2])
        gaussian = gauss_fun(NodeCoords[:,0], NodeCoords[:,1], NodeCoords[:,2])

    MaxPrincipal = mean + np.sqrt(np.maximum(mean**2-gaussian,0))
    MinPrincipal = mean - np.sqrt(np.maximum(mean**2-gaussian,0))

    return MaxPrincipal, MinPrincipal, mean, gaussian   

def MeanCurvature(MaxPrincipal,MinPrincipal):
    """
    Calculate the mean curvature from the maximum and 
    minimum principal curvatures.

    Parameters
    ----------
    MaxPrincipal : list, float
        Single value or list of maximum principal curvatures.
    MinPrincipal : list, float
        Single value or list of principal curvatures.

    Returns
    -------
    mean : list, float
        Single value or list of mean curvatures.
    """    
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
    """
    Calculate the Gaussian curvature from the maximum and 
    minimum principal curvatures.

    Parameters
    ----------
    MaxPrincipal : list, float
        Single value or list of maximum principal curvatures.
    MinPrincipal : list, float
        Single value or list of principal curvatures.

    Returns
    -------
    gaussian : list, float
        Single value or list of Gaussian curvatures.
    """  
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
    """
    Calculate the curvedness from the maximum and minimum principal curvatures.
    From Koenderink, J.J. and Van Doorn, A.J., 1992.
    :cite:p:`Koenderink1992a`

    Parameters
    ----------
    MaxPrincipal : list, float
        Single value or list of maximum principal curvatures.
    MinPrincipal : list, float
        Single value or list of principal curvatures.

    Returns
    -------
    curvedness
        Single value or list of curvedness values.
    """

    curvedness = np.sqrt((np.asarray(MaxPrincipal)**2 + np.asarray(MinPrincipal)**2)/2)
    if len(curvedness) == 1 and type(MaxPrincipal) is int or type(MaxPrincipal) is float:
        curvedness = curvedness[0]
        
    
    return curvedness
 
def ShapeIndex(MaxPrincipal,MinPrincipal):
    """
    Calculate shape indices from the maximum and minimum principal curvatures.
    From Koenderink, J.J. and Van Doorn, A.J., 1992.
    :cite:p:`Koenderink1992a`

    Parameters
    ----------
    MaxPrincipal : list, float
        Single value or list of maximum principal curvatures.
    MinPrincipal : list, float
        Single value or list of principal curvatures.

    Returns
    -------
    shape : list, float
        Single value or list of shape indices.
    """
    # Note: the equation from Koenderink & van Doorn has the equation: pi/2*arctan((min+max)/(min-max)), but this doesn't
    # seem to give values consistent with what are described as cups/caps - instead using pi/2*arctan((max+min)/(max-min))
    
    MaxPrincipal = np.asarray(MaxPrincipal) 
    MinPrincipal = np.asarray(MinPrincipal) 
    with np.errstate(divide='ignore', invalid='ignore'):
        shape = (2/np.pi) * np.arctan((MaxPrincipal + MinPrincipal)/(MaxPrincipal - MinPrincipal))
    shape[MaxPrincipal == MinPrincipal] = 1*np.sign(MaxPrincipal[MaxPrincipal == MinPrincipal])
    
    if len(shape) == 1 and type(MaxPrincipal) is int or type(MaxPrincipal) is float:
        shape = shape[0]
    
    return shape

def ShapeCategory(shapeindex):
    """
    Categorize shape indices into a nine point scale.

    0 = Spherical Cup

    1 = Trough

    2 = Rut

    3 = Saddle Rut

    4 = Saddle

    5 = Saddle Ridge

    6 = Ridge

    7 = Dome

    8 = Spherical Cap
    
    From Koenderink, J.J. and Van Doorn, A.J., 1992.
    :cite:p:`Koenderink1992a`

    Parameters
    ----------
    shapeindex : list
        List of shape indices.

    Returns
    -------
    shape : list
        List of shape categories.
    """   
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
                       