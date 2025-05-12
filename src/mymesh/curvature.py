# -*- coding: utf-8 -*-
# Created on Wed Sep 29 14:10:08 2021
# @author: toj
"""
Tools for calculating curvature

See also :ref:`Curvature` in the :ref:`Theory Guide`.


.. currentmodule:: mymesh.curvature

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
from . import utils, converter, check_numba
from . import try_njit
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
        if np.array_equal(n,k):
            rotAxis = k
            angle = 0
        elif np.array_equal(n, [-1*i for i in k]):
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
    
    # Get nodes to evaluate curvature
    SurfNodes = np.unique(SurfConn)
    
    # Pad node neighborhoods to be a rectangular array
    RHoods = utils.PadRagged(NodeNeighbors, fillval=-1)[SurfNodes]
    
    # Pad arrays for indexing with padded ragged node neighbors array
    ArrayCoords = np.append(NodeCoords,[[np.nan,np.nan,np.nan]],axis=0)
    N = np.append(NodeNormals,[[np.nan,np.nan,np.nan]],axis=0)
    SurfCoords = np.append(ArrayCoords[SurfNodes],[[np.nan,np.nan,np.nan]],axis=0)
    SurfNormals = np.append(N[SurfNodes],[[np.nan,np.nan,np.nan]],axis=0)
    
    SurfNeighborCoords = np.append(ArrayCoords,np.transpose([np.ones(len(ArrayCoords))]),axis=1)[RHoods]

    # Rotate to align the surface normals to [0,0,1]
    TargetAxis = np.array([0,0,-1])
    Bool = ((SurfNormals[:,0]!=0) | (SurfNormals[:,1]!=0)) & ~np.any(np.isnan(SurfNormals),axis=1)
    Cross = np.cross(TargetAxis,SurfNormals) 
    CrossNorm = np.linalg.norm(Cross,axis=1)
    NonZeroCross = CrossNorm != 0
    RotAxes = np.nan*np.ones_like(Cross)
    RotAxes[NonZeroCross] = Cross[NonZeroCross]/CrossNorm[NonZeroCross,None]
    RotAxes[np.all(SurfNormals == -TargetAxis,axis=1)] = [1,0,0]
    RotAxes[np.all(SurfNormals == TargetAxis,axis=1)] = TargetAxis
    # Rotation Angles
    Angles = np.zeros(len(SurfCoords))
    Angles[np.all(SurfNormals == -TargetAxis,axis=1)] = np.pi
    Angles = np.arccos(np.sum(TargetAxis*SurfNormals,axis=1))

    # Quaternion
    Q = np.hstack([np.transpose([np.cos(Angles/2)]), RotAxes*np.sin(Angles/2)[:,None]])[:-1]
    # Quaternion to rotation matrix
    R = np.zeros((len(SurfNodes),4,4))
    R[:,0,0] = 2*(Q[:,0]**2+Q[:,1]**2)-1
    R[:,0,1] = 2*(Q[:,1]*Q[:,2]-Q[:,0]*Q[:,3])
    R[:,0,2] = 2*(Q[:,1]*Q[:,3]+Q[:,0]*Q[:,2])
    R[:,1,0] = 2*(Q[:,1]*Q[:,2]+Q[:,0]*Q[:,3])
    R[:,1,1] = 2*(Q[:,0]**2+Q[:,2]**2)-1
    R[:,1,2] = 2*(Q[:,2]*Q[:,3]-Q[:,0]*Q[:,1])
    R[:,2,0] = 2*(Q[:,1]*Q[:,3]-Q[:,0]*Q[:,2])
    R[:,2,1] = 2*(Q[:,2]*Q[:,3]+Q[:,0]*Q[:,1])
    R[:,2,2] = 2*(Q[:,0]**2+Q[:,3]**2)-1
    R[:,3,3] = 1

    # Translation matrix
    T = np.repeat([np.eye(4)], len(SurfNodes), axis=0)
    T[:,0,3] = -SurfCoords[:-1,0]
    T[:,1,3] = -SurfCoords[:-1,1]
    T[:,2,3] = -SurfCoords[:-1,2]

    TRCoords = np.matmul(np.matmul(T,SurfNeighborCoords.swapaxes(1,2)).swapaxes(1,2),R)

    xjs = TRCoords[:,:,0]
    yjs = TRCoords[:,:,1]
    zjs = TRCoords[:,:,2]

    nNeighbors = RHoods.shape[1]
    
    Amat = np.array([1/2*xjs**2, xjs*yjs, 1/2*yjs**2]).transpose(1,2,0)
    Bmat = zjs[:,:,None]

    MaxPrincipal = np.repeat(np.nan,len(NodeCoords))
    MinPrincipal = np.repeat(np.nan,len(NodeCoords))
    for i,idx in enumerate(SurfNodes):
        amat = Amat[i,~np.any(np.isnan(Amat[i]),axis=1) & ~np.any(np.isnan(Bmat[i]),axis=1)]
        bmat = Bmat[i,~np.any(np.isnan(Amat[i]),axis=1) & ~np.any(np.isnan(Bmat[i]),axis=1)]
        A = np.matmul(amat.T,amat)
        if np.linalg.det(A) != 0:
            B = np.matmul(amat.T,bmat)
            X = np.linalg.solve(A,B).T[0]
            W = np.array([[X[0],X[1]],
                            [X[1],X[2]]])
            if np.any(np.isnan(W)):
                MaxPrincipal[idx] = np.nan
                MinPrincipal[idx] = np.nan
            else:
                [v,x] = np.linalg.eig(W)
                MaxPrincipal[idx] = max(v)
                MinPrincipal[idx] = min(v)
    return MaxPrincipal,MinPrincipal

def CubicFit(NodeCoords,SurfConn,NodeNeighborhoods,NodeNormals,jit=True,return_directions=False):
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
    jit : bool
        If True, will attempt to use numba jit compilation for curvature 
        calculation. In general, the jit version should perform noticeably faster 
        than the alternative vectorized approach, which encounters overhead due
        to the non-rectangular nature of the neighborhoods list. Numba must
        be installed for this option to work, and mymesh.check_numba() should
        return True. 
    return_directions : bool, optional
        If True, will return the principal curvature directions as well as 
        magnitudes
    Returns
    -------
    MaxPrincipal : list
        List of maximum principal curvatures for each node.
    MinPrincipal : list
        List of minimum principal curvatures for each node.
    MaxPrincipalDirection : np.ndarray, optional
        Unit vector associated with the maximum principal curvature. Returned if
        return_directions is True.
    MinPrincipalDirection : np.ndarray, optional
        Unit vector associated with the minimum principal curvature. Returned if
        return_directions is True.
    """    
     # Get nodes to evaluate curvature
    SurfNodes = np.array(list({i for elem in SurfConn for i in elem}))
    if jit and check_numba():
        NodeCoords = np.asarray(NodeCoords, dtype=np.float64)
        NodeNormals = np.asarray(NodeNormals, dtype=np.float64)
        MaxPrincipal = np.repeat(np.nan,len(NodeCoords))
        MinPrincipal = np.repeat(np.nan,len(NodeCoords))
        if return_directions:
            MaxPrincipalDirection = np.full_like(NodeCoords, np.nan)
            MinPrincipalDirection = np.full_like(NodeCoords, np.nan)
        for i in SurfNodes:
            neighborhood = np.append([i],NodeNeighborhoods[i]).astype(np.int64)
            normals = NodeNormals[neighborhood]
            if return_directions:
                MaxPrincipal[i], MinPrincipal[i], MaxPrincipalDirection[i], MinPrincipalDirection[i] = _CubicFit(NodeCoords, neighborhood, normals, True)
            else:
                MaxPrincipal[i], MinPrincipal[i], _, _ = _CubicFit(NodeCoords, neighborhood, normals, False)
    else:
        # Pad node neighborhoods to be a rectangular array
        if len(SurfNodes) != len(NodeCoords):
            RHoods = utils.PadRagged([NodeNeighborhoods[i] for i in SurfNodes], fillval=-1)
        else:
            RHoods = utils.PadRagged(NodeNeighborhoods, fillval=-1)
        # Pad arrays for indexing with padded ragged node neighbors array
        ArrayCoords = np.append(NodeCoords,[[np.nan,np.nan,np.nan]],axis=0)
        N = np.append(NodeNormals,[[np.nan,np.nan,np.nan]],axis=0)
        SurfCoords = np.append(ArrayCoords[SurfNodes],[[np.nan,np.nan,np.nan]],axis=0)
        SurfNormals = np.append(N[SurfNodes],[[np.nan,np.nan,np.nan]],axis=0)
        
        SurfNeighborCoords = np.append(ArrayCoords,np.transpose([np.ones(len(ArrayCoords))]),axis=1)[RHoods]
        SurfNeighborNormals = np.append(N,np.transpose([np.ones(len(ArrayCoords))]),axis=1)[RHoods]

        # Rotate to align the surface normals to [0,0,1]
        TargetAxis = np.array([0,0,-1])
        Bool = ((SurfNormals[:,0]!=0) | (SurfNormals[:,1]!=0)) & ~np.any(np.isnan(SurfNormals),axis=1)
        Cross = np.cross(TargetAxis,SurfNormals) 
        CrossNorm = np.linalg.norm(Cross,axis=1)
        NonZeroCross = CrossNorm != 0
        RotAxes = np.nan*np.ones_like(Cross)
        RotAxes[NonZeroCross] = Cross[NonZeroCross]/CrossNorm[NonZeroCross,None]
        RotAxes[np.all(SurfNormals == -TargetAxis,axis=1)] = [1,0,0]
        RotAxes[np.all(SurfNormals == TargetAxis,axis=1)] = TargetAxis
        # Rotation Angles
        Angles = np.zeros(len(SurfCoords))
        Angles[np.all(SurfNormals == -TargetAxis,axis=1)] = np.pi
        Angles = np.arccos(np.sum(TargetAxis*SurfNormals,axis=1))

        # Quaternion
        Q = np.hstack([np.transpose([np.cos(Angles/2)]), RotAxes*np.sin(Angles/2)[:,None]])[:-1]
        # Quaternion to rotation matrix
        R = np.zeros((len(SurfNodes),4,4))
        R[:,0,0] = 2*(Q[:,0]**2+Q[:,1]**2)-1
        R[:,0,1] = 2*(Q[:,1]*Q[:,2]-Q[:,0]*Q[:,3])
        R[:,0,2] = 2*(Q[:,1]*Q[:,3]+Q[:,0]*Q[:,2])
        R[:,1,0] = 2*(Q[:,1]*Q[:,2]+Q[:,0]*Q[:,3])
        R[:,1,1] = 2*(Q[:,0]**2+Q[:,2]**2)-1
        R[:,1,2] = 2*(Q[:,2]*Q[:,3]-Q[:,0]*Q[:,1])
        R[:,2,0] = 2*(Q[:,1]*Q[:,3]-Q[:,0]*Q[:,2])
        R[:,2,1] = 2*(Q[:,2]*Q[:,3]+Q[:,0]*Q[:,1])
        R[:,2,2] = 2*(Q[:,0]**2+Q[:,3]**2)-1
        R[:,3,3] = 1

        # Translation matrix
        T = np.repeat([np.eye(4)], len(SurfNodes), axis=0)
        T[:,0,3] = -SurfCoords[:-1,0]
        T[:,1,3] = -SurfCoords[:-1,1]
        T[:,2,3] = -SurfCoords[:-1,2]

        TRCoords = np.matmul(np.matmul(T,SurfNeighborCoords.swapaxes(1,2)).swapaxes(1,2),R)
        RNormals = np.matmul(SurfNeighborNormals,R)

        xjs = TRCoords[:,:,0]
        yjs = TRCoords[:,:,1]
        zjs = TRCoords[:,:,2]

        ajs = RNormals[:,:,0]
        bjs = RNormals[:,:,1]
        cjs = RNormals[:,:,2]

        nNeighbors = RHoods.shape[1]

        Amat = np.zeros((len(SurfNodes),nNeighbors*3,7))
        Amat[:,:nNeighbors] = np.array([1/2*xjs**2, xjs*yjs, 1/2*yjs**2, xjs**3, xjs**2*yjs, xjs*yjs**2, yjs**3]).transpose(1,2,0)
        Amat[:,nNeighbors:2*nNeighbors] = np.array([xjs, yjs, np.zeros(xjs.shape), 3*xjs**2, 2*xjs*yjs, yjs**2, np.zeros(xjs.shape)]).transpose(1,2,0)
        Amat[:,2*nNeighbors:3*nNeighbors] = np.array([np.zeros(xjs.shape), xjs, yjs, np.zeros(xjs.shape), xjs**2, 2*xjs*yjs, 3*yjs**2]).transpose(1,2,0)

        Bmat = np.zeros((len(SurfNodes),nNeighbors*3,1))
        Bmat[:,:nNeighbors,0] = zjs
        Bmat[:,nNeighbors:2*nNeighbors,0] = -ajs/cjs
        Bmat[:,2*nNeighbors:3*nNeighbors,0] = -bjs/cjs

        MaxPrincipal = np.repeat(np.nan,len(NodeCoords))
        MinPrincipal = np.repeat(np.nan,len(NodeCoords))
        MaxPrincipalDirection = np.full((len(NodeCoords),3), np.nan)
        MinPrincipalDirection = np.full((len(NodeCoords),3), np.nan)
        for i,idx in enumerate(SurfNodes):
            amat = Amat[i,~np.any(np.isnan(Amat[i]),axis=1) & ~np.any(np.isnan(Bmat[i]),axis=1)]
            bmat = Bmat[i,~np.any(np.isnan(Amat[i]),axis=1) & ~np.any(np.isnan(Bmat[i]),axis=1)]
            A = np.matmul(amat.T,amat)
            if np.linalg.det(A) != 0:
                B = np.matmul(amat.T,bmat)
                X = np.linalg.solve(A,B).T[0]
                W = np.array([[X[0],X[1]],
                                [X[1],X[2]]])
                if np.any(np.isnan(W)):
                    MaxPrincipal[idx] = np.nan
                    MinPrincipal[idx] = np.nan
                else:
                    [v,x] = np.linalg.eig(W)
                    MaxPrincipal[idx] = max(v)
                    MinPrincipal[idx] = min(v)     
                    if return_directions:
                        MaxPrincipalDirection[idx] = np.matmul(np.append(x[:,np.argmax(v)],0), np.linalg.inv(R[i,:3,:3]))
                        MinPrincipalDirection[idx] = np.matmul(np.append(x[:,np.argmin(v)],0), np.linalg.inv(R[i,:3,:3]))

    if return_directions:
        return MaxPrincipal, MinPrincipal, MaxPrincipalDirection, MinPrincipalDirection
    return MaxPrincipal,MinPrincipal

@try_njit#(cache=True)
def _CubicFit(NodeCoords, neighborhood, normals, return_directions=False):
    """
    Calculate cubic fit curvatures for a single node neighborhood. This is for 
    internal use by :func:`CubicFit` only.

    Parameters
    ----------
    NodeCoords : np.ndarray(dtype=np.float64)
        Array of all node coordinates in the mesh
    neighborhood : np.ndarray(dtype=np.int64)
        Indices of nodes in the neighborhood, including the central node as the
        first entry in the array.
    normals : np.ndarray(dtype=np.float64)
        Node normal vectors of the nodes in the neighborhood.
    return_directions : bool, optional
        If True, will return the principal curvature directions as well as 
        magnitudes

    Returns
    -------
    MaxPrincipal : np.float64
        Maximum principal curvature for the first node specified in the neighborhood
    MinPrincipal : np.float64
        Minimum principal curvature for the first node specified in the neighborhood
    MaxPrincipalDirection : np.ndarray, optional
        Unit vector associated with the maximum principal curvature. Returned if
        return_directions is True.
    MinPrincipalDirection : np.ndarray, optional
        Unit vector associated with the minimum principal curvature. Returned if
        return_directions is True.
    """

    LocalCoords, R = utils.RotateNormalToVector(NodeCoords[neighborhood], normals[0], np.array([0.,0.,-1.]))

    LocalNormals = (R @ normals.T).T

    xjs = LocalCoords[1:,0] - LocalCoords[0,0]
    yjs = LocalCoords[1:,1] - LocalCoords[0,1]
    zjs = LocalCoords[1:,2] - LocalCoords[0,2]

    ajs = LocalNormals[1:,0]
    bjs = LocalNormals[1:,1]
    cjs = LocalNormals[1:,2]

    nNeighbors = len(neighborhood) - 1

    Amat = np.zeros((nNeighbors*3,7))
    Amat[:nNeighbors,:] = np.column_stack((1/2*xjs**2, xjs*yjs, 1/2*yjs**2, xjs**3, xjs**2*yjs, xjs*yjs**2, yjs**3))
    Amat[nNeighbors:2*nNeighbors,:] = np.column_stack((xjs, yjs, np.zeros(xjs.shape), 3*xjs**2, 2*xjs*yjs, yjs**2, np.zeros(xjs.shape)))
    Amat[2*nNeighbors:3*nNeighbors,:] = np.column_stack((np.zeros(xjs.shape), xjs, yjs, np.zeros(xjs.shape), xjs**2, 2*xjs*yjs, 3*yjs**2))

    Bmat = np.zeros((nNeighbors*3,1))
    Bmat[:nNeighbors,0] = zjs
    Bmat[nNeighbors:2*nNeighbors,0] = -ajs/cjs
    Bmat[2*nNeighbors:3*nNeighbors,0] = -bjs/cjs

    if np.any(np.isnan(Amat)) or np.any(np.isnan(Bmat)):
        MaxPrincipal = np.nan
        MinPrincipal = np.nan
    else:
        A = Amat.T@Amat
        B = Amat.T@Bmat
        try:
            X = np.linalg.solve(A,B).T[0]
            W = np.array([[X[0],X[1]],
                            [X[1],X[2]]])
            
            [v,x] = np.linalg.eig(W)
            MaxPrincipal = np.max(v)
            MinPrincipal = np.min(v)
            if return_directions:
                # MaxPrincipalDirection = x[:,np.argmax(v)]
                # MinPrincipalDirection = x[:,np.argmin(v)]
                MaxPrincipalDirection = np.dot(np.linalg.inv(R[:3,:3]), np.append(x[:,np.argmax(v)],0)[:,None])[:,0]
                MinPrincipalDirection = np.dot(np.linalg.inv(R[:3,:3]), np.append(x[:,np.argmin(v)],0)[:,None])[:,0]
            else:
                MaxPrincipalDirection = np.zeros(3)
                MinPrincipalDirection = np.zeros(3)
        except:
            # For singular matrix errors
            MaxPrincipal = np.nan
            MinPrincipal = np.nan
            if return_directions:
                MaxPrincipalDirection = np.repeat(np.nan, 3)
                MinPrincipalDirection = np.repeat(np.nan, 3)
    return MaxPrincipal, MinPrincipal, MaxPrincipalDirection, MinPrincipalDirection

def AnalyticalCurvature(func,NodeCoords):
    """
    Calculate curvature of an implicit function. Curvature is sampled at the provided list of node coordinates, which should lie on the surface of an isosurface of the function.
    Based on Curvature formulas for implicit curves and surfaces, Ron Goldman (2005). 
    :cite:p:`Goldman2005`

    Parameters
    ----------
    func : function
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
    np.seterr(divide='ignore', invalid='ignore')
    try:
        import sympy as sp
    except:
        raise ImportError('AnalyticalCurvature requires sympy. Install with: pip install sympy.')

    x, y, z = sp.symbols('x y z', real=True)
    if type(NodeCoords) is list: NodeCoords = np.asarray(NodeCoords)

    if callable(func):
        try: 
            F = func(x,y,z)
        except:
            raise TypeError('func must accept three arguments (x,y,z) and be a \n sympy symbolic function or convertible to one. Functions that \n utilize  numpy functions like np.sin, np.max, or similar may \n not be compatible. Consider replacing with sympy equivalents. \n Test by running: `x, y, z = sp.symbols("x y z", real=True); func(x,y,z)`')
    elif isinstance(func, sp.Basic):
        F = func
    else:
        raise TypeError('func must be a sympy function or callable function of three arguments (x,y,z).')

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
        if Hess.is_diagonal():
            xcomp = sp.lambdify((x,y,z),Hess[0,0],['numpy',{'DiracDelta':DiracDelta}])(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2])
            ycomp = sp.lambdify((x,y,z),Hess[1,1],['numpy',{'DiracDelta':DiracDelta}])(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2])
            zcomp = sp.lambdify((x,y,z),Hess[2,2],['numpy',{'DiracDelta':DiracDelta}])(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2])

            h = np.zeros((3,3,len(NodeCoords)))
            h[0,0,:] = xcomp
            h[1,1,:] = ycomp
            h[2,2,:] = zcomp
        else:
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
        if Cof.is_diagonal():
            xcomp = sp.lambdify((x,y,z),Cof[0,0],['numpy',{'DiracDelta':DiracDelta}])(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2])
            ycomp = sp.lambdify((x,y,z),Cof[1,1],['numpy',{'DiracDelta':DiracDelta}])(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2])
            zcomp = sp.lambdify((x,y,z),Cof[2,2],['numpy',{'DiracDelta':DiracDelta}])(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2])

            c = np.zeros((3,3,len(NodeCoords)))
            c[0,0,:] = xcomp
            c[1,1,:] = ycomp
            c[2,2,:] = zcomp
        else:
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

def ImageCurvature(I,NodeCoords=None,gaussian_sigma=1,voxelsize=1,brightobject=True, mode='wrap'):
    """
    Calculate curvatures based on a 3D image. Curvature values are calculated for all voxels in the image,
    however, these curvature values are only meaningful at the surface of the imaged object(s). This can be used with 
    utils.grid2fun, converter.im2voxel, contours.MarchingCubesImage to evaluate the voxel curvatures at points on the surface
    of the imaged object(s).
    Based on Curvature formulas for implicit curves and surfaces, Ron Goldman (2005).
    :cite:p:`Goldman2005`

    .. note:: 
        Errors can occur if surface is too close to the boundary of the image. Consider building in padding based on gaussian_sigma.

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
    mode : str, optional
        Method used for handling edges of the image. See scipy.ndimage.gaussian_filter

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
    
    Fx = ndimage.gaussian_filter(I,gaussian_sigma,order=(1,0,0), mode=mode)
    Fy = ndimage.gaussian_filter(I,gaussian_sigma,order=(0,1,0), mode=mode)
    Fz = ndimage.gaussian_filter(I,gaussian_sigma,order=(0,0,1), mode=mode)

    Fxx = ndimage.gaussian_filter(Fx,gaussian_sigma,order=(1,0,0), mode=mode)
    Fxy = ndimage.gaussian_filter(Fx,gaussian_sigma,order=(0,1,0), mode=mode)
    Fxz = ndimage.gaussian_filter(Fx,gaussian_sigma,order=(0,0,1), mode=mode)
    
    Fyx = ndimage.gaussian_filter(Fy,gaussian_sigma,order=(1,0,0), mode=mode)
    Fyy = ndimage.gaussian_filter(Fy,gaussian_sigma,order=(0,1,0), mode=mode)
    Fyz = ndimage.gaussian_filter(Fy,gaussian_sigma,order=(0,0,1), mode=mode)
    
    Fzx = ndimage.gaussian_filter(Fz,gaussian_sigma,order=(1,0,0), mode=mode)
    Fzy = ndimage.gaussian_filter(Fz,gaussian_sigma,order=(0,1,0), mode=mode)
    Fzz = ndimage.gaussian_filter(Fz,gaussian_sigma,order=(0,0,1), mode=mode)


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

    curvedness = np.sqrt((np.atleast_1d(MaxPrincipal)**2 + np.atleast_1d(MinPrincipal)**2)/2)
    if len(curvedness) == 1 and isinstance(MaxPrincipal, (int, float)):
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
    
    MaxP = np.atleast_1d(MaxPrincipal) 
    MinP = np.atleast_1d(MinPrincipal) 
    with np.errstate(divide='ignore', invalid='ignore'):
        shape = (2/np.pi) * np.arctan((MaxP + MinP)/(MaxP - MinP))
    shape[MaxP == MinP] = 1*np.sign(MaxP[MaxP == MinP])
    
    if len(shape) == 1 and isinstance(MaxPrincipal, (int, float)):
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
    s = np.atleast_1d(shapeindex) 
    shape = [-1 for i in range(len(s))]
    for i in range(len(s)):
        if s[i] < -7/8:
            shape[i] = 0
        elif s[i] < -5/8:
            shape[i] = 1
        elif s[i] < -3/8:
            shape[i] = 2
        elif s[i] < -1/8:
            shape[i] = 3
        elif s[i] < 1/8:
            shape[i] = 4
        elif s[i] < 3/8:
            shape[i] = 5
        elif s[i] < 5/8:
            shape[i] = 6
        elif s[i] < 7/8:
            shape[i] = 7
        elif s[i] <= 1:
            shape[i] = 8
    if len(shape) == 1 and isinstance(shapeindex, (int, float)):
        shape = shape[0]
    return shape
                       