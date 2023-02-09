# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 23:58:18 2022

@author: toj
"""
import numpy as np

import sys, copy, warnings
from . import MeshUtils, converter
try:
    from joblib import Parallel, delayed
except:
    warnings.warn('Optional dependencies not found - some functions may not work properly')

# Finite element modeling mesh quality, energy balance and validation methods: A review with recommendations associated with the modeling of bone tissue - Burkhart et al.

def AspectRatio(NodeCoords,NodeConn,verbose=False):
    """
    AspectRatio Calculates element aspect ratios for each element in the mesh.
    For all element types, the aspect ratio is calculated as the length of the longest
    edge divided by the length of the shortest edge of an element.

    Aspect ratio is >= 1, with 1 being the optimal element quality.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.
    verbose : bool, optional
        If true, will print min, max, and mean element quality, by default False.

    Returns
    -------
    aspect : np.ndarray
        Array of aspect ratios for each element.
    """
    ArrayCoords = np.asarray(NodeCoords)
    Edges,EdgeConn = converter.solid2edges(NodeCoords,NodeConn,return_EdgeConn=True,return_EdgeElem=False)
    EdgePoints = ArrayCoords[Edges]
    EdgeVec = EdgePoints[:,1] - EdgePoints[:,0]
    lengths = np.append(np.linalg.norm(EdgeVec,axis=1),[np.nan])
    REdgeConn = MeshUtils.PadRagged(EdgeConn,fillval=-1)
    ElemEdgeLengths = lengths[REdgeConn]
    aspect = np.nanmax(ElemEdgeLengths,axis=1)/np.nanmin(ElemEdgeLengths,axis=1)

    if verbose:
        minAspect = min(aspect)
        maxAspect = max(aspect)
        meanAspect = np.mean(aspect)
        print('------------------------------------------')
        print(f'Minimum Aspect Ratio: {minAspect:.3f} on Element {np.where(aspect==minAspect)[0][0]:.0f}')
        print(f'Maximum Aspect Ratio: {maxAspect:.3f} on Element {np.where(aspect==maxAspect)[0][0]:.0f}')
        print(f'Mean Aspect Ratio: {meanAspect:.3f}')
        print('------------------------------------------')
    return aspect

def Orthogonality(NodeCoords,NodeConn,verbose=False):
    """
    Orthogonality Calculates element orthogonality for each element in the mesh.
    For all element types, orthogonality is calculated by determing the minimum 
    of the angle cosines between face normal vectors (Ai) and the element centroid
    to face centroid vectors (fi) and the angle cosines between Ai and the element
    centroid to neighbor element centroid (ci).

    Orthogonality ranges from 0 to 1, with 0 being the worst element quality
    and 1 being the best.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.
    verbose : bool, optional
        If true, will print min, max, and mean element quality, by default False.

    Returns
    -------
    ortho : np.ndarray
        Array of orthogonalities for each element.
    """
    Faces,FaceConn,FaceElem = converter.solid2faces(NodeCoords,NodeConn, return_FaceConn=True, return_FaceElem=True)
    RFaceConn = MeshUtils.PadRagged(FaceConn,fillval=-1)
    FaceElemConn, UFaces, UFaceConn, UFaceElem, idx, inv = converter.faces2faceelemconn(Faces,FaceConn,FaceElem,return_UniqueFaceInfo=True)


    ElemCentroids = MeshUtils.Centroids(NodeCoords,NodeConn)
    FaceCentroids = MeshUtils.Centroids(NodeCoords,Faces)

    # Face Normal Vectors
    A = np.append(MeshUtils.CalcFaceNormal(NodeCoords,Faces),[[np.nan,np.nan,np.nan]],axis=0)
    Ai = A[RFaceConn]

    # Vectors from element centroid to face centroid
    ConnectedFaceCentroids = np.append(FaceCentroids,[[np.nan,np.nan,np.nan]],axis=0)[RFaceConn]
    fi = ConnectedFaceCentroids - ElemCentroids[:,None,:]
    Aifi = np.nansum(Ai*fi,axis=2)/np.linalg.norm(fi,axis=2)

    # Vectors from element centroid to adjacent element centroids
    ArrayFaceElemConn = np.append(FaceElemConn,[[np.nan,np.nan]],axis=0)
    ArrayFaceElemConn[np.isnan(ArrayFaceElemConn)] = -1
    ArrayFaceElemConn = ArrayFaceElemConn.astype(int)
    
    aElemCentroids = np.append(ElemCentroids,[[np.nan,np.nan,np.nan]],axis=0)
    c = aElemCentroids[ArrayFaceElemConn[inv][:,0]] - aElemCentroids[ArrayFaceElemConn[inv][:,1]]  
    ci = np.append(c,[[np.nan,np.nan,np.nan]],axis=0)[RFaceConn]
    sidx = set(idx)
    sign = np.array([1 if i in sidx else -1 for i in range(len(A))])[RFaceConn]
    Aici = np.nansum(sign[:,:,None]*Ai*ci,axis=2)/np.linalg.norm(ci,axis=2)

    ortho = np.minimum(np.nanmin(Aici,axis=1),np.nanmin(Aifi,axis=1))

    if verbose:
        minOrtho = min(ortho)
        maxOrtho = max(ortho)
        meanOrtho = np.mean(ortho)
        print('------------------------------------------')
        print(f'Minimum Orthogonality: {minOrtho:.3f} on Element {np.where(ortho==minOrtho)[0][0]:.0f}')
        print(f'Maximum Orthogonality: {maxOrtho:.3f} on Element {np.where(ortho==maxOrtho)[0][0]:.0f}')
        print(f'Mean Orthogonality: {meanOrtho:.3f}')
        print('------------------------------------------')

    return ortho

def InverseOrthogonality(NodeCoords,NodeConn,verbose=False):
    """
    InverseOrthogonality Calculates element inverse orthogonality for each element in the mesh.
    For all element types, inverse orthogonality is calculated as 1-orthogonality.

    Inverse orthogonality ranges from 0 to 1, with 0 being the best element quality
    and 1 being the worst.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.
    verbose : bool, optional
        If true, will print min, max, and mean element quality, by default False.

    Returns
    -------
    iortho : np.ndarray
        Array of inverse orthogonalities for each element.
    """
    ortho = Orthogonality(NodeCoords,NodeConn)
    iortho = 1-ortho

    if verbose:
        minIortho = min(iortho)
        maxIortho = max(iortho)
        meanIortho = np.mean(iortho)
        print('------------------------------------------')
        print(f'Minimum Inverse Orthogonality: {minIortho:.3f} on Element {np.where(iortho==minIortho)[0][0]:.0f}')
        print(f'Maximum Inverse Orthogonality: {maxIortho:.3f} on Element {np.where(iortho==maxIortho)[0][0]:.0f}')
        print(f'Mean Inverse Orthogonality: {meanIortho:.3f}')
        print('------------------------------------------')

    return iortho

def OrthogonalQuality(NodeCoords,NodeConn,verbose=False):
    """
    OrthogonalQuality Calculates element orthogonality for each element in the mesh.
    For all element types, orthogonal quality is calculated as 1-InverseOrthogonalQuality.

    Orthogonal quality ranges from 0 to 1, with 1 being the best element quality
    and 0 being the worst.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.
    verbose : bool, optional
        If true, will print min, max, and mean element quality, by default False.

    Returns
    -------
    orthoq : np.ndarray
        Array of orthogonal qualities for each element.
    """
    orthoq = 1-InverseOrthogonalQuality(NodeCoords,NodeConn,verbose=False)

    if verbose:
        minOrthoq = min(orthoq)
        maxOrthoq = max(orthoq)
        meanOrthoq = np.mean(orthoq)
        print('------------------------------------------')
        print(f'Minimum Orthogonal Quality: {minOrthoq:.3f} on Element {np.where(orthoq==minOrthoq)[0][0]:.0f}')
        print(f'Maximum Orthogonal Quality: {maxOrthoq:.3f} on Element {np.where(orthoq==maxOrthoq)[0][0]:.0f}')
        print(f'Mean Orthogonal Quality: {meanOrthoq:.3f}')
        print('------------------------------------------')

    return orthoq

def InverseOrthogonalQuality(NodeCoords,NodeConn,verbose=False):
    """
    InverseOrthogonalQuality Calculates element inverse orthogonal quality for each 
    element in the mesh. For tetrahedral, wedge, and pyramidal elements, inverse orthogonal
    quality is calculated as the maximum of skewness and inverse orthogonality. For hexahedral
    elements, inverse orthogonal quality is simply the inverse orthogonality.

    Inverse orthogonal quality ranges from 0 to 1, with 0 being the best element quality
    and 1 being the worst.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.
    verbose : bool, optional
        If true, will print min, max, and mean element quality, by default False.

    Returns
    -------
    iorthoq : np.ndarray
        Array of inverse orthogonal quality for each element.
    """
    iortho = InverseOrthogonality(NodeCoords,NodeConn,verbose=False)
    skew = Skewness(NodeCoords,NodeConn,verbose=False)
    nElem = np.array([len(elem) for elem in NodeConn])
    TetWdgPyr = nElem < 8
    iorthoq = copy.copy(iortho)
    iorthoq[TetWdgPyr] = np.maximum(skew[TetWdgPyr],iortho[TetWdgPyr])

    if verbose:
        minIorthoq = min(iorthoq)
        maxIorthoq = max(iorthoq)
        meanIorthoq = np.mean(iorthoq)
        print('------------------------------------------')
        print(f'Minimum Inverse Orthogonal Quality: {minIorthoq:.3f} on Element {np.where(iorthoq==minIorthoq)[0][0]:.0f}')
        print(f'Maximum Inverse Orthogonal Quality: {maxIorthoq:.3f} on Element {np.where(iorthoq==maxIorthoq)[0][0]:.0f}')
        print(f'Mean Inverse Orthogonal Quality: {meanIorthoq:.3f}')
        print('------------------------------------------')

    return iorthoq

def Skewness(NodeCoords,NodeConn,verbose=False,tetmethod='volume'):
    """
    Skewness Calculates element skewness for each element in the mesh. 
    For triangular, hexahedral, wedge, and pyramidal elements, skewness is 
    calculated by the equiangular skewness method. 
    For tetrahedral elements, skewness is calculated by either the equiangular
    skewness method or the equilateral volume skewness method.

    Skewness ranges from 0 to 1, with 0 being the best element quality
    and 1 being the worst.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.
    verbose : bool, optional
        If true, will print min, max, and mean element quality, as well 
        as the number of 'slivers' i.e. elements with skewness above 0.9, by default False.
    tetmethod : str, optional
        Method to be used for tetrahedral skewness, by default 'volume'.
        'volume' - uses equilateral volume skewness method.
        'angle' - uses equiangular skewness method.

    Returns
    -------
    skew : np.ndarray
        Array of skewness for each element.
    """
    # tetmethod: 'volume' or 'angle'

    if tetmethod == 'angle':
        skew = equiangular_skewness(NodeCoords,NodeConn)
    elif tetmethod == 'volume':
        skew = np.zeros(len(NodeConn))

        Ls = np.array([len(elem) for elem in NodeConn])
        tetIdx = np.where(Ls == 4)[0]
        otherIdx = np.where(Ls != 4)[0]
        if len(tetIdx) > 0:
            Tets = [NodeConn[i] for i in tetIdx]
            TetSkew = tet_vol_skewness(NodeCoords,Tets)
            skew[tetIdx] = TetSkew
        if len(otherIdx) > 0:
            Others = [NodeConn[i] for i in otherIdx]
            OtherSkew = equiangular_skewness(NodeCoords,Others)
            skew[otherIdx] = OtherSkew

    else:
        raise Exception('Invalid tetmethod argument. Must be "angle" or "volume".')
    
    
    if verbose:
        minSkew = min(skew)
        maxSkew = max(skew)
        meanSkew = np.mean(skew)
        nSliver = sum(skew>0.9)
        print('------------------------------------------')
        print(f'Minimum Skewness: {minSkew:.3f} on Element {np.where(skew==minSkew)[0][0]:.0f}')
        print(f'Maximum Skewness: {maxSkew:.3f} on Element {np.where(skew==maxSkew)[0][0]:.0f}')
        print(f'Mean Skewness: {meanSkew:.3f}')
        print(f'{nSliver:d} Slivers with Skewness > 0.9')
        print('------------------------------------------')

    return skew

def tri_skewness(NodeCoords,NodeConn):
    """
    tri_skewness Calculates triangular skewness for each triangle in the mesh. 
    Mesh should be strictly triangular.

    Skewness ranges from 0 to 1, with 0 being the best element quality
    and 1 being the worst.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.

    Returns
    -------
    skew : np.ndarray
        Array of skewness for each element.
    """
    points = np.asarray(NodeCoords)[np.asarray(NodeConn)]
    A = points[:,0]
    B = points[:,1]
    C = points[:,2]

    a2 = (B[:,0]-C[:,0])**2 + (B[:,1]-C[:,1])**2 + (B[:,2]-C[:,2])**2
    a = np.sqrt(a2)
    b2 = (C[:,0]-A[:,0])**2 + (C[:,1]-A[:,1])**2 + (C[:,2]-A[:,2])**2
    b = np.sqrt(b2)
    c2 = (A[:,0]-B[:,0])**2 + (A[:,1]-B[:,1])**2 + (A[:,2]-B[:,2])**2
    c = np.sqrt(c2)

    # Law of cosines
    alpha = np.arccos(np.clip((b2+c2-a2)/(2*b*c),-1,1))
    beta = np.arccos(np.clip((a2+c2-b2)/(2*a*c),-1,1))
    gamma = np.arccos(np.clip((a2+b2-c2)/(2*a*b),-1,1))

    # Normalized Equiangular Skewness
    thetaMax = np.max([alpha,beta,gamma],axis=0)
    thetaMin = np.min([alpha,beta,gamma],axis=0)
    thetaEqui = np.pi/3

    skew = np.max([(thetaMax-thetaEqui)/(np.pi-thetaEqui),(thetaEqui-thetaMin)/(thetaEqui)],axis=0)
    return skew

def quad_skewness(NodeCoords,NodeConn):
    """
    quad_skewness Calculates quadrilateral skewness for each triangle in the mesh. 
    Mesh should be strictly quadrilateral.

    Skewness ranges from 0 to 1, with 0 being the best element quality
    and 1 being the worst.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.

    Returns
    -------
    skew : np.ndarray
        Array of skewness for each element.
    """
    points = np.asarray(NodeCoords)[np.asarray(NodeConn)]
    A = points[:,0]
    B = points[:,1]
    C = points[:,2]
    D = points[:,3]

    # Diagonals
    BD2 = (B[:,0]-D[:,0])**2 + (B[:,1]-D[:,1])**2 + (B[:,2]-D[:,2])**2
    AC2 = (C[:,0]-A[:,0])**2 + (C[:,1]-A[:,1])**2 + (C[:,2]-A[:,2])**2
    # Sides
    AB2 = (B[:,0]-A[:,0])**2 + (B[:,1]-A[:,1])**2 + (B[:,2]-A[:,2])**2
    AB = np.sqrt(AB2)
    BC2 = (B[:,0]-C[:,0])**2 + (B[:,1]-C[:,1])**2 + (B[:,2]-C[:,2])**2
    BC = np.sqrt(BC2)
    CD2 = (D[:,0]-C[:,0])**2 + (D[:,1]-C[:,1])**2 + (D[:,2]-C[:,2])**2
    CD = np.sqrt(CD2)
    AD2 = (D[:,0]-A[:,0])**2 + (D[:,1]-A[:,1])**2 + (D[:,2]-A[:,2])**2
    AD = np.sqrt(AD2)

    # Law of cosines
    alpha = np.arccos(np.clip((AB2+AD2-BD2)/(2*AB*AD),-1,1))
    beta = np.arccos(np.clip((AB2+BC2-AC2)/(2*AB*BC),-1,1))
    gamma = np.arccos(np.clip((BC2+CD2-BD2)/(2*BC*CD),-1,1))
    delta = np.arccos(np.clip((AD2+CD2-AC2)/(2*AD*CD),-1,1))
    # Normalized Equiangular Skewness
    thetaMax = np.max([alpha,beta,gamma,delta],axis=0)
    thetaMin = np.min([alpha,beta,gamma,delta],axis=0)
    thetaEqui = np.pi/2

    skew = np.max([(thetaMax-thetaEqui)/(np.pi-thetaEqui),(thetaEqui-thetaMin)/(thetaEqui)],axis=0)
    return skew

def tet_vol_skewness(NodeCoords,NodeConn):
    """
    tet_vol_skewness Calculates element skewness for each tetrahedral element in the mesh
    using the equilateral volume skewness method.

    Skewness ranges from 0 to 1, with 0 being the best element quality
    and 1 being the worst.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.

    Returns
    -------
    skew : np.ndarray
        Array of skewness for each element.
    """
    
    # Volume-based
    V = Volume(NodeCoords,NodeConn)
    points = np.asarray(NodeCoords)[np.asarray(NodeConn)]
    # edge lengths
    a = np.linalg.norm(points[:,0] - points[:,3],axis=1)
    b = np.linalg.norm(points[:,1] - points[:,3],axis=1)
    c = np.linalg.norm(points[:,2] - points[:,3],axis=1)
    A = np.linalg.norm(points[:,1] - points[:,2],axis=1)
    B = np.linalg.norm(points[:,2] - points[:,0],axis=1)
    C = np.linalg.norm(points[:,0] - points[:,1],axis=1)
    # Circumradius
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.sqrt((a*A+b*B+c*C)*(a*A+b*B-c*C)*(a*A-b*B+c*C)*(-a*A+b*B+c*C))/(24*V)

        Videal = 8*np.sqrt(3)/27 * R**3
        skew = (Videal - V)/Videal
    return skew

def equiangular_skewness(NodeCoords,NodeConn):
    """
    equiangular_skewness Calculates element skewness for each element in the mesh
    using the equiangular skewness method.

    Skewness ranges from 0 to 1, with 0 being the best element quality
    and 1 being the worst.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.

    Returns
    -------
    skew : np.ndarray
        Array of skewness for each element.
    """
    Faces,FaceConn = converter.solid2faces(NodeCoords,NodeConn,return_FaceConn=True)

    FaceSkew = np.zeros(len(Faces))
    Ls = np.array([len(elem) for elem in Faces])
    triIdx = np.where(Ls == 3)[0]
    if len(triIdx) > 0:
        Tris = [Faces[i] for i in triIdx]
        TriSkew = tri_skewness(NodeCoords,Tris)
        FaceSkew[triIdx] = TriSkew
    quadIdx = np.where(Ls == 4)[0]
    if len(quadIdx) > 0:
        Quads = [Faces[i] for i in quadIdx]
        QuadSkew = quad_skewness(NodeCoords,Quads)
        FaceSkew[quadIdx] = QuadSkew

    RFaceConn = MeshUtils.PadRagged(FaceConn)
    FaceSkew = np.append(FaceSkew,np.nan)
    skew = np.nanmax(FaceSkew[RFaceConn],axis=1)

    return skew

def Area(NodeCoords,NodeConn):
    # Currently only for triangular meshes
    Points = np.asarray(NodeCoords)[np.asarray(NodeConn)]
    Area = np.linalg.norm(np.cross(Points[:,1]-Points[:,0],Points[:,2]-Points[:,0]),axis=1)/2 
    return Area

def Volume(NodeCoords,NodeConn,verbose=False):
    """
    Volme Calculates element volumes for each element in the mesh.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.
    verbose : bool, optional
        If true, will print min, max, and mean element volume, by default False.

    Returns
    -------
    V : np.ndarray
        Array of volumes for each element.
    """
    if len(NodeConn) == 0:
        return []
    ArrayCoords = np.asarray(NodeCoords)    
    TetConn,ElemIds = converter.solid2tets(NodeCoords,NodeConn,return_ids=True)     
    ArrayConn = np.array(TetConn)
    pt0 = ArrayCoords[ArrayConn][:,0]
    pt1 = ArrayCoords[ArrayConn][:,1]
    pt2 = ArrayCoords[ArrayConn][:,2]
    pt3 = ArrayCoords[ArrayConn][:,3]
    vol = -np.sum((pt0-pt1)*np.cross((pt1-pt3),(pt2-pt3)),axis=1)/6
    vol = np.append(vol,0)
    V = np.sum(vol[MeshUtils.PadRagged(ElemIds)],axis=1)

    if verbose:
        minVol = min(V)
        minOrd = np.floor(np.log10(minVol))
        maxVol = max(V)
        maxOrd = np.floor(np.log10(maxVol))
        meanVol = np.mean(V)
        meanOrd = np.floor(np.log10(meanVol))
        print('------------------------------------------')
        print(f'Minimum Volume: {minVol/10**minOrd:.3f}e{minOrd:.0f} on Element {np.where(V==minVol)[0][0]:.0f}')
        print(f'Maximum Volume: {maxVol/10**maxOrd:.3f}e{maxOrd:.0f} on Element {np.where(V==maxVol)[0][0]:.0f}')
        print(f'Mean Volume: {meanVol/10**meanOrd:.3f}e{meanOrd:.0f}')
        print('------------------------------------------')
    return V

def MinDihedral(NodeCoords,NodeConn,verbose=False):

    Faces, FaceConn, FaceElem = converter.solid2faces(NodeCoords,NodeConn,return_FaceConn=True,return_FaceElem=True)
    Normals = np.asarray(MeshUtils.CalcFaceNormal(NodeCoords,Faces))

    tetkey = np.array([[0,1],[0,2],[0,3],[1,2],[2,3],[3,1]])
    pyrkey = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]])
    wdgkey = np.array([[0,1],[0,2],[0,3],[1,2],[2,3],[3,1],[1,4],[2,4],[3,4]])
    hexkey = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1],[1,5],[2,5],[3,5],[4,5]])

    elemkeys = [tetkey if len(elem)==4 else pyrkey if len(elem)==5 else wdgkey if len(elem)==6 else hexkey if len(elem)==8 else [] for elem in NodeConn]
    MinAngles = np.array([np.min(dihedralAngles(Normals[FaceConn[i]][elemkeys[i][:,0]],Normals[FaceConn[i]][elemkeys[i][:,1]],Abs=True)) for i in range(len(NodeConn))])

    if verbose:
        minAngle = min(MinAngles)
        maxAngle = max(MinAngles)
        meanAngle = np.mean(MinAngles)
        print('------------------------------------------')
        print(f'Minimum Minimum Dihedral Angle: {minAngle*180/np.pi:.3f}° on Element {np.where(MinAngles==minAngle)[0][0]:.0f}')
        print(f'Maximum Minimum Dihedral Angle: {maxAngle*180/np.pi:.3f}° on Element {np.where(MinAngles==maxAngle)[0][0]:.0f}')
        print(f'Mean Minimum Dihedral Angle: {meanAngle*180/np.pi:.3f}°')
        print('------------------------------------------')
    return MinAngles

def MaxDihedral(NodeCoords,NodeConn,verbose=False):

    Faces, FaceConn, FaceElem = converter.solid2faces(NodeCoords,NodeConn,return_FaceConn=True,return_FaceElem=True)
    Normals = np.asarray(MeshUtils.CalcFaceNormal(NodeCoords,Faces))

    tetkey = np.array([[0,1],[0,2],[0,3],[1,2],[2,3],[3,1]])
    pyrkey = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]])
    wdgkey = np.array([[0,1],[0,2],[0,3],[1,2],[2,3],[3,1],[1,4],[2,4],[3,4]])
    hexkey = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1],[1,5],[2,5],[3,5],[4,5]])

    elemkeys = [tetkey if len(elem)==4 else pyrkey if len(elem)==5 else wdgkey if len(elem)==6 else hexkey if len(elem)==8 else [] for elem in NodeConn]
    MaxAngles = np.array([np.max(dihedralAngles(Normals[FaceConn[i]][elemkeys[i][:,0]],Normals[FaceConn[i]][elemkeys[i][:,1]],Abs=True)) for i in range(len(NodeConn))])

    if verbose:
        minAngle = min(MaxAngles)
        maxAngle = max(MaxAngles)
        meanAngle = np.mean(MaxAngles)
        print('------------------------------------------')
        print(f'Minimum Maximum Dihedral Angle: {minAngle*180/np.pi:.3f}° on Element {np.where(MaxAngles==minAngle)[0][0]:.0f}')
        print(f'Maximum Maximum Dihedral Angle: {maxAngle*180/np.pi:.3f}° on Element {np.where(MaxAngles==maxAngle)[0][0]:.0f}')
        print(f'Mean Maximum Dihedral Angle: {meanAngle*180/np.pi:.3f}°')
        print('------------------------------------------')
    return MaxAngles

def dihedralAngle(Ni,Nj,Abs=True):
    # (In Radians) based on element normals Ni, Nj
    if Abs:
        return np.arccos(min([1,abs(np.dot(Ni,Nj)/(np.linalg.norm(Ni)*np.linalg.norm(Nj)))]))
    else:
        return np.arccos(np.round(np.dot(Ni,Nj)/(np.linalg.norm(Ni)*np.linalg.norm(Nj)),16))

def dihedralAngles(Nis,Njs,Abs=False):
    # Vectorized dihedral angles between two sets of normal vectors
    if Abs:
        angles = np.arccos(np.clip(np.abs(np.sum((np.asarray(Nis)*np.asarray(Njs)),axis=1)),0,1))
    else:
        angles = np.arccos(np.clip(np.sum((np.asarray(Nis)*np.asarray(Njs)),axis=1),-1,1))
    return angles

def DihedralAngles(ElemNormals,ElemNeighbors):
    # Dihedral angles between (surface/face) elements and their neighbors
    ElemNormals = np.asarray(ElemNormals)
    ElemNeighbors = np.asarray(ElemNeighbors)
    NeighborNormals = ElemNormals[ElemNeighbors]
    angles = np.arccos(np.clip(np.sum((np.array(ElemNormals)[:,None]*NeighborNormals),axis=2),-1,1))
    return angles
    
def tet_volume(nodes):
    return -np.dot(np.subtract(nodes[0],nodes[1]),np.cross(np.subtract(nodes[1],nodes[3]),np.subtract(nodes[2],nodes[3])))/6


    
