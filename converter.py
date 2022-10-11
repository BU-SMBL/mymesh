# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 17:48:50 2021

@author: toj
"""
#%%
import numpy as np
import pandas as pd
from scipy import ndimage
import sys, os, warnings, glob, gc, cv2
from . import MeshUtils, Rays
from joblib import Parallel, delayed

def solid2surface(NodeCoords,NodeConn):
    """
    solid2surface Extract the 2D surface elemenets from a 3D volume mesh

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates. Ex. [[x0,y0,z0],[x1,y1,z1],...]
    NodeConn : list
        Nodal connectivity list. Ex. [[n1,n2,n3,n4],[n2,n3,n4,n5],...]

    Returns
    -------
    SurfConn : list
        Nodal connectivity list of the extracted surface. Node IDs correspond to the original node list NodeCoords
    """    
    Faces = solid2faces(NodeCoords,NodeConn,return_FaceConn=False)
    SurfConn = faces2surface(Faces)
    
    return SurfConn

def solid2faces(NodeCoords,NodeConn,return_FaceConn=False,return_FaceElem=False):
    """
    solid2faces Convert solid mesh to faces. The will be one face for each side of each element,
    i.e. there will be duplicate faces for non-surface faces. Use faces2surface(Faces) to extract
    only the surface faces or face2unique(Faces) to remove duplicates.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.
    return_FaceConn : bool, optional
        If true, will return FaceConn, the Face Connectivity of each element.
        For each element, FaceConn has the indices of the faces connected to 
        that element, by default False
    return_FaceElem : bool, optional
        If true, will return FaceElem, the Element Connectivity of each face.
        For each face, FaceElem has the index of the element that the face
        is a part of, by default False

    Returns
    -------
    Faces : list
        List of mesh faces.
    FaceConn : list, optional
        The face connectivity of each element.
    FaceElem : list, optional
        The element index that each face is taken from.
    """     
       
    Ls = np.array([len(elem) for elem in NodeConn])
    triIdx = np.where(Ls == 3)[0]
    tetIdx = np.where((Ls == 4) | (Ls == 10))[0]
    pyrIdx = np.where(Ls == 5)[0]
    wdgIdx = np.where(Ls == 6)[0]
    hexIdx = np.where(Ls == 8)[0]
    tris = [NodeConn[i] for i in triIdx]
    tets = [NodeConn[i] for i in tetIdx]
    pyrs = [NodeConn[i] for i in pyrIdx]
    wdgs = [NodeConn[i] for i in wdgIdx]
    hexs = [NodeConn[i] for i in hexIdx]

    Faces = tris + tet2faces([],tets) + pyramid2faces([],pyrs) + wedge2faces([],wdgs) + hex2faces([],hexs)
    ElemIds_i = np.concatenate((triIdx,np.repeat(tetIdx,4),np.repeat(pyrIdx,5),np.repeat(wdgIdx,5),np.repeat(hexIdx,6)))
    FaceElem = ElemIds_i
    ElemIds_j = np.concatenate((np.repeat(0,len(triIdx)), 
            np.repeat([[0,1,2,3]],len(tetIdx),axis=0).reshape(len(tetIdx)*4),  
            np.repeat([[0,1,2,3,4]],len(pyrIdx),axis=0).reshape(len(pyrIdx)*5),                   
            np.repeat([[0,1,2,3,4]],len(wdgIdx),axis=0).reshape(len(wdgIdx)*5),   
            np.repeat([[0,1,2,3,4,5]],len(hexIdx),axis=0).reshape(len(hexIdx)*6),                    
            ))
    FaceConn = -1*np.ones((len(NodeConn),6))
    FaceConn[ElemIds_i,ElemIds_j] = np.arange(len(Faces))
    FaceConn = MeshUtils.ExtractRagged(FaceConn,dtype=int)
    
    if return_FaceConn and return_FaceElem:
        return Faces,FaceConn,FaceElem
    elif return_FaceConn:
        return Faces,FaceConn
    elif return_FaceElem:
        return Faces,FaceConn,FaceElem
    else:
        return Faces

def solid2edges(NodeCoords,NodeConn,ElemType='auto',return_EdgeConn=False,return_EdgeElem=False):
    """
    solid2edges Convert solid mesh to edges. The will be one edge for each edge of each element,
    i.e. there will be multiple entries for shared edges. Solid2Edges is also suitable for use 
    with 2D or surface meshes. It differes from surface2edges in that surface2edges returns only 
    exposed edges of unclosed surfaces.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.
    ElemType : str, optional
        Specifies the element type contained within the mesh, by default 'auto'.
        'auto' or 'mixed' - Will detect element type by the number of nodes present in each element. NOTE that 4-node elements are assumed to be tets, not quads
        'tri' - All elements treated as 3-node triangular elements.
        'quad' - All elements treated as 4-node quadrilateral elements.
        'tet' - All elements treated as 4-node tetrahedral elements.
        'pyramid' - All elements treated as 5-node wedge elements.
        'wedge' - All elements treated as 6-node quadrilateral elements.
        'hex' - All elements treated as 8-node quadrilateral elements.
        'polygon' - All elements treated as n-node polygonal elements. TODO: add support for return_EdgeConn and return_EdgeElem

    return_EdgeConn : bool, optional
        If true, will return EdgeConn, the Edge Connectivity of each element.
        For each element, EdgeConn has the indices of the edges connected to 
        that element, by default False
    return_EdgeElem : bool, optional
        If true, will return EdgeElem, the Element Connectivity of each edge.
        For each face, EdgeElem has the index of the element that the edge
        is a part of, by default False

    Returns
    -------
    Edges : list
        List of node connectivity of the edges in the mesh. Ex. [[n0,n1],[n1,n2],...]
    EdgeConn : list, optional
        The edge connectivity of each element. Ex. [[e0,e1,e2,e3,e4,e5],[e6,e7,e8,e9,e10],...]
    EdgeElem : list, optional
        The element index that each edge is taken from. Ex. [E0,E0,E0,E0,E0,E0,E1,E1,E1,...]
    """     
    
    if ElemType=='auto' or ElemType=='mixed':
        Ls = np.array([len(elem) for elem in NodeConn])
        triIdx = np.where(Ls == 3)[0]
        tetIdx = np.where(Ls == 4)[0]
        pyrIdx = np.where(Ls == 5)[0]
        wdgIdx = np.where(Ls == 6)[0]
        hexIdx = np.where(Ls == 8)[0]
        tris = [NodeConn[i] for i in triIdx]
        tets = [NodeConn[i] for i in tetIdx]
        pyrs = [NodeConn[i] for i in pyrIdx]
        wdgs = [NodeConn[i] for i in wdgIdx]
        hexs = [NodeConn[i] for i in hexIdx]

        Edges = tri2edges([],tris) + tet2edges([],tets) + pyramid2edges([],pyrs) + wedge2edges([],wdgs) + hex2edges([],hexs)
        if return_EdgeElem or return_EdgeConn:
            EdgeElem = np.concatenate((np.repeat(triIdx,3),np.repeat(tetIdx,6),np.repeat(pyrIdx,8),np.repeat(wdgIdx,9),np.repeat(hexIdx,12)))
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2]],len(triIdx),axis=0).reshape(len(triIdx)*3), 
                np.repeat([[0,1,2,3,4,5]],len(tetIdx),axis=0).reshape(len(tetIdx)*6),  
                np.repeat([[0,1,2,3,4,5,6,7]],len(pyrIdx),axis=0).reshape(len(pyrIdx)*8),                   
                np.repeat([[0,1,2,3,4,5,6,7,8]],len(wdgIdx),axis=0).reshape(len(wdgIdx)*9),   
                np.repeat([[0,1,2,3,4,5,6,7,8,9,10,11]],len(hexIdx),axis=0).reshape(len(hexIdx)*12),                    
                ))
            EdgeConn = -1*np.ones((len(NodeConn),12))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            EdgeConn = MeshUtils.ExtractRagged(EdgeConn,dtype=int)
    elif ElemType=='tri':
        Edges = tri2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            triIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(triIdx,3)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2]],len(tetIdx),axis=0).reshape(len(tetIdx)*3), 
                ))
            EdgeConn = -1*np.ones((len(NodeConn),3))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            EdgeConn = EdgeConn.astype(int).tolist()
    elif ElemType=='quad':
        Edges = quad2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            quadIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(quadIdx,4)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2]],len(tetIdx),axis=0).reshape(len(tetIdx)*3), 
                ))
            EdgeConn = -1*np.ones((len(NodeConn),4))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            EdgeConn = EdgeConn.astype(int).tolist()
    elif ElemType=='tet':
        Edges = tet2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            tetIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(tetIdx,6)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5]],len(tetIdx),axis=0).reshape(len(tetIdx)*6),  
                ))
            EdgeConn = -1*np.ones((len(NodeConn),6))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            EdgeConn = EdgeConn.astype(int).tolist()
    elif ElemType=='pyramid':
        Edges = pyramid2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            pyrIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(pyrIdx,8)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5,6,7]],len(pyrIdx),axis=0).reshape(len(pyrIdx)*8),                   
                ))
            EdgeConn = -1*np.ones((len(NodeConn),8))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            EdgeConn = EdgeConn.astype(int).tolist()
    elif ElemType=='wedge':
        Edges = wedge2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            wdgIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(wdgIdx,9)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5,6,7,8]],len(wdgIdx),axis=0).reshape(len(wdgIdx)*9),   
                ))
            EdgeConn = -1*np.ones((len(NodeConn),9))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            EdgeConn = EdgeConn.astype(int).tolist()
    elif ElemType=='hex':
        Edges = hex2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            hexIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(hexIdx,12)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5,6,7,8,9,10,11]],len(hexIdx),axis=0).reshape(len(hexIdx)*12),                    
                ))
            EdgeConn = -1*np.ones((len(NodeConn),12))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            EdgeConn = EdgeConn.astype(int).tolist()
    elif ElemType=='polygon':
        Edges = polygon2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            raise Exception('EdgeElem not implemented for ElemType=polygon')
    else:
        raise Exception('Invalid ElemType')
    
    if return_EdgeElem and return_EdgeConn:
        return Edges, EdgeConn, EdgeElem
    elif return_EdgeElem:
        return Edges, EdgeElem
    elif return_EdgeConn:
        return Edges, EdgeConn
    return Edges

def EdgesByElement(NodeCoords,NodeConn,ElemType='auto'):
    """
    EdgesByElement Returns edges grouped by the element from which they came.
    TODO: This can should be rewritten based on solid2edges using EdgeConn

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.
    ElemType : str, optional
        Specifies the element type contained within the mesh, by default 'auto'.
        'auto' or 'mixed' - Will detect element type by the number of nodes present in each element. NOTE that 4-node elements are assumed to be tets, not quads
        'tri' - All elements treated as 3-node triangular elements.
        'quad' - All elements treated as 4-node quadrilateral elements.
        'tet' - All elements treated as 4-node tetrahedral elements.
        'pyramid' - All elements treated as 5-node wedge elements.
        'wedge' - All elements treated as 6-node quadrilateral elements.
        'hex' - All elements treated as 8-node quadrilateral elements.
        'polygon' - All elements treated as n-node polygonal elements.

    Returns
    -------
    Edges, list
        Edge connectivity, grouped by element
    """    
    Edges = [[] for i in range(len(NodeConn))]
    for i,elem in enumerate(NodeConn):
        if (ElemType=='auto' and len(elem) == 3) or ElemType == 'tri':
            # Tri
            Edges[i] = tri2edges(NodeCoords,[elem])
        if (ElemType=='auto' and len(elem) == 4) or ElemType == 'tet':
            # Tet
            Edges[i] = tet2edges(NodeCoords,[elem])
        elif (ElemType=='auto' and len(elem) == 5) or ElemType == 'pyramid':
            # Pyramid
            Edges[i] = pyramid2edges(NodeCoords,[elem])
        elif (ElemType=='auto' and len(elem) == 6) or ElemType == 'wedge':
            # Wedge
            Edges[i] = wedge2edges(NodeCoords,[elem])
        elif (ElemType=='auto' and len(elem) == 8) or ElemType == 'hex':
            # Hex
            Edges[i] = hex2edges(NodeCoords,[elem])
        elif ElemType=='polygon':
            Edges[i] = polygon2edges(NodeCoords,[elem])
    return Edges

def solid2tets(NodeCoords,NodeConn,return_ids=False):
    """
    solid2tets Decompose all elements of a 3D volume mesh to tetrahedra.
    NOTE the generated tetrahedra will not generally be continuously oriented, i.e.
    edges of child tetrahedra may not be aligned between one parent element 
    and its neighbor, and thus the resulting mesh will typically be invalid.
    The primary use-case for this method is for methods like Quality.Volume
    which utilize the geometric properties of tetrahedra to determine properties of 
    the parent elements.
    

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.
    return_ids : bool, optional
        Element ids of the tets connected to the original elements, by default False

    Returns
    -------
    TetConn, list
        Nodal connectivity list of generated tetrahedra
    """    
    Ls = np.array([len(elem) for elem in NodeConn])
    tetIdx = np.where(Ls == 4)[0]
    pyrIdx = np.where(Ls == 5)[0]
    wdgIdx = np.where(Ls == 6)[0]
    hexIdx = np.where(Ls == 8)[0]
    tets = [NodeConn[i] for i in tetIdx]
    pyrs = [NodeConn[i] for i in pyrIdx]
    wdgs = [NodeConn[i] for i in wdgIdx]
    hexs = [NodeConn[i] for i in hexIdx]

    TetConn = tets + pyramid2tet([],pyrs) + wedge2tet([],wdgs) + hex2tet([],hexs)
    if return_ids:
        # Element ids of the tets connected to the original elements
        ElemIds_i = np.concatenate((tetIdx,np.repeat(pyrIdx,2),np.repeat(wdgIdx,3),np.repeat(hexIdx,5)))
        ElemIds_j = np.concatenate((np.repeat(0,len(tetIdx)), 
                np.repeat([[0,1]],len(pyrIdx),axis=0).reshape(len(pyrIdx)*2),                   
                np.repeat([[0,1,2]],len(wdgIdx),axis=0).reshape(len(wdgIdx)*3),   
                np.repeat([[0,1,2,3,4]],len(hexIdx),axis=0).reshape(len(hexIdx)*5),                    
                ))
        ElemIds = -1*np.ones((len(NodeConn),6))
        ElemIds[ElemIds_i,ElemIds_j] = np.arange(len(TetConn))
        ElemIds = MeshUtils.ExtractRagged(ElemIds,dtype=int)
    
    # TetConn = []
    # for i,elem in enumerate(NodeConn):
    #     if len(elem) == 4:
    #         tets = [elem]
    #     elif len(elem) == 5:
    #         tets = pyramid2tet(elem)
    #     elif len(elem) == 6:
    #         tets = wedge2tet(elem)
    #     elif len(elem) == 8:
    #         tets = hex2tet(elem)
    #     else:
    #         raise Exception('Unable to identify element type')
    #     TetConn += tets
    if return_ids:
        return TetConn, ElemIds
    return TetConn

def hex2tet(NodeCoords,NodeConn):
    """
    hex2tet Decompose all elements of a 3D hexahedral mesh to tetrahedra.
    Generally solid2tets should be used rather than hex2tet directly
    NOTE the generated tetrahedra will not generally be continuously oriented, i.e.
    edges of child tetrahedra may not be aligned between one parent element 
    and its neighbor, and thus the resulting mesh will typically be invalid.
    The primary use-case for this method is for methods like Quality.Volume
    which utilize the geometric properties of tetrahedra to determine properties of 
    the parent elements.
    

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list. All elements should be 8-Node hexahedral elements.

    Returns
    -------
    TetConn, list
        Nodal connectivity list of generated tetrahedra
    """

    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Tets = -1*np.ones((len(NodeConn)*5,4))
        t1 = ArrayConn[:,[0,1,3,4]]
        t2 = ArrayConn[:,[1,2,3,6]]
        t3 = ArrayConn[:,[4,6,5,1]]
        t4 = ArrayConn[:,[4,7,6,3]]
        t5 = ArrayConn[:,[4,6,1,3]]
        Tets[0::5] = t1
        Tets[1::5] = t2
        Tets[2::5] = t3
        Tets[3::5] = t4
        Tets[4::5] = t5
        Tets = Tets.astype(int).tolist()
    else:
        Tets = []


    # idx = np.array([
    #     [0,1,3,4],
    #     [1,2,3,6],
    #     [4,6,5,1],
    #     [4,7,6,3],
    #     [4,6,1,3]
    #     ])

    # Tets = np.array(elem)[idx].tolist()
    return Tets

def wedge2tet(NodeCoords,NodeConn):
    """
    wedge2tet Decompose all elements of a 3D wedge-element mesh to tetrahedra.
    Generally solid2tets should be used rather than wedge2tet directly
    NOTE the generated tetrahedra will not generally be continuously oriented, i.e.
    edges of child tetrahedra may not be aligned between one parent element 
    and its neighbor, and thus the resulting mesh will typically be invalid.
    The primary use-case for this method is for methods like Quality.Volume
    which utilize the geometric properties of tetrahedra to determine properties of 
    the parent elements.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list. All elements should be 6-Node wedge elements.

    Returns
    -------
    TetConn, list
        Nodal connectivity list of generated tetrahedra
    """
    if len(NodeConn) == 0:
        return []

    ArrayConn = np.asarray(NodeConn)
    Tets = -1*np.ones((len(NodeConn)*3,4))
    t1 = ArrayConn[:,[0,1,2,3]]
    t2 = ArrayConn[:,[1,2,3,4]]
    t3 = ArrayConn[:,[4,5,2,3]]
    Tets[0::3] = t1
    Tets[1::3] = t2
    Tets[2::3] = t3
    Tets = Tets.astype(int).tolist()

    return Tets

def pyramid2tet(NodeCoords,NodeConn):
    """
    pyramid2tet Decompose all elements of a 3D pyramidal mesh to tetrahedra.
    Generally solid2tets should be used rather than pyramid2tet directly
    NOTE the generated tetrahedra will not generally be continuously oriented, i.e.
    edges of child tetrahedra may not be aligned between one parent element 
    and its neighbor, and thus the resulting mesh will typically be invalid.
    The primary use-case for this method is for methods like Quality.Volume
    which utilize the geometric properties of tetrahedra to determine properties of 
    the parent elements.
    

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list. All elements should be 5-Node pyramidal elements.

    Returns
    -------
    TetConn, list
        Nodal connectivity list of generated tetrahedra
    """
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Tets = -1*np.ones((len(NodeConn)*2,4))
        t1 = ArrayConn[:,[0,1,2,4]]
        t2 = ArrayConn[:,[0,2,3,4]]
        Tets[0::2] = t1
        Tets[1::2] = t2
        Tets = Tets.astype(int).tolist()
    else:
        Tets = []
    # idx = np.array([
    #     [0,1,2,4],
    #     [0,2,3,4]
    # ])
    # Tets = np.array(elem)[idx].tolist()
    return Tets

def faces2surface(Faces):
    """
    faces2surface Identify surface elements, i.e. faces that aren't shared between two elements

    Parameters
    ----------
    Faces : list
        Nodal connectivity of mesh faces (as obtained by solid2faces)

    Returns
    -------
    SurfConn
        Nodal connectivity of the surface mesh.
    """    
    
    sortedConn = [sorted(face) for face in Faces]
    arr = np.empty(len(sortedConn),dtype=object)
    arr[:] = sortedConn
    unique,indices,counts = np.unique(arr,return_counts=True,return_index=True)
    SurfIdx = indices[np.where(counts==1)]
    SurfConn = np.array(Faces,dtype=object)[SurfIdx].tolist()
    return SurfConn

def faces2unique(Faces,return_idx=False,return_inv=False):
    """
    faces2unique reduce set of mesh faces to contain only unique faces, i.e. there will only
    be one entry to indicate a face shared between two elements.

    Parameters
    ----------
    Faces : list
        Nodal connectivity of mesh faces (as obtained by solid2faces)
    return_idx : bool, optional
        If true, will return idx, the array of indices that relate the original list of
        faces to the list of unique faces (UFaces = Faces[idx]), by default False.
        See numpy.unique for additional details.
    return_inv : bool, optional
        If true, will return inv, the array of indices that relate the unique list of
        faces to the list of original faces (Faces = UFaces[inv]), by default False
        See numpy.unique for additional details.

    Returns
    -------
    UFaces : list
        Nodal connectivity of unique mesh faces.
    idx : np.ndarray, optional
        The array of indices that relate the unique list of
        faces to the list of original faces.
    inv : np.ndarray, optional
        The array of indices that relate the unique list of
        faces to the list of original faces.

    """
    # Returns only the unique faces (not duplicated for each element)
    Rfaces = MeshUtils.PadRagged(Faces)
    # Get all unique element faces (accounting for flipped versions of faces)
    _,idx,inv = np.unique(np.sort(Rfaces,axis=1),axis=0,return_index=True,return_inverse=True)
    RFaces = Rfaces[idx]
    UFaces = MeshUtils.ExtractRagged(RFaces,dtype=int)
    if return_idx and return_inv:
        return UFaces,idx,inv
    elif return_idx:
        return UFaces,idx
    elif return_inv:
        return UFaces,inv
    else:
        return UFaces

def faces2faceelemconn(Faces,FaceConn,FaceElem,return_UniqueFaceInfo=False):
    """
    faces2faceelemconn 
    FaceElemConn gives the elements connected to each face (max 2), ordered such that the element that the face
    is facing (based on face normal direction) is listed first. If the face is only attached to one element (such
    as on the surface), the other entry will be np.nan. 
    Assumes Faces, FaceConn, and FaceElem are directly from solid2faces, i.e. faces aren't yet the unique
    faces obtained by faces2unique


    Parameters
    ----------
    Faces : list
        Nodal connectivity of element faces (as obtained by solid2faces).
    FaceConn : list
        Face connectivity of the mesh elements i.e. indices of Faces connected to each element,
        (as obtained by solid2faces).
    FaceElem : list
        List of elements that each face originated on (as obtained by solid2faces).
    return_UniqueFaceInfo : bool, optional
        If true, will return data obtained from faces2unique (with all optional outputs)
        to reduce redundant call to faces2unique.

    Returns
    -------
    FaceElemConn : list
        List of elements connected to each face. 
    UFaces : list
        Nodal connectivity of unique mesh faces. (see faces2unique)
    UFaceConn : list, optional
        FaceConn transformed to properly index UFaces rather than Faces.
    UFaceElem : list, optional
        FaceElem transformed to coorespond with UFaces rather than Faces.
    idx : np.ndarray, optional
        The array of indices that relate the unique list of
        faces to the list of original faces (see faces2unique).
    inv : np.ndarray, optional
        The array of indices that relate the unique list of
        faces to the list of original faces (see faces2unique).
    """    
    # 
    UFaces,idx,inv = faces2unique(Faces,return_idx=True,return_inv=True)

    UFaces = MeshUtils.PadRagged(Faces)[idx]
    UFaceElem = np.asarray(FaceElem)[idx]
    UFaces = np.append(UFaces, np.repeat(-1,UFaces.shape[1])[None,:],axis=0)
    inv = np.append(inv,-1)
    UFaceConn = inv[MeshUtils.PadRagged(FaceConn)] # Faces attached to each element
    # Face-Element Connectivity
    FaceElemConn = np.nan*(np.ones((len(UFaces),2))) # Elements attached to each face

    FaceElemConn = np.nan*(np.ones((len(UFaces),2))) # Elements attached to each face
    FECidx = (UFaceElem[UFaceConn] == np.repeat(np.arange(len(FaceConn))[:,None],UFaceConn.shape[1],axis=1)).astype(int)
    FaceElemConn[UFaceConn,FECidx] = np.repeat(np.arange(len(FaceConn))[:,None],UFaceConn.shape[1],axis=1)
    FaceElemConn = [[int(x) if not np.isnan(x) else x for x in y] for y in FaceElemConn[:-1]]

    if return_UniqueFaceInfo:
        UFaces = MeshUtils.ExtractRagged(UFaces)[:-1]
        UFaceConn = MeshUtils.ExtractRagged(UFaceConn)
        return FaceElemConn, UFaces, UFaceConn, UFaceElem, idx, inv[:-1]

    return FaceElemConn

def edges2unique(Edges,return_idx=False,return_inv=False):
    """
    edges2unique reduce set of mesh edges to contain only unique edges, i.e. there will only
    be one entry to indicate a edge shared between multiple elements.

    Parameters
    ----------
    Edges : list
        Nodal connectivity of mesh edges (as obtained by solid2edges).
    return_idx : bool, optional
        If true, will return idx, the array of indices that relate the original list of
        edges to the list of unique edges (UEaces = Eaces[idx]), by default False.
        See numpy.unique for additional details.
    return_inv : bool, optional
        If true, will return inv, the array of indices that relate the unique list of
        faces to the list of original faces (Faces = UFaces[inv]), by default False
        See numpy.unique for additional details.

    Returns
    -------
    UEdges : list
        Nodal connectivity of unique mesh faces.
    idx : np.ndarray, optional
        The array of indices that relate the unique list of
        faces to the list of original faces.
    inv : np.ndarray, optional
        The array of indices that relate the unique list of
        faces to the list of original faces.
    """
    # Returns only the unique edges (not duplicated for each element)
    # Get all unique element edges (accounting for flipped versions of edges)
    _,idx,inv = np.unique(np.sort(Edges,axis=1),axis=0,return_index=True,return_inverse=True)
    UEdges = np.array(Edges)[idx]
    if return_idx and return_inv:
        return UEdges,idx,inv
    elif return_idx:
        return UEdges,idx
    elif return_inv:
        return UEdges,inv
    else:
        return UEdges

def tet2faces(NodeCoords,NodeConn):
    """
    tet2faces extract triangular faces from all elements of a purely 4-Node tetrahedral mesh.
    All faces will be ordered such that the nodes are in counter-clockwise order when
    viewed from outside of the element. Best practice is to use solid2faces, rather than 
    using tet2faces directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.

    Returns
    -------
    Faces : list
        List of nodal connectivity of the mesh faces.
    """
    
    # Explode volume tet mesh into triangles, 4 triangles per tet, ensuring
    # that triangles are ordered in counter-clockwise order when viewed
    # from the outside, assuming the tetrahedral node numbering scheme
    # ref: https://abaqus-docs.mit.edu/2017/English/SIMACAETHERefMap/simathe-c-tritetwedge.htm#simathe-c-tritetwedge-t-Interpolation-sma-topic1__simathe-c-stmtritet-iso-master
    if len(NodeConn) > 0:
        if len(NodeConn[0]) == 4:
            ArrayConn = np.asarray(NodeConn)
            Faces = -1*np.ones((len(NodeConn)*4,3))
            f1 = ArrayConn[:,[0,2,1]]
            f2 = ArrayConn[:,[0,1,3]]
            f3 = ArrayConn[:,[1,2,3]]
            f4 = ArrayConn[:,[0,3,2]]
            Faces[0::4] = f1
            Faces[1::4] = f2
            Faces[2::4] = f3
            Faces[3::4] = f4
            Faces = Faces.astype(int).tolist()
        elif len(NodeConn[0]) == 10:
            ArrayConn = np.asarray(NodeConn)
            Faces = -1*np.ones((len(NodeConn)*4,6))
            f1 = ArrayConn[:,[0,2,1,6,5,4]]
            f2 = ArrayConn[:,[0,1,3,4,8,7]]
            f3 = ArrayConn[:,[1,2,3,5,9,8]]
            f4 = ArrayConn[:,[0,3,2,7,9,6]]
            Faces[0::4] = f1
            Faces[1::4] = f2
            Faces[2::4] = f3
            Faces[3::4] = f4
            Faces = Faces.astype(int).tolist()
        else:
            raise Exception('Must be 4 or 10 node tetrahedral mesh.')
    else:
        Faces = []

    return Faces

def hex2faces(NodeCoords,NodeConn):
    """
    hex2faces extract quadrilateral faces from all elements of a purely 8-Node hexahedral mesh.
    All faces will be ordered such that the nodes are in counter-clockwise order when
    viewed from outside of the element. Best practice is to use solid2faces, rather than 
    using hex2faces directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.

    Returns
    -------
    Faces : list
        List of nodal connectivity of the mesh faces.
    """
    # Explode volume hex mesh into quads, 6 quads per hex, 
    # assuming the hexahedral node numbering scheme of abaqus
    # ref: https://abaqus-docs.mit.edu/2017/English/SIMACAEELMRefMap/simaelm-c-solidcont.htm
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Faces = -1*np.ones((len(NodeConn)*6,4))
        f1 = ArrayConn[:,[0,3,2,1]]
        f2 = ArrayConn[:,[0,1,5,4]]
        f3 = ArrayConn[:,[1,2,6,5]]
        f4 = ArrayConn[:,[2,3,7,6]]
        f5 = ArrayConn[:,[3,0,4,7]]
        f6 = ArrayConn[:,[4,5,6,7]]
        Faces[0::6] = f1
        Faces[1::6] = f2
        Faces[2::6] = f3
        Faces[3::6] = f4
        Faces[4::6] = f5
        Faces[5::6] = f6
        Faces = Faces.astype(int).tolist()
    else:
        Faces = []
    # for i in range(len(NodeConn)):
    #     Faces[6*i+0] = [NodeConn[i][j] for j in [0,3,2,1]]
    #     Faces[6*i+1] = [NodeConn[i][j] for j in [0,1,5,4]]
    #     Faces[6*i+2] = [NodeConn[i][j] for j in [1,2,6,5]]
    #     Faces[6*i+3] = [NodeConn[i][j] for j in [2,3,7,6]]
    #     Faces[6*i+4] = [NodeConn[i][j] for j in [3,0,4,7]]
    #     Faces[6*i+5] = [NodeConn[i][j] for j in [4,5,6,7]]
    return Faces

def pyramid2faces(NodeCoords,NodeConn):
    """
    pyramid2faces extract triangular and quadrilateral faces from all elements of a 
    purely 5-Node pyramidal mesh. All faces will be ordered such that the nodes are in 
    counter-clockwise order when viewed from outside of the element. Best practice is to 
    use solid2faces, rather than using pyramid2faces directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.

    Returns
    -------
    Faces : list
        List of nodal connectivity of the mesh faces.
    """
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Faces = -1*np.ones((len(NodeConn)*5,4))
        f1 = ArrayConn[:,[0,3,2,1]]
        f2 = ArrayConn[:,[0,1,4]]
        f3 = ArrayConn[:,[1,2,4]]
        f4 = ArrayConn[:,[2,3,4]]
        f5 = ArrayConn[:,[3,0,4]]
        Faces[0::5] = f1
        Faces[1::5,:3] = f2
        Faces[2::5,:3] = f3
        Faces[3::5,:3] = f4
        Faces[4::5,:3] = f5
        Faces = MeshUtils.ExtractRagged(Faces,delval=-1,dtype=int)
    else:
        Faces = []
    # Faces = [[] for i in range(5*len(NodeConn))]    
    # for i in range(len(NodeConn)):
    #     Faces[5*i+0] = [NodeConn[i][j] for j in [0,1,2,3]]
    #     Faces[5*i+1] = [NodeConn[i][j] for j in [0,1,4]]
    #     Faces[5*i+2] = [NodeConn[i][j] for j in [1,2,4]]
    #     Faces[5*i+3] = [NodeConn[i][j] for j in [2,3,4]]
    #     Faces[5*i+4] = [NodeConn[i][j] for j in [3,0,4]]
    return Faces
        
def wedge2faces(NodeCoords,NodeConn):
    """
    wedge2faces extract triangular and quadrilateral faces from all elements of a purely 
    6-Node wedge elemet mesh. All faces will be ordered such that the nodes are in 
    counter-clockwise order when viewed from outside of the element. Best practice is 
    to use solid2faces, rather than using wedge2faces directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.

    Returns
    -------
    Faces : list
        List of nodal connectivity of the mesh faces.
    """
    if len(NodeConn):
        ArrayConn = np.asarray(NodeConn)
        Faces = -1*np.ones((len(NodeConn)*5,4))
        f1 = ArrayConn[:,[2,1,0]]
        f2 = ArrayConn[:,[0,1,4,3]]
        f3 = ArrayConn[:,[1,2,5,4]]
        f4 = ArrayConn[:,[2,0,3,5]]
        f5 = ArrayConn[:,[3,4,5]]
        Faces[0::5,:3] = f1
        Faces[1::5] = f2
        Faces[2::5] = f3
        Faces[3::5] = f4
        Faces[4::5,:3] = f5
        Faces = MeshUtils.ExtractRagged(Faces,delval=-1,dtype=int)
    else:
        Faces = []
    # Faces = [[] for i in range(5*len(NodeConn))]    
    # for i in range(len(NodeConn)):
    #     Faces[5*i+0] = [NodeConn[i][j] for j in [0,1,2]]
    #     Faces[5*i+1] = [NodeConn[i][j] for j in [0,1,4,3]]
    #     Faces[5*i+2] = [NodeConn[i][j] for j in [1,2,5,4]]
    #     Faces[5*i+3] = [NodeConn[i][j] for j in [2,0,3,5]]
    #     Faces[5*i+4] = [NodeConn[i][j] for j in [3,4,5]]
    return Faces

def tri2edges(NodeCoords,NodeConn):
    """
    tri2edges extract edges from all elements of a purely 3-Node triangular mesh.
    Best practice is to use solid2edges, rather than using tri2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """
    # Note that some code relies on these edges being in the order that they're currently in
    # edges = [[] for i in range(3*len(NodeConn))]
    
    # Explode surface elements into edges
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*3,2))
        e1 = ArrayConn[:,[0,1]]
        e2 = ArrayConn[:,[1,2]]
        e3 = ArrayConn[:,[2,0]]
        Edges[0::3] = e1
        Edges[1::3] = e2
        Edges[2::3] = e3
        Edges = Edges.astype(int).tolist()
    else:
        Edges = []
    
    return Edges

def quad2edges(NodeCoords,NodeConn):
    """
    quad2edges extract edges from all elements of a purely 4-Node quadrilateral mesh.
    Best practice is to use solid2edges, rather than using quad2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*4,2))
        e1 = ArrayConn[:,[0,1]]
        e2 = ArrayConn[:,[1,2]]
        e3 = ArrayConn[:,[2,3]]
        e4 = ArrayConn[:,[3,0]]
        Edges[0::4] = e1
        Edges[1::4] = e2
        Edges[2::4] = e3
        Edges[3::4] = e4

        Edges = Edges.astype(int).tolist()
    else:
        Edges = []
    
    return Edges

def polygon2edges(NodeCoords,NodeConn):
    """
    polygon2edges extract edges from all elements of a polygonal mesh.
    Best practice is to use solid2edges, rather than using polygon2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """
    edges = []
    for i,elem in enumerate(NodeConn):
        for j,n in enumerate(elem):
            edges.append([elem[j-1],n])
    return edges   

def tet2edges(NodeCoords,NodeConn):
    """
    tet2edges extract edges from all elements of a purely 4-Node tetrahedral mesh.
    Best practice is to use solid2edges, rather than using tet2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """

    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*6,2))
        e1 = ArrayConn[:,[0,1]]
        e2 = ArrayConn[:,[1,2]]
        e3 = ArrayConn[:,[2,0]]
        e4 = ArrayConn[:,[0,3]]
        e5 = ArrayConn[:,[1,3]]
        e6 = ArrayConn[:,[2,3]]
        Edges[0::6] = e1
        Edges[1::6] = e2
        Edges[2::6] = e3
        Edges[3::6] = e4
        Edges[4::6] = e5
        Edges[5::6] = e6
        Edges = Edges.astype(int).tolist()
    else:
        Edges = []
    return Edges

def pyramid2edges(NodeCoords,NodeConn):
    """
    pyramid2edges extract edges from all elements of a purely 5-Node pyramidal mesh.
    Best practice is to use solid2edges, rather than using pyramid2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*8,2))
        e1 = ArrayConn[:,[0,1]]
        e2 = ArrayConn[:,[1,2]]
        e3 = ArrayConn[:,[2,3]]
        e4 = ArrayConn[:,[3,0]]
        e5 = ArrayConn[:,[0,4]]
        e6 = ArrayConn[:,[1,4]]
        e7 = ArrayConn[:,[2,4]]
        e8 = ArrayConn[:,[3,4]]
        Edges[0::8] = e1
        Edges[1::8] = e2
        Edges[2::8] = e3
        Edges[3::8] = e4
        Edges[4::8] = e5
        Edges[5::8] = e6
        Edges[6::8] = e7
        Edges[7::8] = e8
        Edges = Edges.astype(int).tolist()
    else:
        Edges = []
    return Edges

def wedge2edges(NodeCoords,NodeConn):
    """
    wedge2edges extract edges from all elements of a purely 6-Node wedge element mesh.
    Best practice is to use solid2edges, rather than using wedge2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*9,2))
        e1 = ArrayConn[:,[0,1]]
        e2 = ArrayConn[:,[1,2]]
        e3 = ArrayConn[:,[2,0]]
        e4 = ArrayConn[:,[0,3]]
        e5 = ArrayConn[:,[1,4]]
        e6 = ArrayConn[:,[2,5]]
        e7 = ArrayConn[:,[3,4]]
        e8 = ArrayConn[:,[4,5]]
        e9 = ArrayConn[:,[5,3]]
        Edges[0::9] = e1
        Edges[1::9] = e2
        Edges[2::9] = e3
        Edges[3::9] = e4
        Edges[4::9] = e5
        Edges[5::9] = e6
        Edges[6::9] = e7
        Edges[7::9] = e8
        Edges[8::9] = e9
        Edges = Edges.astype(int).tolist()
    else:
        Edges = []
    return Edges

def hex2edges(NodeCoords,NodeConn):
    """
    hex2edges extract edges from all elements of a purely 8-Node hexahedral mesh.
    Best practice is to use solid2edges, rather than using hex2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*12,2))
        e1 = ArrayConn[:,[0,1]]
        e2 = ArrayConn[:,[1,2]]
        e3 = ArrayConn[:,[2,3]]
        e4 = ArrayConn[:,[3,0]]
        e5 = ArrayConn[:,[0,4]]
        e6 = ArrayConn[:,[1,5]]
        e7 = ArrayConn[:,[2,6]]
        e8 = ArrayConn[:,[3,7]]
        e9 = ArrayConn[:,[4,5]]
        e10 = ArrayConn[:,[5,6]]
        e11 = ArrayConn[:,[6,7]]
        e12 = ArrayConn[:,[7,4]]
        Edges[0::12] = e1
        Edges[1::12] = e2
        Edges[2::12] = e3
        Edges[3::12] = e4
        Edges[4::12] = e5
        Edges[5::12] = e6
        Edges[6::12] = e7
        Edges[7::12] = e8
        Edges[8::12] = e9
        Edges[9::12] = e10
        Edges[10::12] = e11
        Edges[11::12] = e12
        Edges = Edges.astype(int).tolist()
    else:
        Edges = []
    return Edges

def quad2tri(QuadNodeConn):
    """
    quad2tri Converts a quadrilateral mesh to a triangular mesh by splitting each quad into 2 tris  

    Parameters
    ----------
    QuadNodeConn : list
        list of nodal connectivities for a strictly quadrilateral mesh.

    Returns
    -------
    TriNodeConn : list
        list of nodal connectivities for the new triangular mesh.
    """
    TriNodeConn = [[] for i in range(2*len(QuadNodeConn))]
    for i in range(len(QuadNodeConn)):
        TriNodeConn[2*i] = [QuadNodeConn[i][0], QuadNodeConn[i][1], QuadNodeConn[i][3]]
        TriNodeConn[2*i+1] = [QuadNodeConn[i][1], QuadNodeConn[i][2], QuadNodeConn[i][3]]
    return TriNodeConn
        
def tet102tet4(Tet10NodeConn):
    """
    tet102tet4 Converts a 10 node tetradehdral mesh to a 4 node tetradehedral mesh.
    Assumes a 10 node tetrahedral numbering scheme where the first 4 nodes define the
    tetrahedral vertices, the remaining nodes are thus neglected.

    Parameters
    ----------
    Tet10NodeConn : list
        Nodal connectivities for a 10-Node tetrahedral mesh

    Returns
    -------
    Tet4NodeConn
        Nodal connectivities for the equivalent 4-Node tetrahedral mesh
    """
    # 
    # Assumes the 10 node tetrahedral element scheme used by Abaqus
    Tet4NodeConn = [[Tet10NodeConn[i][0],Tet10NodeConn[i][1],Tet10NodeConn[i][2],Tet10NodeConn[i][3]] for i in range(len(Tet10NodeConn))]
    
    return Tet4NodeConn
     
def surf2edges(NodeCoords,NodeConn):
    """
    surf2edges Extract the edges of an unclosed surface mesh.
    This differs from solid2edges in that it doesn't return any
    interior mesh edges, and for a volume mesh or closed surface,
    surf2edges will return [].

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.

    Returns
    -------
    Edges : list
        List of nodal connectivities for exposed edges.
    """
    # TODO: this should be revamped to utilize solid2edges, edges2unique

    edges = [[0,0] for i in range(3*len(NodeConn))]
    if len(NodeConn) == 0:
        return edges
    
    # Explode surface elements into edges
    for i in range(len(NodeConn)):
        edges[3*i+0] = [NodeConn[i][j] for j in [0,1]]
        edges[3*i+1] = [NodeConn[i][j] for j in [1,2]]
        edges[3*i+2] = [NodeConn[i][j] for j in [0,2]]
    # Identify surface elements, i.e. triangles that aren't shared between two elements
    sortedConn = np.sort(edges,axis=1)
    unique,counts = np.unique(sortedConn,axis=0,return_counts=True)
    uedges = unique[np.where(counts==1)].tolist()
    sC = sortedConn.tolist()
    # SurfIdx = [np.where(np.all(surfs[i] == sC,axis=1))[0].tolist()[0] for i in range(len(surfs))]
    EdgeIdx = [sC.index(uedges[i]) for i in range(len(uedges))]
    Edges = np.array(edges)[EdgeIdx].tolist()

    return Edges

#%% -----------------------------------------------------------
# TODO: Below functions need to be revisitied, may be unstable.
# -------------------------------------------------------------

def edge2corners(NodeCoords,EdgeConn,angle=150):
    corners = []
    EdgeNodeNeighbors,EdgeElemConn = MeshUtils.getNodeNeighbors(NodeCoords,EdgeConn)
    for i in range(len(NodeCoords)):
        if len(EdgeNodeNeighbors[i]) == 2:
            A = NodeCoords[i]
            B = NodeCoords[EdgeNodeNeighbors[i][0]]
            C = NodeCoords[EdgeNodeNeighbors[i][1]]
                
            a2 = (B[0]-C[0])**2 + (B[1]-C[1])**2 + (B[2]-C[2])**2
            b2 = (C[0]-A[0])**2 + (C[1]-A[1])**2 + (C[2]-A[2])**2
            b = np.sqrt(b2)
            c2 = (A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2
            c = np.sqrt(c2)
            
            # Law of cosines
            alpha = np.arccos((b2+c2-a2)/(2*b*c))*180/np.pi
            if alpha < angle:
                corners.append(i)
    return corners
        
def GridMesh(xlims,ylims,zlims,h):
    """
    GridMesh Generate structured hexahedral mesh with element size h

    Parameters
    ----------
    xlims : list
        [xmin, xmax]
    ylims : list
        [ymin, ymax].
    zlims : list
        [zmin, zmax].
    h : numeric
        Element side length.
                
    Returns
    -------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivities.

    """        
    nX = int(np.round((xlims[1]-xlims[0])/h))
    nY = int(np.round((ylims[1]-ylims[0])/h))
    nZ = int(np.round((zlims[1]-zlims[0])/h))
    X, Y, Z = np.mgrid[xlims[0]:xlims[1]:nX*1j, 
                        ylims[0]:ylims[1]:nY*1j, 
                        zlims[0]:zlims[1]:nZ*1j]
    Xshape = X.shape
    # print(0)
    # x = np.round(X.flatten(),decimals=16)
    # y = np.round(Y.flatten(),decimals=16)
    # z = np.round(Z.flatten(),decimals=16)
    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    del X, Y, Z
    gc.collect()
    # print(1)
    ids = np.arange(len(x))
    Ids = np.reshape(ids,Xshape)
    # print(2)
    del ids
    gc.collect()
    NodeCoords = np.vstack([x,y,z]).transpose()
    pd.DataFrame(NodeCoords).to_csv('NodeCoords.csv')
    # print(3)
    del x, y, z
    gc.collect()
    NodeConn = [[] for i in range(nX-1) for j in range(nY-1) for k in range(nZ-1)] 
    l = 0
    for i in range(nX-1):
        # print(i)
        for j in range(nY-1):
            for k in range(nZ-1):
                idxs = [Ids[i,j,k],Ids[i+1,j,k],Ids[i+1,j+1,k],Ids[i,j+1,k],Ids[i,j,k+1],Ids[i+1,j,k+1],Ids[i+1,j+1,k+1],Ids[i,j+1,k+1]]
                NodeConn[l] = idxs
                # f.write(','.join([str(x) for x in idxs]) + '\n')
                l += 1
            
    return NodeCoords,NodeConn

def Surf2Voxel(SurfCoords,SurfConn,h):
    # Very Slow right now
    arrayCoords = np.array(SurfCoords)
    xlims = [min(arrayCoords[:,0]), max(arrayCoords[:,0])]
    ylims = [min(arrayCoords[:,1]), max(arrayCoords[:,1])]
    zlims = [min(arrayCoords[:,2]), max(arrayCoords[:,2])]
    GridCoords,GridConn = GridMesh(xlims,ylims,zlims,h)
    
    VoxelConn = []
    k = 0
    for elem in GridConn:
        k+=1 
        print(k)
        for trielem in SurfConn:
            xlim = [GridCoords[elem[0]][0],GridCoords[elem[6]][0]]
            ylim = [GridCoords[elem[0]][1],GridCoords[elem[6]][1]]
            zlim = [GridCoords[elem[0]][2],GridCoords[elem[6]][2]]
            if Rays.TriangleBoxIntersection(arrayCoords[trielem], xlim, ylim, zlim):
                VoxelConn.append(elem)
                break
    VoxelCoords,VoxelConn,_ = removeNodes(GridCoords,VoxelConn)
    return VoxelCoords,VoxelConn

def makeGrid(xlims, ylims, zlims, VoxelSize):
    
    h = VoxelSize
    nX = int(np.round((xlims[1]-xlims[0])/h))
    nY = int(np.round((ylims[1]-ylims[0])/h))
    nZ = int(np.round((zlims[1]-zlims[0])/h))
    xs = np.arange(xlims[0],xlims[1]+h,h)
    ys = np.arange(ylims[0],ylims[1]+h,h)
    zs = np.arange(zlims[0],zlims[1]+h,h)
    X, Y, Z = np.meshgrid(xs,ys,zs,indexing='ij')
    Xshape = X.shape
    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    del X, Y, Z, xs, ys, zs
    gc.collect()
    ids = np.arange(len(x))
    Ids = np.reshape(ids,Xshape)
    del ids
    gc.collect()
    VoxelCoords = np.vstack([x,y,z]).transpose()
    del x, y, z
    gc.collect()
    VoxelConn = [[] for i in range(nX-1) for j in range(nY-1) for k in range(nZ-1)] 
    l = 0
    for i in range(nX-1):
        for j in range(nY-1):
            for k in range(nZ-1):
                idxs = [Ids[i,j,k],Ids[i+1,j,k],Ids[i+1,j+1,k],Ids[i,j+1,k],Ids[i,j,k+1],Ids[i+1,j,k+1],Ids[i+1,j+1,k+1],Ids[i,j+1,k+1]]
                VoxelConn[l] = idxs
                l += 1
    VoxelCoords,VoxelConn,_ = removeNodes(VoxelCoords,VoxelConn)
    return VoxelCoords, VoxelConn

def im2voxel(img, voxelsize, scalefactor=1, scaleorder=1, threshold=None, n_jobs=4):
    
    if type(img) == list:
        img = np.array(img)
    elif type(img) == str:
        pool = Parallel(n_jobs=n_jobs)
        path = img
        files = glob.glob(os.path.join(path,'*'))
        files = [file for file in files if '.tiff' in file or '.TIFF' in file or '.DCM' in file]
        img = np.array(pool(delayed(cv2.imread)(file,0) for file in files))

    if scalefactor != 1:
        img = ndimage.zoom(img,scalefactor,order=scaleorder)
        voxelsize /= scalefactor

    (nz,ny,nx) = img.shape
    xlims = [0,(nx)*voxelsize]
    ylims = [0,(ny)*voxelsize]
    zlims = [0,(nz)*voxelsize]
    VoxelCoords, VoxelConn = makeGrid(xlims, ylims, zlims, voxelsize)
    NodeVals = img.flatten(order='F')
    if threshold is not None:
        VoxelConn = [elem for elem in VoxelConn if not all(NodeVals[elem] < threshold)]

    return VoxelCoords, VoxelConn, NodeVals


# def im2voxel(img, voxelsize, scalefactor=1, scaleorder=1, threshold=None, n_jobs=4):
    
#     if type(img) == list:
#         img = np.array(img)
#     elif type(img) == str:
#         pool = Parallel(n_jobs=n_jobs)
#         path = img
#         files = glob.glob(os.path.join(path,'*'))
#         files = [file for file in files if '.tiff' in file or '.TIFF' in file or '.DCM' in file]
#         img = np.array(pool(delayed(cv2.imread)(file for file in files)))

#     if scalefactor != 1:
#         img = ndimage.zoom(img,scalefactor,order=scaleorder)
#         voxelsize /= scalefactor

#     # (nz,nx,ny) = img.shape
#     (nz,ny,nx) = img.shape
#     xlims = [0,(nx-1)*voxelsize]
#     ylims = [0,(ny-1)*voxelsize]
#     zlims = [0,(nz-1)*voxelsize]
    
#     h = np.float64(voxelsize)
#     nX = int(np.round((xlims[1]-xlims[0])/h))
#     nY = int(np.round((ylims[1]-ylims[0])/h))
#     nZ = int(np.round((zlims[1]-zlims[0])/h))
#     xs = np.arange(xlims[0],nX*h+h/10,h)
#     ys = np.arange(ylims[0],nY*h+h/10,h)
#     zs = np.arange(zlims[0],nZ*h+h/10,h)

#     # X, Y, Z = np.mgrid[xlims[0]:xlims[1]:nX*1j, 
#     #                     ylims[0]:ylims[1]:nY*1j, 
#     #                     zlims[0]:zlims[1]:nZ*1j]
#     X, Y, Z = np.meshgrid(xs,ys,zs,indexing='ij')
#     Xshape = X.shape
#     x = X.flatten()
#     y = Y.flatten()
#     z = Z.flatten()
#     del X, Y, Z, xs, ys, zs
#     gc.collect()
#     ids = np.arange(len(x))
#     Ids = np.reshape(ids,Xshape)
#     del ids
#     gc.collect()
#     VoxelCoords = np.vstack([x,y,z]).transpose()
#     del x, y, z
#     gc.collect()
    
#     NodeVals = img.flatten(order='F')
#     del img
#     gc.collect()
#     VoxelConn = [[] for i in range(nX-1) for j in range(nY-1) for k in range(nZ-1)] 
#     l = 0
    
#     if threshold:
#         for i in range(nX-1):
#             # print(i)
#             for j in range(nY-1):
#                 for k in range(nZ-1):
#                     idxs = [Ids[i,j,k],Ids[i+1,j,k],Ids[i+1,j+1,k],Ids[i,j+1,k],Ids[i,j,k+1],Ids[i+1,j,k+1],Ids[i+1,j+1,k+1],Ids[i,j+1,k+1]]
#                     if any(NodeVals[idxs] > threshold):
#                         VoxelConn[l] = idxs
#                     l += 1
#         VoxelConn = [elem for elem in VoxelConn if elem != []]
        
#     else:
#         for i in range(nX-1):
#             # print(i)
#             for j in range(nY-1):
#                 for k in range(nZ-1):
#                     idxs = [Ids[i,j,k],Ids[i+1,j,k],Ids[i+1,j+1,k],Ids[i,j+1,k],Ids[i,j,k+1],Ids[i+1,j,k+1],Ids[i+1,j+1,k+1],Ids[i,j+1,k+1]]
#                     VoxelConn[l] = idxs
#                     l += 1
    
#     del Ids, idxs
#     gc.collect()
    
#     return VoxelCoords, VoxelConn, NodeVals

def voxel2im(VoxelCoords, VoxelConn, NodeVals):
    if type(VoxelCoords) == list: VoxelCoords = np.array(VoxelCoords)
    shape = (len(np.unique(VoxelCoords[:,2])),len(np.unique(VoxelCoords[:,1])),len(np.unique(VoxelCoords[:,0])))
    I = np.reshape(NodeVals,shape,order='F')
    
    return I

def removeNodes(NodeCoords,NodeConn):
    # removeNodes Removes nodes that aren't held by any element
    OriginalIds, inverse = np.unique([n for elem in NodeConn for n in elem],return_inverse=True)
    NewNodeCoords = [NodeCoords[i] for i in OriginalIds]
    # NewNodeConn = np.reshape(inverse,np.shape(NodeConn)).tolist()
    
    NewNodeConn = [[] for elem in NodeConn]
    k = 0
    for i,elem in enumerate(NodeConn):
        temp = []
        for e in elem:
            temp.append(inverse[k])
            k += 1
        NewNodeConn[i] = temp
    
    
    return NewNodeCoords, NewNodeConn, OriginalIds

def surf2dual(NodeCoords,SurfConn,Centroids=None,ElemConn=None,NodeNormals=None,sort='ccw'):
    if not Centroids:
        Centroids = MeshUtils.Centroids(NodeCoords,SurfConn)
    if not ElemConn:
        _,ElemConn = MeshUtils.getNodeNeighbors(NodeCoords,SurfConn,ElemType='polygon')
    if not NodeNormals and (sort == 'ccw' or sort == 'CCW' or sort == 'cw' or sort == 'CW'):
        ElemNormals = MeshUtils.CalcFaceNormal(NodeCoords,SurfConn)
        NodeNormals = MeshUtils.Face2NodeNormal(NodeCoords,SurfConn,ElemConn,ElemNormals)
    
    DualCoords = Centroids
    if sort == 'ccw' or sort == 'CCW' or sort == 'cw' or sort == 'CW':
        DualConn = [[] for i in range(len(NodeCoords))]
        for i,P in enumerate(NodeCoords):
            E = ElemConn[i]
            N = NodeNormals[i]
            C = np.array([Centroids[e] for e in E])
            # Transform to local coordinate system
            # Rotation matrix from global z (k=[0,0,1]) to local z (N)
            k = [0,0,1]
            if N == k:
                rotAxis = k
                angle = 0
            elif np.all(N == [0,0,-1]):
                rotAxis = [1,0,0]
                angle = np.pi
            else:
                cross = np.cross(k,N)
                rotAxis = cross/np.linalg.norm(cross)
                angle = np.arccos(np.dot(k,N))
                
            sinhalf = np.sin(angle/2)
            q = [np.cos(angle/2),               # Quaternion Rotation
                 rotAxis[0]*sinhalf,
                 rotAxis[1]*sinhalf,
                 rotAxis[2]*sinhalf]
        
            R = [[2*(q[0]**2+q[1]**2)-1,   2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2]), 0],
                 [2*(q[1]*q[2]+q[0]*q[3]), 2*(q[0]**2+q[2]**2)-1,   2*(q[2]*q[3]-q[0]*q[1]), 0],
                 [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 2*(q[0]**2+q[3]**2)-1,   0],
                 [0,                       0,                       0,                       1]
                 ]
            # Translation to map p to (0,0,0)
            T = [[1,0,0,-P[0]],
                 [0,1,0,-P[1]],
                 [0,0,1,-P[2]],
                 [0,0,0,1]]
            
            localCentroids = [np.matmul(np.matmul(T,[c[0],c[1],c[2],1]),R)[0:3] for c in C]
            # projCentroids = [np.subtract(c,np.multiply(np.dot(np.subtract(c,P),k),k)) for c in localCentroids]
            angles = [np.arctan2(c[1],c[0]) for c in localCentroids]
            
            zipped = [(angles[j],E[j]) for j in range(len(E))]
            zipped.sort()
            DualConn[i] = [z[1] for z in zipped]
            
            if sort == 'cw' or sort == 'CW':
                DualConn[i].reverse()
    elif sort == 'None' or sort == 'none' or sort == None:
        DualConn = ElemConn
    else:
        raise Exception('Invalid input for sort')
    
    return DualCoords,DualConn
# %%
