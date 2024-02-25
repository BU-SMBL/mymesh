# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 17:48:50 2021

@author: toj
"""

import numpy as np

from scipy import ndimage, sparse
import sys, os, warnings, glob, gc, tempfile
from . import utils, rays, primitives

def solid2surface(NodeCoords,NodeConn):
    """
    Extract the 2D surface elements from a 3D volume mesh

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
    Convert solid mesh to faces. The will be one face for each side of each element,
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
    edgIdx = np.where(Ls == 2)[0]
    triIdx = np.where(Ls == 3)[0]
    tetIdx = np.where((Ls == 4) | (Ls == 10))[0]
    pyrIdx = np.where(Ls == 5)[0]
    wdgIdx = np.where(Ls == 6)[0]
    hexIdx = np.where(Ls == 8)[0]
    edgs = [NodeConn[i] for i in edgIdx]
    tris = [NodeConn[i] for i in triIdx]
    tets = [NodeConn[i] for i in tetIdx]
    pyrs = [NodeConn[i] for i in pyrIdx]
    wdgs = [NodeConn[i] for i in wdgIdx]
    hexs = [NodeConn[i] for i in hexIdx]

    Faces = edgs + tris + tet2faces([],tets) + pyramid2faces([],pyrs) + wedge2faces([],wdgs) + hex2faces([],hexs)
    ElemIds_i = np.concatenate((edgIdx,triIdx,np.repeat(tetIdx,4),np.repeat(pyrIdx,5),np.repeat(wdgIdx,5),np.repeat(hexIdx,6)))
    FaceElem = ElemIds_i
    ElemIds_j = np.concatenate((np.repeat(0,len(edgIdx)),np.repeat(0,len(triIdx)), 
            np.repeat([[0,1,2,3]],len(tetIdx),axis=0).reshape(len(tetIdx)*4),  
            np.repeat([[0,1,2,3,4]],len(pyrIdx),axis=0).reshape(len(pyrIdx)*5),                   
            np.repeat([[0,1,2,3,4]],len(wdgIdx),axis=0).reshape(len(wdgIdx)*5),   
            np.repeat([[0,1,2,3,4,5]],len(hexIdx),axis=0).reshape(len(hexIdx)*6),                    
            ))
    FaceConn = -1*np.ones((len(NodeConn),6))
    FaceConn[ElemIds_i,ElemIds_j] = np.arange(len(Faces))
    FaceConn = utils.ExtractRagged(FaceConn,dtype=int)
    
    if return_FaceConn and return_FaceElem:
        return Faces,FaceConn,FaceElem
    elif return_FaceConn:
        return Faces,FaceConn
    elif return_FaceElem:
        return Faces,FaceConn,FaceElem
    else:
        return Faces

def solid2edges(NodeCoords,NodeConn,ElemType='auto',ReturnType=list,return_EdgeConn=False,return_EdgeElem=False,):
    """
    Convert solid mesh to edges. The will be one edge for each edge of each element,
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
        edgIdx = np.where(Ls == 2)[0]
        triIdx = np.where(Ls == 3)[0]
        tetIdx = np.where(Ls == 4)[0]
        pyrIdx = np.where(Ls == 5)[0]
        wdgIdx = np.where(Ls == 6)[0]
        hexIdx = np.where(Ls == 8)[0]
        if len(edgIdx) > 0:
            edgs = np.array([NodeConn[i] for i in edgIdx]).astype(int)
        else:
            edgs = np.empty((0,2),dtype=int)
        tris = [NodeConn[i] for i in triIdx]
        tets = [NodeConn[i] for i in tetIdx]
        pyrs = [NodeConn[i] for i in pyrIdx]
        wdgs = [NodeConn[i] for i in wdgIdx]
        hexs = [NodeConn[i] for i in hexIdx]
        
        Edges = np.concatenate((edgs,tri2edges([],tris,ReturnType=np.ndarray), 
                                tet2edges([],tets,ReturnType=np.ndarray), 
                                pyramid2edges([],pyrs,ReturnType=np.ndarray), 
                                wedge2edges([],wdgs,ReturnType=np.ndarray), 
                                hex2edges([],hexs,ReturnType=np.ndarray)))
        if return_EdgeElem or return_EdgeConn:
            EdgeElem = np.concatenate((np.repeat(edgIdx,1),np.repeat(triIdx,3),np.repeat(tetIdx,6),np.repeat(pyrIdx,8),np.repeat(wdgIdx,9),np.repeat(hexIdx,12)))
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0]],len(edgIdx),axis=0).reshape(len(edgIdx)*1),
                np.repeat([[0,1,2]],len(triIdx),axis=0).reshape(len(triIdx)*3), 
                np.repeat([[0,1,2,3,4,5]],len(tetIdx),axis=0).reshape(len(tetIdx)*6),  
                np.repeat([[0,1,2,3,4,5,6,7]],len(pyrIdx),axis=0).reshape(len(pyrIdx)*8),                   
                np.repeat([[0,1,2,3,4,5,6,7,8]],len(wdgIdx),axis=0).reshape(len(wdgIdx)*9),   
                np.repeat([[0,1,2,3,4,5,6,7,8,9,10,11]],len(hexIdx),axis=0).reshape(len(hexIdx)*12),                    
                ))
            EdgeConn = -1*np.ones((len(NodeConn),12))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            EdgeConn = utils.ExtractRagged(EdgeConn,dtype=int)
        if ReturnType is list:
            Edges = Edges.tolist()
            if return_EdgeElem or return_EdgeConn:
                EdgeElem = EdgeElem.tolist()

    elif ElemType=='tri':
        Edges = tri2edges(NodeCoords,NodeConn,ReturnType=ReturnType)
        if return_EdgeElem or return_EdgeConn:
            triIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(triIdx,3)
            if ReturnType is list:
                EdgeElem = EdgeElem.tolist()
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2]],len(tetIdx),axis=0).reshape(len(tetIdx)*3), 
                ))
            EdgeConn = -1*np.ones((len(NodeConn),3))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            if ReturnType is list:
                EdgeElem = EdgeElem.tolist()
                EdgeConn = EdgeConn.tolist()
    elif ElemType=='quad':
        Edges = quad2edges(NodeCoords,NodeConn,ReturnType=ReturnType)
        if return_EdgeElem or return_EdgeConn:
            quadIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(quadIdx,4)
            if ReturnType is list:
                EdgeElem = EdgeElem.tolist()
                EdgeConn = EdgeConn.tolist()
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2]],len(tetIdx),axis=0).reshape(len(tetIdx)*3), 
                ))
            EdgeConn = -1*np.ones((len(NodeConn),4))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            if ReturnType is list:
                EdgeElem = EdgeElem.tolist()
                EdgeConn = EdgeConn.tolist()
    elif ElemType=='tet':
        Edges = tet2edges(NodeCoords,NodeConn,ReturnType=ReturnType)
        if return_EdgeElem or return_EdgeConn:
            tetIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(tetIdx,6)
            if ReturnType is list:
                EdgeElem = EdgeElem.tolist()
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5]],len(tetIdx),axis=0).reshape(len(tetIdx)*6),  
                ))
            EdgeConn = -1*np.ones((len(NodeConn),6))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            if ReturnType is list:
                EdgeElem = EdgeElem.tolist()
    elif ElemType=='pyramid':
        Edges = pyramid2edges(NodeCoords,NodeConn,ReturnType=ReturnType)
        if return_EdgeElem or return_EdgeConn:
            pyrIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(pyrIdx,8)
            if ReturnType is list:
                EdgeElem = EdgeElem.tolist()
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5,6,7]],len(pyrIdx),axis=0).reshape(len(pyrIdx)*8),                   
                ))
            EdgeConn = -1*np.ones((len(NodeConn),8))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            if ReturnType is list:
                EdgeElem = EdgeElem.tolist()
    elif ElemType=='wedge':
        Edges = wedge2edges(NodeCoords,NodeConn,ReturnType=ReturnType)
        if return_EdgeElem or return_EdgeConn:
            wdgIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(wdgIdx,9)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5,6,7,8]],len(wdgIdx),axis=0).reshape(len(wdgIdx)*9),   
                ))
            EdgeConn = -1*np.ones((len(NodeConn),9))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            if ReturnType is list:
                EdgeElem = EdgeElem.tolist()
    elif ElemType=='hex':
        Edges = hex2edges(NodeCoords,NodeConn,ReturnType=ReturnType)
        if return_EdgeElem or return_EdgeConn:
            hexIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(hexIdx,12)
            if ReturnType is list:
                EdgeElem = EdgeElem.tolist()
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5,6,7,8,9,10,11]],len(hexIdx),axis=0).reshape(len(hexIdx)*12),                    
                ))
            EdgeConn = -1*np.ones((len(NodeConn),12))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            if ReturnType is list:
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
    Returns edges grouped by the element from which they came.
    TODO: This can/should be rewritten based on solid2edges using EdgeConn

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
    Decompose all elements of a 3D volume mesh to tetrahedra.
    NOTE the generated tetrahedra will not necessarily be continuously oriented, i.e.
    edges of child tetrahedra may not be aligned between one parent element 
    and its neighbor, and thus the resulting mesh will typically be invalid.
    The primary use-case for this method is for methods like quality.Volume
    which utilize the geometric properties of tetrahedra to determine properties of 
    the parent elements.
    

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.
    return_ids : bool, optional
        Element IDs of the tets connected to the original elements, by default False

    Returns
    -------
    TetCoords : list
        Nodal coordinates of the generated tetrahedra. Depending on what elements are in the original mesh,
        new nodes may have been added.
    TetConn : list
        Nodal connectivity list of generated tetrahedra
    ElemIds : list, optional
        If return_ids = True, a list of the element ids of the new tetrahedra for each 
        of the original elements.
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


    if len(pyrIdx) > 0 or len(wdgIdx) > 0:
        hexmethod = '1to14'
    else:
        hexmethod = '1to6'
    TetCoords,fromhex = hex2tet(NodeCoords,hexs,method=hexmethod)
    TetCoords,fromwdg = wedge2tet(TetCoords,wdgs,method='1to20')
    TetCoords,frompyr = pyramid2tet(TetCoords,pyrs,method='1to4')
    TetConn = tets + frompyr + fromwdg + fromhex
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
        ElemIds = utils.ExtractRagged(ElemIds,dtype=int)
    
    if return_ids:
        return TetCoords, TetConn, ElemIds
    return TetCoords, TetConn

def hex2tet(NodeCoords,NodeConn,method='1to6'):
    """
    Decompose all elements of a 3D hexahedral mesh to tetrahedra.
    Generally solid2tets should be used rather than hex2tet directly

    NOTE the generated tetrahedra of the '1to5' method will not be continuously oriented, 
    i.e. edges of child tetrahedra may not be aligned between one parent element 
    and its neighbor, and thus the resulting mesh will typically be invalid.
    The primary use-case for this method is for methods like quality.Volume
    which utilize the geometric properties of tetrahedra to determine properties of 
    the parent elements. '1to6' or '1to24' will generate continuously oriented tetrahedra.
    

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list. All elements should be 8-Node hexahedral elements.
    method : str, optional
        Method of decomposition to use for tetrahedralization.
        '1to5'  - Not continuously oriented, no nodes added
        '1to6'  - Continuously oriented, no nodes added
        '1to24' - Continuously oriented, nodes added at center of element and element faces
        
    Returns
    -------
    NewCoords : list
        New list of nodal coordinates. For '1to5' or '1to6', this will be unchanged from
        the input.
    TetConn, list
        Nodal connectivity list of generated tetrahedra


    Examples
    --------
    >>> # A single hexahedral element
    >>> HexCoords, HexConn = primitives.Grid([0,1,0,1,0,1],1)
    >>> TetCoords1to5, TetConn1to5 = converter.hex2tet(HexCoords, HexConn, method='1to5')
    >>> TetCoords1to6, TetConn1to6 = converter.hex2tet(HexCoords, HexConn, method='1to6')
    >>> TetCoords1to24, TetConn1to24 = converter.hex2tet(HexCoords, HexConn, method='1to24')

    """
    if len(NodeConn) == 0:
        return NodeCoords, NodeConn

    if method == '1to5':
        ArrayConn = np.asarray(NodeConn)
        TetConn = -1*np.ones((len(NodeConn)*5,4))
        TetConn[0::5] = ArrayConn[:,[0,1,3,4]]
        TetConn[1::5] = ArrayConn[:,[1,2,3,6]]
        TetConn[2::5] = ArrayConn[:,[4,6,5,1]]
        TetConn[3::5] = ArrayConn[:,[4,7,6,3]]
        TetConn[4::5] = ArrayConn[:,[4,6,1,3]]
        TetConn = TetConn.astype(int).tolist()
        NewCoords = NodeCoords
    elif method == '1to6':
        ArrayConn = np.asarray(NodeConn)
        TetConn = -1*np.ones((len(NodeConn)*6,4))
        TetConn[0::6] = ArrayConn[:,[0,1,3,5]]
        TetConn[1::6] = ArrayConn[:,[5,2,3,6]]
        TetConn[2::6] = ArrayConn[:,[0,5,3,4]]
        TetConn[3::6] = ArrayConn[:,[3,7,4,5]]
        TetConn[4::6] = ArrayConn[:,[1,2,3,5]]
        TetConn[5::6] = ArrayConn[:,[5,7,6,3]]

        TetConn = TetConn.astype(int).tolist()
        NewCoords = NodeCoords
    elif method == '1to24':
        ArrayCoords = np.asarray(NodeCoords)
        ArrayConn = np.asarray(NodeConn)

        Centroids = utils.Centroids(ArrayCoords,NodeConn)
        Face0Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,1,2,3]]],axis=1)
        Face1Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,1,5,4]]],axis=1)
        Face2Centroids = np.mean(ArrayCoords[ArrayConn[:,[1,2,6,5]]],axis=1)
        Face3Centroids = np.mean(ArrayCoords[ArrayConn[:,[2,3,7,6]]],axis=1)
        Face4Centroids = np.mean(ArrayCoords[ArrayConn[:,[3,0,4,7]]],axis=1)
        Face5Centroids = np.mean(ArrayCoords[ArrayConn[:,[4,5,6,7]]],axis=1)

        CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*0,len(NodeCoords)+len(NodeConn)*1)
        Face0CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*1,len(NodeCoords)+len(NodeConn)*2)
        Face1CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*2,len(NodeCoords)+len(NodeConn)*3)
        Face2CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*3,len(NodeCoords)+len(NodeConn)*4)
        Face3CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*4,len(NodeCoords)+len(NodeConn)*5)
        Face4CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*5,len(NodeCoords)+len(NodeConn)*6)
        Face5CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*6,len(NodeCoords)+len(NodeConn)*7)
        
        NewCoords = np.vstack([ArrayCoords,Centroids,Face0Centroids,Face1Centroids,Face2Centroids,Face3Centroids,Face4Centroids,Face5Centroids])        
        
        TetConn = -1*np.ones((len(NodeConn)*24,4))
        TetConn[0::24] = np.hstack([ArrayConn[:,[0,1]],Face0CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[1::24] = np.hstack([ArrayConn[:,[1,2]],Face0CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[2::24] = np.hstack([ArrayConn[:,[2,3]],Face0CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[3::24] = np.hstack([ArrayConn[:,[3,0]],Face0CentroidIds[:,None],CentroidIds[:,None]])

        TetConn[4::24] = np.hstack([ArrayConn[:,[1,0]],Face1CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[5::24] = np.hstack([ArrayConn[:,[5,1]],Face1CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[6::24] = np.hstack([ArrayConn[:,[4,5]],Face1CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[7::24] = np.hstack([ArrayConn[:,[0,4]],Face1CentroidIds[:,None],CentroidIds[:,None]])

        TetConn[8::24] = np.hstack([ArrayConn[:,[2,1]],Face2CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[9::24] = np.hstack([ArrayConn[:,[6,2]],Face2CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[10::24] = np.hstack([ArrayConn[:,[5,6]],Face2CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[11::24] = np.hstack([ArrayConn[:,[1,5]],Face2CentroidIds[:,None],CentroidIds[:,None]])

        TetConn[12::24] = np.hstack([ArrayConn[:,[3,2]],Face3CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[13::24] = np.hstack([ArrayConn[:,[7,3]],Face3CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[14::24] = np.hstack([ArrayConn[:,[6,7]],Face3CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[15::24] = np.hstack([ArrayConn[:,[2,6]],Face3CentroidIds[:,None],CentroidIds[:,None]])

        TetConn[16::24] = np.hstack([ArrayConn[:,[0,3]],Face4CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[17::24] = np.hstack([ArrayConn[:,[4,0]],Face4CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[18::24] = np.hstack([ArrayConn[:,[7,4]],Face4CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[19::24] = np.hstack([ArrayConn[:,[3,7]],Face4CentroidIds[:,None],CentroidIds[:,None]])

        TetConn[20::24] = np.hstack([ArrayConn[:,[5,4]],Face5CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[21::24] = np.hstack([ArrayConn[:,[6,5]],Face5CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[22::24] = np.hstack([ArrayConn[:,[7,6]],Face5CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[23::24] = np.hstack([ArrayConn[:,[4,7]],Face5CentroidIds[:,None],CentroidIds[:,None]])
    

    return NewCoords, TetConn

def wedge2tet(NodeCoords, NodeConn, method='1to3'):
    """
    Decompose all elements of a 3D wedge-element mesh to tetrahedra.
    Generally solid2tets should be used rather than wedge2tet directly
    NOTE the generated tetrahedra of the '1to3' method will not be continuously oriented, 
    i.e. edges of child tetrahedra may not be aligned between one parent element 
    and its neighbor, and thus the resulting mesh will typically be invalid.
    The primary use-case for this method is for methods like quality.Volume
    which utilize the geometric properties of tetrahedra to determine properties of 
    the parent elements. '1to14' or '1to36' will generate continuously oriented tetrahedra.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list. All elements should be 6-Node wedge elements.
    method : str, optional
        Method of decomposition to use for tetrahedralization.
        '1to3'  - Not continuously oriented, no nodes added
        '1to14' - Continuously oriented, nodes added at center of element and element faces
        '1to36' - Continuously oriented, nodes added at center of element, element faces, and element edges

    Returns
    -------
    TetConn, list
        Nodal connectivity list of generated tetrahedra

    Examples
    --------
    >>> # A single wedge element
    >>> WedgeCoords = np.array([[0., 0., 0.],
                                [1., 0., 0.],
                                [0.5, 1., 0.],
                                [0., 0., 1.],
                                [1., 0., 1.],
                                [0.5, 1., 1.]])
    >>> WedgeConn = [[0,1,2,3,4,5]]
    >>> TetCoords1to3, TetConn1to3 = converter.wedge2tet(WedgeCoords, WedgeConn, method='1to3')
    >>> TetCoords1to14, TetConn1to14 = converter.wedge2tet(WedgeCoords, WedgeConn, method='1to14')
    >>> TetCoords1to36, TetConn1to36 = converter.wedge2tet(WedgeCoords, WedgeConn, method='1to36')
    """
    if len(NodeConn) == 0:
        return NodeCoords, NodeConn

    ArrayConn = np.asarray(NodeConn)
    if method == '1to3':
        TetConn = -1*np.ones((len(NodeConn)*3,4))
        TetConn[0::3] = ArrayConn[:,[0,1,2,3]]
        TetConn[1::3] = ArrayConn[:,[1,2,3,4]]
        TetConn[2::3] = ArrayConn[:,[4,5,2,3]]
        TetConn = TetConn.astype(int).tolist()
        NewCoords = NodeCoords

    elif method == '1to14':
        
        ArrayCoords = np.asarray(NodeCoords)
        ArrayConn = np.asarray(NodeConn)

        Centroids = utils.Centroids(ArrayCoords,NodeConn)
        Face0Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,1,4,3]]],axis=1)
        Face1Centroids = np.mean(ArrayCoords[ArrayConn[:,[1,4,5,2]]],axis=1)
        Face2Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,2,5,3]]],axis=1)


        CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*0,len(NodeCoords)+len(NodeConn)*1)
        Face0CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*1,len(NodeCoords)+len(NodeConn)*2)
        Face1CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*2,len(NodeCoords)+len(NodeConn)*3)
        Face2CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*3,len(NodeCoords)+len(NodeConn)*4)
        
        NewCoords = np.vstack([ArrayCoords,Centroids,Face0Centroids,Face1Centroids,Face2Centroids])        
        
        TetConn = 0*np.ones((len(NodeConn)*14,4))
        TetConn[0::14] = np.hstack([ArrayConn[:,[0,1,2]],CentroidIds[:,None]])
        TetConn[1::14] = np.hstack([ArrayConn[:,[5,4,3]],CentroidIds[:,None]])

        TetConn[2::14] = np.hstack([ArrayConn[:,[0,1]],CentroidIds[:,None],Face0CentroidIds[:,None]])
        TetConn[3::14] = np.hstack([ArrayConn[:,[1,2]],CentroidIds[:,None],Face1CentroidIds[:,None]])
        TetConn[4::14] = np.hstack([ArrayConn[:,[2,0]],CentroidIds[:,None],Face2CentroidIds[:,None]])

        TetConn[5::14] = np.hstack([ArrayConn[:,[4,3]],CentroidIds[:,None],Face0CentroidIds[:,None]])
        TetConn[6::14] = np.hstack([ArrayConn[:,[5,4]],CentroidIds[:,None],Face1CentroidIds[:,None]])
        TetConn[7::14] = np.hstack([ArrayConn[:,[3,5]],CentroidIds[:,None],Face2CentroidIds[:,None]])

        TetConn[8::14] = np.hstack([ArrayConn[:,[0,3]],Face0CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[9::14] = np.hstack([ArrayConn[:,[0,3]],CentroidIds[:,None],Face2CentroidIds[:,None]])
        TetConn[10::14] = np.hstack([ArrayConn[:,[2,5]],Face2CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[11::14] = np.hstack([ArrayConn[:,[2,5]],CentroidIds[:,None],Face1CentroidIds[:,None]])
        TetConn[12::14] = np.hstack([ArrayConn[:,[1,4]],Face1CentroidIds[:,None],CentroidIds[:,None]])
        TetConn[13::14] = np.hstack([ArrayConn[:,[1,4]],CentroidIds[:,None],Face0CentroidIds[:,None]])


        TetConn = TetConn.astype(int).tolist()

    elif method == '1to36':
        
        ArrayCoords = np.asarray(NodeCoords)
        ArrayConn = np.asarray(NodeConn)

        Centroids = utils.Centroids(ArrayCoords,NodeConn)
        Face0Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,1,4,3]]],axis=1)
        Face1Centroids = np.mean(ArrayCoords[ArrayConn[:,[1,4,5,2]]],axis=1)
        Face2Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,2,5,3]]],axis=1)
        Face3Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,1,2]]],axis=1)
        Face4Centroids = np.mean(ArrayCoords[ArrayConn[:,[3,4,5]]],axis=1)

        Edge0Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,3]]],axis=1)
        Edge1Centroids = np.mean(ArrayCoords[ArrayConn[:,[1,4]]],axis=1)
        Edge2Centroids = np.mean(ArrayCoords[ArrayConn[:,[2,5]]],axis=1)
        Edge3Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,1]]],axis=1)
        Edge4Centroids = np.mean(ArrayCoords[ArrayConn[:,[1,2]]],axis=1)
        Edge5Centroids = np.mean(ArrayCoords[ArrayConn[:,[2,0]]],axis=1)
        Edge6Centroids = np.mean(ArrayCoords[ArrayConn[:,[3,4]]],axis=1)
        Edge7Centroids = np.mean(ArrayCoords[ArrayConn[:,[4,5]]],axis=1)
        Edge8Centroids = np.mean(ArrayCoords[ArrayConn[:,[5,3]]],axis=1)

        CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*0,len(NodeCoords)+len(NodeConn)*1)
        Face0CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*1,len(NodeCoords)+len(NodeConn)*2)
        Face1CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*2,len(NodeCoords)+len(NodeConn)*3)
        Face2CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*3,len(NodeCoords)+len(NodeConn)*4)
        Face3CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*4,len(NodeCoords)+len(NodeConn)*5)
        Face4CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*5,len(NodeCoords)+len(NodeConn)*6)
        Edge0CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*6,len(NodeCoords)+len(NodeConn)*7)
        Edge1CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*7,len(NodeCoords)+len(NodeConn)*8)
        Edge2CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*8,len(NodeCoords)+len(NodeConn)*9)
        Edge3CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*9,len(NodeCoords)+len(NodeConn)*10)
        Edge4CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*10,len(NodeCoords)+len(NodeConn)*11)
        Edge5CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*11,len(NodeCoords)+len(NodeConn)*12)
        Edge6CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*12,len(NodeCoords)+len(NodeConn)*13)
        Edge7CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*13,len(NodeCoords)+len(NodeConn)*14)
        Edge8CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*14,len(NodeCoords)+len(NodeConn)*15)
        
        NewCoords = np.vstack([ArrayCoords,Centroids,Face0Centroids,Face1Centroids,Face2Centroids,Face3Centroids,
        Face4Centroids,Edge0Centroids,Edge1Centroids,Edge2Centroids,Edge3Centroids,Edge4Centroids,Edge5Centroids, 
        Edge6Centroids,Edge7Centroids,Edge8Centroids])        
        
        TetConn = 0*np.ones((len(NodeConn)*36,4))

        TetConn[0::36] = np.hstack([ArrayConn[:,[0]],Edge3CentroidIds[:,None],CentroidIds[:,None],Face0CentroidIds[:,None]])
        TetConn[1::36] = np.hstack([ArrayConn[:,[1]],CentroidIds[:,None],Edge3CentroidIds[:,None],Face0CentroidIds[:,None]])
        TetConn[2::36] = np.hstack([ArrayConn[:,[1]],Edge4CentroidIds[:,None],CentroidIds[:,None],Face1CentroidIds[:,None]])
        TetConn[3::36] = np.hstack([ArrayConn[:,[2]],CentroidIds[:,None],Edge4CentroidIds[:,None],Face1CentroidIds[:,None]])
        TetConn[4::36] = np.hstack([ArrayConn[:,[2]],Edge5CentroidIds[:,None],CentroidIds[:,None],Face2CentroidIds[:,None]])
        TetConn[5::36] = np.hstack([ArrayConn[:,[0]],CentroidIds[:,None],Edge5CentroidIds[:,None],Face2CentroidIds[:,None]])

        TetConn[6::36] = np.hstack([ArrayConn[:,[3]],CentroidIds[:,None],Edge6CentroidIds[:,None],Face0CentroidIds[:,None]])
        TetConn[7::36] = np.hstack([ArrayConn[:,[4]],Edge6CentroidIds[:,None],CentroidIds[:,None],Face0CentroidIds[:,None]])
        TetConn[8::36] = np.hstack([ArrayConn[:,[4]],CentroidIds[:,None],Edge7CentroidIds[:,None],Face1CentroidIds[:,None]])
        TetConn[9::36] = np.hstack([ArrayConn[:,[5]],Edge7CentroidIds[:,None],CentroidIds[:,None],Face1CentroidIds[:,None]])
        TetConn[10::36] = np.hstack([ArrayConn[:,[5]],CentroidIds[:,None],Edge8CentroidIds[:,None],Face2CentroidIds[:,None]])
        TetConn[11::36] = np.hstack([ArrayConn[:,[3]],Edge8CentroidIds[:,None],CentroidIds[:,None],Face2CentroidIds[:,None]])

        TetConn[12::36] = np.hstack([Face0CentroidIds[:,None],Edge0CentroidIds[:,None],CentroidIds[:,None],ArrayConn[:,[0]]])
        TetConn[13::36] = np.hstack([Edge0CentroidIds[:,None],Face2CentroidIds[:,None],CentroidIds[:,None],ArrayConn[:,[0]]])
        TetConn[14::36] = np.hstack([Face1CentroidIds[:,None],Edge1CentroidIds[:,None],CentroidIds[:,None],ArrayConn[:,[1]]])
        TetConn[15::36] = np.hstack([Edge1CentroidIds[:,None],Face0CentroidIds[:,None],CentroidIds[:,None],ArrayConn[:,[1]]])
        TetConn[16::36] = np.hstack([Face2CentroidIds[:,None],Edge2CentroidIds[:,None],CentroidIds[:,None],ArrayConn[:,[2]]])
        TetConn[17::36] = np.hstack([Edge2CentroidIds[:,None],Face1CentroidIds[:,None],CentroidIds[:,None],ArrayConn[:,[2]]])

        TetConn[18::36] = np.hstack([CentroidIds[:,None],Edge0CentroidIds[:,None],Face0CentroidIds[:,None],ArrayConn[:,[3]]])
        TetConn[19::36] = np.hstack([CentroidIds[:,None],Face2CentroidIds[:,None],Edge0CentroidIds[:,None],ArrayConn[:,[3]]])
        TetConn[20::36] = np.hstack([CentroidIds[:,None],Edge1CentroidIds[:,None],Face1CentroidIds[:,None],ArrayConn[:,[4]]])
        TetConn[21::36] = np.hstack([CentroidIds[:,None],Face0CentroidIds[:,None],Edge1CentroidIds[:,None],ArrayConn[:,[4]]])
        TetConn[22::36] = np.hstack([CentroidIds[:,None],Edge2CentroidIds[:,None],Face2CentroidIds[:,None],ArrayConn[:,[5]]])
        TetConn[23::36] = np.hstack([CentroidIds[:,None],Face1CentroidIds[:,None],Edge2CentroidIds[:,None],ArrayConn[:,[5]]])

        TetConn[24::36] = np.hstack([CentroidIds[:,None],Edge3CentroidIds[:,None],ArrayConn[:,[0]],Face3CentroidIds[:,None]])
        TetConn[25::36] = np.hstack([CentroidIds[:,None],ArrayConn[:,[0]],Edge5CentroidIds[:,None],Face3CentroidIds[:,None]])
        TetConn[26::36] = np.hstack([CentroidIds[:,None],Edge4CentroidIds[:,None],ArrayConn[:,[1]],Face3CentroidIds[:,None]])
        TetConn[27::36] = np.hstack([CentroidIds[:,None],ArrayConn[:,[1]],Edge3CentroidIds[:,None],Face3CentroidIds[:,None]])
        TetConn[28::36] = np.hstack([CentroidIds[:,None],Edge5CentroidIds[:,None],ArrayConn[:,[2]],Face3CentroidIds[:,None]])
        TetConn[29::36] = np.hstack([CentroidIds[:,None],ArrayConn[:,[2]],Edge4CentroidIds[:,None],Face3CentroidIds[:,None]])

        TetConn[30::36] = np.hstack([CentroidIds[:,None],ArrayConn[:,[3]],Edge6CentroidIds[:,None],Face4CentroidIds[:,None]])
        TetConn[31::36] = np.hstack([CentroidIds[:,None],Edge8CentroidIds[:,None],ArrayConn[:,[3]],Face4CentroidIds[:,None]])
        TetConn[32::36] = np.hstack([CentroidIds[:,None],ArrayConn[:,[4]],Edge7CentroidIds[:,None],Face4CentroidIds[:,None]])
        TetConn[33::36] = np.hstack([CentroidIds[:,None],Edge6CentroidIds[:,None],ArrayConn[:,[4]],Face4CentroidIds[:,None]])
        TetConn[34::36] = np.hstack([CentroidIds[:,None],ArrayConn[:,[5]],Edge8CentroidIds[:,None],Face4CentroidIds[:,None]])
        TetConn[35::36] = np.hstack([CentroidIds[:,None],Edge7CentroidIds[:,None],ArrayConn[:,[5]],Face4CentroidIds[:,None]])

        TetConn = TetConn.astype(int).tolist()

    return NewCoords, TetConn

def pyramid2tet(NodeCoords,NodeConn, method='1to2'):
    """
    Decompose all elements of a 3D pyramidal mesh to tetrahedra.
    Generally solid2tets should be used rather than pyramid2tet directly
    NOTE the generated tetrahedra from 1 to 2 will not necessarily be continuously 
    oriented, i.e. edges of child tetrahedra may not be aligned between one 
    parent element and its neighbor, and thus the resulting mesh will typically be invalid.
    1 to 4 splitting is guaranteed to be consistent with other pyramids as well as as hexs
    split with 1to24 and wedges split with 1to20.
    

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

    Examples
    --------
    >>> # A single pyramid element
    >>> PyramidCoords = np.array([[0., 0., 0.],
                                [1., 0., 0.],
                                [1., 1., 0.],
                                [0., 1., 0.],
                                [0.5, 0.5, 1.]])
    >>> PyramidConn = [[0,1,2,3,4]]
    >>> TetCoords1to2, TetConn1to2 = converter.pyramid2tet(PyramidCoords, PyramidConn, method='1to2')
    >>> TetCoords1to4, TetConn1to4 = converter.pyramid2tet(PyramidCoords, PyramidConn, method='1to4')
    """
    if len(NodeConn) == 0:
        return NodeCoords, NodeConn
    
    if method == '1to2':
        ArrayConn = np.asarray(NodeConn)
        TetConn = -1*np.ones((len(NodeConn)*2,4))
        TetConn[0::2] = ArrayConn[:,[0,1,2,4]]
        TetConn[1::2] = ArrayConn[:,[0,2,3,4]]
        TetConn = TetConn.astype(int).tolist()
        NewCoords = NodeCoords
    
    elif method == '1to4':
        
        ArrayCoords = np.asarray(NodeCoords)
        ArrayConn = np.asarray(NodeConn)

        Face0Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,1,2,3]]],axis=1)
        Face0CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*0,len(NodeCoords)+len(NodeConn)*1)
        
        NewCoords = np.vstack([ArrayCoords,Face0Centroids])        
        
        TetConn = -1*np.ones((len(NodeConn)*4,4))
        TetConn[0::4] = np.hstack([Face0CentroidIds[:,None],ArrayConn[:,[0,1,4]]])
        TetConn[1::4] = np.hstack([Face0CentroidIds[:,None],ArrayConn[:,[1,2,4]]])
        TetConn[2::4] = np.hstack([Face0CentroidIds[:,None],ArrayConn[:,[2,3,4]]])
        TetConn[3::4] = np.hstack([Face0CentroidIds[:,None],ArrayConn[:,[3,0,4]]])

        TetConn = TetConn.astype(int).tolist()

    return NewCoords, TetConn

def faces2surface(Faces):
    """
    Identify surface elements, i.e. faces that aren't shared between two elements

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
    if len(Faces) == len(SurfConn):
        SurfConn=Faces
    return SurfConn

def faces2unique(Faces,return_idx=False,return_inv=False):
    """
    Reduce set of mesh faces to contain only unique faces, i.e. there will only
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
    Rfaces = utils.PadRagged(Faces)
    # Get all unique element faces (accounting for flipped versions of faces)
    _,idx,inv = np.unique(np.sort(Rfaces,axis=1),axis=0,return_index=True,return_inverse=True)
    RFaces = Rfaces[idx]
    UFaces = utils.ExtractRagged(RFaces,dtype=int)
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
        FaceElem transformed to correspond with UFaces rather than Faces.
    idx : np.ndarray, optional
        The array of indices that relate the unique list of
        faces to the list of original faces (see faces2unique).
    inv : np.ndarray, optional
        The array of indices that relate the unique list of
        faces to the list of original faces (see faces2unique).
    """    
    # 
    UFaces,idx,inv = faces2unique(Faces,return_idx=True,return_inv=True)

    UFaces = utils.PadRagged(Faces)[idx]
    UFaceElem = np.asarray(FaceElem)[idx]
    UFaces = np.append(UFaces, np.repeat(-1,UFaces.shape[1])[None,:],axis=0)
    inv = np.append(inv,-1)
    UFaceConn = inv[utils.PadRagged(FaceConn)] # Faces attached to each element
    # Face-Element Connectivity
    FaceElemConn = np.nan*(np.ones((len(UFaces),2))) # Elements attached to each face

    FaceElemConn = np.nan*(np.ones((len(UFaces),2))) # Elements attached to each face
    FECidx = (UFaceElem[UFaceConn] == np.repeat(np.arange(len(FaceConn))[:,None],UFaceConn.shape[1],axis=1)).astype(int)
    FaceElemConn[UFaceConn,FECidx] = np.repeat(np.arange(len(FaceConn))[:,None],UFaceConn.shape[1],axis=1)
    FaceElemConn = [[int(x) if not np.isnan(x) else x for x in y] for y in FaceElemConn[:-1]]

    if return_UniqueFaceInfo:
        UFaces = utils.ExtractRagged(UFaces)[:-1]
        UFaceConn = utils.ExtractRagged(UFaceConn)
        return FaceElemConn, UFaces, UFaceConn, UFaceElem, idx, inv[:-1]

    return FaceElemConn

def edges2unique(Edges,return_idx=False,return_inv=False,return_counts=False):
    """
    Reduce set of mesh edges to contain only unique edges, i.e. there will only
    be one entry to indicate a edge shared between multiple elements.

    Parameters
    ----------
    Edges : list
        Nodal connectivity of mesh edges (as obtained by solid2edges).
    return_idx : bool, optional
        If true, will return idx, the array of indices that relate the original list of
        edges to the list of unique edges (UEdges = Edges[idx]), by default False.
        See numpy.unique for additional details.
    return_inv : bool, optional
        If true, will return inv, the array of indices that relate the unique list of
        edges to the list of original edges (Edges = UEdges[inv]), by default False
        See numpy.unique for additional details.

    Returns
    -------
    UEdges : list
        Nodal connectivity of unique mesh edges.
    idx : np.ndarray, optional
        The array of indices that relate the unique list of
        faces to the list of original edges.
    inv : np.ndarray, optional
        The array of indices that relate the unique list of
        edges to the list of original edges.
    """
    # Returns only the unique edges (not duplicated for each element)
    # Get all unique element edges (accounting for flipped versions of edges)
    if len(Edges) > 0:
        _,idx,inv,counts = np.unique(np.sort(Edges,axis=1),axis=0,return_index=True,return_inverse=True,return_counts=True)
        UEdges = np.asarray(Edges)[idx]
    else:
        UEdges = np.array([])
        idx = np.array([])
        inv = np.array([])
        counts = np.array([])

    check = [True,return_idx,return_inv,return_counts]
    out = [o for i,o in enumerate([UEdges, idx, inv, counts]) if check[i]]

    return out

def tet2faces(NodeCoords,NodeConn):
    """
    Extract triangular faces from all elements of a purely 4-Node tetrahedral mesh.
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
            Faces[0::4] = ArrayConn[:,[0,2,1]]
            Faces[1::4] = ArrayConn[:,[0,1,3]]
            Faces[2::4] = ArrayConn[:,[1,2,3]]
            Faces[3::4] = ArrayConn[:,[0,3,2]]
            Faces = Faces.astype(int).tolist()
        elif len(NodeConn[0]) == 10:
            ArrayConn = np.asarray(NodeConn)
            Faces = -1*np.ones((len(NodeConn)*4,6))
            Faces[0::4] = ArrayConn[:,[0,2,1,6,5,4]]
            Faces[1::4] = ArrayConn[:,[0,1,3,4,8,7]]
            Faces[2::4] = ArrayConn[:,[1,2,3,5,9,8]]
            Faces[3::4] = ArrayConn[:,[0,3,2,7,9,6]]
            Faces = Faces.astype(int).tolist()
        else:
            raise Exception('Must be 4 or 10 node tetrahedral mesh.')
    else:
        Faces = []

    return Faces

def hex2faces(NodeCoords,NodeConn):
    """
    Extract quadrilateral faces from all elements of a purely 8-Node hexahedral mesh.
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
        Faces[0::6] = ArrayConn[:,[0,3,2,1]]
        Faces[1::6] = ArrayConn[:,[0,1,5,4]]
        Faces[2::6] = ArrayConn[:,[1,2,6,5]]
        Faces[3::6] = ArrayConn[:,[2,3,7,6]]
        Faces[4::6] = ArrayConn[:,[3,0,4,7]]
        Faces[5::6] = ArrayConn[:,[4,5,6,7]]
        Faces = Faces.astype(int).tolist()
    else:
        Faces = []
    return Faces

def pyramid2faces(NodeCoords,NodeConn):
    """
    Extract triangular and quadrilateral faces from all elements of a 
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
        Faces[0::5] = ArrayConn[:,[0,3,2,1]]
        Faces[1::5,:3] = ArrayConn[:,[0,1,4]]
        Faces[2::5,:3] = ArrayConn[:,[1,2,4]]
        Faces[3::5,:3] = ArrayConn[:,[2,3,4]]
        Faces[4::5,:3] = ArrayConn[:,[3,0,4]]
        Faces = utils.ExtractRagged(Faces,delval=-1,dtype=int)
    else:
        Faces = []
    return Faces
        
def wedge2faces(NodeCoords,NodeConn):
    """
    Extract triangular and quadrilateral faces from all elements of a purely 
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
        Faces[0::5,:3] = ArrayConn[:,[2,1,0]]
        Faces[1::5] = ArrayConn[:,[0,1,4,3]]
        Faces[2::5] = ArrayConn[:,[1,2,5,4]]
        Faces[3::5] = ArrayConn[:,[2,0,3,5]]
        Faces[4::5,:3] = ArrayConn[:,[3,4,5]]
        Faces = utils.ExtractRagged(Faces,delval=-1,dtype=int)
    else:
        Faces = []
    return Faces

def tri2edges(NodeCoords,NodeConn,ReturnType=list):
    """
    Extract edges from all elements of a purely 3-Node triangular mesh.
    Best practice is to use solid2edges, rather than using tri2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.
    ReturnType : type, optional.
        Specifies the format of the returned list-like object.
        Can be list or np.ndarray. Default is list.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """
    # Note that some code relies on these edges being in the order that they're currently in
    
    # Explode surface elements into edges
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*3,2))
        Edges[0::3] = ArrayConn[:,[0,1]]
        Edges[1::3] = ArrayConn[:,[1,2]]
        Edges[2::3] = ArrayConn[:,[2,0]]
        Edges = Edges.astype(int)
        if ReturnType is list:
            Edges = Edges.tolist()
    else:
        if ReturnType is list:
            Edges = []
        else:
            Edges = np.empty((0,2),dtype=int)
    
    return Edges

def quad2edges(NodeCoords,NodeConn,ReturnType=list):
    """
    Extract edges from all elements of a purely 4-Node quadrilateral mesh.
    Best practice is to use solid2edges, rather than using quad2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.
    ReturnType : type, optional.
        Specifies the format of the returned list-like object.
        Can be list or np.ndarray. Default is list.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*4,2))
        Edges[0::4] = ArrayConn[:,[0,1]]
        Edges[1::4] = ArrayConn[:,[1,2]]
        Edges[2::4] = ArrayConn[:,[2,3]]
        Edges[3::4] = ArrayConn[:,[3,0]]

        Edges = Edges.astype(int)
        if ReturnType is list:
            Edges = Edges.tolist()
    else:
        if ReturnType is list:
            Edges = []
        else:
            Edges = np.empty((0,2),dtype=int)
    
    return Edges

def polygon2edges(NodeCoords,NodeConn,ReturnType=list):
    """
    Extract edges from all elements of a polygonal mesh.
    Best practice is to use solid2edges, rather than using polygon2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.
    ReturnType : type, optional.
        Specifies the format of the returned list-like object.
        Can be list or np.ndarray. Default is list.

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

def tet2edges(NodeCoords,NodeConn,ReturnType=list):
    """
    Extract edges from all elements of a purely 4-Node tetrahedral mesh.
    Best practice is to use solid2edges, rather than using tet2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.
    ReturnType : type, optional.
        Specifies the format of the returned list-like object.
        Can be list or np.ndarray. Default is list.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """

    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*6,2))
        Edges[0::6] = ArrayConn[:,[0,1]]
        Edges[1::6] = ArrayConn[:,[1,2]]
        Edges[2::6] = ArrayConn[:,[2,0]]
        Edges[3::6] = ArrayConn[:,[0,3]]
        Edges[4::6] = ArrayConn[:,[1,3]]
        Edges[5::6] = ArrayConn[:,[2,3]]
        Edges = Edges.astype(int)
        if ReturnType is list:
            Edges = Edges.tolist()
    else:
        if ReturnType is list:
            Edges = []
        else:
            Edges = np.empty((0,2),dtype=int)
    return Edges

def pyramid2edges(NodeCoords,NodeConn,ReturnType=list):
    """
    Extract edges from all elements of a purely 5-Node pyramidal mesh.
    Best practice is to use solid2edges, rather than using pyramid2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.
    ReturnType : type, optional.
        Specifies the format of the returned list-like object.
        Can be list or np.ndarray. Default is list.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*8,2))
        Edges[0::8] = ArrayConn[:,[0,1]]
        Edges[1::8] = ArrayConn[:,[1,2]]
        Edges[2::8] = ArrayConn[:,[2,3]]
        Edges[3::8] = ArrayConn[:,[3,0]]
        Edges[4::8] = ArrayConn[:,[0,4]]
        Edges[5::8] = ArrayConn[:,[1,4]]
        Edges[6::8] = ArrayConn[:,[2,4]]
        Edges[7::8] = ArrayConn[:,[3,4]]
        Edges = Edges.astype(int)
        if ReturnType is list:
            Edges = Edges.tolist()
    else:
        if ReturnType is list:
            Edges = []
        else:
            Edges = np.empty((0,2),dtype=int)
    return Edges

def wedge2edges(NodeCoords,NodeConn,ReturnType=list):
    """
    Extract edges from all elements of a purely 6-Node wedge element mesh.
    Best practice is to use solid2edges, rather than using wedge2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.
    ReturnType : type, optional.
        Specifies the format of the returned list-like object.
        Can be list or np.ndarray. Default is list.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*9,2))
        Edges[0::9] = ArrayConn[:,[0,1]]
        Edges[1::9] = ArrayConn[:,[1,2]]
        Edges[2::9] = ArrayConn[:,[2,0]]
        Edges[3::9] = ArrayConn[:,[0,3]]
        Edges[4::9] = ArrayConn[:,[1,4]]
        Edges[5::9] = ArrayConn[:,[2,5]]
        Edges[6::9] = ArrayConn[:,[3,4]]
        Edges[7::9] = ArrayConn[:,[4,5]]
        Edges[8::9] = ArrayConn[:,[5,3]]
        Edges = Edges.astype(int)
        if ReturnType is list:
            Edges = Edges.tolist()
    else:
        if ReturnType is list:
            Edges = []
        else:
            Edges = np.empty((0,2),dtype=int)
    return Edges

def hex2edges(NodeCoords,NodeConn,ReturnType=list):
    """
    Extract edges from all elements of a purely 8-Node hexahedral mesh.
    Best practice is to use solid2edges, rather than using hex2edges directly.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        List of nodal connectivity.
    ReturnType : type, optional.
        Specifies the format of the returned list-like object.
        Can be list or np.ndarray. Default is list.

    Returns
    -------
    Edges : list
        List of nodal connectivity of the mesh edges.
    """
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*12,2))
        Edges[0::12] = ArrayConn[:,[0,1]]
        Edges[1::12] = ArrayConn[:,[1,2]]
        Edges[2::12] = ArrayConn[:,[2,3]]
        Edges[3::12] = ArrayConn[:,[3,0]]
        Edges[4::12] = ArrayConn[:,[0,4]]
        Edges[5::12] = ArrayConn[:,[1,5]]
        Edges[6::12] = ArrayConn[:,[2,6]]
        Edges[7::12] = ArrayConn[:,[3,7]]
        Edges[8::12] = ArrayConn[:,[4,5]]
        Edges[9::12] = ArrayConn[:,[5,6]]
        Edges[10::12] = ArrayConn[:,[6,7]]
        Edges[11::12] = ArrayConn[:,[7,4]]
        Edges = Edges.astype(int)
        if ReturnType is list:
            Edges = Edges.tolist()
    else:
        if ReturnType is list:
            Edges = []
        else:
            Edges = np.empty((0,2),dtype=int)
    return Edges

def quad2tri(QuadNodeConn):
    """
    Converts a quadrilateral mesh to a triangular mesh by splitting each quad into 2 tris  

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
    Converts a 10 node tetradehdral mesh to a 4 node tetradehedral mesh.
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
     
def surf2edges(NodeCoords,NodeConn,ElemType='auto'):
    """
    Extract the edges of an unclosed surface mesh.
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

    edges = solid2edges(NodeCoords, NodeConn, ElemType=ElemType)
    UEdges, indices, counts = edges2unique(edges, return_idx=True, return_counts=True)

    EdgeIdx = indices[np.where(counts==1)]
    Edges = np.asarray(edges)[EdgeIdx]

    return Edges

def im2voxel(img, voxelsize, scalefactor=1, scaleorder=1, return_nodedata=False, return_gradient=False, gaussian_sigma=1, threshold=None, crop=None, threshold_direction=1):
    """
    Convert 3D image data to a cubic mesh. Each voxel will be represented by an element.

    Parameters
    ----------
    img : str or np.ndarray
        If a str, should be the directory to an image stack of tiff or dicom files.
        If an array, shoud be a 3D array of image data.
    voxelsize : float, or tuple
        Size of voxel (based on image resolution).
        If a tuple, should be specified as (hx,hy,hz)
    scalefactor : float, optional
        Scale factor for resampling the image. If greater than 1, there will be more than
        1 elements per voxel. If less than 1, will coarsen the image, by default 1.
    scaleorder : int, optional
        Interpolation order for scaling the image (see scipy.ndimage.zoom), by default 1.
        Must be 0-5.
    threshold : float, optional
        Voxel intensity threshold, by default None.
        If given, elements with all nodes less than threshold will be discarded.

    Returns
    -------
    VoxelCoords : list
        Node coordinates for the voxel mesh
    VoxelConn : list
        Node connectivity for the voxel mesh
    VoxelData : numpy.ndarray
        Image intensity data for each voxel.
    NodeData : numpy.ndarray
        Image intensity data for each node, averaged from connected voxels.

    """    
    multichannel = False

    if type(voxelsize) is list or type(voxelsize) is tuple:
        assert len(voxelsize) == 3, 'If specified as a list or tuple, voxelsize must have a length of 3.'
        xscale = voxelsize[0]
        yscale = voxelsize[1]
        zscale = voxelsize[2]
        voxelsize = 1
        rectangular_elements=True
    else:
        rectangular_elements=False
    if type(img) == list:
        img = np.array(img)
    if type(img) == np.ndarray:
        assert len(img.shape) == 3, 'Image data must be a 3D array.'
        if scalefactor != 1:
            img = ndimage.zoom(img,scalefactor,order=scaleorder)
            voxelsize /= scalefactor
    elif type(img) == str:
        path = img
        tiffs = glob.glob(os.path.join(path,'*.tiff')) + glob.glob(os.path.join(path,'*.tif')) + glob.glob(os.path.join(path,'*.jpg')) + glob.glob(os.path.join(path,'*.jpeg')) + glob.glob(os.path.join(path,'*.png'))
        tiffs.sort()
        dicoms = glob.glob(os.path.join(path,'*.dcm'))
        dicoms.sort()
        if len(dicoms) == 0:
            dicoms = glob.glob(os.path.join(path,'*.DCM'))
            dicoms.sort()
        if len(tiffs) > 0 & len(dicoms) > 0:
            warnings.warn('Image directory: "{:s}" contains .dcm files as well as other image file types - only loading dcm files.')
            files = dicoms
            ftype = 'dcm'
        elif len(tiffs) > 0:
            try:
                import cv2
            except:
                raise ImportError('im2voxel requires opencv-python (cv2) to load tiff, jpg, or png files. Install with: pip install opencv-python')
            files = tiffs
            ftype = 'tiff'
        elif len(dicoms) > 0:
            try:
                import cv2
            except:
                raise ImportError('im2voxel requires pydicom to load DICOM files. Install with: pip install pydicom')
            files = dicoms
            ftype = 'dcm'
        else:
            warnings.warn('Image directory is empty.')
            if return_nodedata:
                return [], [], [], []
            else:
                return [], [], []
        print('Loading image data from {:s}...'.format(img))
        if ftype == 'tiff':
            
            temp = cv2.imread(files[0])
            if len(temp.shape) > 2:
                multichannel = True
                multiimgs = [np.array([cv2.imread(file)[:,:,i] for file in files]) for i in range(temp.shape[2])]
                imgs = multiimgs[0]
            else:
                multichannel = False
                imgs = np.array([cv2.imread(file) for file in files])
        else:
            # temp = pydicom.dcmread(files[0]).pixel_array
            imgs = np.array([pydicom.dcmread(file).pixel_array for file in files])
        
        if scalefactor != 1:
            if multichannel:
                multiimg = [ndimage.zoom(I,scalefactor,order=scaleorder) for I in multiimgs]
                img = multiimg[0]
            else:
                img = ndimage.zoom(imgs,scalefactor,order=scaleorder)
            voxelsize /= scalefactor
        else:
            if multichannel:
                multiimg = multiimgs
            img = imgs
    else:
        raise Exception('Type {:s} not supported'.format(str(type(img))))
    
    
    if crop is None:
        (nz,ny,nx) = img.shape
        xlims = [0,(nx)*voxelsize]
        ylims = [0,(ny)*voxelsize]
        zlims = [0,(nz)*voxelsize]
        bounds = [xlims[0],xlims[1],ylims[0],ylims[1],zlims[0],zlims[1]]
        VoxelCoords, VoxelConn = primitives.Grid(bounds, voxelsize, exact_h=False, meshobj=False)
        if multichannel:
            VoxelData = np.column_stack([I.flatten(order='F') for I in multiimg])
        else:
            VoxelData = img.flatten(order='F')
        if return_gradient:
            gradx = ndimage.gaussian_filter(img,gaussian_sigma,order=(1,0,0))
            grady = ndimage.gaussian_filter(img,gaussian_sigma,order=(0,1,0))
            gradz = ndimage.gaussian_filter(img,gaussian_sigma,order=(0,0,1))
            GradData = np.vstack([gradx.flatten(order='F'),grady.flatten(order='F'),gradz.flatten(order='F')]).T
    else:
        # Adjust crop values to only get whole voxels
        crop[:-1:2] = np.floor(np.asarray(crop)[:-1:2]/voxelsize)*voxelsize
        crop[1::2] = np.ceil(np.asarray(crop)[1::2]/voxelsize)*voxelsize
        if rectangular_elements:
            bounds = [crop[0]/xscale,crop[1]/xscale,
                    crop[2]/yscale,crop[3]/yscale,
                    crop[4]/zscale,crop[5]/zscale]
        else:
            bounds = crop
        VoxelCoords, VoxelConn = primitives.Grid(bounds, voxelsize, exact_h=False, meshobj=False)
        mins = np.round(np.min(VoxelCoords,axis=0)/voxelsize).astype(int)
        maxs = np.round(np.max(VoxelCoords,axis=0)/voxelsize).astype(int)
        if multichannel:
            cropimg = [I[mins[2]:maxs[2],mins[1]:maxs[1],mins[0]:maxs[0]] for I in multiimg]
            VoxelData = np.column_stack([I.flatten(order='F') for I in cropimg])
        else:
            cropimg = img[mins[2]:maxs[2],mins[1]:maxs[1],mins[0]:maxs[0]]
            VoxelData = cropimg.flatten(order='F')
        if return_gradient:
            gradx = ndimage.gaussian_filter(cropimg,gaussian_sigma,order=(1,0,0))
            grady = ndimage.gaussian_filter(cropimg,gaussian_sigma,order=(0,1,0))
            gradz = ndimage.gaussian_filter(cropimg,gaussian_sigma,order=(0,0,1))
            GradData = np.vstack([gradx.flatten(order='F'),grady.flatten(order='F'),gradz.flatten(order='F')]).T
    if rectangular_elements:
        VoxelCoords[:,0] = VoxelCoords[:,0]*xscale
        VoxelCoords[:,1] = VoxelCoords[:,1]*yscale
        VoxelCoords[:,2] = VoxelCoords[:,2]*zscale
        
    if threshold is not None:
        if threshold_direction == 1:
            VoxelConn = VoxelConn[VoxelData>=threshold]
            VoxelData = VoxelData[VoxelData>=threshold]
            if return_gradient: GradData = GradData[VoxelData>=threshold]
            VoxelCoords,VoxelConn,_ = removeNodes(VoxelCoords,VoxelConn)
            VoxelConn = np.asarray(VoxelConn)

        elif threshold_direction == -1:
            VoxelConn = VoxelConn[VoxelData<=threshold]
            VoxelData = VoxelData[VoxelData<=threshold]
            if return_gradient: GradData = GradData[VoxelData<=threshold]
            VoxelCoords,VoxelConn,_ = removeNodes(VoxelCoords,VoxelConn)
            VoxelConn = np.asarray(VoxelConn)
        else:
            raise Exception('threshold_direction must be 1 or -1, where 1 indicates that values >= threshold will be kept and -1 indicates that values <= threshold will be kept.')
    if return_nodedata:
        rows = VoxelConn.flatten()
        cols = np.repeat(np.arange(len(VoxelConn)),8)
        data = np.ones(len(rows))
        M = sparse.coo_matrix((data,(rows,cols))).tolil()
        M = M.multiply(1/(M*np.ones((M.shape[1],1))))

        if multichannel:
            NodeData = np.column_stack([M*VoxelData[:,i] for i in range(VoxelData.shape[1])])
        else:
            NodeData = M*VoxelData
        
        if return_gradient:
            NodeGrad = M*GradData            
            VoxelData = (VoxelData,GradData)
            NodeData = (NodeData,NodeGrad)
            
        return VoxelCoords, VoxelConn, VoxelData, NodeData
    if return_gradient:
        VoxelData = (VoxelData,GradData)
    return VoxelCoords, VoxelConn, VoxelData
        
def surf2voxel(SurfCoords,SurfConn,h,Octree='generate',mode='any'):
    """
    Convert a surface mesh to a filled voxel mesh. The surface must be 
    closed to work properly, unexpected behavior could occur with unclosed
    surfaces.

    Parameters
    ----------
    SurfCoords : array_like
        Node coordinates of the surface mesh
    SurfConn : array_like
        Node connectivity of the surface mesh
    h : float
        Voxel size for the output mesh.
    Octree : str, octree.OctreeNode, None, optional
        Octree setting, by default 'generate'. 
        'generate' will construct an octree for use in creating the voxel mesh, None will not use an octree. Alternatively, if an existing 
        octree structure exists, that can be provided.
    mode : str, optional
        Mode for determining which elements are kept, by default 'any'.
        Voxels will be kept if:
        'any' - if any node of a voxel is inside the surface, 
        'all' - if all nodes of a voxel are inside the surface, 
        'centroid' - if the centroid of the voxel is inside the surface.

    Returns
    -------
    VoxelCoords : np.ndarray
        Node coordinates of the voxel mesh
    VoxelConn : np.ndarray
        Node connectivity of the voxel mesh

    """    

    arrayCoords = np.asarray(SurfCoords)
    bounds = np.column_stack([np.min(arrayCoords,axis=0),np.max(arrayCoords,axis=0)]).flatten()
    GridCoords,GridConn = primitives.Grid(bounds,h)
    
    if mode.lower() == 'centroid':
        centroids = utils.Centroids(GridCoords, GridConn)
        Inside = rays.isInsidesSurf(centroids, SurfCoords, SurfConn, Octree=Octree)
        VoxelConn = GridConn[Inside]
    else:
        Inside = rays.isInsidesSurf(GridCoords, SurfCoords, SurfConn, Octree=Octree)
        ElemInsides = Inside[GridConn]

        if mode.lower() == 'any':
            VoxelConn = GridConn[np.any(ElemInsides,axis=1)]
        elif mode.lower() == 'all':
            VoxelConn = GridConn[np.all(ElemInsides,axis=1)]
        else:
            raise Exception('mode must be "any", "all", or "centroid".')

    VoxelCoords,VoxelConn,_ = removeNodes(GridCoords,VoxelConn)
    return VoxelCoords,VoxelConn

def voxel2im(VoxelCoords, VoxelConn, Vals):
    """
    Convert a rectilinear voxel mesh (grid) to an 3D image matrix.

    Parameters
    ----------
    VoxelCoords : array_like
        Node coordinates of the voxel mesh
    VoxelConn : array_like
        Node connectivity of the voxel mesh
    Vals : array_like
        Values associated with either the nodes (len(Vals)=len(VoxelCoords)) or elements (len(Vals)=len(VoxelConn)) that will be stored in the image matrix.

    Returns
    -------
    I : np.ndarray
        3D array of image data. This data is ordered such that the three axes of the
        array (0, 1, 2) correspond to (z, y, x). I.e. I[i,:,:] will give a 2D array in the yx plane at z-position i. 

    """
    if type(VoxelCoords) == list: VoxelCoords = np.array(VoxelCoords)
    if len(Vals) == len(VoxelCoords):
        # Node values
        shape = (len(np.unique(VoxelCoords[:,2])),len(np.unique(VoxelCoords[:,1])),len(np.unique(VoxelCoords[:,0])))
    elif len(Vals) == len(VoxelConn):
        # Voxel values
        shape = (len(np.unique(VoxelCoords[:,2]))-1,len(np.unique(VoxelCoords[:,1]))-1,len(np.unique(VoxelCoords[:,0]))-1)
    else:
        raise Exception('Vals must be equal in length to either VoxelCoords or VoxelConn')
    I = np.reshape(Vals,shape,order='F')
    
    return I

def removeNodes(NodeCoords,NodeConn):
    """
    Removes nodes that aren't held by any element

    Parameters
    ----------
    NodeCoords : array_like
        Node coordinates
    NodeConn : array_like
        Node connectivity

    Returns
    -------
    NewNodeCoords : list
        New set of node coordinates where unused nodes have been removed
    NewNodeConn : list
        Renumbered set of node connectivities to be consistent with NewNodeCoords
    OriginalIds : np.ndarray
        The indices the original IDs of the nodes still in the mesh. This can be used
        to remove entries in associated node data (ex. new_data = old_data[OriginalIds]).
    """    
    # removeNodes 
    OriginalIds, inverse = np.unique([n for elem in NodeConn for n in elem],return_inverse=True)
    NewNodeCoords = [NodeCoords[i] for i in OriginalIds]
    
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
    """
    Convert a surface mesh to it's dual mesh.
    NOTE: this function has undergone limited testing and hasn't been optimized.
    NOTE: The polygonal meshes that result from this function aren't well supported 
    by most of the other functions of this library.

    Parameters
    ----------
    NodeCoords : array_like
        Node coordinates
    SurfConn : array_like
        Surface mesh node connectivity
    Centroids : array_like, optional
        Element centroids, by default None. If none are provided, they will
        be calculated for use within this function.
    ElemConn : list, optional
        Node-Element connectivity, by default None. If none are provided, they will
        be calculated for use within this function. ElemConn can be obtained from
        utils.getElemConnectivity()
    NodeNormals : array_like, optional
        Normal vectors for each node in the surface mesh, by default None. If none are provided, they will be calculated for use within this function. These are needed
        for proper ordering of the nodes in the dual mesh.
    sort : str, optional
        Which direction to sort the dual mesh elements about their centroid.
        Options are clockwise ('cw') or counter-clockwise ('ccw'), by default 'ccw'. 

    Returns
    -------
    DualCoords : list
        Node coordinates of the dual mesh
    DualConn : list
        Node connectivity of the dual mesh

    """    
    if not Centroids:
        Centroids = utils.Centroids(NodeCoords,SurfConn)
    if not ElemConn:
        ElemConn = utils.getElemConnectivity(NodeCoords,SurfConn,ElemType='auto')
    if not NodeNormals:
        ElemNormals = utils.CalcFaceNormal(NodeCoords,SurfConn)
        NodeNormals = utils.Face2NodeNormal(NodeCoords,SurfConn,ElemConn,ElemNormals)
    
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
            if np.array_equal(N, k):
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