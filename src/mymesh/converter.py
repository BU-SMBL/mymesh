# -*- coding: utf-8 -*-
# Created on Sun Aug  1 17:48:50 2021
# @author: toj
"""
Mesh conversion tools.
This module provides functions for converting between mesh types (e.g. a solid
volumetric mesh to a surface mesh) element types (e.g. hexahedral to 
tetrahedral), and connectivity representations (e.g. element node connectivities
to element faces or edges).


.. currentmodule:: mymesh.converter


Mesh type conversion
====================
.. autosummary::
    :toctree: submodules/

    solid2surface
    im2voxel
    mesh2im
    voxel2im
    surf2voxel
    surf2dual
    
Connectivity conversion
=======================
.. autosummary::
    :toctree: submodules/

    solid2faces
    solid2edges
    surf2edges
    EdgesByElement
    faces2surface
    faces2unique
    edges2unique
    tet2faces
    hex2faces
    pyramid2faces
    wedge2faces
    tri2edges
    quad2edges
    polygon2edges
    tet2edges
    pyramid2edges
    wedge2edges
    hex2edges

Element type conversion
=======================
.. autosummary::
    :toctree: submodules/

    solid2tets
    surf2tris
    linear2quadratic
    quadratic2linear
    hex2tet
    wedge2tet
    pyramid2tet
    quad2tri    
    edge32linear
    edge2quadratic
    tri62linear
    tri2quadratic
    quad82linear
    quad2quadratic
    tet102linear
    tet2quadratic
    pyr132linear
    pyr2quadratic
    wdg152linear
    wdg2quadratic
    hex202linear
    hex2quadratic
    hexsubdivide

"""

import numpy as np

from scipy import ndimage, sparse
import sys, os, warnings, glob, gc, tempfile
from . import utils, rays, primitives, image, tree

def solid2surface(NodeCoords,NodeConn,return_SurfElem=False):
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
    if return_SurfElem:
        Faces, FaceElem = solid2faces(NodeCoords,NodeConn,return_FaceElem=True)
        SurfConn, SurfIds = faces2surface(Faces, return_ids=True)
        ids = FaceElem[SurfIds]
        return SurfConn, ids
    else:
        Faces = solid2faces(NodeCoords,NodeConn)
        SurfConn = faces2surface(Faces)
        
        return SurfConn

def solid2faces(NodeCoords,NodeConn,return_FaceConn=False,return_FaceElem=False,ElemType='auto'):
    """
    Convert solid mesh to faces. The will be one face for each side of each element,
    i.e. there will be duplicate faces for non-surface faces. Use faces2surface(Faces) to extract only the surface faces or face2unique(Faces) to remove duplicates.

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
    ElemType : str, optional
        Specifies the element type contained within the mesh, by default 'auto'.

        - 'auto' or 'mixed' - Will detect element type by the number of nodes present in each element using :func:`~mymesh.utils.identify_type`. 

        - 'surf' - Will detect element type by the number of nodes present in each 
        element, assuming four node elements are quads

        - 'vol' - Will detect element type by the number of nodes present in each 
        element, assuming four node elements are tets (functionally the ame as 'auto')

        - 'tri' - All elements treated as 3-node triangular elements.

        - 'quad' - All elements treated as 4-node quadrilateral elements.

        - 'tet' - All elements treated as 4-node tetrahedral elements.

        - 'pyramid' - All elements treated as 5-node wedge elements.

        - 'wedge' - All elements treated as 6-node quadrilateral elements.

        - 'hex' - All elements treated as 8-node quadrilateral elements.

    Returns
    -------
    Faces : list
        List of mesh faces.
    FaceConn : list, optional
        The face connectivity of each element.
    FaceElem : list, optional
        The element index that each face is taken from.
    """     
    
    if ElemType in ('auto','mixed','line','surf','vol'):
        # check if a single element mesh to avoid unnecessary overhead
        if ElemType in ('line','surf','vol'):
            t = ElemType
        else:
            t = None
        types = utils.identify_elem(NodeCoords, NodeConn, Type=t)
        if len(types) == 1:
            ElemType = types[0]
    
    if ElemType in ('auto','mixed','line','surf','vol'):
        Ls = np.array(list(map(len, NodeConn)))
        edgIdx = np.where(Ls == 2)[0]
        triIdx = np.where(Ls == 3)[0]
        tetIdx = np.where(Ls == 4)[0]
        tet10Idx = np.where(Ls == 10)[0]
        pyrIdx = np.where(Ls == 5)[0]
        wdgIdx = np.where(Ls == 6)[0]
        hexIdx = np.where(Ls == 8)[0] 
        hex20Idx = np.where(Ls == 20)[0]
        edgs = [NodeConn[i] for i in edgIdx]
        tris = [NodeConn[i] for i in triIdx]
        tets = [NodeConn[i] for i in tetIdx]
        tet10s = [NodeConn[i] for i in tet10Idx]
        pyrs = [NodeConn[i] for i in pyrIdx]
        wdgs = [NodeConn[i] for i in wdgIdx]
        hexs = [NodeConn[i] for i in hexIdx]
        hex20s = [NodeConn[i] for i in hex20Idx]

        Faces = edgs + tris + tet2faces([],tets).tolist() + tet102faces([],tet10s).tolist() + pyramid2faces([],pyrs) + wedge2faces([],wdgs) + hex2faces([],hexs).tolist()+ hex202faces([],hex20s).tolist()
        if return_FaceConn or return_FaceElem:
            ElemIds_i = np.concatenate((edgIdx,triIdx,np.repeat(tetIdx,4),np.repeat(tet10Idx,4),np.repeat(pyrIdx,5),np.repeat(wdgIdx,5),np.repeat(hexIdx,6),np.repeat(hex20Idx,6)))
            FaceElem = ElemIds_i
            ElemIds_j = np.concatenate((np.repeat(0,len(edgIdx)),np.repeat(0,len(triIdx)), 
                    np.repeat([[0,1,2,3]],len(tetIdx),axis=0).reshape(len(tetIdx)*4),  
                    np.repeat([[0,1,2,3]],len(tet10Idx),axis=0).reshape(len(tet10Idx)*4),  
                    np.repeat([[0,1,2,3,4]],len(pyrIdx),axis=0).reshape(len(pyrIdx)*5),                   
                    np.repeat([[0,1,2,3,4]],len(wdgIdx),axis=0).reshape(len(wdgIdx)*5),   
                    np.repeat([[0,1,2,3,4,5]],len(hexIdx),axis=0).reshape(len(hexIdx)*6), 
                    np.repeat([[0,1,2,3,4,5]],len(hex20Idx),axis=0).reshape(len(hex20Idx)*6),                    
                    ))
            FaceConn = -1*np.ones((len(NodeConn),6))
            FaceConn[ElemIds_i,ElemIds_j] = np.arange(len(Faces))
            FaceConn = utils.ExtractRagged(FaceConn,dtype=int)
    elif ElemType=='tri':
        Faces = NodeConn
        if return_FaceElem or return_FaceConn:
            triIdx = np.arange(len(NodeConn))
            FaceElem = triIdx
        if return_FaceConn:
            FaceConn = triIdx[:,None]
    elif ElemType=='quad':
        Faces = NodeConn
        if return_FaceElem or return_FaceConn:
            quadIdx = np.arange(len(NodeConn))
            FaceElem = quadIdx
        if return_FaceConn:
            FaceConn = quadIdx[:,None]
    elif ElemType=='tet':
        Faces = tet2faces(NodeCoords,NodeConn)
        if return_FaceElem or return_FaceConn:
            tetIdx = np.arange(len(NodeConn))
            FaceElem = np.repeat(tetIdx,4)
        if return_FaceConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3]],len(tetIdx),axis=0).reshape(len(tetIdx)*4),  
                ))
            FaceConn = -1*np.ones((len(NodeConn),4), dtype=int)
            FaceConn[FaceElem,ElemIds_j] = np.arange(len(Faces))
    elif ElemType=='tet10':
        Faces = tet102faces(NodeCoords,NodeConn)
        if return_FaceElem or return_FaceConn:
            tetIdx = np.arange(len(NodeConn))
            FaceElem = np.repeat(tetIdx,4)
        if return_FaceConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3]],len(tetIdx),axis=0).reshape(len(tetIdx)*4),  
                ))
            FaceConn = -1*np.ones((len(NodeConn),4), dtype=int)
            FaceConn[FaceElem,ElemIds_j] = np.arange(len(Faces))
    elif ElemType=='pyramid':
        Faces = pyramid2faces(NodeCoords,NodeConn)
        if return_FaceElem or return_FaceConn:
            pyrIdx = np.arange(len(NodeConn))
            FaceElem = np.repeat(pyrIdx,5)
        if return_FaceConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4]],len(pyrIdx),axis=0).reshape(len(pyrIdx)*5),                   
                ))
            FaceConn = -1*np.ones((len(NodeConn),5), dtype=int)
            FaceConn[FaceElem,ElemIds_j] = np.arange(len(Faces))
    elif ElemType=='wedge':
        Facees = wedge2faces(NodeCoords,NodeConn)
        if return_FaceElem or return_FaceConn:
            wdgIdx = np.arange(len(NodeConn))
            FaceElem = np.repeat(wdgIdx,5)
        if return_FaceConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4]],len(wdgIdx),axis=0).reshape(len(wdgIdx)*5),   
                ))
            FaceConn = -1*np.ones((len(NodeConn),5), dtype=int)
            FaceConn[FaceElem,ElemIds_j] = np.arange(len(Faces))
    elif ElemType=='hex':
        Faces = hex2faces(NodeCoords,NodeConn)
        if return_FaceElem or return_FaceConn:
            hexIdx = np.arange(len(NodeConn))
            FaceElem = np.repeat(hexIdx,6)
        if return_FaceConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5]],len(hexIdx),axis=0).reshape(len(hexIdx)*6),                    
                ))
            FaceConn = -1*np.ones((len(NodeConn),6), dtype=int)
            FaceConn[FaceElem,ElemIds_j] = np.arange(len(Faces))
    elif ElemType=='hex20':
        Faces = hex202faces(NodeCoords,NodeConn)
        if return_FaceElem or return_FaceConn:
            hexIdx = np.arange(len(NodeConn))
            FaceElem = np.repeat(hexIdx,6)
        if return_FaceConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5]],len(hexIdx),axis=0).reshape(len(hexIdx)*6),                    
                ))
            FaceConn = -1*np.ones((len(NodeConn),6), dtype=int)
            FaceConn[FaceElem,ElemIds_j] = np.arange(len(Faces))


    if return_FaceConn and return_FaceElem:
        return Faces,FaceConn,FaceElem
    elif return_FaceConn:
        return Faces,FaceConn
    elif return_FaceElem:
        return Faces,FaceElem
    else:
        return Faces

def solid2edges(NodeCoords,NodeConn,ElemType='auto',return_EdgeConn=False,return_EdgeElem=False,):
    """
    Convert a solid mesh to edges. There will be one edge for each edge of each element,
    i.e. there will be multiple entries for shared edges. solid2edges is also suitable for use 
    with 2D or surface meshes. It differs from surface2edges in that surface2edges returns only exposed edges of unclosed surfaces.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.
    ElemType : str, optional
        Specifies the element type contained within the mesh, by default 'auto'.

        - 'auto' or 'mixed' - Will detect element type by the number of nodes present in each element using :func:`~mymesh.utils.identify_type`. 

        - 'surf' - Will detect element type by the number of nodes present in each 
        element, assuming four node elements are quads

        - 'vol' - Will detect element type by the number of nodes present in each 
        element, assuming four node elements are tets (functionally the ame as 'auto')

        - 'tri' - All elements treated as 3-node triangular elements.

        - 'quad' - All elements treated as 4-node quadrilateral elements.

        - 'tet' - All elements treated as 4-node tetrahedral elements.

        - 'pyramid' - All elements treated as 5-node wedge elements.

        - 'wedge' - All elements treated as 6-node quadrilateral elements.

        - 'hex' - All elements treated as 8-node quadrilateral elements.

        - 'polygon' - All elements treated as n-node polygonal elements. TODO: add support for return_EdgeConn and return_EdgeElem

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
    if ElemType in ('auto','mixed','line','surf','vol'):
        # check if a single element mesh to avoid unnecessary overhead
        if ElemType in ('line','surf','vol'):
            t = ElemType
        else:
            t = None
        types = utils.identify_elem(NodeCoords, NodeConn, Type=t)
        if len(types) == 1:
            ElemType = types[0]

    if ElemType in ('auto','mixed','line','surf','vol'):
        Ls = np.array([len(elem) for elem in NodeConn])
        edgIdx = np.where(Ls == 2)[0]
        triIdx = np.where(Ls == 3)[0]
        tetIdx = np.where(Ls == 4)[0]
        pyrIdx = np.where(Ls == 5)[0]
        wdgIdx = np.where(Ls == 6)[0]
        hexIdx = np.where(Ls == 8)[0]
        tet10Idx = np.where(Ls == 10)[0]
        if len(edgIdx) > 0:
            edgs = np.array([NodeConn[i] for i in edgIdx]).astype(int)
        else:
            edgs = np.empty((0,2),dtype=int)
        tris = [NodeConn[i] for i in triIdx]
        tets = [NodeConn[i] for i in tetIdx]
        pyrs = [NodeConn[i] for i in pyrIdx]
        wdgs = [NodeConn[i] for i in wdgIdx]
        hexs = [NodeConn[i] for i in hexIdx]
        tet10s = [NodeConn[i] for i in tet10Idx]
        
        if ElemType == 'vol':
            fournodefunc = tet2edges
            fournodeedgenum = 6
        elif ElemType == 'surf':
            fournodefunc = quad2edges
            fournodeedgenum = 4
        else:
            if len(tetIdx) > 0:
                Type = utils.identify_type(NodeCoords, NodeConn)
                if Type == 'vol':
                    fournodefunc = tet2edges
                    fournodeedgenum = 6
                else:
                    fournodefunc = quad2edges
                    fournodeedgenum = 4
            else:
                fournodefunc = tet2edges
                fournodeedgenum = 6

        Edges = np.concatenate((edgs,tri2edges([],tris), 
                                fournodefunc([],tets), 
                                pyramid2edges([],pyrs), 
                                wedge2edges([],wdgs), 
                                hex2edges([],hexs)))
        QuadEdges = np.concatenate((tet102edges([],tet10s),))
        if len(QuadEdges) > 0:
            if len(Edges) > 0:
                Edges = Edges.tolist() + QuadEdges.tolist()
            else:
                Edges = QuadEdges
    
        if return_EdgeElem or return_EdgeConn:
            EdgeElem = np.concatenate((np.repeat(edgIdx,1),np.repeat(triIdx,3),np.repeat(tetIdx,fournodeedgenum),np.repeat(pyrIdx,8),np.repeat(wdgIdx,9),np.repeat(hexIdx,12),np.repeat(tet10Idx,6)))
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0]],len(edgIdx),axis=0).reshape(len(edgIdx)*1),
                np.repeat([[0,1,2]],len(triIdx),axis=0).reshape(len(triIdx)*3), 
                np.repeat([np.arange(fournodeedgenum)],len(tetIdx),axis=0).reshape(len(tetIdx)*fournodeedgenum),  
                np.repeat([[0,1,2,3,4,5,6,7]],len(pyrIdx),axis=0).reshape(len(pyrIdx)*8),                   
                np.repeat([[0,1,2,3,4,5,6,7,8]],len(wdgIdx),axis=0).reshape(len(wdgIdx)*9),   
                np.repeat([[0,1,2,3,4,5,6,7,8,9,10,11]],len(hexIdx),axis=0).reshape(len(hexIdx)*12),   
                np.repeat([[0,1,2,3,4,5]],len(tet10Idx),axis=0).reshape(len(tet10Idx)*6)                 
                ))
            EdgeConn = -1*np.ones((len(NodeConn),12))
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
            EdgeConn = utils.ExtractRagged(EdgeConn,dtype=int)

    elif ElemType=='tri':
        Edges = tri2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            triIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(triIdx,3)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2]],len(triIdx),axis=0).reshape(len(triIdx)*3), 
                ))
            EdgeConn = -1*np.ones((len(NodeConn),3), dtype=int)
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
    elif ElemType=='quad':
        Edges = quad2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            quadIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(quadIdx,4)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2]],len(tetIdx),axis=0).reshape(len(tetIdx)*3), 
                ))
            EdgeConn = -1*np.ones((len(NodeConn),4), dtype=int)
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
    elif ElemType=='tet':
        Edges = tet2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            tetIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(tetIdx,6)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5]],len(tetIdx),axis=0).reshape(len(tetIdx)*6),  
                ))
            EdgeConn = -1*np.ones((len(NodeConn),6), dtype=int)
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
    elif ElemType=='pyramid':
        Edges = pyramid2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            pyrIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(pyrIdx,8)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5,6,7]],len(pyrIdx),axis=0).reshape(len(pyrIdx)*8),                   
                ))
            EdgeConn = -1*np.ones((len(NodeConn),8), dtype=int)
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
    elif ElemType=='wedge':
        Edges = wedge2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            wdgIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(wdgIdx,9)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5,6,7,8]],len(wdgIdx),axis=0).reshape(len(wdgIdx)*9),   
                ))
            EdgeConn = -1*np.ones((len(NodeConn),9), dtype=int)
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
    elif ElemType=='hex':
        Edges = hex2edges(NodeCoords,NodeConn)
        if return_EdgeElem or return_EdgeConn:
            hexIdx = np.arange(len(NodeConn))
            EdgeElem = np.repeat(hexIdx,12)
        if return_EdgeConn:
            ElemIds_j = np.concatenate((
                np.repeat([[0,1,2,3,4,5,6,7,8,9,10,11]],len(hexIdx),axis=0).reshape(len(hexIdx)*12),                    
                ))
            EdgeConn = -1*np.ones((len(NodeConn),12), dtype=int)
            EdgeConn[EdgeElem,ElemIds_j] = np.arange(len(Edges))
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

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.
    ElemType : str, optional
        Specifies the element type contained within the mesh, by default 'auto'.
        See :func:`solid2edges` for options.

    Returns
    -------
    ElementEdges : list
        Edge connectivity, grouped by element
    """    
    Edges, EdgeConn = converter.solid2edges(*S, return_EdgeConn=True, ElemType=ElemType)
    ElementEdges = [Edges[ec] for ec in EdgeConn]
    return ElementEdges

def solid2tets(NodeCoords,NodeConn,return_ids=False,return_inv=False):
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
    return_inv : bool, optional
        If True, return element IDs of the original elements that correspond to 
        the tetrahedral elements, by default False

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
    inv : np.ndarray, optional
        Array of indices that point from the tetrahedral mesh back to the original
        mesh
    """    
    if type(NodeConn) is np.ndarray:
        Ls = np.repeat(NodeConn.shape[1], NodeConn.shape[0])
    else:
        Ls = np.array([len(elem) for elem in NodeConn])

    if np.all(Ls == 4):
        if return_inv:
            inv = np.arange(len(NodeConn))
        if return_ids:
            ElemIds = np.arange(len(NodeConn)).reshape(len(NodeConn),1)
            if return_inv:
                return NodeCoords, NodeConn, ElemIds, inv    
            return NodeCoords, NodeConn, ElemIds
        elif return_inv:
            return NodeCoords, NodeConn, inv
        else:
            return NodeCoords, NodeConn
            
    tetIdx = np.where(Ls == 4)[0]
    pyrIdx = np.where(Ls == 5)[0]
    wdgIdx = np.where(Ls == 6)[0]
    hexIdx = np.where(Ls == 8)[0]
    tet10Idx = np.where(Ls == 10)[0]

    tets = np.array([NodeConn[i] for i in tetIdx])
    if len(tets) == 0:
        tets = np.empty((0,4))
    pyrs = np.array([NodeConn[i] for i in pyrIdx])
    wdgs = np.array([NodeConn[i] for i in wdgIdx])
    hexs = np.array([NodeConn[i] for i in hexIdx])
    tet10 = np.array([NodeConn[i] for i in tet10Idx])

    if len(hexIdx) > 0 and (len(pyrIdx) > 0 or len(wdgIdx) > 0):
        # if 2/3 of hexs, pyrs, and wdgs use decomposition guaranteed to produce
        #aligned edges
        hexmethod = '1to24'
        hexn = 24
        wdgmethod = '1to14'
        wdgn = 14
        pyrmethod = '1to4'
        pyrn = 4
    else:
        hexmethod = '1to6'
        hexn = 6
        wdgmethod = '1to3c'
        wdgn = 3
        pyrmethod = '1to2c'
        pyrn = 2

    TetCoords,fromhex = hex2tet(NodeCoords,hexs,method=hexmethod)
    TetCoords,fromwdg = wedge2tet(TetCoords,wdgs,method=wdgmethod)
    TetCoords,frompyr = pyramid2tet(TetCoords,pyrs,method=pyrmethod)
    TetCoords,fromtet10 = tet102linear(TetCoords,tet10)
    # TetConn = tets + frompyr + fromwdg + fromhex + fromtet10
    TetConn = np.vstack([tets, frompyr, fromwdg, fromhex, fromtet10]).astype(int)
    if return_ids or return_inv:
        inv = np.concatenate((tetIdx,np.repeat(pyrIdx,pyrn),np.repeat(wdgIdx,wdgn),np.repeat(hexIdx,hexn),tet10Idx))
    if return_ids:
        # Element ids of the tets connected to the original elements
        ElemIds_i = inv
        ElemIds_j = np.concatenate((np.repeat(0,len(tetIdx)), 
                np.repeat([np.arange(pyrn)],len(pyrIdx),axis=0).reshape(len(pyrIdx)*pyrn),                   
                np.repeat([np.arange(wdgn)],len(wdgIdx),axis=0).reshape(len(wdgIdx)*wdgn),   
                np.repeat([np.arange(hexn)],len(hexIdx),axis=0).reshape(len(hexIdx)*hexn),
                np.repeat(0,len(tet10Idx)),                 
                ))
        ElemIds = -1*np.ones((len(NodeConn),np.max([4,wdgn,hexn])))
        ElemIds[ElemIds_i,ElemIds_j] = np.arange(len(TetConn))
        ElemIds = utils.ExtractRagged(ElemIds,dtype=int)
    
    if return_ids:
        if return_inv:
            return TetCoords, TetConn, ElemIds, inv
        return TetCoords, TetConn, ElemIds
    elif return_inv:
        return TetCoords, TetConn, inv
    return TetCoords, TetConn

def surf2tris(NodeCoords,NodeConn,return_ids=False,return_inv=False):
    """
    Decompose all elements of a surface mesh to triangles.

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list.
    return_ids : bool, optional
        Element IDs of the tris connected to the original elements, by default False
    return_inv : bool, optional
        If True, return element IDs of the original elements that correspond to 
        the triangular elements, by default False

    Returns
    -------
    TriCoords : list
        Nodal coordinates of the generated triangles. 
    TriConn : list
        Nodal connectivity list of generated triangles.
    ElemIds : list, optional
        If return_ids = True, a list of the element ids of the new triangles for each 
        of the original elements.
    inv : np.ndarray, optional
        Array of indices that point from the triangular mesh back to the original
        mesh
    """    
    Ls = np.array([len(elem) for elem in NodeConn])
    triIdx = np.where(Ls == 3)[0]
    quadIdx = np.where(Ls == 4)[0]
    tris = [NodeConn[i] for i in triIdx]
    quads = [NodeConn[i] for i in quadIdx]
    if len(tris) == 0:
        tris = np.empty((0,3),dtype=int)

    TriCoords = NodeCoords
    _,fromquad = quad2tri(NodeCoords, quads)
    TriConn = np.vstack([tris, fromquad])
    if return_ids or return_inv:
        inv = np.concatenate((triIdx,np.repeat(quadIdx,2)))
    if return_ids:
        # Element ids of the tris connected to the original elements
        ElemIds_i = inv
        ElemIds_j = np.concatenate((np.repeat(0,len(triIdx)), 
                np.repeat([[0,1]],len(quadIdx),axis=0).reshape(len(quadIdx)*2),                   
                ))
        ElemIds = -1*np.ones((len(NodeConn),6))
        ElemIds[ElemIds_i,ElemIds_j] = np.arange(len(TriConn))
        ElemIds = utils.ExtractRagged(ElemIds,dtype=int)

    if return_ids:
        if return_inv:
            return TriCoords, TriConn, ElemIds, inv
        return TriCoords, TriConn, ElemIds
    elif return_inv:
        return TriCoords, TriConn, inv
    return TriCoords, TriConn

def linear2quadratic(NodeCoords,NodeConn,Type=None):
    """
    Convert linear (first-order) elements to quadratic (second-order) elements.
    See also :ref:`Element Types`.    

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates.
    NodeConn : array_like
        Nodal connectivity list.
    Type : str
        Mesh Type ('surf' or 'vol'). This helps resolve ambiguities between
        4-node quadrilaterals and 4-node tetrahedra. If not provided, Type
        will be detected using :func:`mymesh.utils.identify_type`

    Returns
    -------
    NewCoords : list
        Nodal coordinates of the second order mesh
    NewConn : list
        Nodal connectivity list of the second order mesh
    """    
    if type(NodeConn) is np.ndarray:
        Ls = np.repeat(NodeConn.shape[1], NodeConn.shape[0])
    else:
        Ls = np.array([len(elem) for elem in NodeConn])
    
    if Type is None:
        Type = utils.identify_type(NodeCoords, NodeConn)
    
    edgeIdx = np.where(Ls == 2)[0]
    triIdx = np.where(Ls == 3)[0]
    if Type.lower() == 'surf':
        quadIdx = np.where(Ls == 4)[0]
        tetIdx = np.array([],dtype=int)
    else:
        quadIdx = np.array([],dtype=int)
        tetIdx = np.where(Ls == 4)[0]
    pyrIdx = np.where(Ls == 5)[0]
    wdgIdx = np.where(Ls == 6)[0]
    hexIdx = np.where(Ls == 8)[0]

    edges = np.array([NodeConn[i] for i in edgeIdx])
    tris = np.array([NodeConn[i] for i in triIdx])
    quads = np.array([NodeConn[i] for i in quadIdx])
    tets = np.array([NodeConn[i] for i in tetIdx])
    pyrs = np.array([NodeConn[i] for i in pyrIdx])
    wdgs = np.array([NodeConn[i] for i in wdgIdx])
    hexs = np.array([NodeConn[i] for i in hexIdx])

    QuadCoords,fromedge = edge2quadratic(NodeCoords,edges, cleanup=False)
    QuadCoords,fromtri = tri2quadratic(QuadCoords,tris, cleanup=False)
    QuadCoords,fromquad = quad2quadratic(QuadCoords,quads, cleanup=False)
    QuadCoords,fromtet = tet2quadratic(QuadCoords,tets, cleanup=False)
    QuadCoords,frompyr = pyr2quadratic(QuadCoords,pyrs, cleanup=False)
    QuadCoords,fromwdg = wdg2quadratic(QuadCoords,wdgs, cleanup=False)
    QuadCoords,fromhex = hex2quadratic(QuadCoords,hexs, cleanup=False)

    QuadConn = -1*np.ones((len(NodeConn),20), dtype=int)
    QuadConn[edgeIdx,:3] = fromedge
    QuadConn[triIdx,:6] = fromtri
    QuadConn[quadIdx,:8] = fromquad
    QuadConn[tetIdx,:10] = fromtet
    QuadConn[pyrIdx,:13] = frompyr
    QuadConn[wdgIdx,:15] = fromwdg
    QuadConn[hexIdx,:20] = fromhex
    
    QuadConn = utils.ExtractRagged(QuadConn, delval=-1)
    QuadCoords, QuadConn = utils.DeleteDuplicateNodes(QuadCoords, QuadConn)

    return QuadCoords, QuadConn

def quadratic2linear(NodeCoords, NodeConn, cleanup=True):
    """
    Convert quadratic (second-order) elements to linear (first-order) elements.
    See also :ref:`Element Types`.    

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates.
    NodeConn : array_like
        Nodal connectivity list.
    cleanup : bool
        Remove nodes that are no longer in the mesh. Note that this will renumber
        the nodes and any external arrays that are assoviated with the nodes may
        no longer match.

    Returns
    -------
    NewCoords : list
        Nodal coordinates of the second order mesh
    NewConn : list
        Nodal connectivity list of the second order mesh
    """    
    if type(NodeConn) is np.ndarray:
        Ls = np.repeat(NodeConn.shape[1], NodeConn.shape[0])
    else:
        Ls = np.array([len(elem) for elem in NodeConn])
    
    
    edgeIdx = np.where(Ls == 3)[0]
    
    triIdx = np.where(Ls == 6)[0]
    quadIdx = np.where(Ls == 8)[0]
    tetIdx = np.where(Ls == 10)[0]
    pyrIdx = np.where(Ls == 13)[0]
    wdgIdx = np.where(Ls == 15)[0]
    hexIdx = np.where(Ls == 20)[0]

    edges = np.array([NodeConn[i] for i in edgeIdx])
    tris = np.array([NodeConn[i] for i in triIdx])
    quads = np.array([NodeConn[i] for i in quadIdx])
    tets = np.array([NodeConn[i] for i in tetIdx])
    pyrs = np.array([NodeConn[i] for i in pyrIdx])
    wdgs = np.array([NodeConn[i] for i in wdgIdx])
    hexs = np.array([NodeConn[i] for i in hexIdx])

    LinCoords,fromedge = edge32linear(NodeCoords, edges)
    LinCoords,fromtri = tri62linear(LinCoords, tris)
    LinCoords,fromquad = quad82linear(LinCoords, quads)
    LinCoords,fromtet = tet102linear(LinCoords, tets)
    LinCoords,frompyr = pyr132linear(LinCoords, pyrs)
    LinCoords,fromwdg = wdg152linear(LinCoords, wdgs)
    LinCoords,fromhex = hex202linear(LinCoords, hexs)

    LinConn = -1*np.ones((len(NodeConn),8),dtype=int)
    LinConn[edgeIdx,:2] = fromedge
    LinConn[triIdx,:3] = fromtri
    LinConn[quadIdx,:4] = fromquad
    LinConn[tetIdx,:4] = fromtet
    LinConn[pyrIdx,:5] = frompyr
    LinConn[wdgIdx,:6] = fromwdg
    LinConn[hexIdx,:8] = fromhex
    
    LinConn = utils.ExtractRagged(LinConn, delval=-1)
    if cleanup:
        LinCoords, LinConn, _ = utils.RemoveNodes(LinCoords, LinConn)

    return LinCoords, LinConn

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
    TetConn, np.ndarray
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
        return NodeCoords, np.empty((0,4),dtype=int)

    if method == '1to5':
        ArrayConn = np.asarray(NodeConn, dtype=int)
        TetConn = -1*np.ones((len(NodeConn)*5,4))
        TetConn[0::5] = ArrayConn[:,[0,1,3,4]]
        TetConn[1::5] = ArrayConn[:,[1,2,3,6]]
        TetConn[2::5] = ArrayConn[:,[4,6,5,1]]
        TetConn[3::5] = ArrayConn[:,[4,7,6,3]]
        TetConn[4::5] = ArrayConn[:,[4,6,1,3]]
        TetConn = TetConn.astype(int)
        NewCoords = NodeCoords
    elif method == '1to6':
        ArrayConn = np.asarray(NodeConn, dtype=int)
        TetConn = -1*np.ones((len(NodeConn)*6,4))
        TetConn[0::6] = ArrayConn[:,[0,1,3,5]]
        TetConn[1::6] = ArrayConn[:,[5,2,3,6]]
        TetConn[2::6] = ArrayConn[:,[0,5,3,4]]
        TetConn[3::6] = ArrayConn[:,[3,7,4,5]]
        TetConn[4::6] = ArrayConn[:,[1,2,3,5]]
        TetConn[5::6] = ArrayConn[:,[5,7,6,3]]

        TetConn = TetConn.astype(int)
        NewCoords = NodeCoords
    elif method == '1to24':
        ArrayCoords = np.asarray(NodeCoords)
        ArrayConn = np.asarray(NodeConn, dtype=int)

        Centroids = utils.Centroids(ArrayCoords,NodeConn)
        Face0Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,1,2,3]]],axis=1)
        Face1Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,1,5,4]]],axis=1)
        Face2Centroids = np.mean(ArrayCoords[ArrayConn[:,[1,2,6,5]]],axis=1)
        Face3Centroids = np.mean(ArrayCoords[ArrayConn[:,[2,3,7,6]]],axis=1)
        Face4Centroids = np.mean(ArrayCoords[ArrayConn[:,[3,0,4,7]]],axis=1)
        Face5Centroids = np.mean(ArrayCoords[ArrayConn[:,[4,5,6,7]]],axis=1)

        CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*0,len(NodeCoords)+len(NodeConn)*1, dtype=int)
        Face0CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*1,len(NodeCoords)+len(NodeConn)*2, dtype=int)
        Face1CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*2,len(NodeCoords)+len(NodeConn)*3, dtype=int)
        Face2CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*3,len(NodeCoords)+len(NodeConn)*4, dtype=int)
        Face3CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*4,len(NodeCoords)+len(NodeConn)*5, dtype=int)
        Face4CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*5,len(NodeCoords)+len(NodeConn)*6, dtype=int)
        Face5CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*6,len(NodeCoords)+len(NodeConn)*7, dtype=int)
        
        NewCoords = np.vstack([ArrayCoords,Centroids,Face0Centroids,Face1Centroids,Face2Centroids,Face3Centroids,Face4Centroids,Face5Centroids])        
        
        TetConn = -1*np.ones((len(NodeConn)*24,4), dtype=int)
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

def wedge2tet(NodeCoords, NodeConn, method='1to3c'):
    """
    Decompose all elements of a 3D wedge-element mesh to tetrahedra.
    Generally solid2tets should be used rather than wedge2tet directly
    NOTE the generated tetrahedra of the '1to3' method will not be continuously oriented, 
    i.e. edges of child tetrahedra may not be aligned between one parent element 
    and its neighbor, and thus the resulting mesh will typically be invalid.
    The primary use-case for this method is for methods like quality.Volume
    which utilize the geometric properties of tetrahedra to determine properties of 
    the parent elements. '1to3c', '1to14' or '1to36' will generate continuously oriented tetrahedra.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list. All elements should be 6-Node wedge elements.
    method : str, optional
        Method of decomposition to use for tetrahedralization.
        '1to3'  - Not continuously oriented, no nodes added, all elements decomposed the same way
        '1to3c'  - Continuously oriented, no nodes added (default)
        '1to14' - Continuously oriented, nodes added at center of element and element faces
        '1to36' - Continuously oriented, nodes added at center of element, element faces, and element edges

    Returns
    -------
    NewCoords : array_like
        New list of nodal coordinates. For '1to3' this will be unchanged from
        the input.
    TetConn, np.ndarray
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
        return NodeCoords, np.empty((0,4),dtype=int)

    ArrayConn = np.asarray(NodeConn)
    if method == '1to3':
        TetConn = -1*np.ones((len(NodeConn)*3,4))
        TetConn[0::3] = ArrayConn[:,[0,1,2,3]]
        TetConn[1::3] = ArrayConn[:,[1,2,3,4]]
        TetConn[2::3] = ArrayConn[:,[4,5,2,3]]
        TetConn = TetConn.astype(int)
        NewCoords = NodeCoords
    elif method == '1to3c':
        # Get the three quadrilateral faces
        face0 = ArrayConn[:,[0,1,4,3]]
        face1 = ArrayConn[:,[1,2,5,4]]
        face2 = ArrayConn[:,[0,3,5,2]]

        # Choose the starting point for drawing the diagonal as the smallest 
        # node index of each face
        face0_pt0 = np.argmin(face0, axis=1)
        face1_pt0 = np.argmin(face1, axis=1)
        face2_pt0 = np.argmin(face2, axis=1)

        # Determine which of the diagonals is being used
        ## True if diagonal from node 0 to node 4
        face0_case = ((face0_pt0 == 0) | (face0_pt0 == 2)).astype(int) 
        ## True if diagonal from node 1 to node 5
        face1_case = ((face1_pt0 == 0) | (face1_pt0 == 2)).astype(int) 
        ## True if diagonal from node 0 to node 5
        face2_case = ((face2_pt0 == 0) | (face2_pt0 == 2)).astype(int)

        # Lookup table
        a, b, c, d, e, f = 0, 1, 2, 3, 4, 5
        lookup = np.array([
            # Case 0
            [
                [a, b, c, d],
                [b, c, d, e],
                [c, f, d, e]
            ],
            # Case 1 - INVALID
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1]
            ],
            # Case 2 
            [
                [a, b, c, d],
                [b, f, d, e],
                [b, d, f, c]
            ],
            # Case 3
            [
                [a, b, f, d],
                [b, f, d, e],
                [a, b, c, f]
            ],
            # Case 4
            [
                [a, d, e, c],
                [a, b, c, e],
                [c, f, d, e]
            ],
            # Case 5
            [
                [a, e, f, d],
                [a, b, c, e],
                [a, e, c, f]
            ],
            # Case 6 - INVALID
            [
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1]
            ],
            # Case 7 
            [
                [a, e, f, d],
                [a, b, f, e],
                [a, b, c, f]
            ]
        ])
        
        lookup_keys = np.sum(np.column_stack([face0_case, face1_case, face2_case]) * 2**np.array([2,1,0]), axis=1)
        if np.any((lookup_keys == 1)|(lookup_keys == 6)):
            raise Exception('Invalid configuration encountered in wedge2tet. This most likely resulted from attempting to tetrahedralize a degenerate element with a duplicate node number.')
        configs = lookup[lookup_keys]

        TetConn = np.take_along_axis(ArrayConn[:, None, :], configs, axis=2).reshape(-1,4)
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


        TetConn = TetConn.astype(int)

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

        TetConn = TetConn.astype(int)

    return NewCoords, TetConn

def pyramid2tet(NodeCoords,NodeConn, method='1to2c'):
    """
    Decompose all elements of a 3D pyramidal mesh to tetrahedra.
    Generally solid2tets should be used rather than pyramid2tet directly
    NOTE the generated tetrahedra from 1 to 2 will not necessarily be continuously 
    oriented, i.e. edges of child tetrahedra may not be aligned between one 
    parent element and its neighbor, and thus the resulting mesh will typically be invalid.
    1 to 4 splitting is guaranteed to be consistent with other pyramids, hexs
    split with 1to24, and wedges split with 1to20.
    

    Parameters
    ----------
    NodeCoords : list
        List of nodal coordinates.
    NodeConn : list
        Nodal connectivity list. All elements should be 5-Node pyramidal elements.
    method : str, optional
        Method of decomposition to use for tetrahedralization.
        '1to2'  - Not continuously oriented, no nodes added, all elements decomposed the same way
        '1to2c'  - Continuously oriented, no nodes added (default)
        '1to4' - Continuously oriented, nodes added at center of the quad faces

    Returns
    -------
    NewCoords : array_like
        New list of nodal coordinates. For '1to2' this will be unchanged from
        the input.
    TetConn, np.ndarray
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
        return NodeCoords, np.empty((0,4),dtype=int)
    
    if method == '1to2':
        ArrayConn = np.asarray(NodeConn)
        TetConn = -1*np.ones((len(NodeConn)*2,4))
        TetConn[0::2] = ArrayConn[:,[0,1,2,4]]
        TetConn[1::2] = ArrayConn[:,[0,2,3,4]]
        TetConn = TetConn.astype(int)
        NewCoords = NodeCoords

    elif method == '1to2c':
        ArrayConn = np.asarray(NodeConn)
        # Get the quadrilateral face
        face = ArrayConn[:,[0,1,2,3]]

        # Choose the starting point for drawing the diagonal as the smallest 
        # node index of each face
        a = np.argmin(face, axis=1)
        b = a-3
        c = a-2
        d = a-1
        e = np.repeat(4,len(a))
        TetConn = -1*np.ones((len(ArrayConn)*2,4), dtype=int)
        indices = np.arange(len(ArrayConn))
        TetConn[0::2,0] = face[indices,a]
        TetConn[0::2,1] = face[indices,b]
        TetConn[0::2,2] = face[indices,c]
        TetConn[0::2,3] = ArrayConn[indices,e]
        TetConn[1::2,0] = face[indices,a]
        TetConn[1::2,1] = face[indices,c]
        TetConn[1::2,2] = face[indices,d]
        TetConn[1::2,3] = ArrayConn[indices,e]
        TetConn = TetConn.astype(int)
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

        TetConn = TetConn.astype(int)

    return NewCoords, TetConn

def faces2surface(Faces, return_ids=False):
    """
    Identify surface elements, i.e. faces that aren't shared between two elements

    Parameters
    ----------
    Faces : list
        Nodal connectivity of mesh faces (as obtained by solid2faces)
    return_ids : bool, optional
        If True, will return the face ids that correspond to each surface,
        i.e. `SurfConn = Faces[ids]`. By default, False.

    Returns
    -------
    SurfConn
        Nodal connectivity of the surface mesh.
    
    """    
    
    table = dict()
    for i,face in enumerate(Faces):
        f = tuple(sorted(face))
        if f in table:
            table[f] = -1
        else:
            table[f] = i
           
    SurfConn = [Faces[i] for key, i in table.items() if i != -1]
    if len(Faces) == len(SurfConn):
        SurfConn=Faces
    if return_ids:
        vals = np.array(list(table.values()))
        ids = vals[vals != -1]
        return SurfConn, ids
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
        faces to the list of original faces (UFaces = Faces[idx]).
    inv : np.ndarray, optional
        The array of indices that relate the unique list of
        faces to the list of original faces (Faces = UFaces[inv]).

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
    if len(out) == 1:
        out = out[0]
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
        ArrayConn = np.asarray(NodeConn)
        Faces = -1*np.ones((len(NodeConn)*4,3),dtype=int)
        # Faces = [None]*len(NodeConn)*4
        Faces[0::4] = ArrayConn[:,[0,2,1]]
        Faces[1::4] = ArrayConn[:,[0,1,3]]
        Faces[2::4] = ArrayConn[:,[1,2,3]]
        Faces[3::4] = ArrayConn[:,[0,3,2]]
    else:
        Faces = np.empty((0,3))

    return Faces

def tet102faces(NodeCoords,NodeConn):
    """
    Extract triangular faces from all elements of a purely 10-Node tetrahedral mesh.
    All faces will be ordered such that the nodes are in counter-clockwise order when
    viewed from outside of the element. Best practice is to use solid2faces, rather than 
    using tet102faces directly.

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
        ArrayConn = np.asarray(NodeConn)
        Faces = -1*np.ones((len(NodeConn)*4,6),dtype=int)
        Faces[0::4] = ArrayConn[:,[0,2,1,6,5,4]]
        Faces[1::4] = ArrayConn[:,[0,1,3,4,8,7]]
        Faces[2::4] = ArrayConn[:,[1,2,3,5,9,8]]
        Faces[3::4] = ArrayConn[:,[0,3,2,7,9,6]]
    else:
        Faces = np.empty((0,6))

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
        Faces = Faces.astype(int)
    else:
        Faces = np.empty((0,4))
    return Faces

def hex202faces(NodeCoords,NodeConn):
    """
    Extract quadrilateral faces from all elements of a purely 20-Node hexahedral mesh.
    All faces will be ordered such that the nodes are in counter-clockwise order when
    viewed from outside of the element. Best practice is to use solid2faces, rather than 
    using hex202faces directly.

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
        Faces = -1*np.ones((len(NodeConn)*6,8))
        Faces[0::6] = ArrayConn[:,[0,3,2,1,11,10,9,8]]
        Faces[1::6] = ArrayConn[:,[0,1,5,4,8,17,12,16]]
        Faces[2::6] = ArrayConn[:,[1,2,6,5,9,18,13,17]]
        Faces[3::6] = ArrayConn[:,[2,3,7,6,10,19,14,18]]
        Faces[4::6] = ArrayConn[:,[3,0,4,7,11,16,15,19]]
        Faces[5::6] = ArrayConn[:,[4,5,6,7,12,13,14,15]]
        Faces = Faces.astype(int)
    else:
        Faces = np.empty((0,8))
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

def tri2edges(NodeCoords,NodeConn):
    """
    Extract edges from all elements of a purely 3-Node triangular mesh.
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
    
    # Explode surface elements into edges
    if len(NodeConn) > 0:
        ArrayConn = np.asarray(NodeConn)
        Edges = -1*np.ones((len(NodeConn)*3,2))
        Edges[0::3] = ArrayConn[:,[0,1]]
        Edges[1::3] = ArrayConn[:,[1,2]]
        Edges[2::3] = ArrayConn[:,[2,0]]
        Edges = Edges.astype(int)
    else:
        Edges = np.empty((0,2),dtype=int)
    
    return Edges

def quad2edges(NodeCoords,NodeConn):
    """
    Extract edges from all elements of a purely 4-Node quadrilateral mesh.
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
        Edges[0::4] = ArrayConn[:,[0,1]]
        Edges[1::4] = ArrayConn[:,[1,2]]
        Edges[2::4] = ArrayConn[:,[2,3]]
        Edges[3::4] = ArrayConn[:,[3,0]]

        Edges = Edges.astype(int)
    else:
        Edges = np.empty((0,2),dtype=int)
    
    return Edges

def polygon2edges(NodeCoords,NodeConn):
    """
    Extract edges from all elements of a polygonal mesh.
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
    Extract edges from all elements of a purely 4-Node tetrahedral mesh.
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
        Edges = -1*np.ones((len(NodeConn)*6,2),dtype=np.int64)
        Edges[0::6] = ArrayConn[:,np.array([0,1])]
        Edges[1::6] = ArrayConn[:,np.array([1,2])]
        Edges[2::6] = ArrayConn[:,np.array([2,0])]
        Edges[3::6] = ArrayConn[:,np.array([0,3])]
        Edges[4::6] = ArrayConn[:,np.array([1,3])]
        Edges[5::6] = ArrayConn[:,np.array([2,3])]
    else:
        Edges = np.empty((0,2),dtype=np.int64)
    return Edges

def tet102edges(NodeCoords,NodeConn):
    """
    Extract edges from all elements of a purely 10-Node tetrahedral mesh.

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
        Edges = -1*np.ones((len(NodeConn)*6,3),dtype=np.int64)
        Edges[0::6] = ArrayConn[:,np.array([0,4,1])]
        Edges[1::6] = ArrayConn[:,np.array([1,5,2])]
        Edges[2::6] = ArrayConn[:,np.array([2,6,0])]
        Edges[3::6] = ArrayConn[:,np.array([0,7,3])]
        Edges[4::6] = ArrayConn[:,np.array([1,8,3])]
        Edges[5::6] = ArrayConn[:,np.array([2,9,3])]
    else:
        Edges = np.empty((0,3),dtype=np.int64)
    return Edges

def pyramid2edges(NodeCoords,NodeConn):
    """
    Extract edges from all elements of a purely 5-Node pyramidal mesh.
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
        Edges[0::8] = ArrayConn[:,[0,1]]
        Edges[1::8] = ArrayConn[:,[1,2]]
        Edges[2::8] = ArrayConn[:,[2,3]]
        Edges[3::8] = ArrayConn[:,[3,0]]
        Edges[4::8] = ArrayConn[:,[0,4]]
        Edges[5::8] = ArrayConn[:,[1,4]]
        Edges[6::8] = ArrayConn[:,[2,4]]
        Edges[7::8] = ArrayConn[:,[3,4]]
        Edges = Edges.astype(int)
        Edges = Edges.tolist()
    else:
        Edges = np.empty((0,2),dtype=int)
    return Edges

def wedge2edges(NodeCoords,NodeConn):
    """
    Extract edges from all elements of a purely 6-Node wedge element mesh.
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
    else:
        Edges = np.empty((0,2),dtype=int)
    return Edges

def hex2edges(NodeCoords,NodeConn):
    """
    Extract edges from all elements of a purely 8-Node hexahedral mesh.
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
    else:
        Edges = np.empty((0,2),dtype=int)
    return Edges

def quad2tri(NodeCoords,NodeConn):
    """
    Converts a quadrilateral mesh to a triangular mesh by splitting each quad into 2 tris  

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates.
    NodeConn : array_like
        Nodal connectivity list. All elements should be 4-Node quadrilateral elements.

    Returns
    -------
    TriConn : list
        list of nodal connectivities for the new triangular mesh.
    """
    # TriNodeConn = [[] for i in range(2*len(QuadNodeConn))]
    # for i in range(len(QuadNodeConn)):
    #     TriNodeConn[2*i] = [QuadNodeConn[i][0], QuadNodeConn[i][1], QuadNodeConn[i][3]]
    #     TriNodeConn[2*i+1] = [QuadNodeConn[i][1], QuadNodeConn[i][2], QuadNodeConn[i][3]]

    if len(NodeConn) == 0:
        return NodeCoords, np.empty((0,3),dtype=int)

    ArrayConn = np.asarray(NodeConn, dtype=int)
    TriConn = -1*np.ones((len(NodeConn)*2,3),dtype=int)
    TriConn[0::2] = ArrayConn[:,[0,1,3,]]
    TriConn[1::2] = ArrayConn[:,[1,2,3,]]
    
    return NodeCoords, TriConn

def edge32linear(NodeCoords, NodeConn):
    """
    Converts a 3-node edge mesh to a 2-node edge mesh.
    Assumes a 3-node edge numbering scheme where the first and third nodes define
    the end points, the remaining nodes are thus neglected.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    NodeConn : array_like
        Nodal connectivities for a 3-Node edge mesh

    Returns
    -------
    NodeCoords : array_like
        List of nodal coordinates (unchanged from input). 
    NewConn : np.ndarray
        Nodal connectivities for the equivalent 2-Node edge mesh
    """
    if len(NodeConn) == 0:
        return NodeCoords, np.empty((0,2),dtype=int)
    NewConn = np.asarray(NodeConn)[:, [0,2]]
    
    return NodeCoords, NewConn

def edge2quadratic(NodeCoords, NodeConn, cleanup=True):
    """
    Converts a 2 node edge mesh to 3 node edge mesh. A new node 
    is placed at the midpoint of each edge.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    NodeConn : array_like
        Nodal connectivities for a 2-node edge mesh
    cleanup : bool
        Cleanup/merge duplicate nodes created during the process, by default True

    Returns
    -------
    NodeCoords : np.ndarray
        New list of nodal coordinates. 
    NewConn : np.ndarray
        Nodal connectivities for the 3-Node edge mesh

    """
    if len(NodeConn) > 0:
        NodeCoords = np.asarray(NodeCoords)
        NodeConn = np.asarray(NodeConn)
        Nodes01 = (NodeCoords[NodeConn[:,0]] + NodeCoords[NodeConn[:,1]])/2

        n = len(NodeCoords)
        m = len(NodeConn)
        ids01 = np.arange(    n, n+m)

        NewConn = np.empty((m, 3),dtype=int)
        NewConn[:, [0,2]] = NodeConn
        NewConn[:, 1] = ids01

        NewCoords = np.vstack([NodeCoords, Nodes01])
        if cleanup:
            NewCoords, NewConn = utils.DeleteDuplicateNodes(NewCoords, NewConn)
    else:
        NewCoords = NodeCoords
        NewConn = np.empty((0,3), dtype=int)

    return NewCoords, NewConn

def tri62linear(NodeCoords, NodeConn):
    """
    Converts a 6-node triangular mesh to a 3-node triangular mesh.
    Assumes a 6-node triangular numbering scheme where the first 3 nodes define
    the triangular vertices, the remaining nodes are thus neglected.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    NodeConn : array_like
        Nodal connectivities for a 6-Node triangular mesh

    Returns
    -------
    NodeCoords : array_like
        List of nodal coordinates (unchanged from input). 
    NewConn : np.ndarray
        Nodal connectivities for the equivalent 3-Node triangular mesh
    """
    if len(NodeConn) == 0:
        return NodeCoords, np.empty((0,3),dtype=int)
    NewConn = np.asarray(NodeConn)[:, :3]
    
    return NodeCoords, NewConn

def tri2quadratic(NodeCoords, NodeConn, cleanup=True):
    """
    Converts a 3 node triangular mesh to 6 node triangular mesh. A new node 
    is placed at the midpoint of each edge.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    NodeConn : array_like
        Nodal connectivities for a 3-node triangular mesh
    cleanup : bool
        Cleanup/merge duplicate nodes created during the process, by default True

    Returns
    -------
    NodeCoords : np.ndarray
        New list of nodal coordinates. 
    NewConn : np.ndarray
        Nodal connectivities for the 6-Node triangular mesh

    """
    if len(NodeConn) > 0:
        NodeCoords = np.asarray(NodeCoords)
        NodeConn = np.asarray(NodeConn)
        Nodes01 = (NodeCoords[NodeConn[:,0]] + NodeCoords[NodeConn[:,1]])/2
        Nodes12 = (NodeCoords[NodeConn[:,1]] + NodeCoords[NodeConn[:,2]])/2
        Nodes20 = (NodeCoords[NodeConn[:,2]] + NodeCoords[NodeConn[:,0]])/2

        n = len(NodeCoords)
        m = len(NodeConn)
        ids01 = np.arange(    n, n+m)
        ids12 = np.arange(  n+m, n+2*m)
        ids20 = np.arange(n+2*m, n+3*m)

        NewConn = np.empty((m, 6),dtype=int)
        NewConn[:, :3] = NodeConn
        NewConn[:, 3] = ids01
        NewConn[:, 4] = ids12
        NewConn[:, 5] = ids20

        NewCoords = np.vstack([NodeCoords, Nodes01, Nodes12, Nodes20])
        if cleanup:
            NewCoords, NewConn = utils.DeleteDuplicateNodes(NewCoords, NewConn)
    else:
        NewCoords = NodeCoords
        NewConn = np.empty((0,6), dtype=int)

    return NewCoords, NewConn

def quad82linear(NodeCoords, NodeConn):
    """
    Converts a 8-node quadrilateral mesh to a 4-node quadrilateral mesh.
    Assumes a 8-node quadrilateral numbering scheme where the first 4 nodes define
    the quadrilateral vertices, the remaining nodes are thus neglected.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    NodeConn : array_like
        Nodal connectivities for a 8-Node quadrilateral mesh

    Returns
    -------
    NodeCoords : array_like
        List of nodal coordinates (unchanged from input). 
    NewConn : np.ndarray
        Nodal connectivities for the equivalent 4-Node quadrilateral mesh
    """
    if len(NodeConn) == 0:
        return NodeCoords, np.empty((0,4),dtype=int)
    NewConn = np.asarray(NodeConn)[:, :4]
    
    return NodeCoords, NewConn

def quad2quadratic(NodeCoords, NodeConn, cleanup=True):
    """
    Converts a 4 node quadrilateral mesh to 8 node quadrilateral mesh. A new node 
    is placed at the midpoint of each edge.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    NodeConn : array_like
        Nodal connectivities for a 4-node quadrilateral mesh
    cleanup : bool
        Cleanup/merge duplicate nodes created during the process, by default True

    Returns
    -------
    NodeCoords : np.ndarray
        New list of nodal coordinates. 
    NewConn : np.ndarray
        Nodal connectivities for the 8-Node quadrilateral mesh

    """
    if len(NodeConn) > 0:
        NodeCoords = np.asarray(NodeCoords)
        NodeConn = np.asarray(NodeConn)
        Nodes01 = (NodeCoords[NodeConn[:,0]] + NodeCoords[NodeConn[:,1]])/2
        Nodes12 = (NodeCoords[NodeConn[:,1]] + NodeCoords[NodeConn[:,2]])/2
        Nodes23 = (NodeCoords[NodeConn[:,2]] + NodeCoords[NodeConn[:,3]])/2
        Nodes30 = (NodeCoords[NodeConn[:,3]] + NodeCoords[NodeConn[:,0]])/2

        n = len(NodeCoords)
        m = len(NodeConn)
        ids01 = np.arange(    n, n+m)
        ids12 = np.arange(  n+m, n+2*m)
        ids23 = np.arange(n+2*m, n+3*m)
        ids30 = np.arange(n+3*m, n+4*m)

        NewConn = np.empty((m, 8),dtype=int)
        NewConn[:, :4] = NodeConn
        NewConn[:, 4] = ids01
        NewConn[:, 5] = ids12
        NewConn[:, 6] = ids23
        NewConn[:, 7] = ids30

        NewCoords = np.vstack([NodeCoords, Nodes01, Nodes12, Nodes23, Nodes30])
        if cleanup:
            NewCoords, NewConn = utils.DeleteDuplicateNodes(NewCoords, NewConn)
    else:
        NewCoords = NodeCoords
        NewConn = np.empty((0,8), dtype=int)
    return NewCoords, NewConn

def tet102linear(NodeCoords, Tet10NodeConn):
    """
    Converts a 10-node tetradehdral mesh to a 4-node tetradehedral mesh.
    Assumes a 10-node tetrahedral numbering scheme where the first 4 nodes define
    the tetrahedral vertices, the remaining nodes are thus neglected.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    Tet10NodeConn : array_like
        Nodal connectivities for a 10-Node tetrahedral mesh

    Returns
    -------
    NodeCoords : array_like
        List of nodal coordinates (unchanged from input). 
    Tet4NodeConn : np.ndarray
        Nodal connectivities for the equivalent 4-Node tetrahedral mesh
    """
    if len(Tet10NodeConn) == 0:
        return NodeCoords, np.empty((0,4),dtype=int)
    Tet4NodeConn = np.asarray(Tet10NodeConn)[:, :4]
    
    return NodeCoords, Tet4NodeConn

def tet2quadratic(NodeCoords, Tet4NodeConn, cleanup=True):
    """
    Converts a 4 node tetrahedral mesh to 10 node tetrahedral mesh. A new node 
    is placed at the midpoint of each edge.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    Tet4NodeConn : array_like
        Nodal connectivities for a 4-node tetrahedral mesh
    cleanup : bool
        Cleanup/merge duplicate nodes created during the process, by default True

    Returns
    -------
    NodeCoords : np.ndarray
        New list of nodal coordinates. 
    Tet10NodeConn : np.ndarray
        Nodal connectivities for the 10-Node tetrahedral mesh

    """
    if len(Tet4NodeConn) > 0:
        NodeCoords = np.asarray(NodeCoords)
        Tet4NodeConn = np.asarray(Tet4NodeConn)
        Nodes01 = (NodeCoords[Tet4NodeConn[:,0]] + NodeCoords[Tet4NodeConn[:,1]])/2
        Nodes12 = (NodeCoords[Tet4NodeConn[:,1]] + NodeCoords[Tet4NodeConn[:,2]])/2
        Nodes20 = (NodeCoords[Tet4NodeConn[:,2]] + NodeCoords[Tet4NodeConn[:,0]])/2
        Nodes03 = (NodeCoords[Tet4NodeConn[:,0]] + NodeCoords[Tet4NodeConn[:,3]])/2
        Nodes13 = (NodeCoords[Tet4NodeConn[:,1]] + NodeCoords[Tet4NodeConn[:,3]])/2
        Nodes23 = (NodeCoords[Tet4NodeConn[:,2]] + NodeCoords[Tet4NodeConn[:,3]])/2

        n = len(NodeCoords)
        m = len(Tet4NodeConn)
        ids01 = np.arange(    n, n+m)
        ids12 = np.arange(  n+m, n+2*m)
        ids20 = np.arange(n+2*m, n+3*m)
        ids03 = np.arange(n+3*m, n+4*m)
        ids13 = np.arange(n+4*m, n+5*m)
        ids23 = np.arange(n+5*m, n+6*m)

        Tet10NodeConn = np.empty((m, 10),dtype=int)
        Tet10NodeConn[:, :4] = Tet4NodeConn
        Tet10NodeConn[:, 4] = ids01
        Tet10NodeConn[:, 5] = ids12
        Tet10NodeConn[:, 6] = ids20
        Tet10NodeConn[:, 7] = ids03
        Tet10NodeConn[:, 8] = ids13
        Tet10NodeConn[:, 9] = ids23

        NewCoords = np.vstack([NodeCoords, Nodes01, Nodes12, Nodes20, Nodes03, Nodes13, Nodes23])

        if cleanup:
            NewCoords, Tet10NodeConn = utils.DeleteDuplicateNodes(NewCoords, Tet10NodeConn)
    else:
        NewCoords = NodeCoords
        Tet10NodeConn = np.empty((0,10), dtype=int)
    return NewCoords, Tet10NodeConn

def pyr132linear(NodeCoords, NodeConn):
    """
    Converts a 13-node pyramid mesh to a 5-node pyramid mesh.
    Assumes a 13-node pyramid numbering scheme where the first 5 nodes define
    the pyramid vertices, the remaining nodes are thus neglected.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    NodeConn : array_like
        Nodal connectivities for a 13-Node pyramid mesh

    Returns
    -------
    NodeCoords : array_like
        List of nodal coordinates (unchanged from input). 
    NodeConn : np.ndarray
        Nodal connectivities for the equivalent 5-Node pyramid mesh
    """
    if len(NodeConn) == 0:
        return NodeCoords, np.empty((0,5),dtype=int)
    NewConn = np.asarray(NodeConn)[:, :5]
    
    return NodeCoords, NewConn

def pyr2quadratic(NodeCoords, NodeConn, cleanup=True):
    """
    Converts a 5 node pyramid mesh to 13 node pyramid mesh. A new node 
    is placed at the midpoint of each edge.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    NodeConn : array_like
        Nodal connectivities for a 5-node pyramid mesh
    cleanup : bool
        Cleanup/merge duplicate nodes created during the process, by default True
        
    Returns
    -------
    NodeCoords : np.ndarray
        New list of nodal coordinates. 
    NewConn : np.ndarray
        Nodal connectivities for the 13-Node pyramid mesh

    """
    if len(NodeConn) > 0:
        NodeCoords = np.asarray(NodeCoords)
        NodeConn = np.asarray(NodeConn)
        Nodes01 = (NodeCoords[NodeConn[:,0]] + NodeCoords[NodeConn[:,1]])/2
        Nodes12 = (NodeCoords[NodeConn[:,1]] + NodeCoords[NodeConn[:,2]])/2
        Nodes23 = (NodeCoords[NodeConn[:,2]] + NodeCoords[NodeConn[:,3]])/2
        Nodes30 = (NodeCoords[NodeConn[:,3]] + NodeCoords[NodeConn[:,0]])/2
        Nodes04 = (NodeCoords[NodeConn[:,0]] + NodeCoords[NodeConn[:,4]])/2
        Nodes14 = (NodeCoords[NodeConn[:,1]] + NodeCoords[NodeConn[:,4]])/2
        Nodes24 = (NodeCoords[NodeConn[:,2]] + NodeCoords[NodeConn[:,4]])/2
        Nodes34 = (NodeCoords[NodeConn[:,3]] + NodeCoords[NodeConn[:,4]])/2

        n = len(NodeCoords)
        m = len(NodeConn)
        ids01 = np.arange(    n, n+m)
        ids12 = np.arange(  n+m, n+2*m)
        ids23 = np.arange(n+2*m, n+3*m)
        ids30 = np.arange(n+3*m, n+4*m)
        ids04 = np.arange(n+4*m, n+5*m)
        ids14 = np.arange(n+5*m, n+6*m)
        ids24 = np.arange(n+6*m, n+7*m)
        ids34 = np.arange(n+7*m, n+8*m)

        NewConn = np.empty((m, 13),dtype=int)
        NewConn[:, :5] = NodeConn
        NewConn[:, 5] = ids01
        NewConn[:, 6] = ids12
        NewConn[:, 7] = ids23
        NewConn[:, 8] = ids30
        NewConn[:, 9] = ids04
        NewConn[:, 10] = ids14
        NewConn[:, 11] = ids24
        NewConn[:, 12] = ids34

        NewCoords = np.vstack([NodeCoords, Nodes01, Nodes12, Nodes23, Nodes30, Nodes04, Nodes14, Nodes24, Nodes34])

        if cleanup:
            NewCoords, NewConn = utils.DeleteDuplicateNodes(NewCoords, NewConn)
    else:
        NewCoords = NodeCoords
        NewConn = np.empty((0,13), dtype=int)
    return NewCoords, NewConn

def wdg152linear(NodeCoords, NodeConn):
    """
    Converts a 15-node wedge mesh to a 6-node wedge mesh.
    Assumes a 15-node wedge numbering scheme where the first 6 nodes define
    the wedge vertices, the remaining nodes are thus neglected.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    NodeConn : array_like
        Nodal connectivities for a 15-Node pyramid mesh

    Returns
    -------
    NodeCoords : array_like
        List of nodal coordinates (unchanged from input). 
    NodeConn : np.ndarray
        Nodal connectivities for the equivalent 6-Node wedge mesh
    """
    if len(NodeConn) == 0:
        return NodeCoords, np.empty((0,6),dtype=int)
    NewConn = np.asarray(NodeConn)[:, :6]
    
    return NodeCoords, NewConn

def wdg2quadratic(NodeCoords, NodeConn, cleanup=True):
    """
    Converts a 6 node wedge mesh to 15 node wedge mesh. A new node 
    is placed at the midpoint of each edge.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    NodeConn : array_like
        Nodal connectivities for a 6-node wedge mesh
    cleanup : bool
        Cleanup/merge duplicate nodes created during the process, by default True

    Returns
    -------
    NodeCoords : np.ndarray
        New list of nodal coordinates. 
    NewConn : np.ndarray
        Nodal connectivities for the 15-Node wedge mesh

    """
    if len(NodeConn) > 0:
        NodeCoords = np.asarray(NodeCoords)
        NodeConn = np.asarray(NodeConn)
        Nodes01 = (NodeCoords[NodeConn[:,0]] + NodeCoords[NodeConn[:,1]])/2
        Nodes12 = (NodeCoords[NodeConn[:,1]] + NodeCoords[NodeConn[:,2]])/2
        Nodes20 = (NodeCoords[NodeConn[:,2]] + NodeCoords[NodeConn[:,0]])/2
        Nodes34 = (NodeCoords[NodeConn[:,3]] + NodeCoords[NodeConn[:,4]])/2
        Nodes45 = (NodeCoords[NodeConn[:,4]] + NodeCoords[NodeConn[:,5]])/2
        Nodes53 = (NodeCoords[NodeConn[:,5]] + NodeCoords[NodeConn[:,3]])/2
        Nodes03 = (NodeCoords[NodeConn[:,0]] + NodeCoords[NodeConn[:,3]])/2
        Nodes14 = (NodeCoords[NodeConn[:,1]] + NodeCoords[NodeConn[:,4]])/2
        Nodes25 = (NodeCoords[NodeConn[:,2]] + NodeCoords[NodeConn[:,5]])/2
        

        n = len(NodeCoords)
        m = len(NodeConn)
        ids01 = np.arange(    n, n+m)
        ids12 = np.arange(  n+m, n+2*m)
        ids20 = np.arange(n+2*m, n+3*m)
        ids34 = np.arange(n+3*m, n+4*m)
        ids45 = np.arange(n+4*m, n+5*m)
        ids53 = np.arange(n+5*m, n+6*m)
        ids03 = np.arange(n+6*m, n+7*m)
        ids14 = np.arange(n+7*m, n+8*m)
        ids25 = np.arange(n+8*m, n+9*m)

        NewConn = np.empty((m, 15),dtype=int)
        NewConn[:, :6] = NodeConn
        NewConn[:, 6] = ids01
        NewConn[:, 7] = ids12
        NewConn[:, 8] = ids20
        NewConn[:, 9] = ids34
        NewConn[:, 10] = ids45
        NewConn[:, 11] = ids53
        NewConn[:, 12] = ids03
        NewConn[:, 13] = ids14
        NewConn[:, 14] = ids25

        NewCoords = np.vstack([NodeCoords, Nodes01, Nodes12, Nodes20, Nodes34, Nodes45, Nodes53, Nodes03, Nodes14, Nodes25])

        if cleanup:
            NewCoords, NewConn = utils.DeleteDuplicateNodes(NewCoords, NewConn)
    else:
        NewCoords = NodeCoords
        NewConn = np.empty((0,15), dtype=int)
    return NewCoords, NewConn

def hex202linear(NodeCoords, Hex20NodeConn):
    """
    Converts a 20-node hexahedral mesh to an 8-node hexahedral mesh.
    Assumes a 20-node hexahedral numbering scheme where the first 8 nodes define
    the hexahedral vertices, the remaining nodes are thus neglected.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    Hex20NodeConn : array_like
        Nodal connectivities for a 20-Node hexahedral mesh

    Returns
    -------
    NodeCoords : array_like
        List of nodal coordinates (unchanged from input). 
    Hex8NodeConn : np.ndarray
        Nodal connectivities for the equivalent 8-Node hexahedral mesh
    """
    if len(Hex20NodeConn) == 0:
        return NodeCoords, np.empty((0,8),dtype=int)
    Hex8NodeConn = np.asarray(Hex20NodeConn)[:, :8]
    
    return NodeCoords, Hex8NodeConn

def hex2quadratic(NodeCoords, Hex8NodeConn, cleanup=True):
    """
    Converts a 4 node tetrahedral mesh to 10 node tetrahedral mesh. A new node 
    is placed at the midpoint of each edge.

    Parameters
    ----------
    NodeCoords : array_like
        List of nodal coordinates. 
    Hex8NodeConn : array_like
        Nodal connectivities for a 4-node tetrahedral mesh
    cleanup : bool
        Cleanup/merge duplicate nodes created during the process, by default True

    Returns
    -------
    NodeCoords : np.ndarray
        New list of nodal coordinates. 
    Hex20NodeConn : np.ndarray
        Nodal connectivities for the 20-Node tetrahedral mesh

    """
    if len(Hex8NodeConn) > 0:
        NodeCoords = np.asarray(NodeCoords)
        Hex8NodeConn = np.asarray(Hex8NodeConn)
        Nodes01 = (NodeCoords[Hex8NodeConn[:,0]] + NodeCoords[Hex8NodeConn[:,1]])/2
        Nodes12 = (NodeCoords[Hex8NodeConn[:,1]] + NodeCoords[Hex8NodeConn[:,2]])/2
        Nodes23 = (NodeCoords[Hex8NodeConn[:,2]] + NodeCoords[Hex8NodeConn[:,3]])/2
        Nodes30 = (NodeCoords[Hex8NodeConn[:,3]] + NodeCoords[Hex8NodeConn[:,0]])/2

        Nodes45 = (NodeCoords[Hex8NodeConn[:,4]] + NodeCoords[Hex8NodeConn[:,5]])/2
        Nodes56 = (NodeCoords[Hex8NodeConn[:,5]] + NodeCoords[Hex8NodeConn[:,6]])/2
        Nodes67 = (NodeCoords[Hex8NodeConn[:,6]] + NodeCoords[Hex8NodeConn[:,7]])/2
        Nodes74 = (NodeCoords[Hex8NodeConn[:,7]] + NodeCoords[Hex8NodeConn[:,4]])/2

        Nodes04 = (NodeCoords[Hex8NodeConn[:,0]] + NodeCoords[Hex8NodeConn[:,4]])/2
        Nodes15 = (NodeCoords[Hex8NodeConn[:,1]] + NodeCoords[Hex8NodeConn[:,5]])/2
        Nodes26 = (NodeCoords[Hex8NodeConn[:,2]] + NodeCoords[Hex8NodeConn[:,6]])/2
        Nodes37 = (NodeCoords[Hex8NodeConn[:,3]] + NodeCoords[Hex8NodeConn[:,7]])/2

        n = len(NodeCoords)
        m = len(Hex8NodeConn)
        ids01 = np.arange(    n, n+m)
        ids12 = np.arange(  n+m, n+2*m)
        ids23 = np.arange(n+2*m, n+3*m)
        ids30 = np.arange(n+3*m, n+4*m)
        ids45 = np.arange(n+4*m, n+5*m)
        ids56 = np.arange(n+5*m, n+6*m)
        ids67 = np.arange(n+6*m, n+7*m)
        ids74 = np.arange(n+7*m, n+8*m)
        ids04 = np.arange(n+8*m, n+9*m)
        ids15 = np.arange(n+9*m, n+10*m)
        ids26 = np.arange(n+10*m, n+11*m)
        ids37 = np.arange(n+11*m, n+12*m)

        Hex20NodeConn = np.empty((m, 20),dtype=int)
        Hex20NodeConn[:, :8] = Hex8NodeConn
        Hex20NodeConn[:, 8] = ids01
        Hex20NodeConn[:, 9] = ids12
        Hex20NodeConn[:,10] = ids23
        Hex20NodeConn[:,11] = ids30
        Hex20NodeConn[:,12] = ids45
        Hex20NodeConn[:,13] = ids56
        Hex20NodeConn[:,14] = ids67
        Hex20NodeConn[:,15] = ids74
        Hex20NodeConn[:,16] = ids04
        Hex20NodeConn[:,17] = ids15
        Hex20NodeConn[:,18] = ids26
        Hex20NodeConn[:,19] = ids37

        NewCoords = np.vstack([NodeCoords, Nodes01, Nodes12, Nodes23, Nodes30, Nodes45, Nodes56, Nodes67, Nodes74, Nodes04, Nodes15, Nodes26, Nodes37])

        if cleanup:
            NewCoords, Hex20NodeConn = utils.DeleteDuplicateNodes(NewCoords, Hex20NodeConn)
    else:
        NewCoords = NodeCoords
        Hex20NodeConn = np.empty((0,20), dtype=int)
    return NewCoords, Hex20NodeConn

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
    if len(edges) == 0:
        Edges = edges
        return Edges
    UEdges, indices, counts = edges2unique(edges, return_idx=True, return_counts=True)

    EdgeIdx = indices[np.where(counts==1)]
    Edges = np.asarray(edges)[EdgeIdx]

    return Edges

def hexsubdivide(NodeCoords, NodeConn):
    """
    Subdivide hexahedra into 8 sub-hexahedra, connecting corners to the 
    element, face, and edge centroids.

    Parameters
    ----------
    NodeCoords : array_like
        List of node coordinates
    NodeConn : array_like
        List of node connectivities (must be purely hexahedra, shape=(n,8))

    Returns
    -------
    NewCoords : np.ndarray
        Node coordinates of the subdivided mesh. The new nodes are 
        appended to the original nodes, so the node numbers of the original 
        mesh will refer to the same nodes in the new mesh.
    NewConn : np.ndarray
        Node connectivities of the subdivided mesh. The elements are ordered 
        so that the first element in the original mesh is subdivided into
        the first 8 elements in the new mesh, the second element in the mesh
        becomes the next 8 elements, and so on.
    """
    ArrayCoords = np.asarray(NodeCoords)
    ArrayConn = np.asarray(NodeConn, dtype=int)

    Centroids = utils.Centroids(ArrayCoords,NodeConn)
    Face0Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,1,2,3]]],axis=1)
    Face1Centroids = np.mean(ArrayCoords[ArrayConn[:,[0,1,5,4]]],axis=1)
    Face2Centroids = np.mean(ArrayCoords[ArrayConn[:,[1,2,6,5]]],axis=1)
    Face3Centroids = np.mean(ArrayCoords[ArrayConn[:,[2,3,7,6]]],axis=1)
    Face4Centroids = np.mean(ArrayCoords[ArrayConn[:,[3,0,4,7]]],axis=1)
    Face5Centroids = np.mean(ArrayCoords[ArrayConn[:,[4,5,6,7]]],axis=1)
    Edge01 = np.mean(ArrayCoords[ArrayConn[:,[0,1]]],axis=1)
    Edge12 = np.mean(ArrayCoords[ArrayConn[:,[1,2]]],axis=1)
    Edge23 = np.mean(ArrayCoords[ArrayConn[:,[2,3]]],axis=1)
    Edge30 = np.mean(ArrayCoords[ArrayConn[:,[3,0]]],axis=1)
    Edge04 = np.mean(ArrayCoords[ArrayConn[:,[0,4]]],axis=1)
    Edge15 = np.mean(ArrayCoords[ArrayConn[:,[1,5]]],axis=1)
    Edge26 = np.mean(ArrayCoords[ArrayConn[:,[2,6]]],axis=1)
    Edge37 = np.mean(ArrayCoords[ArrayConn[:,[3,7]]],axis=1)
    Edge45 = np.mean(ArrayCoords[ArrayConn[:,[4,5]]],axis=1)
    Edge56 = np.mean(ArrayCoords[ArrayConn[:,[5,6]]],axis=1)
    Edge67 = np.mean(ArrayCoords[ArrayConn[:,[6,7]]],axis=1)
    Edge74 = np.mean(ArrayCoords[ArrayConn[:,[7,4]]],axis=1)

    CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*0,len(NodeCoords)+len(NodeConn)*1, dtype=int)
    Face0CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*1,len(NodeCoords)+len(NodeConn)*2, dtype=int)
    Face1CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*2,len(NodeCoords)+len(NodeConn)*3, dtype=int)
    Face2CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*3,len(NodeCoords)+len(NodeConn)*4, dtype=int)
    Face3CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*4,len(NodeCoords)+len(NodeConn)*5, dtype=int)
    Face4CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*5,len(NodeCoords)+len(NodeConn)*6, dtype=int)
    Face5CentroidIds = np.arange(len(NodeCoords)+len(NodeConn)*6,len(NodeCoords)+len(NodeConn)*7, dtype=int)
    Edge01Ids = np.arange(len(NodeCoords)+len(NodeConn)*7,len(NodeCoords)+len(NodeConn)*8, dtype=int)
    Edge12Ids = np.arange(len(NodeCoords)+len(NodeConn)*8,len(NodeCoords)+len(NodeConn)*9, dtype=int)
    Edge23Ids = np.arange(len(NodeCoords)+len(NodeConn)*9,len(NodeCoords)+len(NodeConn)*10, dtype=int)
    Edge30Ids = np.arange(len(NodeCoords)+len(NodeConn)*10,len(NodeCoords)+len(NodeConn)*11, dtype=int)
    Edge04Ids = np.arange(len(NodeCoords)+len(NodeConn)*11,len(NodeCoords)+len(NodeConn)*12, dtype=int)
    Edge15Ids = np.arange(len(NodeCoords)+len(NodeConn)*12,len(NodeCoords)+len(NodeConn)*13, dtype=int)
    Edge26Ids = np.arange(len(NodeCoords)+len(NodeConn)*13,len(NodeCoords)+len(NodeConn)*14, dtype=int)
    Edge37Ids = np.arange(len(NodeCoords)+len(NodeConn)*14,len(NodeCoords)+len(NodeConn)*15, dtype=int)
    Edge45Ids = np.arange(len(NodeCoords)+len(NodeConn)*15,len(NodeCoords)+len(NodeConn)*16, dtype=int)
    Edge56Ids = np.arange(len(NodeCoords)+len(NodeConn)*16,len(NodeCoords)+len(NodeConn)*17, dtype=int)
    Edge67Ids = np.arange(len(NodeCoords)+len(NodeConn)*17,len(NodeCoords)+len(NodeConn)*18, dtype=int)
    Edge74Ids = np.arange(len(NodeCoords)+len(NodeConn)*18,len(NodeCoords)+len(NodeConn)*19, dtype=int)
    
    NewCoords = np.vstack([ArrayCoords,Centroids,Face0Centroids,Face1Centroids,Face2Centroids,Face3Centroids,Face4Centroids,Face5Centroids, Edge01, Edge12, Edge23, Edge30, Edge04, Edge15, Edge26, Edge37, Edge45, Edge56, Edge67, Edge74])        
    
    SubConn = -1*np.ones((len(NodeConn)*8,8), dtype=int)
    SubConn[0::8] = np.column_stack([ArrayConn[:,0], Edge01Ids, Face0CentroidIds, Edge30Ids, Edge04Ids, Face1CentroidIds, CentroidIds, Face4CentroidIds])
    SubConn[1::8] = np.column_stack([Edge01Ids, ArrayConn[:,1], Edge12Ids, Face0CentroidIds, Face1CentroidIds, Edge15Ids, Face2CentroidIds, CentroidIds])
    SubConn[2::8] = np.column_stack([Face0CentroidIds, Edge12Ids, ArrayConn[:,2], Edge23Ids, CentroidIds, Face2CentroidIds, Edge26Ids, Face3CentroidIds])
    SubConn[3::8] = np.column_stack([Edge30Ids, Face0CentroidIds, Edge23Ids, ArrayConn[:,3], Face4CentroidIds, CentroidIds, Face3CentroidIds, Edge37Ids])

    SubConn[4::8] = np.column_stack([Edge04Ids, Face1CentroidIds, CentroidIds, Face4CentroidIds, ArrayConn[:,4], Edge45Ids, Face5CentroidIds, Edge74Ids])
    SubConn[5::8] = np.column_stack([Face1CentroidIds, Edge15Ids, Face2CentroidIds, CentroidIds, Edge45Ids, ArrayConn[:,5], Edge56Ids, Face5CentroidIds])
    SubConn[6::8] = np.column_stack([CentroidIds, Face2CentroidIds, Edge26Ids, Face3CentroidIds, Face5CentroidIds, Edge56Ids, ArrayConn[:,6], Edge67Ids])
    SubConn[7::8] = np.column_stack([Face4CentroidIds, CentroidIds, Face3CentroidIds, Edge37Ids, Edge74Ids, Face5CentroidIds, Edge67Ids, ArrayConn[:,7]])

    return NewCoords, SubConn

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
        xscale = voxelsize[0] / scalefactor
        yscale = voxelsize[1] / scalefactor
        zscale = voxelsize[2] / scalefactor
        voxelsize = 1
        rectangular_elements=True
    else:
        voxelsize /= scalefactor
        rectangular_elements=False
    
    if isinstance(img, tuple):
        if scalefactor != 1:
            img = tuple([image.read(i, scalefactor, scaleorder) for i in img])
        multichannel = True
        multiimg = img
        img = multiimg[0]
    else:
        img = image.read(img, scalefactor, scaleorder)
    
    if crop is None:
        (nz,ny,nx) = img.shape
        xlims = [0,(nx)*voxelsize]
        ylims = [0,(ny)*voxelsize]
        zlims = [0,(nz)*voxelsize]
        bounds = [xlims[0],xlims[1],ylims[0],ylims[1],zlims[0],zlims[1]]
        VoxelCoords, VoxelConn = primitives.Grid(bounds, voxelsize, exact_h=False)
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
        VoxelCoords, VoxelConn = primitives.Grid(bounds, voxelsize, exact_h=False)
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
            VoxelCoords,VoxelConn,_ = utils.RemoveNodes(VoxelCoords,VoxelConn)
            VoxelConn = np.asarray(VoxelConn)

        elif threshold_direction == -1:
            VoxelConn = VoxelConn[VoxelData<=threshold]
            VoxelData = VoxelData[VoxelData<=threshold]
            if return_gradient: GradData = GradData[VoxelData<=threshold]
            VoxelCoords,VoxelConn,_ = utils.RemoveNodes(VoxelCoords,VoxelConn)
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
    Octree : str, tree.OctreeNode, None, optional
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
        Inside = rays.PointsInSurf(centroids, SurfCoords, SurfConn, Octree=Octree)
        VoxelConn = GridConn[Inside]
    else:
        Inside = rays.PointsInSurf(GridCoords, SurfCoords, SurfConn, Octree=Octree)
        ElemInsides = Inside[GridConn]

        if mode.lower() == 'any':
            VoxelConn = GridConn[np.any(ElemInsides,axis=1)]
        elif mode.lower() == 'all':
            VoxelConn = GridConn[np.all(ElemInsides,axis=1)]
        else:
            raise ValueError('mode must be "any", "all", or "centroid".')

    VoxelCoords,VoxelConn,_ = utils.RemoveNodes(GridCoords,VoxelConn)
    return VoxelCoords,VoxelConn

def voxel2im(VoxelCoords, VoxelConn, Vals):
    """
    Convert a rectilinear voxel mesh (grid) to a 3D image matrix.

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

def mesh2im(NodeCoords, NodeConn, voxelsize, fill=True, sdf=False, Type=None):
    """
    Convert a mesh to a binarized image. 

    Parameters
    ----------
    NodeCoords : array_like
        Coordinates of the nodes in the mesh
    NodeConn : array_like, list
        Elements in the mesh, defined by the connectivity of nodes
    voxelsize : float
        Voxel size of the image, i.e. image resolution in units/voxel
    fill : bool, optional
        Option to fill in the volume of the mesh, by default True.
        If False, only the boundary will be present in the binarized image.
        The image will only be able to be filled if the surface mesh is closed
        to the resolution of the image (i.e. if defects/holes in the mesh are
        sufficiently smaller than the `voxelsize` such that they aren't 
        resolved by the voxelization, then the image will still be able to be
        filled).
    sdf : bool, optional
        Option to make the image a signed distance field using a Euclidean distance
        transform (:func:`scipy.ndimage.distance_transform_edt`)
        on the binarized image. The returned image will have values
        less than zero inside the surface and greater than zero outside the 
        surface. The magnitude of the values is equal to the distance from the
        surface. This is intended for use with `fill=True`.
    Type : str, NoneType, optional
        Mesh Type ('surf', 'vol'), by default None.
        If not provided, the Type will be inferred using :func:`~mymesh.utils.identify_type`.

    Returns
    -------
    img : np.ndarray
        Three dimensional binarized (0,1) image of the mesh

    Examples
    --------
    To convert a surface mesh into a binary image:
    
    .. plot::

        import matplotlib.pyplot as plt

        M = primitives.Torus([0,0,0], 1, .5)

        img = converter.mesh2im(M.NodeCoords, M.NodeConn, 0.05)
        plt.imshow(img[10], cmap='gray')

    To voxelize only the surface of the mesh, the `fill=False` option can be used:

    .. plot::

        import matplotlib.pyplot as plt

        M = primitives.Torus([0,0,0], 1, .5)

        img = converter.mesh2im(M.NodeCoords, M.NodeConn, 0.05, fill=False)
        plt.imshow(img[10], cmap='gray')

    The binarized image can be converted to a signed distance field (values are
    the distance to the surface, sign is negative inside, positive outside) using
    the `sdf=True` option:

    .. plot::

        import matplotlib.pyplot as plt

        M = primitives.Torus([0,0,0], 1, .5)

        img = converter.mesh2im(M.NodeCoords, M.NodeConn, 0.05, sdf=True)
        plt.imshow(img[10], cmap='gray')

    """
    if Type is None:
        Type = utils.identify_type(NodeCoords, NodeConn)    
        
    if Type.lower() == 'surf':
        ElemTypes = utils.identify_elem(NodeCoords, NodeConn)
        if len(ElemTypes) > 1 or 'tri' not in ElemTypes:
            TriCoords, TriConn = surf2tris(NodeCoords, NodeConn)
        else:
            TriCoords, TriConn = NodeCoords, NodeConn

        root = tree.Surface2Octree(TriCoords, TriConn, minsize=voxelsize)
        leaves = tree.getAllLeaf(root)
        # flipping so x,y,z -> 2,1,0
        centroids = np.array([leaf.centroid for leaf in leaves])[:,::-1] 
        mins = np.min(centroids, axis=0)
        maxs = np.max(centroids, axis=0)

        indices = np.round((centroids - mins)/voxelsize).astype(int)
        shape = np.round((maxs-mins)/voxelsize).astype(int)
        img = np.zeros(shape+1)
        img[indices[:,0],indices[:,1],indices[:,2]] = 1

    if fill:
        img = ndimage.binary_fill_holes(img)

    if sdf:
        edt = -ndimage.distance_transform_edt(img==1)*voxelsize
        edt2 = (ndimage.distance_transform_edt(img==0)-1)*voxelsize
        edt[img==0] = edt2[img==0]
        return edt
    
    return img



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