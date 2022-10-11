# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:35:34 2021

@author: toj
"""


def parseSTL(stlFile):
    """
    parseSTL _summary_

    Parameters
    ----------
    stlFile : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    NodeCoords = []
    NodeConn = []
    ElemNormals = []
    
    
    
    with open(stlFile) as file:
        txtlines = file.readlines()
       
    solidName = txtlines[0][6:]
    
    for i in range(1,len(txtlines)-1,7):
        ElemNormals.append([float(j) for j in txtlines[i].strip().split('facet normal ')[1].split(' ')])
        n1 = [float(j) for j in txtlines[i+2].strip().split('vertex')[1].strip().split(' ')]
        n2 = [float(j) for j in txtlines[i+3].strip().split('vertex')[1].strip().split(' ')]
        n3 = [float(j) for j in txtlines[i+4].strip().split('vertex')[1].strip().split(' ')]
        
        if n1 in NodeCoords:
            n1num = NodeCoords.index(n1)
        else:
            n1num = len(NodeCoords)
            NodeCoords.append(n1)
        
        if n2 in NodeCoords:
            n2num = NodeCoords.index(n2)
        else:
            n2num = len(NodeCoords)
            NodeCoords.append(n2)
        
        if n3 in NodeCoords:
            n3num = NodeCoords.index(n3)
        else:
            n3num = len(NodeCoords)
            NodeCoords.append(n3)
        
        NodeConn.append([n1num, n2num, n3num])
        
    
    return NodeCoords, NodeConn, ElemNormals
