# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 12:02:26 2022

@author: user
"""

import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import sys
from . import MeshUtils, converter
from scipy import spatial


def Sci(NodeCoords):
    """
    Sci _summary_

    Parameters
    ----------
    NodeCoords : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    return spatial.Delaunay(NodeCoords).simplices.tolist()

def BowyerWatson3d(NodeCoords):
    # Someting isn't working right
    NodeConn = []
    TempConn = []
    
    # Bounding Box
    minx = min([NodeCoords[i][0] for i in range(len(NodeCoords))])
    maxx = max([NodeCoords[i][0] for i in range(len(NodeCoords))])
    miny = min([NodeCoords[i][1] for i in range(len(NodeCoords))])
    maxy = max([NodeCoords[i][1] for i in range(len(NodeCoords))])
    minz = min([NodeCoords[i][2] for i in range(len(NodeCoords))])
    maxz = max([NodeCoords[i][2] for i in range(len(NodeCoords))])
    
    # Bounding Tetrahedron (with slight buffer)
    minX = minx+minx/10-10*np.finfo(float).eps
    maxX = maxx+maxx/10+10*np.finfo(float).eps
    minY = miny+miny/10-10*np.finfo(float).eps
    maxY = maxy+maxy/10+10*np.finfo(float).eps
    minZ = minz+minz/10-10*np.finfo(float).eps
    maxZ = maxz+maxz/10+10*np.finfo(float).eps
    tetPts = [[minX,maxY,minZ],
              [minX,minY-max(maxX-minX,maxZ-minZ),minZ],
              [maxX+max(maxY-minY,maxZ-minz),maxY,minZ],
              [minX,maxY,maxZ+max(maxX-minX,maxY-minY)]]
    tetPts = [[tetPts[i][j] + np.random.rand()*np.finfo(float).eps for j in range(len(tetPts[i]))] for i in range(len(tetPts))]
    tempCoords = NodeCoords + tetPts
    tempIdx = [len(NodeCoords),len(NodeCoords)+1,len(NodeCoords)+2,len(NodeCoords)+3]
    TempConn.append(tempIdx)
    
    for i in range(len(NodeCoords)):
        # For Each Node
        badElems = []
        Pt = NodeCoords[i]
        for j in range(len(TempConn)):
            # Check all elements
            Nodes = np.array(tempCoords)[TempConn[j]]
            try:
                C, R = TetCircumsphere(Nodes)
            except:
                print('merp')
            if dist(C, Pt) < R:
                # If new point is inside the circumsphere of an existing element
                badElems.append(j)
            
        # Identify the polyhedral hole
        poly = []        
        badFaces = Converter.tetGetFaces(tempCoords,[TempConn[k] for k in badElems])
        sortedBadFaces = [np.sort(badFaces[j]) for j in range(len(badFaces))]
        for j in range(len(badElems)):
            theseFaces = Converter.tetGetFaces(tempCoords,[TempConn[badElems[j]]])
            for k in range(len(theseFaces)):
                if sum(np.all(np.array(np.sort(theseFaces[k])) == sortedBadFaces,axis=1)) == 1:
                    poly.append(theseFaces[k])
                    
        # Remove old elements  
        sortBad = np.sort(badElems)         
        for j in range(len(sortBad)-1,-1,-1):
            del TempConn[sortBad[j]]
        
        # Retriangulate the polyhedral hole          
        for face in poly:            
            newTet = face + [i] # create a new tet between the face and the new node
            TempConn.append(newTet)
            
        print(len(TempConn))
        fig = go.Figure()
        for elem in TempConn:
            fig = plotTet(fig,tempCoords,elem)
        plot(fig)        
        sciCoords = [tempCoords[j] for j in range(len(tempCoords)) if j in np.unique(TempConn)]
        sciConn = spatial.Delaunay(sciCoords)
        print(len(sciConn.simplices))
        fig = go.Figure()
        for elem in sciConn.simplices:
            fig = plotTet(fig,sciCoords,elem)       
        plot(fig)

    NodeConn = []
    for i in range(len(TempConn)):
        add = True
        for j in tempIdx:
            if j in TempConn[i]:
                add = False
        if add:
            NodeConn.append(TempConn[i])
    
    return NodeConn
    
    
   
    
def plotTet(fig,NodeCoords,elem):
    x = [NodeCoords[elem[0]][0], NodeCoords[elem[1]][0], NodeCoords[elem[2]][0],
         NodeCoords[elem[3]][0], NodeCoords[elem[0]][0], NodeCoords[elem[2]][0],
         NodeCoords[elem[1]][0], NodeCoords[elem[3]][0]]
    y = [NodeCoords[elem[0]][1], NodeCoords[elem[1]][1], NodeCoords[elem[2]][1],
         NodeCoords[elem[3]][1], NodeCoords[elem[0]][1], NodeCoords[elem[2]][1],
         NodeCoords[elem[1]][1], NodeCoords[elem[3]][1]]
    z = [NodeCoords[elem[0]][2], NodeCoords[elem[1]][2], NodeCoords[elem[2]][2],
         NodeCoords[elem[3]][2], NodeCoords[elem[0]][2], NodeCoords[elem[2]][2],
         NodeCoords[elem[1]][2], NodeCoords[elem[3]][2]]
         
    fig.add_trace(go.Scatter3d(x=x,y=y,z=z))
    return fig
   
## Utils ##
def dist(pt1,pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)

def TetCircumsphere(Nodes):
    
    A = np.array([np.subtract(Nodes[1],Nodes[0]),
         np.subtract(Nodes[2],Nodes[0]),
         np.subtract(Nodes[3],Nodes[0])])
    if np.linalg.det(A) == 0:
        Nodes = [[Nodes[i][j] + np.random.rand()*np.finfo(float).eps for j in range(len(Nodes[i]))] for i in range(len(Nodes))]
        A = np.array([np.subtract(Nodes[1],Nodes[0]),
             np.subtract(Nodes[2],Nodes[0]),
             np.subtract(Nodes[3],Nodes[0])])
    B = 0.5 * np.array([[np.linalg.norm(Nodes[1])**2 - np.linalg.norm(Nodes[0])**2],
                        [np.linalg.norm(Nodes[2])**2 - np.linalg.norm(Nodes[0])**2],
                        [np.linalg.norm(Nodes[3])**2 - np.linalg.norm(Nodes[0])**2],
                        ])
    C = np.transpose(np.linalg.solve(A,B)).tolist()[0]
    
    
    a1 = dist(Nodes[0],Nodes[1])
    b1 = dist(Nodes[0],Nodes[2])
    c1 = dist(Nodes[0],Nodes[3])
    a2 = dist(Nodes[2],Nodes[3])
    b2 = dist(Nodes[1],Nodes[3])
    c2 = dist(Nodes[1],Nodes[2])    
    
    V = np.linalg.norm(np.dot(np.subtract(Nodes[0],Nodes[3]),
                              (np.cross(np.subtract(Nodes[1],Nodes[3]),
                                        np.subtract(Nodes[2],Nodes[3]))
                                  )))/6
    R = np.sqrt((a1*a2 + b1*b2 + c1*c2) *
                (a1*a2 + b1*b2 - c1*c2) *
                (a1*a2 - b1*b2 + c1*c2) *
                (-a1*a2 + b1*b2 + c1*c2))/(24*V)
    
    return C, R
