# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:45:14 2022

@author: toj

To Install tetgen with Anaconda: "conda install -c conda-forge tetgen"

"""

import os, sys, subprocess, tempfile
import numpy as np

def tetgen(NodeCoords, NodeConn, holes=[], switches=[''], verbose=False, BoundingBox=False):
    
    # Use BoundingBox if the the mesh is potentially discontinuous and there are holes
    if type(NodeCoords) == np.ndarray: NodeCoords = NodeCoords.tolist()
    if type(holes) == np.ndarray: holes = holes.tolist()
    if BoundingBox:
        arrayCoords = np.array(NodeCoords)
        xmin = np.min(arrayCoords[:,0])
        xmax = np.max(arrayCoords[:,0])
        dx = xmax-xmin
        ymin = np.min(arrayCoords[:,1])
        ymax = np.max(arrayCoords[:,1])
        dy = ymax-ymin
        zmin = np.min(arrayCoords[:,2])
        zmax = np.max(arrayCoords[:,2])
        dz = zmax-zmin
        
        boxcoords = [[xmin-dx/10,ymin-dy/10,zmin-dz/10],
                     [xmax+dx/10,ymin-dy/10,zmin-dz/10],
                     [xmax+dx/10,ymax+dy/10,zmin-dz/10],
                     [xmin-dx/10,ymax+dy/10,zmin-dz/10],
                     
                     [xmin-dx/10,ymin-dy/10,zmax+dz/10],
                     [xmax+dx/10,ymin-dy/10,zmax+dz/10],
                     [xmax+dx/10,ymax+dy/10,zmax+dz/10],
                     [xmin-dx/10,ymax+dy/10,zmax+dz/10]]
        
        boxconn = [[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,4,7],[3,0,4],[0,2,1],[0,3,2],[4,5,7],[5,6,7]]        
        NodeConn = NodeConn + (np.array(boxconn)+len(NodeCoords)).tolist()
        NodeCoords = NodeCoords + boxcoords
        # NodeConn = boxconn
        # NodeCoords = boxcoords
        hole = [xmin-dx/20,ymin-dy/20,zmin-dz/20]
        holes.append(hole)
        
    with tempfile.TemporaryDirectory() as tempdir:       
        filename = os.path.join(tempdir,'mesh.smesh')
        writeSmesh(filename, NodeCoords, NodeConn, holes=holes)
        process = subprocess.run(['tetgen'] + switches + [filename], capture_output=True, text=True)
        if verbose: print(process.stdout)
        node = os.path.join(tempdir,'mesh.1.node')
        ele = os.path.join(tempdir,'mesh.1.ele')
        assert os.path.exists(node), 'Failed to Generate Node File, Verify Inputs, Run with verbose=True for Diagnostics'
        TetCoords = readNode(node)
        assert os.path.exists(ele), 'Failed to Generate Element File, Verify Inputs, Run with verbose=True for Diagnostics'
        TetConn = readEle(ele)        
    
    return TetCoords, TetConn

def remesh(NodeCoords, NodeConn, switches=[''], verbose=False):

    with tempfile.TemporaryDirectory() as tempdir:   
        nodename = os.path.join(tempdir,'mesh.node')
        elename = os.path.join(tempdir,'mesh.ele')
        writeNode(nodename, NodeCoords)
        writeEle(elename, NodeConn)
        if '-r' not in switches:
            switches.append('-r')
        process = subprocess.run(['tetgen'] + switches + [nodename] + [elename], capture_output=True, text=True)
        if verbose: print(process.stdout)
        node = os.path.join(tempdir,'mesh.1.node')
        ele = os.path.join(tempdir,'mesh.1.ele')
        assert os.path.exists(node), 'Failed to Generate Node File, Verify Inputs, Run with verbose=True for Diagnostics'
        TetCoords = readNode(node)
        assert os.path.exists(ele), 'Failed to Generate Element File, Verify Inputs, Run with verbose=True for Diagnostics'
        TetConn = readEle(ele)        
    return TetCoords, TetConn

def writeNode(filename, NodeCoords):
    
    if os.path.splitext(filename)[1] != '.node':
        filename += '.node'
        
    with open(filename,'w') as f:
        # Write Nodes
        f.write(' '.join([str(len(NodeCoords)), str(len(NodeCoords[0])), str(0), str(0)]))  # TODO: No attributes or boundary markers
        f.write('\n')
        for i,coord in enumerate(NodeCoords):
            f.write(' '.join([str(i+1), str(coord[0]), str(coord[1]), str(coord[2])]))
            f.write('\n')

def writeEle(filename,NodeConn):
    if os.path.splitext(filename)[1] != '.ele':
        filename += '.ele'
        
    with open(filename,'w') as f:
        f.write(' '.join([str(len(NodeConn)), str(len(NodeConn[0])), str(0)]))     # TODO: No boundary markers
        f.write('\n')
        for elem in NodeConn:
            f.write(' '.join([str(len(elem))] + [str(node+1) for node in elem]))
            f.write('\n')

def writeSmesh(filename, NodeCoords, NodeConn, holes=[]):
    
    if os.path.splitext(filename)[1] != '.smesh':
        filename += '.smesh'
        
    with open(filename,'w') as f:
        # Write Nodes
        f.write('# Part 1 - the node list.\n')        
        f.write(' '.join([str(len(NodeCoords)), str(len(NodeCoords[0])), str(0), str(0)]))  # TODO: No attributes or boundary markers
        f.write('\n')
        for i,coord in enumerate(NodeCoords):
            f.write(' '.join([str(i+1), str(coord[0]), str(coord[1]), str(coord[2])]))
            f.write('\n')
        
        # Write Facets
        f.write('# Part 2 - the facet list.\n')
        f.write(' '.join([str(len(NodeConn)), str(0)]))     # TODO: No boundary markers
        f.write('\n')
        for elem in NodeConn:
            f.write(' '.join([str(len(elem))] + [str(node+1) for node in elem]))
            f.write('\n')
        
        # Write Holes
        f.write('# Part 3 - the hole list.\n')        
        f.write(str(len(holes)))  
        f.write('\n')
        for i,hole in enumerate(holes):
            f.write(' '.join([str(i+1), str(hole[0]), str(hole[1]), str(hole[2])]))
            f.write('\n')
        
        # TODO: Write Regional Attributes
        f.write('# Part 4 - the region list.\n')        
        f.write(str(0))
        
    
def readNode(filename):
    NodeCoords = []
    with open(filename,'r') as f:
        lines = f.readlines()
    
    checkheader = True
    for line in lines:
        l = line.split('#')[0].strip()
        if len(l) == 0:
            continue
        s = l.split()
        if checkheader:
            NNode = int(s[0])       # Number of Nodes
            nD = int(s[1])          # Number of Dimensions
            nAttrb = int(s[2])      # Number of Attributes
            checkheader = False     #
            continue
        
        NodeCoords.append([float(coord) for coord in s[1:1+nD]])
    return NodeCoords
    
def readEle(filename):    
    NodeConn = []
    with open(filename,'r') as f:
        lines = f.readlines()
    
    checkheader = True
    for line in lines:
        l = line.split('#')[0].strip()
        if len(l) == 0:
            continue
        s = l.split()
        if checkheader:
            NElem = int(s[0])       # Number of Elements
            nNode = int(s[1])       # Number of Nodes per Element
            regAttrb = int(s[2])    # Region Attribute (0 or 1)
            checkheader = False
            continue
        
        NodeConn.append([int(node)-1 for node in s[1:]])
        
    return NodeConn
