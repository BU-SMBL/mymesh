# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:20:47 2021

@author: toj
"""

from . import MeshUtils, Improvement, converter, Quality, Rays, curvature
from sys import getsizeof
import scipy
import numpy as np
import copy, warnings, pickle
import meshio
import json

class mesh:
    def __init__(self,*args):
        # Primary attributes
        self.NodeCoords = []
        self.NodeConn = []

        # Properties:
        self._SurfConn = []
        self._NodeNeighbors = []
        self._ElemConn = []
        self._SurfNodeNeighbors = []
        self._SurfElemConn = []
        self._ElemNormals = []
        self._NodeNormals = []
        self._Centroids = []
        self._Faces = []  # Used for FVM/CFD meshes, not for use in 2D
        self._FaceConn = [] # For each element, gives the indices of connected faces
        self._FaceElemConn = [] # For each face, gives the indices of connected elements (nan -> surface)
        self._Edges = []    
        self._EdgeConn = []
        self._EdgeElemConn = []
        self._NodeNormalsMethod = 'Angle'
        self._NFace = 0
        self._NEdge = 0
        self._NNode = 0
        self._NElem = 0
        self.nNode = []
        
        # Sets:
        self.NodeSets = {}
        self.ElemSets = {}
        self.EdgeSets = {}
        self.FaceSets = {}

        # Data:
        self.NodeData = {}
        self.ElemData = {}
        
        self.nD = 0
        for i,arg in enumerate(args):
            if i == 0:
                self.NodeCoords = arg
            elif i == 1:
                self.NodeConn = arg
                if len(self.NodeConn) > 0:
                    self.nNode = len(self.NodeConn[0])
        
        self.verbose = True
        self._printlevel = 0
        self.initialize(cleanup=False)
    def __sizeof__(self):
        size = 0
        size += getsizeof(self.NodeCoords)
        size += getsizeof(self.NodeConn)
        size += getsizeof(self._Faces)
        size += getsizeof(self._FaceConn)
        size += getsizeof(self._FaceElemConn)
        size += getsizeof(self._Edges)
        size += getsizeof(self._EdgeConn)
        size += getsizeof(self._EdgeElemConn)
        size += getsizeof(self._NElem)
        size += getsizeof(self.nNode)
        size += getsizeof(self._NNode)
        size += getsizeof(self.nD)
        size += getsizeof(self._SurfConn)
        size += getsizeof(self._NodeNeighbors)
        size += getsizeof(self._ElemConn)
        size += getsizeof(self._SurfNodeNeighbors)
        size += getsizeof(self._SurfElemConn)
        size += getsizeof(self._ElemNormals)
        size += getsizeof(self._NodeNormals)
        size += getsizeof(self._Centroids)
        return size
    def __repr__(self):
        return 'Mesh Object\n{0:d} Nodes\n{1:d} Elements'.format(self.NNode,self.NElem)
    def __iter__(self):
        return iter((self.NodeCoords,self.NodeConn))
    def initialize(self,cleanup=True):
        if cleanup: self.cleanup()
        if len(self.NodeCoords) > 0:
            self.setnD()
    def reset(self,properties=None,keep=[]):
        """
        reset Reset all or specified properties

        Parameters
        ----------
        properties : NoneType, str, or list, optional
            If specified as a string or list of strings, will reset the properties specified by those names, the default is None
        """        
        # Reset calculated mesh attributes
        if type(keep) is str:
            keep = [keep]
        if properties == None:
            if 'SurfConn' not in keep: self._SurfConn = []
            if 'NodeNeighbors' not in keep: self._NodeNeighbors = []
            if 'ElemConn' not in keep: self._ElemConn = []
            if 'SurfNodeNeighbors' not in keep: self._SurfNodeNeighbors = []
            if 'SurfElemConn' not in keep: self._SurfElemConn = []
            if 'ElemNormals' not in keep: self._ElemNormals = []
            if 'NodeNormals' not in keep: self._NodeNormals = []
            if 'Centroids' not in keep: self._Centroids = []
            if 'Edges' not in keep: self._Edges = []
            if 'Edges' not in keep or 'EdgeConn' not in keep: self._FaceConn = []
            if 'Edges' not in keep or 'EdgeElemConn' not in keep: self._FaceElemConn = [] 
            if 'Faces' not in keep: self._Faces = []
            if 'Faces' not in keep or 'FaceConn' not in keep: self._FaceConn = []
            if 'Faces' not in keep or 'FaceElemConn' not in keep: self._FaceElemConn = [] 
        elif type(properties) is list:
            for prop in properties:
                if prop[0] != '_':
                    prop = '_'+prop
                setattr(self,prop,[])
        elif type(properties) is str:
            if properties[0] != '_':
                properties = '_'+properties
            setattr(self,properties,[])
        else:
            raise Exception('Invalid input.')
    def cleanup(self,tol=1e-12,angletol=1e-3,strict=False):
        # TODO: This needs to be improved so other variables that point to nodes or elements are updated accordingly
        
        self.reset()
        self.NodeCoords,self.NodeConn,_ = MeshUtils.DeleteDuplicateNodes(self.NodeCoords,self.NodeConn,tol=tol)
        if self.NElem > 0 and len(self.NodeConn[0]) == 3:
            # Currently only valid for tris
            self.NodeCoords,self.NodeConn = MeshUtils.DeleteDegenerateElements(*self,tol=tol,angletol=angletol,strict=strict)
        elif self.NElem > 0:
            self.NodeCoords,self.NodeConn = MeshUtils.DeleteDegenerateElements(*self,angletol=angletol,strict=True)
        self.NodeCoords,self.NodeConn,_ = converter.removeNodes(self.NodeCoords,self.NodeConn)
        
    def validate(self):
        assert type(self.NodeCoords) == list, 'Invalid type for model.mesh.NodeCoords'
        assert len(self.NodeCoords) > 0, 'Undefined Node Coordinates'
        assert type(self.NodeConn) == list, 'Invalid type for model.mesh.NodeConn'
        assert len(self.NodeConn), 'Undefined Nodal Connectivity'
        assert max([max(elem) for elem in self.NodeConn]) <= len(self.NodeCoords), 'NodeConn references undefined nodes'
        v = Quality.Volume(*self)
        if np.nanmin(v) < 0:
            warnings.warn('VALIDATION WARNING: Mesh has inverted elements')
    def setnD(self):
        if len(self.NodeCoords[0]) == 3:
            self.nD = 3
        elif len(self.NodeCoords[0]) == 2:
            self.nD = 2

    def defineMesh(self,NodeCoords,NodeConn):
        self.addNodes(NodeCoords)
        self.addElems(NodeConn)
        self.nNode = len(NodeConn[0])
        self.setnD()
    def addNodes(self,NewNodeCoords,NodeSet=None):
        if type(NewNodeCoords) is np.ndarray:
            NewNodeCoords = NewNodeCoords.tolist()
        if type(self.NodeCoords) is np.ndarray:
            self.NodeCoords = self.NodeCoords.tolist()
        assert type(NewNodeCoords) is list, 'Supplied NodeCoords must be list or np.ndarray'
        
        nnode = self.NNode
        self.NodeCoords += NewNodeCoords
        if NodeSet in self.NodeSets.keys():
            self.NodeSets[NodeSet] = list(self.NodeSets[NodeSet]) + list(range(nnode,self.NNode))
        elif NodeSet:
            self.NodeSets[NodeSet] = range(nnode,self.NNode)            
    def addFaces(self,NewFaces,FaceSet=None):
        if type(NewFaces) is np.ndarray:
            NewFaces = NewFaces.tolist()
        assert type(NewFaces) is list, 'Supplied Faces must be list or np.ndarray'
        nface = self.NFace
        self._Faces += NewFaces
        if FaceSet in self.FaceSets.keys():
            self.FaceSets[FaceSet] = list(self.FaceSets[FaceSet]) + list(range(nface,self.NFace))
        elif FaceSet:
            self.FaceSets[FaceSet] = range(nface,self.NFace)            
    def addEdges(self,NewEdges,EdgeSet=None):
        if type(NewEdges) is np.ndarray:
            NewEdges = NewEdges.tolist()
        assert type(NewEdges) is list, 'Supplied Edges must be list or np.ndarray'
        nedge = self.NEdge
        self._Edges += NewEdges
        if EdgeSet in self.EdgeSets.keys():
            self.EdgeSets[EdgeSet] = list(self.EdgeSets[EdgeSet]) + list(range(nedge,self.NEdge))
        elif EdgeSet:
            self.EdgeSets[EdgeSet] = range(nedge,self.NEdge) 
    def addElems(self,NewNodeConn,ElemSet=None):
        if type(NewNodeConn) is np.ndarray:
            NewNodeConn = NewNodeConn.tolist()
        if type(self.NodeConn) is np.ndarray:
            self.NodeConn = self.NodeConn.tolist()
        assert type(NewNodeConn) is list, 'Supplied NodeConn must be list or np.ndarray'
        nelem = self.NElem
        self.NodeConn += NewNodeConn
        if ElemSet in self.ElemSets.keys():
            self.ElemSets[ElemSet] = list(self.ElemSets[ElemSet]) + list(range(nelem,self.NElem))
        elif ElemSet:
            self.ElemSets[ElemSet] = range(nelem,self.NElem) 
        
    def copy(self):
        
        M = mesh()
        M.NodeCoords = copy.copy(self.NodeCoords)
        M.NodeConn = copy.copy(self.NodeConn)
        
        M.nNode = copy.copy(self.nNode)
        M.nD = copy.copy(self.nD)
        
        M._Faces = copy.copy(self._Faces)
        M._FaceConn = copy.copy(self._FaceConn)
        M._FaceElemConn = copy.copy(self._FaceElemConn)
        M._Edges = copy.copy(self._Edges)
        
        M.NodeSets = copy.copy(self.NodeSets)
        M.EdgeSets = copy.copy(self.EdgeSets)
        M.FaceSets = copy.copy(self.FaceSets)
        M.ElemSets = copy.copy(self.ElemSets)

        M.NodeData = copy.copy(self.NodeData)
        M.ElemData = copy.copy(self.ElemData)
        
        M._SurfConn = copy.copy(self._SurfConn)
        M._NodeNeighbors = copy.copy(self._NodeNeighbors)
        M._ElemConn = copy.copy(self._ElemConn)
        M._SurfNodeNeighbors = copy.copy(self._SurfNodeNeighbors)
        M._SurfElemConn = copy.copy(self._SurfElemConn)
        M._ElemNormals = copy.copy(self._ElemNormals)
        M._NodeNormals = copy.copy(self._NodeNormals)
        M._Centroids = copy.copy(self._Centroids)
        M.verbose = copy.copy(self.verbose)
        
        return M
    def merge(self,Mesh2,tol=1e-14,cleanup=True):
        self.initialize(cleanup=cleanup)
        if type(Mesh2) is list:
            MeshList = Mesh2
        else:
            MeshList = [Mesh2]
        for M in MeshList:
            # Original Stats
            NNode = self.NNode
            NElem = self.NElem
            NFace = self._NFace
            NEdge = self._NEdge
            
            # Add Nodes
            if len(M.NodeSets) > 1:
                keys = list(M.NodeSets.keys())
                for i in range(len(keys)):
                    keyName = keys[i]
                    self.addNodes([M.NodeCoords[node] for node in M.NodeSets[keyName]],NodeSet=keyName)
            else:
                self.addNodes(M.NodeCoords)
            # Add Edges 
            # if len(M.EdgeSets) > 1:
            #     keys = list(M.EdgeSets.keys())
            #     for i in range(len(keys)):
            #         keyName = keys[i]
            #         self.addEdges([[node+NNode for node in M.Edges()[edge]] for edge in M.EdgeSets[keyName]],EdgeSet=keyName)
            # else:
            #     self.addEdges([[node+NNode for node in M.Edges()[edge]] for edge in range(len(M.Edges()))])
            # # Add Faces  
            # if len(M.FaceSets) > 1:
            #     keys = list(M.FaceSets.keys())
            #     for i in range(len(keys)):
            #         keyName = keys[i]
            #         self.addFaces([[node+NNode for node in M.Faces()[face]] for face in M.FaceSets[keyName]],FaceSet=keyName)
            # else:
            #     self.addFaces([[node+NNode for node in M.Faces()[face]] for face in range(len(M.Faces()))])
            # Add Elems
            if len(M.ElemSets) > 1:
                keys = list(M.ElemSets.keys())
                for i in range(len(keys)):
                    keyName = keys[i]
                    self.addElems([[node+NNode for node in M.NodeConn[elem]] for elem in M.ElemSets[keyName]],ElemSet=keyName)
            else:
                self.addElems([[node+NNode for node in M.NodeConn[elem]] for elem in range(len(M.NodeConn))])
                    
            
            self._FaceElemConn = self._FaceElemConn + [[elem + NElem for elem in elemconn] for elemconn in M._FaceElemConn]
            self._FaceConn = self._FaceConn + [[face + NFace for face in faceconn] for faceconn in M._FaceConn]
            # TODO: EdgeELemConn, EdgeConn
        # Cleanup
        if cleanup:
            self.cleanup(tol=tol)

    @property
    def NNode(self):
        self._NNode = len(self.NodeCoords)
        return self._NNode
    @property
    def NElem(self):
        self._NElem = len(self.NodeConn)
        return self._NElem
    @property
    def NEdge(self):
        self._NEdge = len(self.Edges)
        return self._NEdge
    @property
    def NFace(self):
        self._NFace = len(self.Faces)
        return self._NFace
    @property
    def NEdge(self):
        self._NEdge = len(self.Edges)
        return self._NEdge
    def __get_faces(self):
        if self.NElem > 0:
            # Get all element faces
            faces,faceconn,faceelem = converter.solid2faces(self.NodeCoords,self.NodeConn,return_FaceConn=True,return_FaceElem=True)
            # Pad Ragged arrays in case of mixed-element meshes
            Rfaces = MeshUtils.PadRagged(faces)
            Rfaceconn = MeshUtils.PadRagged(faceconn)
            # Get all unique element faces (accounting for flipped versions of faces)
            _,idx,inv = np.unique(np.sort(Rfaces,axis=1),axis=0,return_index=True,return_inverse=True)
            RFaces = Rfaces[idx]
            FaceElem = faceelem[idx]
            RFaces = np.append(RFaces, np.repeat(-1,RFaces.shape[1])[None,:],axis=0)
            inv = np.append(inv,-1)
            RFaceConn = inv[Rfaceconn] # Faces attached to each element
            # Face-Element Connectivity
            FaceElemConn = np.nan*(np.ones((len(RFaces),2))) # Elements attached to each face

            FECidx = (FaceElem[RFaceConn] == np.repeat(np.arange(self.NElem)[:,None],RFaceConn.shape[1],axis=1)).astype(int)
            FaceElemConn[RFaceConn,FECidx] = np.repeat(np.arange(self.NElem)[:,None],RFaceConn.shape[1],axis=1)
            FaceElemConn = [[int(x) if not np.isnan(x) else x for x in y] for y in FaceElemConn[:-1]]


            Faces = MeshUtils.ExtractRagged(RFaces[:-1],dtype=int)
            FaceConn = MeshUtils.ExtractRagged(RFaceConn,dtype=int)
            return Faces, FaceConn, FaceElemConn
        else:
            return [], [], []
    def __get_edges(self):
        # TODO: This might not work properly with mixed element types - but I think it shoud
        if self.NElem > 0:
            # Get all element edges
            edges, edgeconn, edgeelem = converter.solid2edges(self.NodeCoords,self.NodeConn,return_EdgeConn=True,return_EdgeElem=True)
            # Convert to unique edges
            Edges, UIdx, UInv = converter.edges2unique(edges,return_idx=True,return_inv=True)
            EdgeElem = np.asarray(edgeelem)[UIdx]
            EdgeConn = UInv[MeshUtils.PadRagged(edgeconn)]
            
            rows = EdgeConn.flatten()
            cols = np.repeat(np.arange(self.NElem),[len(x) for x in EdgeConn])
            data = np.ones(len(rows))
            
            mat = scipy.sparse.coo_matrix((data,(rows,cols)),shape=(len(Edges),self.NElem)).tocsr()
            EdgeElemConn = [list(mat.indices[mat.indptr[i]:mat.indptr[i+1]]) for i in range(mat.shape[0])]
            
            return Edges, EdgeConn, EdgeElemConn
        else:
            return [], [], []
    @property
    def Faces(self):
        if self._Faces == []:
            if self.verbose: 
                print('\n'+'\t'*self._printlevel+'Identifying element faces...',end='')
                self._printlevel+=1

            self._Faces, self._FaceConn, self._FaceElemConn = self.__get_faces()

            if self.verbose: 
                self._printlevel-=1
                print('Done', end='\n'+'\t'*self._printlevel)
        return self._Faces
    @property
    def FaceConn(self):
        if self._FaceConn == []:
            if self.verbose: 
                print('\n'+'\t'*self._printlevel+'Identifying element-face connectivity...',end='')
                self._printlevel+=1

            self._Faces, self._FaceConn, self._FaceElemConn = self.__get_faces()
            
            if self.verbose: 
                self._printlevel-=1
                print('Done', end='\n'+'\t'*self._printlevel)
        return self._FaceConn
    @property
    def FaceElemConn(self):
        if self._FaceElemConn == []:
            if self.verbose: 
                print('\n'+'\t'*self._printlevel+'Identifying element face-element connectivity...',end='')
                self._printlevel+=1

            self._Faces, self._FaceConn, self._FaceElemConn = self.__get_faces()
            
            if self.verbose: 
                self._printlevel-=1
                print('Done', end='\n'+'\t'*self._printlevel)
        return self._FaceElemConn
    @property
    def Edges(self):
        if self._Edges == []:
            if self.verbose: 
                print('\n'+'\t'*self._printlevel+'Identifying element edges...',end='')
                self._printlevel+=1

            self._Edges, self._EdgeConn, self._EdgeElemConn = self.__get_edges()

            if self.verbose: 
                self._printlevel-=1
                print('Done', end='\n'+'\t'*self._printlevel)
        return self._Edges
    @property
    def EdgeConn(self):
        if self._FaceConn == []:
            if self.verbose: 
                print('\n'+'\t'*self._printlevel+'Identifying element-edge connectivity...',end='')
                self._printlevel+=1

            self._Edges, self._EdgeConn, self._EdgeElemConn = self.__get_edges()
            
            if self.verbose: 
                self._printlevel-=1
                print('Done', end='\n'+'\t'*self._printlevel)
        return self._EdgeConn
    @property
    def EdgeElemConn(self):
        if self._EdgeElemConn == []:
            if self.verbose: 
                print('\n'+'\t'*self._printlevel+'Identifying element edge-element connectivity...',end='')
                self._printlevel+=1

            self._Edges, self._EdgeConn, self._EdgeElemConn = self.__get_edges()
            
            if self.verbose: 
                self._printlevel-=1
                print('Done', end='\n'+'\t'*self._printlevel)
        return self._EdgeElemConn
    @property
    def SurfConn(self):
        if self._SurfConn == []:
            if self.verbose: print('\n'+'\t'*self._printlevel+'Identifying surface...',end='')
            self._SurfConn = converter.solid2surface(*self)
            if self.verbose: print('Done', end='\n'+'\t'*self._printlevel)
        return self._SurfConn
    @property
    def NodeNeighbors(self):
        if self._NodeNeighbors == []:
            if self.verbose: print('\n'+'\t'*self._printlevel+'Identifying volume node neighbors...',end='')
            self._NodeNeighbors = MeshUtils.getNodeNeighbors(*self)
            if self.verbose: print('Done', end='\n'+'\t'*self._printlevel)
        return self._NodeNeighbors
    @property
    def ElemConn(self):
        if self._ElemConn == []:
            if self.verbose: print('\n'+'\t'*self._printlevel+'Identifying volume node element connectivity...',end='')
            self._ElemConn = MeshUtils.getElemConnectivity(*self)
            if self.verbose: print('Done', end='\n'+'\t'*self._printlevel)
        return self._ElemConn
    @property
    def SurfNodeNeighbors(self):
        if self._SurfNodeNeighbors == []:
            if self.verbose: 
                print('\n'+'\t'*self._printlevel+'Identifying surface node neighbors...',end='')
                self._printlevel+=1
            self._SurfNodeNeighbors = MeshUtils.getNodeNeighbors(self.NodeCoords,self.SurfConn)
            if self.verbose: 
                self._printlevel-=1
                print('Done', end='\n'+'\t'*self._printlevel)
        return self._SurfNodeNeighbors
    @property
    def SurfElemConn(self):
        if self._SurfElemConn == []:
            if self.verbose: 
                print('\n'+'\t'*self._printlevel+'Identifying surface node element connectivity...',end='')
                self._printlevel+=1
            self._SurfElemConn = MeshUtils.getElemConnectivity(self.NodeCoords,self.SurfConn)
            if self.verbose: 
                self._printlevel-=1
                print('Done', end='\n'+'\t'*self._printlevel)
                
        return self._SurfElemConn
    @property
    def ElemNormals(self):
        if self._ElemNormals == []:
            if self.verbose: 
                print('\n'+'\t'*self._printlevel+'Calculating surface element normals...',end='')
                self._printlevel+=1
            self._ElemNormals = MeshUtils.CalcFaceNormal(self.NodeCoords,self.SurfConn)
            if self.verbose: 
                self._printlevel-=1
                print('Done', end='\n'+'\t'*self._printlevel)
        return self._ElemNormals        
    @property
    def NodeNormalsMethod(self):
        return self._NodeNormalsMethod
    @NodeNormalsMethod.setter
    def NodeNormalsMethod(self,method):
        self._NodeNormals = []
        self._NodeNormalsMethod = method
    @property
    def NodeNormals(self):
        if self._NodeNormals == []:
            if self.verbose: 
                print('\n'+'\t'*self._printlevel+'Calculating surface node normals...',end='')
                self._printlevel+=1
            self._NodeNormals = MeshUtils.Face2NodeNormal(self.NodeCoords,self.SurfConn,self.SurfElemConn,self.ElemNormals,method=self.NodeNormalsMethod)
            if self.verbose: 
                self._printlevel-=1
                print('Done', end='\n'+'\t'*self._printlevel)
        return self._NodeNormals
    @property
    def Centroids(self):
        if self._Centroids == []:
            if self.verbose: print('\n'+'\t'*self._printlevel+'Calculating element centroids...',end='')
            self._Centroids = MeshUtils.Centroids(*self)
            if self.verbose: print('Done', end='\n'+'\t'*self._printlevel)
        return self._Centroids
    
    def RenumberNodesBySet(self):
        # Re-organize the order of nodes based on their node sets and make required adjustments to other stored values
        setkeys = self.NodeSets.keys()
        # newIds is a list of node ids where the new index is located at the old index
        newIds = np.repeat(np.nan,len(self.NodeCoords))
        end = 0
        # Renumber nodes in node sets
        for key in setkeys:
            start = end
            end += len(self.NodeSets[key])
            newIds[list(self.NodeSets[key])] = np.arange(start,end)
            self.NodeSets[key] = range(start,end)
            
        # Renumber any nodes that aren't in node sets
        newIds[np.isnan(newIds)] = np.arange(end,len(self.NodeCoords))
        self.NodeCoords, self.NodeConn, self._Faces = MeshUtils.RelabelNodes(self.NodeCoords, self.NodeConn, newIds, faces=self._Faces)
        
        self.reset(keep=['Faces','FaceElemConn','FaceConn'])
    def RenumberFacesBySet(self):
        setkeys = list(self.FaceSets.keys())
        
        if any([len(set(self.FaceSets[key1]).intersection(self.FaceSets[key2]))>0 for i,key1 in enumerate(setkeys) for key2 in setkeys[i+1:]]):
            raise Exception('There must be no overlap between FaceSets')
        
        
        # newIds is a list of face ids where the new index is located at the old index
        newIds = np.repeat(np.nan,len(self.Faces))
        end = 0
        # Renumber faces in face sets
        for key in setkeys:
            start = end
            end += len(self.FaceSets[key])
            newIds[list(self.FaceSets[key])] = np.arange(start,end,dtype=int)
            self.FaceSets[key] = range(start,end)
        # Renumber any faces that aren't in face sets
        newIds[np.isnan(newIds)] = np.arange(end,len(newIds),dtype=int)
        newIds = newIds.astype(int)

        # Reorder faces
        NewFaces = np.zeros(MeshUtils.PadRagged(self.Faces).shape,dtype=int)
        NewFaceElemConn = np.zeros(np.shape(self.FaceElemConn))

        NewFaces[newIds,:] = MeshUtils.PadRagged(self.Faces)
        NewFaceElemConn[newIds] = self.FaceElemConn


        NewFaceConn = newIds[MeshUtils.PadRagged(self.FaceConn)]
        
        self._Faces = MeshUtils.ExtractRagged(NewFaces,dtype=int)
        self._FaceElemConn = NewFaceElemConn.tolist()
        self._FaceConn = MeshUtils.ExtractRagged(NewFaceConn,dtype=int)
    
    def CreateBoundaryLayer(self,nLayers,FixedNodes=set(),StiffnessFactor=1,Thickness=None,OptimizeTets=True,FaceSets='surf'):
        """
        CreateBoundaryLayer Generate boundary layer elements 
        Based partially on 'A Procedure for Tetrahedral Boundary Layer Mesh Generation' - Bottaso and Detomi
        Currently surfaces must be strictly triangular.

        Parameters
        ----------
        nLayers : int
            Number of element layers to generate. 
        FixedNodes : set or list, optional
            Set of nodes that will be held fixed, by default set().
            It is not necessary to specify any fixed nodes, and by default 
            the starting nodes of the boundary layer will be held fixed.
        StiffnessFactor : int or float, optional
            Stiffness factor used for the spring network, by default 1
        Thickness : float or NoneType, optional
            Specified value for the maximum total thickness of the boundary layers. 
            If nLayers > 1, this thickness is subdivided by nLayers, by default None
        OptimizeTets : bool, optional
            If True, will perform tetrahedral mesh optimization
            (see Improvement.TetOpt), by default True.
        FaceSets : str or list, optional
            FaceSet or list of FaceSets to generate boundary later elements on, by default ['surf'].
            While mesh.FaceSets can generally contain any element face, boundary layer face sets
            must be surface faces; however, the face ids contained within the face sets should index
            mesh.Faces, and not mesh.SurfConn. The default value of 'surf' can be used even if no 
            sets exist in mesh.FaceSets and will generate boundary layer elements along the entire 
            surface. If mesh.FaceSets is empty, or doesn't contain a key with the name 'surf', the
            surface mesh will be used, otherwise, mesh.FaceSets['surf'] will be used.
        """        
        if type(FixedNodes) != set:
            FixedNodes = set(FixedNodes)
        if type(FaceSets) is str: 
            FaceSets = [FaceSets]

        # Create first layer with 0 thickness
        self.reset('SurfConn')
        NOrigElem = self.NElem
        OrigConn = copy.copy(self.NodeConn)
        self.NodeNormalsMethod = 'MostVisible'
        NodeNormals = self.NodeNormals
        surfconn = self.SurfConn
        surfnodes = set(np.unique(surfconn).tolist())
        
        if len(self.FaceSets) == 0:
            if len(FaceSets) > 1 or 'surf' not in FaceSets:
                raise Exception('Requested FaceSets are undefined.')
            ForceNodes = copy.copy(surfnodes)            

        else:
            ForceNodes = set()
            for key in FaceSets:
                if key not in self.FaceSets.keys():
                    raise Exception('Requested set "{:s}" is undefined.'.format(key))
                FaceIds = self.FaceSets[key]
                FaceNodes = set([n for i in FaceIds for n in self.Faces[i]])
                ForceNodes.update(FaceNodes)
            NoGrowthNodes = surfnodes.difference(ForceNodes)
            FixedNodes.update(NoGrowthNodes)


        newsurfconn = [[node+len(self.NodeCoords) for node in elem] for elem in surfconn]
        newsurfnodes = np.unique(newsurfconn)
        BLConn = [elem + newsurfconn[i] for i,elem in enumerate(surfconn)]
        self.addNodes(self.NodeCoords)
        self.addElems(BLConn)
        self.reset()

        FixedNodes.update(newsurfnodes)
        Forces = [[0,0,0] if i not in ForceNodes else -1*np.array(NodeNormals[i]) for i in range(self.NNode)]
        allnodes = set([n for elem in self.NodeConn for n in elem])
        FixedNodes.update(set(i for i in range(len(self.NodeCoords)) if i not in allnodes))
        Fixed = np.array([1 if i in FixedNodes else 0 for i in range(self.NNode)])

        # Oriented wedge->tet conversion -  (Bottasso & Detomi)
        # surfedges = converter.solid2edges(self.NodeCoords,surfconn)
        surfedges,surfedgeconn,surfedgeelem = converter.solid2edges(self.NodeCoords,newsurfconn,return_EdgeElem=True,return_EdgeConn=True)

        UEdges,idx,inv = converter.edges2unique(surfedges,return_idx=True,return_inv=True)
        UEdgeConn = inv[surfedgeconn]

        NodeEdges = [[] for i in self.NodeCoords]
        for i,e in enumerate(UEdges):
            NodeEdges[e[0]].append(i)
            NodeEdges[e[1]].append(i)
        
        oriented = np.zeros(len(UEdges)) # 1 will indicate that the edge will be oriented as is, -1 indicates a flip
        for i,node in enumerate(newsurfnodes):
            for edge in NodeEdges[node]:
                if oriented[edge] == 0:
                    if UEdges[edge][0] == node:
                        oriented[edge] = 1
                    else:
                        oriented[edge] = -1
        OrientedEdges = copy.copy(UEdges)
        OrientedEdges[oriented==-1] = np.fliplr(UEdges[oriented==-1])

        surfedges = np.array(surfedges)
        
        # Tetrahedronization:
        # For each triangle, ElemEdgeOrientations will have a 3 entries, corresponding to the orientation of the edges
        # For a particular edge in an element, True -> the oriented edge is oriented clockwise, False -> counterclockwise 
        ElemEdgeOrientations = (OrientedEdges[UEdgeConn] == surfedges[surfedgeconn])[:,:,0] 

        # The are 6 possible combinations
        Cases = -1*np.ones(len(surfconn))
        Cases[np.all(ElemEdgeOrientations==[True,True,False],axis=1)] = 1
        Cases[np.all(ElemEdgeOrientations==[True,False,True],axis=1)] = 2
        Cases[np.all(ElemEdgeOrientations==[True,False,False],axis=1)] = 3
        Cases[np.all(ElemEdgeOrientations==[False,True,True],axis=1)] = 4
        Cases[np.all(ElemEdgeOrientations==[False,True,False],axis=1)] = 5
        Cases[np.all(ElemEdgeOrientations==[False,False,True],axis=1)] = 6

        
        # Each triangle in surfconn lines up with the indices of wedges in BLConn
        ArrayConn = np.asarray(BLConn)
        TetConn = -1*np.ones((len(BLConn)*3,4))
        t1 = np.zeros((len(ArrayConn),4))
        t2 = np.zeros((len(ArrayConn),4))
        t3 = np.zeros((len(ArrayConn),4))
        # Case 1:
        t1[Cases==1] = ArrayConn[Cases==1][:,[0,1,2,5]]
        t2[Cases==1] = ArrayConn[Cases==1][:,[1,4,5,0]]
        t3[Cases==1] = ArrayConn[Cases==1][:,[0,3,4,5]]
        # Case 2:
        t1[Cases==2] = ArrayConn[Cases==2][:,[0,1,2,4]]
        t2[Cases==2] = ArrayConn[Cases==2][:,[0,4,2,3]]
        t3[Cases==2] = ArrayConn[Cases==2][:,[4,5,2,3]]
        # Case 3:
        t1[Cases==3] = ArrayConn[Cases==3][:,[0,1,2,4]]
        t2[Cases==3] = ArrayConn[Cases==3][:,[0,4,2,5]]
        t3[Cases==3] = ArrayConn[Cases==3][:,[0,4,5,3]]
        # Case 4:
        t1[Cases==4] = ArrayConn[Cases==4][:,[0,1,2,3]]
        t2[Cases==4] = ArrayConn[Cases==4][:,[1,2,3,5]]
        t3[Cases==4] = ArrayConn[Cases==4][:,[1,5,3,4]]
        # Case 5:
        t1[Cases==5] = ArrayConn[Cases==5][:,[0,1,2,5]]
        t2[Cases==5] = ArrayConn[Cases==5][:,[1,5,3,4]]
        t3[Cases==5] = ArrayConn[Cases==5][:,[0,1,5,3]]
        # Case 6:
        t1[Cases==6] = ArrayConn[Cases==6][:,[0,1,2,3]]
        t2[Cases==6] = ArrayConn[Cases==6][:,[1,2,3,4]]
        t3[Cases==6] = ArrayConn[Cases==6][:,[4,5,2,3]]
        
        TetConn[0::3] = t1
        TetConn[1::3] = t2
        TetConn[2::3] = t3
        TetConn = TetConn.astype(int).tolist()

        # RelevantElems = [elem for elem in self.NodeConn if not all([n in FixedNodes for n in elem])]
        RelevantElems = TetConn + [elem for elem in self.NodeConn if len(elem)==4]
        RelevantCoords,RelevantConn,NodeIds = converter.removeNodes(self.NodeCoords,RelevantElems) 

        # TetConn = converter.solid2tets(RelevantCoords,RelevantConn)
        RelevantNodeNeighbors = MeshUtils.getNodeNeighbors(RelevantCoords,RelevantConn)
        RelevantElemConn = MeshUtils.getElemConnectivity(RelevantCoords,RelevantConn)
        RelevantForces = np.asarray(Forces)[NodeIds]
        RelevantFixed = Fixed[NodeIds]
        RelevantFixedNodes = set(np.where(RelevantFixed)[0])
        NewCoords = np.asarray(self.NodeCoords)

        if Thickness:
            L0Override = Thickness
        else:
            L0Override = 'min'
        
        # Expand boundary layer
        NewRelevantCoords,U,(K,F) = Improvement.SegmentSpringSmoothing(RelevantCoords,RelevantConn,
            RelevantNodeNeighbors,RelevantElemConn,StiffnessFactor=StiffnessFactor,
            FixedNodes=RelevantFixedNodes,Forces=RelevantForces,L0Override=L0Override,return_KF=True)

        if Thickness:
            # Find stiffness factor that gives desired thickness

            # Full solve:
            # def fun(k):
            #     NewRelevantCoords,_ = Improvement.SegmentSpringSmoothing(RelevantCoords,TetConn,
            #                                 RelevantNodeNeighbors,RelevantElemConn,StiffnessFactor=k,
            #                                 FixedNodes=RelevantFixedNodes,Forces=RelevantForces,L0Override=L0Override)
            #     t = max(np.linalg.norm(U,axis=1))
            #     # print(k,t)
            #     return abs(Thickness - t)
            # res = scipy.optimize.minimize_scalar(fun,(StiffnessFactor,StiffnessFactor/10),tol=ThicknessTol,options={'maxiter':ThicknessMaxIter})

            # Scaled K matrix:
            # def fun(k):
            #     U = scipy.sparse.linalg.spsolve((k/StiffnessFactor)*K.tocsc(), F).toarray()
            #     t = max(np.linalg.norm(U,axis=1))
            #     print(k,t)
            #     return abs(Thickness - t)s
            # res = scipy.optimize.minimize_scalar(fun,(StiffnessFactor,StiffnessFactor/10),tol=ThicknessTol,options={'maxiter':ThicknessMaxIter})
            # k = res.x
            
            # Power Law:
            # t = max(np.linalg.norm(NewCoords[ForceNodes] - NewCoords[SurfNodes],axis=1))
            t = np.nanmax(np.linalg.norm(U,axis=1))
            alpha = t*StiffnessFactor
            k = alpha*Thickness**-1
            U2 = scipy.sparse.linalg.spsolve(k*K.tocsc()/StiffnessFactor, F).toarray()
            # t2 = max(np.linalg.norm(U2,axis=1))
            NewRelevantCoords = np.add(RelevantCoords, U2)

        NewCoords[NodeIds] = NewRelevantCoords
        NewCoords[list(FixedNodes)] = np.array(self.NodeCoords)[list(FixedNodes)]
        # Collapse transition elements

        if OptimizeTets:
            Tets = [elem for elem in self.NodeConn if len(elem)==4]
            skew = Quality.Skewness(NewCoords,Tets)
            BadElems = set(np.where(skew>0.9)[0])
            ElemNeighbors = MeshUtils.getElemNeighbors(NewCoords,Tets)
            BadElems.update([e for i in BadElems for e in ElemNeighbors[i]])
            BadNodes = set([n for i in BadElems for n in Tets[i]])

            SurfConn = converter.solid2surface(NewCoords,Tets)
            SurfNodes = set([n for elem in SurfConn for n in elem])

            FreeNodes = BadNodes.difference(SurfNodes)

            NewCoords = Improvement.TetOpt(NewCoords,Tets,FreeNodes=FreeNodes,objective='eta',method='BFGS',iterate=4)
        
        # Divide the boundary layer to create the specified number of layers
        if nLayers > 1: 
            nNum = len(NewCoords)
            NewCoords2 = np.array(NewCoords)        
            for i in range(NOrigElem,self.NElem):
                elem = self.NodeConn[i]
                NewNodes = []
                NewElems = [[elem[0],elem[1],elem[2],elem[0],elem[1],elem[2]]] + [[] for j in range(nLayers-1)]
                for j in range(1,nLayers):
                    NewNodes += [NewCoords2[elem[0]] + (NewCoords2[elem[3]]-NewCoords2[elem[0]])*j/nLayers, 
                                     NewCoords2[elem[1]] + (NewCoords2[elem[4]]-NewCoords2[elem[1]])*j/nLayers, 
                                     NewCoords2[elem[2]] + (NewCoords2[elem[5]]-NewCoords2[elem[2]])*j/nLayers]
                    NewElems[j-1] = [NewElems[j-1][3],NewElems[j-1][4],NewElems[j-1][5],nNum,nNum+1,nNum+2]
                    NewElems[j] = [nNum,nNum+1,nNum+2,nNum,nNum+1,nNum+2]
                    nNum += 3
                NewElems[-1] = [NewElems[-2][3],NewElems[-2][4],NewElems[-2][5],elem[3],elem[4],elem[5]]
                NewCoords2 = np.append(NewCoords2,np.array(NewNodes),axis=0)
                OrigConn += NewElems
        
            self.NodeCoords = NewCoords2.tolist()
            self.NodeConn = OrigConn
        else:
            self.NodeCoords = NewCoords
        

        # Reduce or remove degenerate wedges -- TODO: This can probably be made more efficient
        # self.cleanup()
        self.NodeCoords,self.NodeConn,_ = MeshUtils.DeleteDuplicateNodes(self.NodeCoords,self.NodeConn)
        Unq = [np.unique(elem,return_index=True,return_inverse=True) for elem in self.NodeConn]
        key = MeshUtils.PadRagged([u[1][u[2]] for u in Unq],fillval=-1)

        Cases = -1*np.ones(self.NElem,dtype=int)
        # Fully degenerate wedges (triangles):
        Cases[np.all(key[:,0:6]==[0,1,2,0,1,2],axis=1)] = 0
        Cases[np.all(key[:,0:6]==[3,4,5,3,4,5],axis=1)] = 0
        # Double-edge degenerate wedges (tetrahedrons):
        Cases[np.all(key[:,0:6]==[0,1,2,3,1,2],axis=1)] = 1
        Cases[np.all(key[:,0:6]==[0,1,2,0,4,2],axis=1)] = 2
        Cases[np.all(key[:,0:6]==[0,1,2,0,1,5],axis=1)] = 3
        Cases[np.all(key[:,0:6]==[0,4,5,3,4,5],axis=1)] = 4
        Cases[np.all(key[:,0:6]==[3,1,5,3,4,5],axis=1)] = 5
        Cases[np.all(key[:,0:6]==[3,4,2,3,4,5],axis=1)] = 6
        # Single-edge degenerate wedges (pyramids):
        Cases[np.all(key[:,0:6]==[0,1,2,3,1,5],axis=1)] = 7
        Cases[np.all(key[:,0:6]==[0,1,2,3,4,2],axis=1)] = 8
        Cases[np.all(key[:,0:6]==[0,1,2,0,4,5],axis=1)] = 9
        Cases[np.all(key[:,0:6]==[0,1,5,3,4,5],axis=1)] = 10
        Cases[np.all(key[:,0:6]==[3,1,2,3,4,5],axis=1)] = 11
        Cases[np.all(key[:,0:6]==[0,4,2,3,4,5],axis=1)] = 12
        # Non-wedges
        nNodes = np.array([len(elem) for elem in self.NodeConn])
        Cases[nNodes!=6] = -1

        ProperKeys = [
            [],             # 0
            [0,1,2,3],      # 1
            [0,1,2,4],      # 2
            [0,1,2,5],      # 3
            [0,4,5,3],      # 4
            [3,1,5,4],      # 5
            [3,4,2,5],      # 6
            [0,2,5,3,1],    # 7
            [0,3,4,1,2],    # 8
            [1,4,5,2,0],    # 9
            [0,3,4,1,5],    # 10
            [1,4,5,2,3],    # 11
            [0,2,5,3,4]     # 12
        ]
        RNodeConn = MeshUtils.PadRagged(self.NodeConn)
        for i,case in enumerate(Cases):
            if case == -1:
                continue
            else:
                self.NodeConn[i] = RNodeConn[i][ProperKeys[case]].tolist()
        self.NodeConn = [elem for elem in self.NodeConn if len(elem)>0]
        # Attempt to fix any element inversions
        # NewCoords = np.asarray(NewCoords)
        # NewRelevantCoords = NewCoords[NodeIds]
        V = Quality.Volume(*self)
        if min(V) < 0:
            # print(sum(V<0))
            # BLConn = [elem for elem in self.NodeConn if len(elem) == 6]
            # self.NodeCoords = Improvement.FixInversions(self.NodeCoords,BLConn,FixedNodes=np.unique(converter.solid2surface(self.NodeCoords,self.NodeConn)))
            BadElems = set(np.where(V<0)[0])
            ElemNeighbors = MeshUtils.getElemNeighbors(*self)
            BadElems.update([e for i in BadElems for e in ElemNeighbors[i]])
            BadNodes = set([n for i in BadElems for n in self.NodeConn[i]])
            # self.reset('SurfConn')
            SurfNodes = set([n for elem in self.SurfConn for n in elem])

            FreeNodes = BadNodes.difference(SurfNodes)
            FixedNodes = set(range(self.NNode)).difference(FreeNodes)
            # NewRelevantCoords = Improvement.TetOpt(NewRelevantCoords,RelevantConn,FreeNodes=FreeNodes,objective='eta',method='BFGS',iterate=4)
            self.NodeCoords = Improvement.FixInversions(*self,FixedNodes=FixedNodes)
            # NewCoords[NodeIds] = NewRelevantCoords
        self.reset()
    
    def getQuality(self,metrics=['Skewness','Aspect Ratio','Inverse Orthogonal Quality','Inverse Orthogonality','Min Dihedral(deg)','Max Dihedral(deg)','Volume']):
        
        quality = {}
        if type(metrics) is str: metrics = [metrics]
        for metric in metrics:
            if metric == 'Skewness':
                quality[metric] = Quality.Skewness(*self,verbose=self.verbose)
            elif metric == 'Aspect Ratio':
                quality[metric] = Quality.AspectRatio(*self,verbose=self.verbose)    
            elif metric == 'Inverse Orthogonal Quality':
                quality[metric] = Quality.InverseOrthogonalQuality(*self,verbose=self.verbose)
            elif metric == 'Orthogonal Quality':
                quality[metric] = Quality.OrthogonalQuality(*self,verbose=self.verbose)
            elif metric == 'Inverse Orthogonality':
                quality[metric] = Quality.InverseOrthogonality(*self,verbose=self.verbose)
            elif metric == 'Orthogonality':
                quality[metric] = Quality.Orthogonality(*self,verbose=self.verbose)
            elif metric == 'Min Dihedral':
                quality[metric] = Quality.MinDihedral(*self,verbose=self.verbose)
            elif metric == 'Min Dihedral(deg)':
                quality[metric] = Quality.MinDihedral(*self,verbose=self.verbose)*180/np.pi
            elif metric == 'Max Dihedral':
                quality[metric] = Quality.MaxDihedral(*self,verbose=self.verbose)
            elif metric == 'Max Dihedral(deg)':
                quality[metric] = Quality.MaxDihedral(*self,verbose=self.verbose)*180/np.pi
            
            elif metric == 'Volume':
                quality[metric] = Quality.Volume(*self,verbose=self.verbose)
            else:
                raise Exception('Invalid quality metric.')
        return quality

    def getCurvature(self,metrics=['Max Principal','Min Principal', 'Curvedness', 'Shape Index', 'Mean', 'Gaussian'], nRings=1, SplitFeatures=False):
        
        if type(metrics) is str: metrics = [metrics]
        Curvature = {}
        if SplitFeatures:
            edges,corners = MeshUtils.DetectFeatures(self.NodeCoords,self.SurfConn)
            FeatureNodes = set(edges).union(corners)
            NodeRegions = MeshUtils.getConnectedNodes(self.NodeCoords,self.SurfConn,BarrierNodes=FeatureNodes)
            MaxPs = np.nan*np.ones((self.NNode,len(NodeRegions))) 
            MinPs = np.nan*np.ones((self.NNode,len(NodeRegions))) 
            for i,region in enumerate(NodeRegions):
                Elems = [elem for elem in self.SurfConn if all([n in region for n in elem])]
                # ElemNormals = [self.ElemNormals[i] for i,elem in enumerate(self.SurfConn) if all([n in region for n in elem])]
                ElemNormals = MeshUtils.CalcFaceNormal(self.NodeCoords,Elems)
                Neighbors,ElemConn = MeshUtils.getNodeNeighbors(self.NodeCoords,Elems)
                if nRings > 1:
                    Neighbors = MeshUtils.getNodeNeighborhood(self.NodeCoords,Elems,nRings)
                NodeNormals = MeshUtils.Face2NodeNormal(self.NodeCoords,Elems,ElemConn,ElemNormals)
                MaxPs[:,i], MinPs[:,i] = curvature.CubicFit(self.NodeCoords,Elems,Neighbors,NodeNormals)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                MaxPrincipal = np.nanmean(MaxPs,axis=1)
                MinPrincipal = np.nanmean(MinPs,axis=1)

        else:
            if nRings == 1:
                Neighbors = self.SurfNodeNeighbors
            else:
                Neighbors = MeshUtils.getNodeNeighborhood(self.NodeCoords,self.SurfConn,nRings=nRings)
            MaxPrincipal,MinPrincipal = curvature.CubicFit(self.NodeCoords,self.SurfConn,Neighbors,self.NodeNormals)
        if 'Max Principal' in metrics:
            Curvature['Max Principal Curvature'] = MaxPrincipal
        if 'Max Principal' in metrics:
            Curvature['Min Principal Curvature'] = MinPrincipal
        if 'Shape Index' in metrics or 'Shape Category' in metrics:
            SI = curvature.ShapeIndex(MaxPrincipal,MinPrincipal)
            if 'Shape Index' in metrics:
                Curvature['Shape Index'] = SI
        if 'Shape Category' in metrics:
            SC = curvature.ShapeCategory(SI)
            Curvature['Shape Category'] = SC
        if 'Curvedness' in metrics:
            C = curvature.Curvedness(MaxPrincipal,MinPrincipal)
            Curvature['Curvedness'] = C
        if 'Gaussian' in metrics:
            G = curvature.GaussianCurvature(MaxPrincipal,MinPrincipal)
            Curvature['Gaussian Curvature'] = G
        if 'Mean' in metrics:
            M = curvature.MeanCurvature(MaxPrincipal,MinPrincipal)
            Curvature['Mean Curvature'] = M
        
        return Curvature

    def save(self,filename,method='pickle'):
        
        self.NodeConn = [[int(n) for n in elem] for elem in self.NodeConn]
        self._Faces = [[int(n) for n in face] for face in self._Faces]
        self._SurfConn = [[int(n) for n in face] for face in self._SurfConn]
        
        if method=='pickle' or '.pickle' in filename:
            if '.pickle' not in filename: filename += '.pickle'
            with open(filename,'wb') as f:
                pickle.dump(self,f)
        elif method=='json' or '.json' in filename:
            if '.json' not in filename: filename += '.json'
            with open(filename,'w') as f:
                json.dump(self.__dict__,f)
                
        else:
            raise Exception('Unknown method')
    def load(self,filename,method='pickle'):
        
        if method=='pickle' or '.pickle' in filename:
            if '.pickle' not in filename: filename += '.pickle'
            with open(filename,'rb') as f:
                temp = pickle.load(f)
            self.__dict__ = temp.__dict__
        elif method=='json' or '.json' in filename:
            if '.json' not in filename: filename += '.json'
            with open(filename,'r') as f:
                self.__dict__ = json.load(f)
        else:
            raise Exception('Unrecognized file')   
    def Mesh2Meshio(self,PointData={},CellData={}):
        
        points = self.NodeCoords

        if type(PointData) is list or type(PointData) is np.ndarray:
            self.NodeData['_NodeVals_'] = PointData
        elif type(PointData) is dict:
            self.NodeData = {**self.NodeData,**PointData}
        
        if type(CellData) is list or type(CellData) is np.ndarray:
            self.ElemData['_ElemVals_'] = CellData
        elif type(CellData) is dict:
            self.ElemData = {**self.ElemData,**CellData}

        celldict = dict()
        if len(self.ElemData) > 0:
            keys = self.ElemData.keys()
            for key in keys:
                celldata = [[],[],[],[],[],[]]
                if self.nD == 3:
                    for i,elem in enumerate(self.NodeConn):
                        n = len(elem)
                        if n == 3:
                            if len(self.ElemData) > 0: celldata[0].append(self.ElemData[key][i])
                        elif n == 4:
                            if len(self.ElemData) > 0: celldata[1].append(self.ElemData[key][i])
                        elif n == 5:
                            if len(self.ElemData) > 0: celldata[2].append(self.ElemData[key][i])
                        elif n == 6:
                            if len(self.ElemData) > 0: celldata[3].append(self.ElemData[key][i])
                        elif n == 8:
                            if len(self.ElemData) > 0: celldata[4].append(self.ElemData[key][i])
                        elif n == 10:
                            if len(self.ElemData) > 0: celldata[5].append(self.ElemData[key][i])
                celldata = [c for c in celldata if len(c) > 0]
                celldict[key] = celldata
        tris = []   # n = 3            
        tets = []   # n = 4
        pyrs = []   # n = 5
        wdgs = []   # n = 6
        hexs = []   # n = 8
        tet10 = []   # n = 10
        for i,elem in enumerate(self.NodeConn):
            n = len(elem)
            if n == 3:
                tris.append(elem)
            elif n == 4:
                tets.append(elem)
            elif n == 5:
                pyrs.append(elem)
            elif n == 6:
                wdgs.append(elem)
            elif n == 8:
                hexs.append(elem)
            elif n == 10:
                tet10.append(elem)
        
        elems = [e for e in [('triangle',tris),('tetra',tets),('pyramid',pyrs),('wedge',wdgs),('hexahedron',hexs),('tetra10',tet10)] if len(e[1]) > 0]
        m = meshio.Mesh(points, elems, point_data=self.NodeData, cell_data=celldict)
        return m

    def write(self,filename,binary=None):
        if self.NNode == 0:
            warnings.warn('Mesh empty - file not written.')
            return
        m = self.Mesh2Meshio()
        if binary is not None:
            m.write(filename,binary=binary)
        else:
            m.write(filename)
    def Meshio2Mesh(m):
        
        if int(meshio.__version__.split('.')[0]) >= 5 and int(meshio.__version__.split('.')[1]) >= 2:
            NodeConn = [elem for cells in m.cells for elem in cells.data.tolist()]
        else:
            # Support for older meshio version
            NodeConn = [elem for cells in m.cells for elem in cells[1].tolist()]
        NodeCoords = m.points.tolist()
        M = mesh(NodeCoords,NodeConn)
        if len(m.point_data) > 0 :
            for key in m.point_data.keys():
                M.NodeData[key] = m.point_data[key]
        if len(m.cell_data) > 0:
            for key in m.cell_data.keys():
                M.ElemData[key] = [data for celldata in m.cell_data[key] for data in celldata]
        M.NodeSets = m.point_sets
        M.ElemSets = m.cell_sets    # TODO: This might not give the expected result

        return M
    
    def read(file):
        """
        read read a mesh file written in any file type supported by meshio

        Parameters
        ----------
        file : str
            File path to a mesh file readable by meshio (.vtu, .vtk, .inp, .stl, ...)

        Returns
        -------
        M : mesh.mesh
            Mesh object
        """        
        m = meshio.read(file)
        M = mesh.Meshio2Mesh(m)

        return M  
            
    def imread(img, voxelsize, scalefactor=1, scaleorder=1, return_nodedata=False, return_gradient=False, gaussian_sigma=1, threshold=None, crop=None):
        """
        imread load a 3d image stack into a voxel mesh  using converter.im2voxel

        Parameters
        ----------
        img : str or np.ndarray
            If a str, should be the directory to an image stack of tiff or dicom files.
            If an array, shoud be a 3D array of image data.
        voxelsize : float
            Size of voxel (based on image resolution).
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
        M : mesh.mesh
            Mesh object, containing image data for elements and nodes in M.ElemData['Image Data'] and M.NodeData['Image Data'].
        """
        if return_nodedata:
            VoxelCoords, VoxelConn, VoxelData, NodeData = converter.im2voxel(img,voxelsize,scalefactor=scalefactor,scaleorder=scaleorder,return_nodedata=return_nodedata,return_gradient=return_gradient, gaussian_sigma=gaussian_sigma,threshold=threshold,crop=crop)
        else:
            VoxelCoords, VoxelConn, VoxelData = converter.im2voxel(img,voxelsize,scalefactor=scalefactor,scaleorder=scaleorder,return_nodedata=return_nodedata,return_gradient=return_gradient,gaussian_sigma=gaussian_sigma,threshold=threshold,crop=crop)
        M = mesh(VoxelCoords,VoxelConn)
        if return_gradient:
            M.ElemData['Image Data'],M.ElemData['Image Gradient'] = VoxelData
            if return_nodedata: M.NodeData['Image Data'], M.NodeData['Image Gradient']  = NodeData
        else:
            M.ElemData['Image Data'] = VoxelData
            if return_nodedata: M.NodeData['Image Data'] = NodeData
        return M
        