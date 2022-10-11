# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:54:44 2021

@author: toj
"""
import pandas as pd
import numpy as np
import os, time, sys, h5py, csv
from . import mesh, MeshUtils

def parseMechanicalResultTxt(resultTxt):
    # To export data from Ansys mechanical, from mechanical, do 
    # File -> Options -> Export, set "Include Node Numbers" and 
    # "Include Node Location" to "Yes"
    
    df = pd.read_csv(resultTxt,sep='\t')
    if ('X Location' in df.columns.tolist()[1]) and ('Y Location' in df.columns.tolist()[2]) and ('Z Location' in df.columns.tolist()[3]):
            # Node Coordinates provided in columns 1, 2, 3
            X = df[df.columns.tolist()[1]]
            Y = df[df.columns.tolist()[2]]
            Z = df[df.columns.tolist()[3]]
            NodeCoords = [[X[i],Y[i],Z[i]] for i in range(len(df))]
    else:
        raise Warning('Nodal Coordinates not Provided')
        NodeCoords = []
    if NodeCoords != [] and len(df.columns.tolist()) == 5:
        # Both Node Coords and Result data are provided
        NodeVals = df[df.columns.tolist()[4]]
    elif NodeCoords == [] and len(df.columns.tolist()) == 2:
        # Node Coords not provided but nodal outputs are
        NodeVals = df[df.columns.tolist()[1]]
    else:
        raise Warning('Nodal Output not Provided')
        NodeVals = []
    return NodeCoords, NodeVals.tolist()

def parseMechanicalElementTxt(ElementTxt):
    # To get elements, right click in the model window in Mechanical and 
    # select "Select Mesh by ID...", then select all elements by selecting 
    # the Element option then entering the id "1-n" where n is the total 
    # number of elements (can be found by looking under Mesh -> Statistics), 
    # then right click on the named selection and chose 
    # "Export... -> Export Text File"
    
    with open(ElementTxt) as file:
        # NodeConn = [[] for i in range(len(file.readlines())-1)]
        NodeConn = []
        tsv = csv.reader(file, delimiter="\t")
        k = 0
        for line in tsv:
            if k != 0:
                Type = line[1]
                if Type == 'Tet10':
                    NodeConn.append([int(line[j])-1 for j in range(2,12)])                
                else:
                    raise Exception('Element Type ' + Type + ' not yet Implemented')
            k += 1
    
    return NodeConn

def parseCFDPostExport(CFDPostCsv):
    
    readmode = 'init'
    add2NodeNum = 0
    
    NodeCoords = []
    NodeConn = []
    NodeVals = []
    
    with open(CFDPostCsv) as file:
        csv_file = csv.reader(file)
        for line in csv_file:      
            if line == []:
                continue
            if line[0] == '[Name]':
                readmode = 'init'
                add2NodeNum = len(NodeCoords)
                continue
            elif line[0] == 'Node Number':
                readmode = 'nodes'
                continue
            elif line[0] == '[Faces]':
                readmode = 'elem'
                continue
            
            if readmode == 'nodes':
                NodeCoords.append([float(line[1]),float(line[2]),float(line[3])])
                NodeVals.append([float(line[i]) if line[i] != ' null' else np.nan for i in range(4,len(line))])
            elif readmode == 'elem':
                NodeConn.append([int(line[i])+add2NodeNum for i in  range(len(line))])
                
    return NodeCoords, NodeConn, NodeVals
            
def writeMechanicalDatFile(Mesh, Material, bc, filename, jobname='job'):
      
    # Currently only works for 4 node tetrahedral mesh with a single material
    
    if '.dat' not in filename:
        filename += '.dat'
    
    assert Mesh.nD == 3, 'Currently only working for 3D'
    assert len(Mesh.NodeConn[0]) == 8, 'Currently only working for 8 node hexahedrons or degenerate variations'
    
    with open(filename, 'w') as f:
            ## Header
            f.write('/batch\n')
            f.write('/config,noeldb,1\n')
            f.write('*get,_wallstrt,active,,time,wall\n')
            f.write('/units,MKS\n')
            f.write('/nopr\n')
            f.write('/wb,file,start\n')
            f.write('/prep7\n')
            f.write('/nolist\n')
            f.write('etcon,set\n')
            
            ## Nodes
            f.write('nblock,3,,{0:d}\n'.format(Mesh.NNode))
            f.write('(1i9,3e21.13e3)\n')
            for i,node in enumerate(Mesh.NodeCoords):
                f.write('{0:9d}{1:21.13E}{2:21.13E}{3:21.13E}\n'.format(
                          i+1,    node[0],    node[1],    node[2]))
            f.write('-1\n')
            
            ## Elements
            f.write('/wb,elem,start\n') # Not sure if needed
            # Element Type
            f.write('et,1,185\n')           # 185 -> 8 Node Element
            ncol = min(11 + len(Mesh.NodeConn[0]),19)   # This might need to change for other element types?
            f.write('eblock,{0:d},solid,,{1:d}\n'.format(ncol,Mesh.NElem))
            f.write('({0:d}i9)\n'.format(ncol))
            for i,elem in enumerate(Mesh.NodeConn):
                f.write('{0:9d}{1:9d}{2:9d}{3:9d}{4:9d}{5:9d}{6:9d}{7:9d}{8:9d}{9:9d}{10:9d}'.format(1,1,1,1,0,0,0,0,len(elem),0,i+1) +
                        ''.join(['{' + str(j) + ':9d}' for j in range(len(elem))]).format(*(np.array(elem)+1)) + '\n')
                # Need to add a second line if 10+len(elem) > 19 for other element types
            f.write('-1\n')
            f.write('/wb,elem,end\n')
            
            f.write('csys,0\n')
            f.write('toffst,273.15,\n') # Temperature offset from absolute zero
            f.write('tref,22.\n')
            ## Material
            f.write('/wb,mat,start\n')
            f.write('MP,DENS,1,{0:f}\n'.format(Material.Mechanical['Elastic']['rho']))
            f.write('MP,EX,1,{0:f}\n'.format(Material.Mechanical['Elastic']['E']))
            f.write('MP,NUXY,1,{0:f}\n'.format(Material.Mechanical['Elastic']['nu']))
            f.write('/wb,mat,end\n')
            
            
            ## BC
            f.write('/wb,load,start\n')
            # Degree-of-freedom constraints at nodes
            labels = ['ux','uy','uz']
            for i,predisp in enumerate(bc.Mechanical['PreDisp']):
                if type(predisp[2]) == list:
                    # Transient load
                    assert len(predisp[2]) == len(bc.TimeTable), 'TimeTable list must be the same length as the load list'
                    f.write('*DIM,_disptable{0:d},TABLE,{1:d},1,1,TIME\n'.format(i,len(predisp[2])))
                    f.write('*TAXIS,_disptable{0:d}(1),1,'.format(i)+','.join([str(x) for x in bc.TimeTable])+'\n')
                    for j,d in enumerate(predisp[2]):
                        f.write('_disptable{0:d}({1:d},1,1) = {2:f}\n'.format(i,j+1,d))
                    f.write('d,{0:d},{1:s},%_disptable{2:d}%\n'.format(predisp[0]+1,labels[predisp[1]],i))
                else:
                    f.write('d,{0:d},{1:s},{2:f}\n'.format(predisp[0]+1,labels[predisp[1]],predisp[2]))
                
            # Force loads at nodes
            labels = ['fx','fy','fz']
            for i,appforce in enumerate(bc.Mechanical['AppForce']):
                if type(appforce[2]) == list:
                    # Transient load
                    assert len(appforce[2]) == len(bc.TimeTable), 'TimeTable list must be the same length as the load list'
                    f.write('*DIM,_forctable{0:d},TABLE,{1:d},1,1,TIME\n'.format(i,len(appforce[2])))
                    f.write('*TAXIS,_forctable{0:d}(1),1,'.format(i)+','.join([str(x) for x in bc.TimeTable])+'\n')
                    for j,d in enumerate(appforce[2]):
                        f.write('_forctable{0:d}({1:d},1,1) = {2:f}\n'.format(i,j+1,d))
                    f.write('f,{0:d},{1:s},%_forctable{2:d}%\n'.format(appforce[0]+1,labels[appforce[1]],i))
                else:
                    f.write('f,{0:d},{1:s},{2:f}\n'.format(appforce[0]+1,labels[appforce[1]],appforce[2]))
                
                
            # FSI Nodes
            if len(bc.Mechanical['FSI']) > 0:
                f.write('CMBLOCK,   FSIN_1,NODE,{:d}\n'.format(len(bc.Mechanical['FSI'])))
                f.write('(8i10)\n')
                for i,node in enumerate(bc.Mechanical['FSI']):
                    f.write('{:10d}'.format(node+1))
                    if (i+1)%8 == 0:
                        f.write('\n')
                f.write('\n')
                f.write('SF,FSIN_1,FSIN,1\n')
                
            f.write('/wb,load,end\n')
            
            ## Solver Settings
            
            f.write('/solu\n')
            f.write('antype,4\n')                             # transient analysis
            f.write('nlgeom,off\n')                           # Turn on/off Large Deformation Effects
            f.write('kbc,1\n')                                # stepped BC's
            f.write('_thickRatio=  0\n')                      # Ratio of thick parts in the model
            f.write('eqsl,sparse,,,,,1\n')
            f.write('cntr,print,1\n')                         # print out contact info and also make no initial contact an error
            f.write('rstsuppress,none\n')                     # don't suppress anything due to presense of FSI loading
            f.write('dmpoption,emat,no\n')                    # Don't combine emat file for DANSYS
            f.write('dmpoption,esav,no\n')                    # Don't combine esav file for DANSYS
            f.write('cmwrite\n')                              # Export components due to presence of FSI loading
            f.write('trnopt,full,,,,,hht\n')                  # HHT time integration method
            f.write('tintp, mosp\n')			                     # Choose transient integration settings for moderate speed application
            f.write('nldiag,cont,iter\n')                     # print out contact info each equilibrium iteration
            f.write('SCOPT,NO\n')                             # Allow negative coefficients due to System Coupling
            f.write('rescontrol,define,last,last,,dele\n')	  # Program Controlled
            
            
            
            # Turn off automatic time stepping
            f.write('autots,off\n')
            # Number of substeps
            f.write('nsub,1,1,1\n') # 1 substeps
            f.write('time,1\n')     # Time for load step
            f.write('timint,on\n')  # Turn on transient
            
            # Set solution-result data written to database:
            f.write('outres,erase\n')
            f.write('outres,all,none\n')
            f.write('outres,nsol,all,\n')
            f.write('outres,rsol,all\n')
            f.write('outres,eangl,all\n')
            f.write('outres,etmp,all\n')
            f.write('outres,veng,all\n')
            f.write('outres,strs,all,\n')
            f.write('outres,epel,all,\n')
            f.write('outres,eppl,all,\n')
            f.write('outres,cont,all,\n')
            f.write('outres,v,all,\n')
            f.write('outres,a,all,\n')
            
            ## Solve
            f.write('solve\n')
            
            ## Post
            f.write('/post1\n')
            f.write('alls\n')
            f.write('*GET,numb_sets,ACTIVE,0,SET,NSET\n')
            f.write('set,first\n')
            f.write('*get,nummax,NODE,,num,max\n') #Max Node id in selected nodes
            f.write('*get,numnode,NODE,,count\n') #Number of selected nodes
            f.write('*dim,mask,array,nummax\n')
            f.write('*vget,mask(1),NODE,,NSEL\n') #mask for selected nodes
            f.write('*dim,nodal_data_full,array,nummax,6\n') #array for pressures of all 1-to-nummax nodes
            f.write('*dim,nodal_data_comp,array,numnode,6\n') #array for pressures of ONLY selected nodes
            f.write('*dim,nodeid_full,array,nummax\n') #array for containing NODE ID
            f.write('*dim,nodeid_comp,array,numnode\n') #array for containing NODE ID for ONLY selected nodes
            f.write('*vfill,nodeid_full,ramp,1,1\n') #array from 1-to-nummnode (1,2,3,....)
            #
            f.write('*do,i_set,1,numb_sets,1\n')
            f.write('*GET,current_time,ACTIVE,0,SET,TIME\n')
            # Displacement
            f.write('*vmask,mask(1)\n')
            f.write('*vget,nodal_data_full(1,1),node,,U,X\n') 
            f.write('*vmask,mask(1)\n')
            f.write('*vget,nodal_data_full(1,2),node,,U,Y\n')
            f.write('*vmask,mask(1)\n')
            f.write('*vget,nodal_data_full(1,3),node,,U,Z\n')
            f.write('*cfopen,'+jobname+'_disp-time_%current_time%,DAT\n')
            f.write('*vwrite,nodal_data_full(1,1), nodal_data_full(1,2), nodal_data_full(1,3)\n')
            f.write('%E %E %E\n')   #formats for each column
            f.write('*cfclos\n')
            # Strain (Elastic)
            f.write('*vmask,mask(1)\n')
            f.write('*vget,nodal_data_full(1,1),node,,EPEL,X\n') 
            f.write('*vmask,mask(1)\n')
            f.write('*vget,nodal_data_full(1,2),node,,EPEL,Y\n')
            f.write('*vmask,mask(1)\n')
            f.write('*vget,nodal_data_full(1,3),node,,EPEL,Z\n')
            f.write('*vmask,mask(1)\n')
            f.write('*vget,nodal_data_full(1,4),node,,EPEL,XY\n') 
            f.write('*vmask,mask(1)\n')
            f.write('*vget,nodal_data_full(1,5),node,,EPEL,YZ\n')
            f.write('*vmask,mask(1)\n')
            f.write('*vget,nodal_data_full(1,6),node,,EPEL,XZ\n')
            f.write('*cfopen,'+jobname+'_strain-time_%current_time%,DAT\n')
            f.write('*vwrite,nodal_data_full(1,1), nodal_data_full(1,2), nodal_data_full(1,3), nodal_data_full(1,4), nodal_data_full(1,5), nodal_data_full(1,6)\n')
            f.write('%E %E %E %E %E %E\n')   #formats for each column
            f.write('*cfclos\n')
            
            #
            f.write('set,next\n') #read next set
            f.write('*ENDDO\n') #end loop
            
            ## End
            f.write('/wb,file,end\n')
    return filename
                
def writeMechanicalScpFile(datFile,workdir=None):
    # Create a .scp file from a .dat file
    if '.dat' not in datFile:
        datFile += '.dat'
    name = os.path.splitext(datFile)[0]
    scpFile = name+'.scp'
    if not workdir:
        workdir = os.path.dirname(datFile)
    
    with open(datFile,'r') as f:
        dat = f.readlines()
    regions = []
    antype = ''
    for line in dat:
        if ',FSIN,' in line:
            split = line.split(',')
            regions.append(split[1])
        if 'antype' in line:
            split = line.split(',')
            ant = str(int(split[1].split(' ')[0]))
            if ant == '0':
                antype = 'Static'
            elif ant == '4':
                antype = 'Transient'
            else:
                raise Exception("I haven't set this up for analysis types other than static or transient")
    if antype == '':
        raise Exception("Analysis Type Not Identified")
    with open(scpFile,'w') as f:
        # Level 0
        f.write('<CouplingParticipant>\n')
        # Level 1
        f.write('\t<ExecutionControl>\n')
        # Level 2
        f.write('\t\t<WorkingDirectory>')
        f.write(workdir)
        f.write('</WorkingDirectory>\n')
        
        f.write('\t\t<InitialInput>'+os.path.basename(datFile)+'</InitialInput>\n')
        # Level 1
        f.write('\t</ExecutionControl>\n')
        f.write('\t<CosimulationControl>\n')
        # Level 2
        f.write('\t\t<Type>MAPDL</Type>\n')
        
        f.write('\t\t<AnalysisType>')
        f.write(antype)
        f.write('</AnalysisType>\n')
        
        f.write('\t\t<DisplayName>MAPDL ')
        f.write(antype)
        f.write('</DisplayName>\n')
        
        f.write('\t\t<Regions>\n')
        for region in regions:
            # Level 3
            f.write('\t\t\t<Region>\n')
            # Level 4
            f.write('\t\t\t\t<Name>')
            f.write(region)
            f.write('</Name>\n')
            
            f.write('\t\t\t\t<DisplayName>')
            f.write(region+'_Fluid Solid Interface')
            f.write('</DisplayName>\n')
            
            f.write('\t\t\t\t<Topology>Surface</Topology>\n')
            
            f.write('\t\t\t\t<OutputVariables>\n')
            # Level 5
            f.write('\t\t\t\t\t<Variable>INCD</Variable>\n')
            # Level 4
            f.write('\t\t\t\t</OutputVariables>\n')
            
            f.write('\t\t\t\t<InputVariables>\n')
            # Level 5
            f.write('\t\t\t\t\t<Variable>FORC</Variable>\n')
            f.write('\t\t\t\t\t<Variable>FDNS</Variable>\n')
            # Level 4
            f.write('\t\t\t\t</InputVariables>\n')
            # Level 3
            f.write('\t\t\t</Region>\n')
        # Level 2
        f.write('\t\t</Regions>\n')
        
        f.write('\t\t<Variables>\n')
        # Level 3
        f.write('\t\t\t<Variable>\n')
        # Level 4
        f.write('\t\t\t\t<Name>FORC</Name>\n')
        f.write('\t\t\t\t<DisplayName>Force</DisplayName>\n')
        f.write('\t\t\t\t<QuantityType>Force</QuantityType>\n')
        f.write('\t\t\t\t<Location>Node</Location>\n')
        # Level 3
        f.write('\t\t\t</Variable>\n')
        
        f.write('\t\t\t<Variable>\n')
        # Level 4
        f.write('\t\t\t\t<Name>INCD</Name>\n')
        f.write('\t\t\t\t<DisplayName>Incremental Displacement</DisplayName>\n')
        f.write('\t\t\t\t<QuantityType>Incremental Displacement</QuantityType>\n')
        f.write('\t\t\t\t<Location>Node</Location>\n')
        # Level 3
        f.write('\t\t\t</Variable>\n')
        
        f.write('\t\t\t<Variable>\n')
        # Level 4
        f.write('\t\t\t\t<Name>FDNS</Name>\n')
        f.write('\t\t\t\t<DisplayName>Force Density</DisplayName>\n')
        f.write('\t\t\t\t<QuantityType>Force</QuantityType>\n')
        f.write('\t\t\t\t<Location>Element</Location>\n')
        # Level 3
        f.write('\t\t\t</Variable>\n')
        
        # Level 2
        f.write('\t\t</Variables>\n')
        
        # Level 1
        f.write('\t</CosimulationControl>\n')
        
        # Level 0
        f.write('</CouplingParticipant>')
        
        return scpFile
                     
def readFluentH5File(mshFile):
    
    Mesh = mesh.mesh()
    h5 = h5py.File(mshFile, 'r')
    
    #TODO: Need a loop to account for multiple meshes, for now assuming only 1
    
    # Parse Nodes
    coords = h5.get('meshes/1/nodes/coords')
    nodesets = list(h5.get('meshes/1/nodes/coords').keys())
    for i in range(len(nodesets)):
        startIdx = len(Mesh.NodeCoords)
        Mesh.addNodes(np.array(coords.get(nodesets[i])).tolist())
        endIdx = len(Mesh.NodeCoords)
        if h5['meshes/1/nodes/zoneTopology/zoneType'][i] == 2:
            name = 'node' + nodesets[i] + '-' + 'boundary'
        else:
            name = 'node' + nodesets[i]
        Mesh.NodeSets[name] = range(startIdx,endIdx)        
        
    # Parse Edges
    edges = h5.get('meshes/1/edges/nodes')    
    edgesets = list(h5.get('meshes/1/edges/nodes').keys())
    for i in range(len(edgesets)):
        startIdx = len(Mesh.Edges)
        nnodes = np.array(edges.get(edgesets[i]+'/nnodes'))
        nodes = np.array(edges.get(edgesets[i]+'/nodes'))
        Edges = []
        count = 0
        for j in range(len(nnodes)):
            edge = [nodes[k] for k in range(count,count+nnodes[j])]
            count += nnodes[j]
            Edges += [edge]
        Mesh.addEdges(Edges)
        endIdx = len(Mesh.Edges)
        if h5['meshes/1/edges/zoneTopology/edgeType'][i] == 5:
            name = 'edge' + edgesets[i] + '-' + 'boundary'
        else:
            name = 'edge' +edgesets[i]
        Mesh.EdgeSets[name] = range(startIdx,endIdx)        
    
    # Parse Faces
    faces = h5.get('meshes/1/faces/nodes')
    c0 = h5.get('meshes/1/faces/c0')
    c1 = h5.get('meshes/1/faces/c1')
    facesets = list(h5.get('meshes/1/faces/nodes').keys())
    CellConn = []    # 2 cells adjacent to the face
    for i in range(len(facesets)):
        startIdx = len(Mesh.Faces)
        nnodes = np.array(faces.get(facesets[i]+'/nnodes'))
        nodes = np.array(faces.get(facesets[i]+'/nodes'))
        Faces = []
        count = 0
        for j in range(len(nnodes)):
            face = [nodes[k]-1 for k in range(count,count+nnodes[j])]
            count += nnodes[j]
            Faces += [face]
        c0s = np.array(c0.get(facesets[i]))-1.0     # By default, cells are labeled starting at 1 with 0 indicating no adjacent cell
        c1s = np.array(c1.get(facesets[i]))-1.0     # Changing s.t. -1 indicates no adjacent cell with labeling starting at 0 for cells
        CellConn += np.vstack((c0s,c1s)).transpose().astype(int).tolist()

        Mesh.addFaces(Faces)
        endIdx = len(Mesh.Faces)
        name = 'face' + facesets[i] + '-' + 'bcID' + str(h5['meshes/1/faces/zoneTopology/zoneType'][i]) 
        Mesh.FaceSets[name] = range(startIdx,endIdx)
    Mesh.FaceElemConn = CellConn
    
    # Parse Cells
    cells = h5.get('meshes/1/cells/ctype')
    cellsets = list(h5.get('meshes/1/cells/ctype').keys())
    celltypes = []
    for i in range(len(cellsets)):
        startIdx = len(celltypes)
        celltype = np.array(cells.get(cellsets[i] + '/cell-types')).tolist()
        celltypes += celltype
        endIdx = len(celltypes)
        name = 'node' + cellsets[i]
        Mesh.ElemSets[name] = range(startIdx,endIdx)       
    NElem = len(celltypes)
    
    FaceConn = [[] for i in range(NElem)]   # Includes the faces adjacent to a given cell
    for i in range(len(CellConn)):
        for j in range(len(CellConn[i])):
            if CellConn[i][j] >= 0:
                FaceConn[CellConn[i][j]] += [i]
    Mesh._FaceConn = FaceConn
    NodeConn = [np.unique([Mesh.Faces[j][k] for j in FaceConn[i] for k in range(len(Mesh.Faces[j]))]).tolist() for i in range(len(FaceConn))]
    Mesh.addElems(NodeConn)
    
    Mesh.ElemType = 'mixed3d'
    Mesh.setnD()
    Mesh.validate()
    h5.close()
        
    return Mesh

def writeFluentH5File(Mesh, filename):
    raise Exception("This code doesn't work properly")
    if '.msh.h5' not in filename:
        filename += '.msh.h5'
    
    with h5py.File(filename,'w') as h5:
        # Create Fluent CFF hierarchy
        coords = h5.create_group('meshes/1/nodes/coords')
        edges = h5.create_group('meshes/1/edges')
        faces = h5.create_group('meshes/1/faces')
        cells = h5.create_group('meshes/1/cells/ctype')
        
        # Write attributes
        h5['meshes/1'].attrs['cellCount'] = np.array([Mesh.NElem])
        h5['meshes/1'].attrs['cellOffset'] = np.array([0])
        h5['meshes/1'].attrs['dimension'] = np.array([Mesh.nD])
        h5['meshes/1'].attrs['edgeCount'] = np.array([Mesh.NEdge])
        h5['meshes/1'].attrs['faceCount'] = np.array([Mesh.NFace])
        h5['meshes/1'].attrs['faceOffset'] = np.array([0])
        h5['meshes/1'].attrs['nodeCount'] = np.array([Mesh.NNode])
        h5['meshes/1'].attrs['nodeOffset'] = np.array([0])
        h5['meshes/1'].attrs['version'] = np.array([2])
        
        # Nodes
        h5.create_group('meshes/1/nodes/zoneTopology')
        h5['meshes/1/nodes/zoneTopology'].attrs['nZones'] = np.array([len(Mesh.NodeSets)])
        setKeys = list(Mesh.NodeSets.keys())
        nodeId = h5['meshes/1/nodes/zoneTopology'].create_dataset('id',(len(Mesh.NodeSets),))
        nodeId.attrs['version'] = np.array([3])
        nodeId[:] = np.array([i+1 for i in range(len(Mesh.NodeSets))])
        nodeDim = h5['meshes/1/nodes/zoneTopology'].create_dataset('dimension',(len(Mesh.NodeSets),))
        nodeDim[:] = np.array([Mesh.nD for i in range(len(Mesh.NodeSets))])
        nodeDim.attrs['version'] = np.array([3])
        nodeMaxId = h5['meshes/1/nodes/zoneTopology'].create_dataset('maxId',(len(Mesh.NodeSets),))
        nodeMaxId[:] = np.array([max(Mesh.NodeSets[setKeys[i]]) for i in range(len(Mesh.NodeSets))])
        nodeMaxId.attrs['version'] = np.array([3])
        nodeMinId = h5['meshes/1/nodes/zoneTopology'].create_dataset('minId',(len(Mesh.NodeSets),))
        nodeMinId[:] = np.array([min(Mesh.NodeSets[setKeys[i]]) for i in range(len(Mesh.NodeSets))])
        nodeMinId.attrs['version'] = np.array([3])
        
        dt = h5py.special_dtype(vlen=bytes)
        nodeName = h5['meshes/1/nodes/zoneTopology'].create_dataset('name',(1,),dtype=dt)
        nodeName[0] = ';'.join(['node-'+str(nodeId[i]) for i in range(len(Mesh.NodeSets))])
        
        nodeName.attrs['version'] = np.array([3])
        nodeType = h5['meshes/1/nodes/zoneTopology'].create_dataset('zoneType',(len(Mesh.NodeSets),))    
        nodeType[:] = np.array([1 for i in range(len(Mesh.NodeSets))])    ### Specifying all nodes as type (1) -> no/any type (This might be a problem)
        nodeType.attrs['version'] = np.array([3])
        nodeFields = h5['meshes/1/nodes/zoneTopology'].create_dataset('fields',(1,),dtype=dt)   
        nodeFields[:] = 'zoneType;'
        nodeFields.attrs['version'] = np.array([3])
        
        for i in range(len(Mesh.NodeSets)):
            setname = list(Mesh.NodeSets.keys())[i]        
            data = np.array([Mesh.NodeCoords[node] for node in Mesh.NodeSets[setname]])
            dset = coords.create_dataset(setname,data.shape, compression="gzip")
            dset[:] = data
            # Attributes
            coords[setname].attrs['maxId'] = np.array([max(Mesh.NodeSets[setname])])+1
            coords[setname].attrs['minId'] = np.array([min(Mesh.NodeSets[setname])])+1
            
            
        #########
        # Edges
        h5.create_group('meshes/1/edges/zoneTopology')
        h5['meshes/1/edges/zoneTopology'].attrs['nZones'] = np.array([len(Mesh.EdgeSets)])
        setKeys = list(Mesh.EdgeSets.keys())
        edgeId = h5['meshes/1/edges/zoneTopology'].create_dataset('id',(len(Mesh.EdgeSets),))
        edgeId[:] = np.array([i+max(nodeId)+1 for i in range(len(Mesh.EdgeSets))])
        edgeDim = h5['meshes/1/edges/zoneTopology'].create_dataset('dimension',(len(Mesh.EdgeSets),))
        edgeDim[:] = np.array([Mesh.nD for i in range(len(Mesh.EdgeSets))])
        edgeMaxId = h5['meshes/1/edges/zoneTopology'].create_dataset('maxId',(len(Mesh.EdgeSets),))
        edgeMaxId[:] = np.array([max(Mesh.EdgeSets[setKeys[i]]) for i in range(len(Mesh.EdgeSets))])
        edgeMinId = h5['meshes/1/edges/zoneTopology'].create_dataset('minId',(len(Mesh.EdgeSets),))
        edgeMinId[:] = np.array([min(Mesh.EdgeSets[setKeys[i]]) for i in range(len(Mesh.EdgeSets))])
        edgeName = h5['meshes/1/edges/zoneTopology'].create_dataset('name',(1,),dtype=dt)
        edgeName[:] = ';'.join(['edge-'+str(edgeId[i]) for i in range(len(Mesh.EdgeSets))])
        edgeType = h5['meshes/1/edges/zoneTopology'].create_dataset('edgeType',(len(Mesh.EdgeSets),))
        edgeType[:] = np.array([6 for i in range(len(Mesh.EdgeSets))]) ### Everying is type 6 = interior for now
        zoneType = h5['meshes/1/edges/zoneTopology'].create_dataset('zoneType',(len(Mesh.EdgeSets),))
        zoneType[:] = np.array([3 for i in range(len(Mesh.EdgeSets))]) 
        edgeFields = h5['meshes/1/edges/zoneTopology'].create_dataset('fields',(1,),dtype=dt)   
        edgeFields[:] = 'edgeType;zoneType;'
        
        edgenodes = edges.create_group('nodes')    
        edgenodes.attrs['nSections'] = np.array([len(Mesh.EdgeSets)])
            
        for i in range(len(Mesh.EdgeSets)):
            setname = list(Mesh.EdgeSets.keys())[i]
            
            setEdges = [Mesh.Edges()[edge] for edge in Mesh.EdgeSets[setname]]
            group = edgenodes.create_group(setname)
            nnodes = [0 for i in range(len(setEdges))]
            nodes = []
            for j in range(len(setEdges)):
                nnodes[j] = len(setEdges[j])
                nodes += setEdges[j]     
                
            nnodes = np.array(nnodes,dtype='u4')
            nodes = np.array(nodes,dtype='u4')
            nnodesdset = group.create_dataset('nnodes',nnodes.shape,dtype='u4', compression="gzip")
            nnodesdset[:] = nnodes
            nodesdset = group.create_dataset('nodes',nodes.shape,dtype='u4', compression="gzip")
            nodesdset[:] = nodes
            
            # Attributes
            edgenodes[setname].attrs['maxId'] = np.array([max(Mesh.EdgeSets[setname])])+1
            edgenodes[setname].attrs['minId'] = np.array([min(Mesh.EdgeSets[setname])])+1
        
        #############
            
        # Faces
        h5.create_group('meshes/1/faces/zoneTopology')
        h5['meshes/1/faces/zoneTopology'].attrs['nZones'] = np.array([len(Mesh.FaceSets)])
        setKeys = list(Mesh.FaceSets.keys())
        faceId = h5['meshes/1/faces/zoneTopology'].create_dataset('id',(len(Mesh.FaceSets),))
        faceId[:] = np.array([i+max(edgeId)+1 for i in range(len(Mesh.FaceSets))])
        faceDim = h5['meshes/1/faces/zoneTopology'].create_dataset('dimension',(len(Mesh.FaceSets),))
        faceDim[:] = np.array([Mesh.nD for i in range(len(Mesh.FaceSets))])
        faceMaxId = h5['meshes/1/faces/zoneTopology'].create_dataset('maxId',(len(Mesh.FaceSets),))
        faceMaxId[:] = np.array([max(Mesh.FaceSets[setKeys[i]]) for i in range(len(Mesh.FaceSets))])
        faceMinId = h5['meshes/1/faces/zoneTopology'].create_dataset('minId',(len(Mesh.FaceSets),))
        faceMinId[:] = np.array([min(Mesh.FaceSets[setKeys[i]]) for i in range(len(Mesh.FaceSets))])
        faceName = h5['meshes/1/faces/zoneTopology'].create_dataset('name',(1,),dtype=dt)
        faceName[:] = ';'.join(['face-'+str(faceId[i]) for i in range(len(Mesh.FaceSets))])
        childZoneId = h5['meshes/1/faces/zoneTopology'].create_dataset('childZoneId',(len(Mesh.FaceSets),))
        childZoneId[:] = np.array([0 for i in range(len(Mesh.FaceSets))])
        faceType = h5['meshes/1/faces/zoneTopology'].create_dataset('faceType',(len(Mesh.FaceSets),))
        faceType[:] = np.array([0 for i in range(len(Mesh.FaceSets))])
        shadowZoneId = h5['meshes/1/faces/zoneTopology'].create_dataset('shadowZoneId',(len(Mesh.FaceSets),))
        shadowZoneId[:] = np.array([0 for i in range(len(Mesh.FaceSets))])
        zoneType = h5['meshes/1/faces/zoneTopology'].create_dataset('zoneType',(len(Mesh.FaceSets),))
        zoneType[:] = np.array([3 for i in range(len(Mesh.FaceSets))])  ### Specifying all faces as walls for now (This might be a problem)
        faceFields = h5['meshes/1/faces/zoneTopology'].create_dataset('fields',(1,),dtype=dt)   
        faceFields[:] = 'childZoneId;faceType;shadowZoneId;zoneType;'
        
        facenodes = faces.create_group('nodes')
        c0grp = faces.create_group('c0')    
        c1grp = faces.create_group('c1')
        
        facenodes.attrs['nSections'] = np.array([len(Mesh.FaceSets)])
        c0grp.attrs['nSections'] = np.array([len(Mesh.FaceSets)])
        c1grp.attrs['nSections'] = np.array([len(Mesh.FaceSets)])
        
        # FaceConn = [[] for i in range(Mesh.NElem)]
        # for i in range(Mesh.NElem):
        #     tic = time.time()
        #     for j in range(Mesh.NFace):
        #         # if all(facenodes in Mesh.NodeConn[i] for facenodes in Mesh.Faces[j]):
        #         if set(Mesh.NodeConn[i]).issuperset(set(Mesh.Faces[j])):
        #             FaceConn[i].append(j)            
        #     print(time.time()-tic)
        
        ElemConn = [[] for i in range(Mesh.NFace)]
        for i in range(Mesh.NElem):
            for j in range(len(Mesh.FaceConn[i])):
                ElemConn[Mesh.FaceConn[i][j]].append(i+1)
        for i in range(len(ElemConn)):
            if len(ElemConn[i]) == 1:
                ElemConn[i].append(0)
        ElemConn = np.array(ElemConn)
        c0 = ElemConn[:,0]
        c1 = ElemConn[:,1]
        Mesh
        
        for i in range(len(Mesh.FaceSets)):
            setname = list(Mesh.FaceSets.keys())[i]
            
            setFaces = [Mesh.Faces[face] for face in Mesh.FaceSets[setname]]
            group = facenodes.create_group(setname)
            nnodes = [0 for i in range(len(setFaces))]
            nodes = []
            for j in range(len(setFaces)):
                nnodes[j] = len(setFaces[j])
                nodes += setFaces[j]     
                
            nnodes = np.array(nnodes,dtype='u4')
            if np.all(nnodes == 2):
                faceType[i] = 2
            elif np.all(nnodes == 3):
                faceType[i] = 3
            elif np.all(nnodes == 4):
                faceType[i] = 4
            elif np.all(nnodes >= 5):
                faceType[i] = 5
            nodes = np.array(nodes,dtype='u4')
            nnodesdset = group.create_dataset('nnodes',nnodes.shape,dtype='u4', compression="gzip")
            nnodesdset[:] = nnodes
            nodesdset = group.create_dataset('nodes',nodes.shape,dtype='u4', compression="gzip")
            nodesdset[:] = nodes
            
            c0data = np.array([c0[face] for face in Mesh.FaceSets[setname]],dtype='u4')
            c0dset = c0grp.create_dataset(setname,c0data.shape,dtype='u4', compression="gzip")
            c0dset[:] = c0data
            
            c1data = np.array([c1[face] for face in Mesh.FaceSets[setname]],dtype='u4')
            c1dset = c1grp.create_dataset(setname,c1data.shape,dtype='u4', compression="gzip")
            c1dset[:] = c1data
            # Attributes
            facenodes[setname].attrs['maxId'] = np.array([max(Mesh.FaceSets[setname])])+1
            facenodes[setname].attrs['minId'] = np.array([min(Mesh.FaceSets[setname])])+1
            c0grp[setname].attrs['maxId'] = np.array([max(Mesh.FaceSets[setname])])+1
            c0grp[setname].attrs['minId'] = np.array([min(Mesh.FaceSets[setname])])+1
            c1grp[setname].attrs['maxId'] = np.array([max(Mesh.FaceSets[setname])])+1
            c1grp[setname].attrs['minId'] = np.array([min(Mesh.FaceSets[setname])])+1
            
            
       
        
        # Cells
        h5.create_group('meshes/1/cells/zoneTopology')
        cells.attrs['nSections'] = np.array([len(Mesh.ElemSets)])
        h5['meshes/1/cells/zoneTopology'].attrs['nZones'] = np.array([len(Mesh.ElemSets)])
        setKeys = list(Mesh.ElemSets.keys())
        cellId = h5['meshes/1/cells/zoneTopology'].create_dataset('id',(len(Mesh.ElemSets),))
        cellId[:] = np.array([i+max(faceId)+1 for i in range(len(Mesh.ElemSets))])
        cellDim = h5['meshes/1/cells/zoneTopology'].create_dataset('dimension',(len(Mesh.ElemSets),))
        cellDim[:] = np.array([Mesh.nD for i in range(len(Mesh.ElemSets))])
        cellMaxId = h5['meshes/1/cells/zoneTopology'].create_dataset('maxId',(len(Mesh.ElemSets),))
        cellMaxId[:] = np.array([max(Mesh.ElemSets[setKeys[i]]) for i in range(len(Mesh.ElemSets))])
        cellMinId = h5['meshes/1/cells/zoneTopology'].create_dataset('minId',(len(Mesh.ElemSets),))
        cellMinId[:] = np.array([min(Mesh.ElemSets[setKeys[i]]) for i in range(len(Mesh.ElemSets))])
        cellName = h5['meshes/1/cells/zoneTopology'].create_dataset('name',(1,),dtype=dt)
        cellName[:] = ';'.join(['cell-'+str(cellId[i]) for i in range(len(Mesh.ElemSets))])
        cellType = h5['meshes/1/cells/zoneTopology'].create_dataset('cellType',(len(Mesh.ElemSets),))
        cellType[:] = np.array([0 for i in range(len(Mesh.ElemSets))])
        childZoneId = h5['meshes/1/cells/zoneTopology'].create_dataset('childZoneId',(len(Mesh.ElemSets),))
        childZoneId[:] = np.array([0 for i in range(len(Mesh.ElemSets))])
        
        for i in range(len(Mesh.ElemSets)):
            setname = list(Mesh.ElemSets.keys())[i]
            setNodeConn = [Mesh.NodeConn[elem] for elem in Mesh.ElemSets[setname]]
            setFaceConn = [Mesh.FaceConn[elem] for elem in Mesh.ElemSets[setname]]
            elemtypes = [0 for j in range(len(setNodeConn))]
            for j in range(len(setNodeConn)):
                if len(setNodeConn[j]) == 3 and len(setFaceConn[j]) == 3:
                    # Triangular
                    elemtypes[j] = 1
                elif len(setNodeConn[j]) == 4 and len(setFaceConn[j]) == 4 and Mesh.nD == 3:
                    # Tetrahedral
                    elemtypes[j] = 2
                elif len(setNodeConn[j]) == 4 and len(setFaceConn[j]) == 4 and Mesh.nD == 2:
                    # Quadrilateral
                    elemtypes[j] = 3
                elif len(setNodeConn[j]) == 8 and len(setFaceConn[j]) == 6:
                    # Hexahedral
                    elemtypes[j] = 4
                elif len(setNodeConn[j]) == 5 and len(setFaceConn[j]) == 5:
                    # Pyramid
                    elemtypes[j] = 5
                elif len(setNodeConn[j]) == 6 and len(setFaceConn[j]) == 5:
                    # Wedge
                    elemtypes[j] = 6
                elif len(setNodeConn[j]) > 6:
                    # Polyhedral
                    elemtypes[j] = 7
                    
            group = cells.create_group(setname)
            elemtypes = np.array(elemtypes,dtype='u4')
            dset = group.create_dataset('cell-types',elemtypes.shape,dtype='u4', compression="gzip")
            dset[:] = elemtypes
            # Attributes
            group.attrs['elementType'] = np.array([elemtypes[0] if len(np.unique(elemtypes))==1 else 0])
            cellType[i] = np.array([elemtypes[0] if len(np.unique(elemtypes))==1 else 0])
            group.attrs['minId'] = np.array([min(Mesh.ElemSets[setname])]) + 1
            group.attrs['maxId'] = np.array([max(Mesh.ElemSets[setname])]) + 1
            
        # Settings
        settings = h5.create_group('settings')
        
        cortex = settings.create_dataset('Cortex Variables',(1,), dtype=dt)
        cortex[0] = '\n(0 "Cortex variables:")\n(38 ((\n(meshing-mode-vars ((re-partition . #f)))\n(reference-frames-display (((name . "global") (display-state . ""))))\n(reference-frames-definition (((name . "global") (origin point 0. 0. 0. (frame . parent)) (orientation two-axis (axis ((axis-from axis-label x (frame . global)) (axis-to vector 1. 0. 0. (frame . parent))) ((axis-from axis-label y (frame . global)) (axis-to vector 0. 1. 0. (frame . parent)))) (auto? . #t)) (transformations))))\n(reference-frames (((name . "global") (id . 1) (parent . 0) (current-state (origin 0. 0. 0.) (quat 1. 0. 0. 0.)) (motion (motion-type . 0) (velocity 0. 0. 0.) (acceleration 0. 0. 0.) (omega (angle . 0.) (axis 0. 0. 0.)) (alpha (angle . 0.) (axis 0. 0. 0.)) (update . "")) (read-only? . #t))))\n(gui-processing? #t)\n(color/grid/color-by? id)\n(color/grid/types? #f)\n(view-list ((isometric ((0.005518403835594654 0.0090425880625844 0.008039513602852821) (-0.0008756716943279715 0. 0.001645438194059462) (-0.4999999701976776 0.7071067690849304 -0.4999999701976776) 0.005115260446791965 0.005115260446791965 "perspective") #(1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1)) (wf-view ((0.00551867950707674 0.009040285833179951 0.008037884719669819) (-0.0008737675962038338 0. 0.001645438140258193) (-0.5000000596046448 0.7071067690849304 -0.5) 0.005113957915455103 0.005113957915455103 "perspective") #(1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1.)) (front ((-0.0008756685511191085 0. 0.01443358388088362) (-0.0008756685511191085 0. 0.001645438216321016) (0. 1. 0.) 0.005115258265825043 0.005115258265825043 "perspective") #(1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1)) (back ((-0.0008756685511191085 0. -0.01114270744824159) (-0.0008756685511191085 0. 0.001645438216321016) (0. 1. 0.) 0.005115258265825043 0.005115258265825043 "perspective") #(1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1)) (right ((0.0119124771134435 0. 0.001645438216321016) (-0.0008756685511191085 0. 0.001645438216321016) (0. 1. 0.) 0.005115258265825043 0.005115258265825043 "perspective") #(1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1)) (left ((-0.01366381421568172 0. 0.001645438216321016) (-0.0008756685511191085 0. 0.001645438216321016) (0. 1. 0.) 0.005115258265825043 0.005115258265825043 "perspective") #(1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1)) (top ((-0.0008756685511191085 0.01278814566456261 0.001645438216321016) (-0.0008756685511191085 0. 0.001645438216321016) (0. 0. 1.) 0.005115258265825043 0.005115258265825043 "perspective") #(1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1)) (bottom ((-0.0008756685511191085 -0.01278814566456261 0.001645438216321016) (-0.0008756685511191085 0. 0.001645438216321016) (0. 0. -1.) 0.005115258265825043 0.005115258265825043 "perspective") #(1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1))))\n(face-displacement 1)\n(xy/bottom -1)\n(cx-mesh-version (21 2 0))\n(cx-file-written-mode "mesher"))))\n'
        origin = settings.create_dataset('Origin',(1,),dtype=dt) 
        origin[:] = 'ANSYS Unknown 2021 R2'
        solver = settings.create_dataset('Solver',(1,),dtype=dt) 
        solver[:] = 'ANSYS_FLUENT_MESHING'
        TGridData = settings.create_dataset('TGrid Meshing Data',(1,),dtype=dt)
        TGridData[:] = '\n(0 "TGrid meshing data:")\n(73 ())\n'
        TGridVar = settings.create_dataset('TGrid Variables',(1,),dtype=dt) 
        TGridVar[0] = '\n(0 "TGrid variables:")\n(60 (\n(mesh-lib #f)\n(cad-hierarchy ())\n(cad-node-zones ())\n(cad-zone-associations ())\n(cad-assoc ())\n(cad/import-options ((single? #t) (file "/projectnb2/spine-fe/Tim/TRI/TRI-350_125/Cylinder_6x12/HalfCylinder_33_files/dp0/FFF-1/DM/FFF-1.agdb.pmdb") (pattern "") (append? #t) (import-curvature? #t) (feature-angle 40) (scale-factor 1) (zone-name-prefix "") (one-zone-per "program-controlled") (one-face-zone-per "body") (export-cad-hierarchy? #f) (extract-feature? #t)))\n(cad/adv-options ((recursive? #f) (continue-on-error? #f) (create-faceting? #t) (savepmdb? #f) (length-unit "m") (import-part-names? #t) (import-body-names? #t) (create-conformal-faceting? #t) (conformal-faceting-minsize 4e-05) (conformal-faceting-maxsize 4e-05) (conformal-faceting-curvature-normal-angle 18.) (sag-control? #f) (sag-tolerance 0) (sag-maxsize 0) (encl-symm-processing #f) (import-ns? #t) (ns-pattern "Layer*") (ns-regex "^(Color|Layer|Material|[0-9]:).*") (conf-faceting-merge-nodes #t) (conf-faceting-make-objs-independent #t) (conf-faceting-edge-proximity #f) (conf-faceting-cells-per-gap 1.) (conf-faceting-size-field-load? #f) (conf-faceting-size-field-file "/tmp/FM_scc-pi6_13919/FFF-1.sf") (conf-faceting-size-field-save? #t) (do-refaceting? #f) (sag-minsize 0) (conformal-tess #f) (conformal-tess-minsize 0) (conformal-tess-maxsize 0) (conformal-tess-tolerance 0) (double-connected-face-label "") (single-connected-edge-label "single") (name-separator-character ":") (use-collection-names "auto") (use-component-names "auto") (object-type "auto") (reconstruct-topology "") (import-named-selection-labels? #t) (separate-features-by-type? #f) (named-selection-failure "failed") (use-body-names? #t) (use-part-names? "auto") (replacement-char "-") (create-regions-per-body? #t) (derive-zone-name? #f) (merge-nodes? #t) (defeaturing? #f) (defeaturing-tol 0) (defeaturing-skew 0.7) (modify-all-duplicate-names? #f) (use-part-or-body-names-as-suffix-to-named-selections? #t) (strip-file-name-extension-from-naming? #f) (import-label-for-body-named-selection? #f) (strip-path-prefix-from-names? "auto") (merge-objects-per-body-named-selection? #f) (use-conformal-faceting-sizing-table? #f) (conf-faceting-filter-edges? #f) (ug-nx-cad-faceting? #f) (cad-ug-tolerance 0.1) (cad-ug-maxsize 0) (cad-ug-angular-deviation 0) (align-edge-tessellation #f) (cfd-surface-mesh-controls-additional ((curvature #t) (proximity #t) (growth-rate 1.2) (proximity-source-type ("EdgeEdge")) (ignore-orientation #t) (ignore-self #f) (scope-sizing? #t))) (sag-refine-faceting? #f) (create-label-per-body-during-cad-faceting? #f) (conformal-tess? #f)))\n(boundary/remove-gaps-settings ((min-gap-dist 0) (gap-dist 0) (percent-margin 20) (crit-angle 30)))\n(svar/growth-factor 0.125)\n(svar/shrink-factor 0.4)\n(compact-entities/threshold 1.125)\n(compact-entities/list-compact? #f)\n(entity/alloc-factor 0.1)\n(entity/alloc-threshold 250000)\n(entity/id-growth-factor 0.125)\n(anm/cell-sort-option 1)\n(anm/seed 42)\n(anm/validity-check 0)\n(anm/check-warp? #t)\n(anm/check-hand? #t)\n(normalized-sphere-tolerance 1e-07)\n(normalized-center-tolerance 1e-15)\n(sliver-size1 3.35068126771918e-25)\n(sliver-size-scale1 1e-12)\n(sliver-size-upper-end3 1e-08)\n(sliver-size-lower-end1 1e-12)\n(node-tolerance 1.956121790057816e-08)\n(remesh-method 0)\n(sliver-skew 0.97)\n(hole-method 3)\n(hole/separate-thread? #f)\n(hole/remesh? #f)\n(max-skew-limit 1.)\n(max-cell-size 2.450465287837588e-14)\n(max-cell-size-factor 2.)\n(max-cell-skew 0.85)\n(max-cell-skew-sv 0.85)\n(max-cell-skew-ss 0.8)\n(max-boundary-cell-skew 0.7)\n(max-boundary-cell-skew-sv 0.7)\n(max-boundary-cell-skew-ss 0.6)\n(min-swap-skew 0.4)\n(min-swap-skew-sv 0.4)\n(min-swap-skew-ss 0.25)\n(boundary-cell-height 1.)\n(min-boundary-closeness 0.85)\n(min-node-closeness 0.3)\n(check-boundary-closeness #t)\n(node-must-reduce-skew #f)\n(node-insertion-method 2)\n(node-insertion-offset 0.01)\n(cell-refine-size-ratio 1.)\n(max-cells 0)\n(max-nodes 0)\n(boundary-recovery-splits 10)\n(boundary-recovery-skew1 0.999)\n(boundary-recovery-skew1-sv 0.999)\n(boundary-recovery-skew1-ss 0.997)\n(surf/boundary-closeness 0.85)\n(surf/node-closeness 0.3)\n(surf/insert-boundary-closeness 0.05)\n(surf/insert-node-closeness 0.05)\n(check-bound-intersections #t)\n(check-bound-intersections-early #f)\n(recover-intersecting-bound #f)\n(front-partition-mesh #t)\n(front-unrefined-iter 1)\n(front-stalled-iter 3)\n(poly/mesh-type 0)\n(poly/prism-growth ((default constant 3 1 5 aspect-ratio #f)))\n(poly/layer-growth ((default constant 2 1 5 aspect-ratio #f)))\n(poly/keep-primary-mesh? #f)\n(poly/merge-boundary? #t)\n(poly/feature-angle 30.)\n(poly/corner-angle 45.)\n(poly/remesh? #t)\n(poly/connect? #f)\n(poly/skip-tet-refine? #f)\n(poly/smooth-improve? #t)\n(poly/laplace-smooth-convergence 0.0001)\n(poly/laplace-smooth-iter 3)\n(poly/edge-smooth-iter 2)\n(poly/centroid-smooth-iter 3)\n(poly/swap-iter 10)\n(poly/smooth-skew 0.8)\n(poly/prism-smooth-skew 0.9)\n(poly/smooth-iter 5)\n(poly/smooth-attempts 50)\n(poly/smooth-bound1? #t)\n(poly/smooth-corner? #f)\n(poly/smooth-on-layer? #t)\n(poly/smooth-improve-iter 5)\n(poly/smooth-improve-tries 5)\n(poly/stage1-improve-skew1 0.95)\n(poly/improve-bound1? #t)\n(poly/improve-aggressive? #t)\n(poly/stage2-improve-skew1 0.925)\n(poly/flat-skew 0.99)\n(poly/merge-skew 0.95)\n(poly/sliver-cell-area-fraction 0.001)\n(poly/cell-area-fraction 0.01)\n(poly/edge-size-ratio 20)\n(poly/face-size-ratio 100)\n(poly/prism-edge-size-ratio 20)\n(poly/split-thin-poly-prism-cell1? #f)\n(poly/split-thin-check-lefthanded? #f)\n(poly/smooth-poly-prism-cell? #t)\n(poly/remesh-target-skew 0.99)\n(poly/remesh-front-extensions 3)\n(poly/remesh-global-extensions 3)\n(poly/remesh-smooth-tries 5)\n(poly/improve-cell-connectivity1? #t)\n(poly/post-improve? #t)\n(collapse-feature-angle 80.)\n(safe-face-collapse #t)\n(rand-quad-split? #t)\n(rand-quad-split-tol 0.1)\n(mesh-check/periodic-tol 0.001)\n(mesh-check/min-face-area-factor 1e-08)\n(skewness-method 27)\n(max-domain-thread-count 10)\n(impose/cell-size-method 2)\n(impose/cell-size-growth 1.2)\n(impose/max-growth 3.)\n(impose/length-based? #t)\n(skew-refine-improve1? #t)\n(front-refine-improve? #t)\n(front/face-closeness 0.5)\n(front/node-closeness 0.7)\n(front/freeze-edge-ratio1 1.55)\n(front/freeze-height-ratio1 1.05)\n(front/freeze-lscale-ratio1 1.5)\n(front/freeze-size-ratio1 2.)\n(simplex-smooth-method 1)\n(simplex-smooth-relaxation 0.5)\n(simplex-opt-smooth-method 11)\n(simplex-inc-smooth-iterations 20)\n(simplex-inc-smooth-relaxation 0.05)\n(simplex-inc-smooth-feature-angle 30)\n(improve-skew-iter 10)\n(improve-skew-attempts 3)\n(remove-sliver-iter 10)\n(remove-sliver-attempts 5)\n(remove-sliver-method 0)\n(refine-improve-iter 5)\n(refine-improve-attempts 5)\n(utils/trace-across-thread? #t)\n(utils/trace-across-feature? #t)\n(backup-info ())\n(backup-object? #t)\n(prism/remesh-edges? #f)\n(prism/smooth-pre-ignore-side? #f)\n(prism/auto-separate-cell-zone? #f)\n(prism/check-quality? #t)\n(prism/check-cell-quality? #f)\n(prism/check-side-overlap? #t)\n(prism/remove-invalid-layer? #f)\n(prism/remove-incomplete-layer? #f)\n(prism/direction-normal? #t)\n(prism/direction-vector (0 0 1))\n(prism/edge-smooth1? #f)\n(prism/edge-swap1? #f)\n(prism/edge-swap-angle 10.)\n(prism/edge-swap-angle-2 30.)\n(prism/edge-smooth-angle 30.)\n(prism/grow-individually? #f)\n(prism/growth ((default constant 1. 1. 1 uniform #f)))\n(prism/layers 1)\n(prism/morph-frequency 5)\n(prism/morph-once #t)\n(prism/max-allowable-skew 0.98)\n(prism/max-allowable-cell-skew 0.99)\n(prism/max-valid-cell-skew 1)\n(prism/transition-cell-skew 0.9)\n(prism/check-size #t)\n(prism/check-allowable-skew #t)\n(prism/left-hand-check1 0)\n(prism/improve-warp? #f)\n(prism/max-warp 0.48)\n(prism/max-node-cell-skew 0.96)\n(prism/warp-iter-per-face 50)\n(prism/warp-iterations 4)\n(prism/improve-left-handed-faces? #t)\n(prism/post-improve? #t)\n(prism/shrink-left-cap? #t)\n(prism/identify-feature-line? #f)\n(prism/improve-transition-cell? #f)\n(prism/improve-node-path-iter 1)\n(prism/improve-node-path-node-iter 5)\n(prism/improve-node-path-angle-threshold 5.)\n(prism/improve-node-path-check-skew? #t)\n(prism/cell-quality-improve2? #f)\n(prism/corner-height-weight? #f)\n(prism/proximity-accuracy-check? #t)\n(prism/face-smooth1? #f)\n(prism/node-smooth1? #f)\n(prism/all-layers-prox? #t)\n(prism/proximity-stairstep? #f)\n(prism/allow-stairstepping? #f)\n(prism/create-transition-elements? #t)\n(prism/merge-transition-elements? #t)\n(prism/expose-internal-quads? #t)\n(prism/diff-adjacent-layers? #f)\n(prism/quality-ignore? #t)\n(prism/ignore-invalid-nodes? #f)\n(prism/ignore-non-manifold-nodes? #t)\n(prism/post-remove-intersections? #f)\n(prism/ignore-local-intersection? #f)\n(prism/quality-ignore-skew 0.9)\n(prism/prox-ignore? #f)\n(prism/ignore-extension 1)\n(prism/smooth-ignore? #t)\n(prism/normal-ignore? #f)\n(prism/growth-ignore? #f)\n(prism/merge-ignored-threads? #t)\n(prism/collision-first-fix? #f)\n(prism/collision-gap-factor 0.25)\n(prism/shrink-factor 0.5)\n(prism/smoothing-rate 2.)\n(prism/first-min-aspect-ratio 0.1)\n(prism/min-aspect-ratio 1.)\n(prism/max-aspect-ratio 25)\n(prism/max-transition-aspect-ratio 20)\n(prism/max-face-aspect-ratio1 2.35)\n(prism/ortho-band-aspect-ratio1 100)\n(prism/allowed-tangency 1.)\n(prism/node-smooth-angle 150.)\n(prism/node-smooth-normal-angle 85.)\n(prism/node-smooth-converged 0.0001)\n(prism/node-smooth-iter 100)\n(prism/node-smooth-local? #f)\n(prism/node-smooth-rings 1)\n(prism/node-project-local? #f)\n(prism/smooth-projected-nodes1? #f)\n(prism/face-smooth-skew 0.7)\n(prism/face-smooth-converged 0.0001)\n(prism/face-smooth-iter 100)\n(prism/face-smooth-rings 5)\n(prism/normal-bisect-angle 70.)\n(prism/normal-local-convergence? #t)\n(prism/normal-calc-method 0)\n(prism/normal-smooth? #t)\n(prism/normal-smoothing-method2 2)\n(prism/normal-smooth-angle1 60.)\n(prism/normal-smooth-factor 0.8)\n(prism/normal-smooth-converged 0.3)\n(prism/normal-smooth-iter 500)\n(prism/normal-smooth-adjust-offsets? #f)\n(prism/normal-surface-angle 30.)\n(prism/offset-method uniform)\n(prism/offset-smooth? #t)\n(prism/prox-adaptive-offset-smooth? #f)\n(prism/offset-smooth-method 0)\n(prism/offset-adjust? #f)\n(prism/offset-smooth-converged 0.01)\n(prism/offset-smooth-iter 100)\n(prism/weigh-offsets? #t)\n(prism/offset-weight 1.)\n(prism/last-aspect-ratio 40)\n(prism/orthogonal-layers 0)\n(prism/project? #t)\n(prism/project-adjacent-angle 80.)\n(prism/direction-project? #t)\n(prism/proximity-detect-angle 60.)\n(prism/project-converged 0.0001)\n(prism/project-iter 100)\n(prism/retriangulate-adjacent? #t)\n(prism/orient-adjacent-zones? #t)\n(prism/detect-short-adjacent-zones1? #t)\n(prism/swap-smooth-layers 100)\n(prism/skewness-method 11)\n(prism/face-skewness-method 0)\n(prism/swap-smooth-skew 0.2)\n(prism/retriangulate-feature-angle 30)\n(prism/smooth-invalid-cap? #f)\n(prism/smooth-layers1? #t)\n(prism/smooth-layer-skew-method1 11)\n(prism/smooth-layer-iter 20)\n(prism/smooth-layer-skew1 0.98)\n(prism/smooth-layer-conv 0.001)\n(prism/smooth-layer-max-weight 1.)\n(prism/smooth-layer-min-weight 1.)\n(prism/smooth-layer-invalid-weight 1.1)\n(prism/smooth-layer-limit-weight 1.)\n(prism/smooth-max-weight 5.)\n(prism/smooth-min-weight 1.33)\n(prism/smooth-invalid-weight 1.)\n(prism/smooth-limit-weight 1.)\n(prism/smooth-sliver-skew 0.98)\n(prism/smooth-sliver-jac-warp 0.98)\n(prism/smooth-aspect-ratio? #t)\n(prism/smooth-face-aspect-ratio? #t)\n(prism/smooth-brute-force? #t)\n(prism/smooth-cell-rings 5)\n(prism/smooth-improve-cell-rings 3)\n(prism/smooth-iterations 0)\n(prism/perturb-smooth? #t)\n(prism/perturb-smooth-iter 30)\n(prism/side-feature-angle 35.)\n(prism/side-feature-align-angle? 60.)\n(prism/side-topology-align-angle? 85.)\n(prism/post-volume-mesh? #f)\n(prism/post/merge-zones? #f)\n(prism/post/delete-redundant-parents? #t)\n(prism/force-match-node-positions? #f)\n(prism/proximity-skip-base-zones? #f)\n(prism/virtual-layers 0)\n(prism/skip-virtual-zone-proximity? #f)\n(prism/mesh-abort? #f)\n(prism/use-layer-height-proximity? #f)\n(prism/improve-base-boundary-face? #f)\n(prism/check-intersections1? #t)\n(prism/identify-intersections? #f)\n(prism/proximity-check-method 0)\n(prism/smooth-manifold-angles1? #t)\n(prism/offset-correct-smooth? #f)\n(prism/smooth-sides? #f)\n(prism/ratio-layer-compression? #f)\n(prism/stairstep-quality-method 11)\n(prism/stairstep-quality-threshold 0.98)\n(prism/stairstep-smooth? #t)\n(prism/stairstep-max-warp 0.48)\n(prism/stairstep-max-aspect-ratio 5)\n(prism/stairstep-gap-factor-scale 0.5)\n(prism/post-mesh-improve? #f)\n(prism/post-improve-option 2)\n(prism/post-improve-quality-method 11)\n(prism/post-improve-max-quality 0.99)\n(prism/post-improve-min-improvement 0.001)\n(prism/post-improve-iterations 5)\n(prism/post-split? #f)\n(prism/post-split-layers 2)\n(prism/post-residual-splits 0)\n(prism/post-split-change-growth-rate? #f)\n(prism/post-split-growth-rate 1.)\n(prism/post-ignore? #f)\n(prism/post-quality-ignore? #t)\n(prism/post-intersect-ignore? #t)\n(prism/post-warp-quality-ignore? #f)\n(prism/post-feature-edge-ignore? #f)\n(prism/post-feature-angle-ignore? #f)\n(prism/post-total-aspect-ratio-ignore? #f)\n(prism/post-ignore-smooth-boundary? #t)\n(prism/post-ignore-smooth-height? #f)\n(prism/post-ignore-quality-method 0)\n(prism/post-ignore-max-cell-skew 0.95)\n(prism/post-ignore-max-face-skew 0.9)\n(prism/post-ignore-max-warp-skew 0.7)\n(prism/post-ignore-feature-edges ())\n(prism/post-ignore-feature-angle 30.)\n(prism/post-ignore-aspect-ratio 5.)\n(prism/post-ignore-expand-rings 1)\n(prism/ignore-region-ext-angle 80)\n(prism/ignore-region-int-angle 60)\n(prism/smooth-manifold-angle-factor 0.5)\n(prism/proximity-extend-factor 2.)\n(prism/max-aspect-ratio-factor 1.)\n(prism-controls (("smooth-transition_1" "aspect-ratio" 3.676470588235294 1 4e-05 1.2 "fff-1" "fluid-regions" "*" "only-walls" "*" #t)))\n(scoped-prism/frozen-threads ())\n(thinvolume/allow-notarget-select? #f)\n(shell/bl-growth ((default constant 1. 1. 1 uniform #f)))\n(shell/bl-offset-method uniform)\n(shell/last-aspect-ratio 40.)\n(pyramid/neighbor-angle 115)\n(pyramid/offset-distance 1.)\n(pyramid/vertex-method 0)\n(pyramid/jiggle-height-factor 0.)\n(pyramid/check-proximity? #f)\n(pyramid/neighbor-node-tolerance 0.5)\n(remesh/write-debug-file? #f)\n(remesh/refine-boundary? #t)\n(remesh/refine-interior? #t)\n(remesh/insert-original? #t)\n(remesh/swap-and-smooth? #t)\n(remesh/must-improve-skew? #t)\n(remesh/smooth-min-skew 0.2)\n(remesh/flipping-angle 10.)\n(remesh/node-insertion-tolerance 0.01)\n(remesh/intersection-tolerance 0.1)\n(remesh/intersection-angle 11.)\n(remesh/smooth-threshold 0.9)\n(remesh/projection-tolerance 1e-05)\n(remesh/extract-voronoi-edge? #t)\n(remesh/extract-feature-edge? #t)\n(remesh/mesh-nonmanifold-zone? #f)\n(intersect/within-tolerance? #t)\n(intersect/delete-overlap? #t)\n(intersect/fill-concavities? #t)\n(intersect/refine-region? #f)\n(intersect/edge-remesh? #t)\n(intersect/single-loop? #t)\n(intersect/abs-tol? #f)\n(intersect/curve-reconstruction? #f)\n(intersect/separate? #f)\n(intersect/retri-improve? #t)\n(intersect/remesh? #t)\n(intersect/retri-shape-method 0)\n(intersect/retri-fix-orientation? #f)\n(intersect/stitch-type 0)\n(intersect/stitch-preserve? #t)\n(intersect/ignore-filtering? #f)\n(intersect/ignore-parallel-faces? #t)\n(intersect/parallel-face-angle 5.)\n(intersect/ignore-feature-overlap? #f)\n(intersect/remesh-angle 40.)\n(intersect/improve-swap-angle 10.)\n(intersect/improve-remesh-method 1)\n(intersect/patch-check? #t)\n(intersect/error-exit? #f)\n(intersect/fast-loop-creation? #t)\n(intersect/tol 0.05)\n(intersect/location-tol 0.001)\n(intersect/feature-location-tol 0.1)\n(intersect/refine-radius 1.)\n(intersect/sliver-size 1e-07)\n(intersect/retri-close-region 0.3)\n(intersect/retri-close-boundary 0.05)\n(intersect/retri-node-region 0.3)\n(intersect/retri-duplicate-nodes 0.01)\n(intersect/retri-sliver-angle 1.)\n(intersect/retri-swap-angle 30.)\n(intersect/feature-angle 40.)\n(intersect/ignore-feature-angle 178.)\n(intersect/shape-angle 10.)\n(intersect/edge-shape-angle 10.)\n(intersect/degenerate-edge-factor 0.2)\n(intersect/join-match-angle 45.)\n(intersect/join-project-angle 45.)\n(intersect/join-check-orientation? #f)\n(intersect/join-backup? #f)\n(intersect/join-backup/dihedral-angle 1)\n(intersect/edge-critical-angle 160)\n(join-marking/use-proximity-tuple? #f)\n(intersect/ignored-tuples ())\n(intersect/remesh-method 0)\n(intersect/stitch-method 0)\n(intersect/stitch-by-region? #f)\n(intersect/remesh-edge-angle 180)\n(hole/retri-improve #t)\n(hole/trace-feature-angle 70.)\n(hole/trace-align-angle 30.)\n(hole/trace-method 0)\n(hole/retri-sliver-angle 1.)\n(hole/retri-swap-angle 30.)\n(hole/retri-shape-angle 10.)\n(hole/retri-feature-angle 40.)\n(hole/loop-feature-angle 60.)\n(nonconf/remesh-interface 2)\n(nonconf/project-interface? #t)\n(nonconf/swap-side-connect-triangle? #t)\n(smooth/laplace/angle 15)\n(smooth/laplace/iterations 4)\n(smooth/laplace/relax 0.5)\n(smooth/laplace/converged 1e-06)\n(smooth/laplace/project? #t)\n(smooth/feature-nodes? #f)\n(smooth/freeze-type 1)\n(smooth/freeze-aspect-ratio 300)\n(boundary/delete-nodes-after-faces? #t)\n(surfer/improve-method 1)\n(surfer/smoothing-iterations 5)\n(surfer/smoothing-relaxation 0.9)\n(surfer/swapping-iterations 1)\n(surfer/swapping-skew 0.4)\n(surfer/swapping-angle 10.)\n(surfer/degree-skew 0.4)\n(surfer/degree-angle 180.)\n(surfer/max-area 1e+100)\n(surfer/degree-iterations 10)\n(surfer/shape-tolerance 0.2)\n(surfer/grading 1.2)\n(surfer/smooth-grading? #f)\n(surfer/trace-tri-on-surface? #t)\n(surfer/validity-method 1)\n(surfer/selection-interval 1.2)\n(surfer/max-faces 1000000)\n(surfer/feature-angle 30.)\n(surfer/lscale-method 1)\n(surfer/lean-and-mean 0)\n(surfer/surface-recovery 0)\n(surfer/sizing-function ())\n(surfer/size-fn-bgrids ())\n(surfer/fixed-sizing-functions ())\n(surfer/proximity-local-search? #t)\n(surfer/advanced-size-functions? #f)\n(surfer/implicit-edge-remesh? #f)\n(surfer/relative-tol 0.01)\n(surfer/parallel-tol-angle-deg 30.)\n(surfer/tol-angle-deg 60.)\n(surfer/check-multi? #f)\n(surfer/intersection-tol 0.)\n(surfer/right-angle-tri? #f)\n(mesh-origin (0 0 0))\n(edge-collapse/mid-point? #f)\n(edge-remesh/method 1)\n(edge-remesh/spacing (1 1))\n(edge/feature-angle 40.)\n(edge/max-feature-angle 180.)\n(edge/enclosed-angle 13.76)\n(edge/sizing-function ())\n(edge-remesh/quadratic-recon? #f)\n(edge-projection/method 0)\n(edge-projection/direction (1 0 0))\n(edge-intersection/tolerance -1.)\n(edge-intersection/delete-overlapped? #f)\n(edge-intersection/merge-overlapping-edge? #t)\n(edge-intersection/parallel-angle 5.)\n(edge-intersection/relative-size-tolerance -1.)\n(edge/smoothing-iterations 5)\n(edge/advanced-size-functions? #f)\n(edge-loops/filter-zero-length-edges? #t)\n(bgmesh/feature/method 1)\n(bgmesh/proximity-factor 0.25)\n(bgmesh/proximity-min-gap 0.)\n(bgmesh/proximity-ignore-self? #f)\n(bgmesh/proximity-exact-eval? #f)\n(bgmesh/curvature-factor 0.5)\n(bgmesh/curvature-thresh-angle 55.)\n(bgmesh/curvature-across-boundary? #f)\n(bgmesh/curvature-patch-expansion-attempts 3)\n(bgmesh/ignore-feature-edge? #f)\n(bgmesh/override-zone-specific? #f)\n(bgmesh/smooth-curvature? #f)\n(bgmesh/ignore-zero-area-faces? #f)\n(hexcore/poly-transition? #f)\n(hexcore/keep-outer-domain? #f)\n(hexcore/delete-old-face-zones? #f)\n(hexcore/outer-domain-size (0 0 0 0 0 0))\n(hexcore/define-hexcore-extents? #f)\n(hexcore/smooth-interface? #t)\n(hexcore/smooth-iteration 0)\n(hexcore/smooth-relax 0.05)\n(hexcore/delete-dead-zones? #t)\n(hexcore/keep-hex-tet-separate? #f)\n(hexcore/auto-align? #f)\n(hexcore/only-hex? #f)\n(hexcore/absolute-island-count 50)\n(hexcore/island-volume-fraction 1e-06)\n(hexcore/peel-layers 1)\n(hexcore/imprint-child-faces #t)\n(hexcore/upto-boundaries? #f)\n(hexcore/outer-box-ids ())\n(hexcore/refinement-regions ())\n(hexcore/refine-region? #f)\n(hexcore/create-cavity? #f)\n(hexcore/jiggle-face-nodes? #f)\n(hexcore/jiggle-factor 0.01)\n(hexcore/skip-tet-refinement? #f)\n(hexcore/init-tet-improve-skewness 0.9)\n(hexcore/init-tet-improve-angle 10.)\n(hexcore/init-tet-improve-iter 3)\n(hexcore/merge-tets-to-pyramids? #f)\n(hexcore/hexcore-by-size-field? #f)\n(hexcore/sf-driven-octree? #t)\n(hexcore/offset-boundary/smooth? #f)\n(hexcore/point-cloud-order 1)\n(hexcore/merge-nonmanifold? #t)\n(hexcore/relative-peeling-offset 0.75)\n(hexcore/octree-offset-intersect-option 1)\n(hexcore/octree-collect-neighbor-option 1)\n(rapid-octree/global-parameter/active-object-name "")\n(rapid-octree/global-parameter/flow-volume-spec (2 "Bound by Geometry" (0 0 0) keep-solid del-dead))\n(rapid-octree/global-parameter/auto-compute-object-regions #t)\n(rapid-octree/global-parameter/bnd-treatment 0)\n(rapid-octree/global-parameter/bbox ((0 0 0) (0 0 0)))\n(rapid-octree/global-parameter/def-angle (0 0))\n(rapid-octree/global-parameter/max-cell-lvl 3)\n(rapid-octree/global-parameter/def-bnd-lvl 5)\n(rapid-octree/global-parameter/buffer-layers 1)\n(rapid-octree/global-parameter/bnd-layers 1)\n(rapid-octree/refinement-regions ())\n(rapid-octree/refine-region? #f)\n(rapid-octree/check-geometry-before-mesh? #t)\n(tri-tet/refinement-regions ())\n(tri-tet/refine-region? #f)\n(octree/based-hexcore? #f)\n(octree/include-cells-from-all-regions? #f)\n(octree/refine-critical-count 3)\n(octree/refine-iterations 20)\n(octree/refine-smoothing-method 1)\n(octree/size-func-refine-factor 1.25)\n(octree/gauss-point-sf-evaluation? #f)\n(octree/remove-over-exposed-octants? #f)\n(octree/shrink-wrap-thin-cut-iterations 1)\n(octree/wrap-region-denoise? #f)\n(octree/max-giga-bytes 50.)\n(octree/region/merge/allow-intersected? #t)\n(octree/draw/parallel-infra? #f)\n(octree/draw/max-rlevel 20)\n(octree/fix-holes/top-down-regioning? #f)\n(top-down-regioning/remove-faulty-holes? #f)\n(octree/fix-holes/open-large-holes? #f)\n(cartesian/init-domain-factor 0.2)\n(cartesian/init-domain-y-factor 0.)\n(cartesian/init-domain-z-factor 0.)\n(cartesian/init-reference-size 8.75671743415296e-05)\n(cartesian/max-aspect-ratio 4.)\n(cartesian/min-gap-factor 0.1)\n(cartesian/variable-ticks #f)\n(cartesian/init-max-size 40)\n(cartesian/init-min-size 0)\n(cartesian/init-max-size-level 1)\n(cartesian/init-min-size-level 4)\n(cartesian/size-func/proximity? #f)\n(cartesian/size-func/curvature? #f)\n(cartesian/max-init-cells 10000000)\n(cartesian/max-refine-level 14)\n(cartesian/refine-thread-id 0)\n(cartesian/mark/buffer-nlayer 1)\n(cartesian/mark/cell-distance? #f)\n(cartesian/mark/size-method 2)\n(cartesian/mark-region/apply-buffer? #t)\n(cartesian/subdivide/max-subdivisions 50)\n(cartesian/project/smooth-face-nodes? #f)\n(cartesian/project/smooth-cell-nodes? #f)\n(cartesian/project/feature? #t)\n(cartesian/project/intersection? #t)\n(coarsen/num-iter 5)\n(coarsen/node-degree 8)\n(coarsen/node-lscale-method 3)\n(coarsen/preserve-boundary-mesh #f)\n(coarsen/max-edge-length 0)\n(local-refine-regions ())\n(group/activate-names ("global" "global"))\n(group/activate? #f)\n(domain/activate? #f)\n(group/face-zone-ids ())\n(group/edge-zone-ids ())\n(render/shell? #t)\n(render/pack-multiple-shells? #t)\n(wrapper/alternating-split #f)\n(wrapper/face-settled-dist 0.05)\n(wrapper/face-settled-dotp 0.95)\n(wrapper/face-far-dist 0.15)\n(wrapper/face-far-dotp 0.8)\n(wrapper/proj-forced-forward? #t)\n(wrapper/proj-with-smoothing? #t)\n(wrapper/proj-num-iter 5)\n(wrapper/proj-edge-crit 0.1)\n(wrapper/crit-stretch 0.1)\n(wrapper/feature/rel-imprint-threshold 2.)\n(wrapper/feature/abs-imprint-threshold 0.)\n(wrapper/feature/max-dihedral-angle 0.)\n(wrapper/feature/enrich? #t)\n(wrapper/feature/incremental-projection? #t)\n(wrapper/feature/fix-detour? #f)\n(wrapper/feature-line/create-thread? #f)\n(wrapper/feature-line/binary-divide? #f)\n(wrapper/feature-line-old-method 0)\n(wrapper/feature/dijkstra-extract-method? #t)\n(wrapper/feature/aggressive-imprint? #t)\n(wrapper/feature/recover-parallel-features? #t)\n(wrapper/feature/spring-stiffness-feature 2.)\n(wrapper/feature/spring-stiffness-neighbor 1.)\n(wrapper/feature/spring-length-feature 0.)\n(wrapper/feature/spring-length-neighbor 0.)\n(wrapper/feature/reuse 0)\n(wrapper/feature/projection-correction 2)\n(wrapper/feature/imprint-correction 15)\n(wrapper/feature/optimize-path? #t)\n(wrapper/feature/optimize-global-path? #f)\n(wrapper/feature/check-reachable? #t)\n(wrapper/feature/trace/duplicate-nodes? #f)\n(wrapper/feature/smooth-neighborhood-levels 3)\n(wrapper/feature/smooth-neighborhood-angle 15)\n(wrapper/feature-path/separate? #f)\n(wrapper/feature/imprint/segments 10)\n(wrapper/imprint/filter-non-high-geom? #f)\n(wrapper/imprint-iteration 3)\n(wrapper/imprint-increments 11)\n(wrapper/gap-tolerance -1.)\n(wrapper/angle-reversal 45.)\n(wrapper/angle-planar 135.)\n(wrapper/angle-corner 225.)\n(wrapper/edge-refine-crit 1.4)\n(wrapper/coarsen-ratio 0.002)\n(wrapper/max-propagation 1.5)\n(wrapper/zone-local-sizes ())\n(wrapper/draw-size-interval 20)\n(wrapper/post-wrap/preserve-feature 1)\n(wrapper/post-wrap/critical-skewness 1.)\n(wrapper/post-wrap/preserve-zoneboundary 0)\n(wrapper/post-wrap/maintain-manifold? #t)\n(wrapper/post-wrap/default-reproj-range 0.1)\n(wrapper/post-wrap/ssr-relative-threshold 0.25)\n(wrapper/post-wrap/ssr-ignore-orient 0)\n(wrapper/contact-pairs/only-cell-zone-boundaries? #t)\n(wrapper/relative-island-count 0.02)\n(wrapper/reachable-factor 0.1)\n(wrapper/projection-method 0)\n(wrapper/region/auto-hole-detection 1)\n(wrapper/region/hole-separation 0)\n(wrapper/region/post-hole-regioning 1)\n(wrapper/trace-through-holes? #t)\n(wrapper/refine-marked-region-only 0)\n(wrapper/wrap-method 0)\n(wrapper/extract-hole-octant-facets? #f)\n(wrapper/rezone/reprojection-iterations 1)\n(wrapper/rezone/island-upper-bound 10)\n(wrapper/preserve-zone-type? #f)\n(wrapper/preserve-zone-attributes? #f)\n(wrapper/zone/separate/priority-based? #f)\n(wrapper/zone/separate/rel-priority-dist 0.05)\n(wrapper/zone/separate/priority-angle 30.)\n(wrapper/quality/relative-deviation? #f)\n(wrapper/resolve-nonmanifold/maximize? #t)\n(copy-attributes? #t)\n(critical-base-height-ratio 50)\n(shrink-wrap/post-hybrid-wrap-improve? #f)\n(shrink-wrap/use-remeshing? #f)\n(shrink-wrap/sym-planes ())\n(shrink-wrap/min-topo-area 0.)\n(shrink-wrap/min-rel-topo-area 1e-05)\n(shrink-wrap/min-topo-fcount 3)\n(shrink-wrap/min-rel-topo-fcount 1e-05)\n(shrink-wrap/rel-feature-tol 3.)\n(shrink-wrap/feature-proj-correction 7)\n(shrink-wrap/resolution-factor 1.)\n(shrink-wrap/separate-zones? #t)\n(shrink-wrap/remesh? #t)\n(shrink-wrap/wrap-level-zones-list (() () (26500 26496 26495 18389 12916 55 54 52 51 47 39 35 27 24 19 16 15 14)))\n(shrink-wrap/debug-bbox-list ())\n(shrink-wrap/feature/filter-far-factor 0.5)\n(shrink-wrap/relaxation-factor 0.1)\n(shrink-wrap/debug-object-list ())\n(shrink-wrap/remesh-always? #t)\n(shrink-wrap/imprint-edges/beta? #f)\n(shrink-wrap/projection/wrap-v2? #f)\n(shrink-wrap/ignore-periodic-octree-refine? #t)\n(shrink-wrap/rezone/resolve-overlaps? #f)\n(shrink-wrap/delay-rezone? #t)\n(shrink-wrap/priority-separation? #t)\n(shrink-wrap/merge-small-zones? #f)\n(shrink-wrap/sf-based? #t)\n(wrap-v2/projection/iterations 3)\n(wrap-v2/projection/init-imprint? #f)\n(wrap-v2/projection/node-removal? #t)\n(wrap-v2/projection/pressure-increment 0.5)\n(wrap-v2/projection/inflate-baffles 0.)\n(wrap-v2/projection/function-mode 0)\n(wrap-v2/imprint/smooth? #f)\n(wrap-v2/imprint/filter-far-features? #f)\n(wrap-v2/contact-prevention/pairs ())\n(wrap-v2/sf-based? #t)\n(wrap-v2/boi-direct-refinement? #t)\n(wrap-v2/use-trace-features-in-object? #f)\n(wrap-v2/size-compute/composite? #f)\n(wrap-v2/size-controls/geodesic-on-edges? #f)\n(wrap-v2/size-controls/initial ())\n(wrap-v2/size-controls/target ())\n(resolve-intersect/attribute-based? #f)\n(resolve-self-intersections/fix-nonmanifold? #f)\n(resolve-intersect/max-node-degree 10)\n(resolve-intersect/smooth? #t)\n(resolve-self-intersections/local-remesh? #t)\n(resolve-intersect/fix-ribbon-like-intersection? #f)\n(resolve-intersect/ribbon/relative-island-size 5.)\n(resolve-intersect/ribbon/max-swing-angle 45.)\n(resolve-intersect/ribbon/clearance-angle 5.)\n(remove-spikes/critical-angle 220.)\n(remove-spikes/iterations 20)\n(model-string-parameter "example")\n(size-func/hash-size-factor 1.)\n(size-func/size-functions ())\n(size-func/refine-factor 2.)\n(size-func/prox-factor 1.)\n(size-func/prox-view-angle 1.)\n(size-func/prox-gap-tolerance 5e-06)\n(size-func/prox-crit-angle-glob 0.8660254040000001)\n(size-func/global-params (4e-05 4e-05 1.2 2.))\n(size-func/auto-recompute? #f)\n(size-func/nodal-curvature 1)\n(size-func/use-clip-algorithm? #f)\n(size-func/meshed-size-method 0)\n(size-func/triangulate-quad-faces? #f)\n(size-func/contour/refine-facets? #f)\n(size-func/contour/facet-refine-size 0)\n(size-func/critical-min-max-size-ratio 1e-05)\n(size-func/controls/hard-meshed-sf? #f)\n(edge-prox/quick-computation? #f)\n(size-func/max-proximity-detection-points 10000000)\n(size-controls (("import_curvature_0" "curvature" 4e-05 4e-05 1.2 18. 1. 12 "Object Faces and Edges" "*" #f #t) ("import_proximity_0" "proximity" 4e-05 4e-05 1.2 18. 1. 12 "Object Edges" "*" #f #t)))\n(scoped-sz/is-modified? #t)\n(powercrust/mesh-feature-edges 0)\n(powercrust/refine-only-good-powerballs 1)\n(powercrust/consider-forward-proximity 1)\n(powercrust/consider-backward-proximity 1)\n(powercrust/use-kdtree 1)\n(powercrust/ignore-normals 1)\n(powercrust/decimate-output 1)\n(powercrust/decimate-heap 1)\n(powercrust/fit-powerballs 1)\n(powercrust/discard-wing-tail 1)\n(gocart/size-fn-bgrids ())\n(gocart/delete-interior-faces? #t)\n(gocart/boundary/split-warp-threshold 0.1)\n(gocart/morph/method lyon)\n(gocart/morph/solver bcgstab)\n(gocart/morph/solver-residue 1e-08)\n(gocart/morph/surfaces? #t)\n(gocart/morph/surface/steps 2)\n(gocart/morph/reduce-support? #t)\n(gocart/morph/diffusion-law dbm_uniform)\n(gocart/morph/constrain-hanging-nodes? #t)\n(gocart/morph/repair-neg-vol? #t)\n(gocart/morph/quality-improve? #t)\n(gocart/morph/improve-threshold 0.9)\n(gocart/morph/improve-type 0)\n(gocart/mesh-geom-tolerance 0.2)\n(gocart/morph/rbf-type 6)\n(gocart/morph/cavity-size 3)\n(gocart/core/cutcell-meshing? #f)\n(gocart/core/crit-isolated-cell-count 10)\n(gocart/core/process-isolated-cells? #t)\n(gocart/obj-attrib-names (name cell-zone-type priority face-threads edge-threads object-type object-ref-point))\n(gocart/obj-attrib-assoc ((name "" object-name ()) (cell-zone-type (fluid solid dead) cell-zone-type ()) (priority 3 priority ()) (face-threads () object-face-zones (face boundary #f #f)) (edge-threads () object-edge-zones (edge #f #f #f)) (object-type (geom mesh) object-type ()) (object-ref-point #f object-ref-point ())))\n(objects ((fff-1 solid 12 (14 15 16 19 24 27 35 39 47 51 52 54 55 12916 18389) (25583 25581 25579 25577 25575 25573 25571 25569 25567 25565 25563 25561 25559 25557 25555 25553 25551 25549 25547 25545 25543 25541 25539 25537 25535 25533 25531 25529 25527 25525 25523 25521 25519 25517 25515 25513 25511 25509 25507 25505) mesh (("fluid1" fluid (-0.001517972354303868 0.0002154782575568108 0.002454863634448264) (12916 27 24 19 14) ((26494) ())) ("fluid2" fluid (-0.001471884509181453 0.0004768982764315247 0.002768956085183324) (18389 47 39 35 15) ((26490) ())) ("solid" fluid (-0.001603548493483346 -0.0002699277287743593 0.001871655215657299) (18389 12916 55 54 52 51 16) ((26489) ()))) ((fluid1 solid 12 (12916 27 24 19 14) () geom () () () body) (fluid2 solid 11 (18389 47 39 35 15) () geom () () () body) (solid solid 10 (18389 12916 55 54 52 51 16) () geom () () () body)) ((26494 26490 26489) ()) (()) ())))\n(gocart/delete-exterior-zone? #f)\n(gocart/delete-dead-zones? #t)\n(gocart/auto-delete-solid-zones? #f)\n(cutcell/face-group-sources fluid)\n(gocart/auto-hole-fixing? #f)\n(gocart/hdm-improve-threshold 0.99)\n(gocart/material-points ())\n(gocart/crit-mconnected-face-count 0)\n(gocart/crit-isolated-cell-count 5)\n(gocart/skewness-method 12)\n(gocart/grid-snapped? #f)\n(gocart/recover-contact? #f)\n(gocart/thin-cut-face-zones ())\n(gocart/thin-cut-edge-zones ())\n(gocart/contact-sizes ())\n(gocart/contact-rel-tolerance 0.)\n(gocart/prevent-contact-leakage #f)\n(gocart/recovery-method 0)\n(gocart/post-snap-anm-limit 0.85)\n(gocart/post-snap-rem-limit 0.9)\n(gocart/post-snap-anm-glob-iters 1)\n(gocart/post-snap-anm-node-iters 50)\n(gocart/post-snap-anm-restrict? #t)\n(gocart/post-morph-anm-limit 0.9)\n(gocart/post-morph-rem-limit 0.95)\n(gocart/post-morph-anm-glob-iters 1)\n(gocart/post-morph-anm-node-iters 50)\n(gocart/post-morph-anm-restrict? #t)\n(gocart/anm-feature-angle 120.)\n(gocart/fix-degenerate-quads? #t)\n(gocart/fix-baffle-holes? #t)\n(gocart/ignore-cavity-cell-threads? #t)\n(gocart/beta? #f)\n(scissor-mconnected-edges-threshold 5)\n(prism/step-min-aspect-ratio 5.)\n(asm/assembly-meshing? #f)\n(asm/wrap-collectively? #f)\n(asm/wrapping? #f)\n(asm/sewing? #f)\n(asm/two-stage-projection? #f)\n(asm/sew-local-thincut? #t)\n(asm/improve/imprint-features? #t)\n(object/edge-feature-angle 40.)\n(object/remove-gaps-and-thickness? #f)\n(object/show-faces? #t)\n(object/show-edges? #f)\n(object/remove-gaps/ignore-orientation? #t)\n(object/remove-gaps/imprint-features? #t)\n(object/inherit-object-as-label-on-merge? #f)\n(object/remove-gaps/post-improve? #t)\n(object/improve-quality/feature-angle 30.)\n(object/improve-quality/aggressively? #f)\n(object/cell-zone-boundary-labels ())\n(wrap/edge-extraction-option "geometry")\n(wrap/delete-far-edges? #t)\n(wrap/delete-far-edges-factor 1)\n(wrap/use-ray-tracing? #f)\n(wrap/use-smooth-folded-faces? #f)\n(wrap/include-thin-cut-edges-and-faces? #f)\n(wrap/rezone/activate-separation? #f)\n(wrap/rezone/thread-separation-angle 40)\n(wrap/zone-name-prefix "")\n(wrap/sort-regions? #t)\n(wrap/region-threshold-count 10)\n(wrap/auto-patch-holes? #f)\n(wrap/hole-detection-threshold-level 2)\n(wrap/no-go-mpts ())\n(wrap/auto-create-patches? #f)\n(wrap/auto-report-holes? #f)\n(wrap/max-free-edge-loop-length 10)\n(wrap/max-island-face-count 20)\n(wrap/scissor-mconnected-along-edges? #t)\n(wrap/label/inherit-geom-objects? #t)\n(wrap/compact-entities-frequency 5)\n(wrap/initial-bounding-box global)\n(sew/coarsen-factor 1.)\n(sew/improve-level 2)\n(sew/post-remove-intersections? #f)\n(sew/critical-angle 40.)\n(sew/include-thin-cut-edges-and-faces? #f)\n(sew/zone-name-prefix "")\n(sew/process-slit-as-baffle? #f)\n(sew/scissor-mconnected-along-edges? #t)\n(sew/slit-rel-thickness 0.1)\n(sew/slit-angle 10.)\n(sew/flood-fill-from-nodes? #t)\n(sew/max-void-cell-count 2000)\n(sew/void-length-ratio 0.25)\n(sew/remove-void-dead-only? #t)\n(sew/prox-gap-tolerance 0.5)\n(local-sew/faster-find-join-pairs? #t)\n(display/search-invisible-neighbourhood? #t)\n(display/search-invisible-for-group? #t)\n(display/create-closed-edge-loop? #f)\n(display/highlight-tree-selection? #t)\n(display/help-text ((title "") (description "")))\n(nodeloop/trace-along-free? #f)\n(nodeloop/trace-along-multi? #f)\n(nodeloop/trace-along-feature-angle? #f)\n(nodeloop/trace-along-zone-boundary? #f)\n(remesh/reassociate-topo-faces? #t)\n(remesh/dump-failures? #f)\n(remesh/edge-angle-deg 20.)\n(remesh/imprint-closest-node-factor 0.5)\n(remesh/max-path-length-factor 3.)\n(remesh/voronoi-seed-angle-deg 1.)\n(remesh/conformal-remesh-mode 0)\n(remesh/add-retri-postfix? #t)\n(remesh/conformal-robust-mode? #f)\n(remesh/preserve-free-edges? #f)\n(remesh/explicit-advanced-post-pro? #t)\n(remesh/debug-write-face -1)\n(remesh/retain-explicit-remesh-zone? #f)\n(display/ftype-in-list-print-format #f)\n(display/auto-apply-obj-opt #t)\n(display/use-cutplanes-for-bounds #f)\n(display/advanced-rendering-options? #f)\n(display/black-edges? #t)\n(display/simple-shadow? #f)\n(display-help? #t)\n(display/quick-moves-algorithm "on")\n(display/icons? #t)\n(beta-palette? #t)\n(graphics/facezone-separate-angle 40.)\n(graphics/edgezone-separate-angle 40.)\n(graphics/neighbourhood-tol 5e-05)\n(graphics/similar-area/curvature-tol 0.2)\n(graphics/interactive-surface-rows 5)\n(graphics/interactive-surface-columns 5)\n(contact-pair/smoothen? #t)\n(contact-pair/smooth-iterations 2)\n(flag-slit-faces-skip-already-marked? #f)\n(flag-slit-faces-break-loop-on-marking? #f)\n(flag-slit-faces-remove-unmarked-from-tree? #f)\n(boundary-recover-iter 5)\n(join-intersect-marking/bbox-padding-factor 0.05)\n(ait/rename-baffles? #f)\n(ait/overlap-tol 1e-12)\n(ait/parallel-face-angle 1e-05)\n(parallel/write-partition-ids? #f)\n(parallel/fast-neighborhood? #f)\n(parallel/smooth/scheme 2)\n(cavity/ignore-prisms? #f)\n(edge-proximity/quick-computation? #f)\n(all-cells? #f)\n(all-faces? #t)\n(all-nodes? #f)\n(all-edges? #t)\n(free? #f)\n(left-handed? #f)\n(multi? #f)\n(unmeshed? #f)\n(unused? #f)\n(marked? #f)\n(tagged? #f)\n(refine? #f)\n(boundary-cells? #f)\n(labels? #f)\n(normals? #f)\n(normal-scale 0.1)\n(face-quality ())\n(cell-quality ())\n(face-size ())\n(cell-size ())\n(draw-x-range ())\n(draw-y-range ())\n(draw-z-range ())\n(display-range-delta 0)\n(label-alignment "*")\n(label-font "sans serif")\n(label-scale 1.)\n(node-size 0.25)\n(node-symbol "(*)")\n(xy/scale/percent/y? #f)\n(wrapper/auto-draw-sizes? #f)\n(select-visible-entities? #f)\n(fileio/double-precision? #t)\n(merge-tolerance 9.780608950289082e-09)\n(periodic-transform ())\n(periodic-info ())\n(model-periodic-info ())\n(non-fluid-default-cell-type dead)\n(spy-level 0)\n(progress-reports 2)\n(diagnostics #t)\n(verbosity-level 1)\n(domains ((1 _hexcore_input_domain_ () (14 15 16 19 24 27 35 39 47 51 52 54 55 12916 18389) ())))\n(init-nodes ())\n(face-edge-association ())\n(user/thread-color ())\n(mode/thread-color ())\n(color-mode thread)\n(interrupt-check-interval 2)\n(license-check-interval 2)\n(parallel/partition-method "")\n(auto/merge-free-nodes? #t)\n(auto/merge-free-nodes-args (#t #t))\n(auto/delete-unused-nodes? #t)\n(auto/merge-cell-zones? #t)\n(auto/quad-tet-transition 1)\n(auto/create-prisms? #t)\n(auto/grow-prisms "scoped")\n(auto/fill-method 2)\n(auto/delete-dead-zones? #t)\n(auto/preserve-old-mesh? #f)\n(auto/prefix-cell-zones? #f)\n(auto/cell-zones-prefix "")\n(auto/topo-identify? #t)\n(auto/merge-cells-per-body1? #t)\n(boundary/merge-edge-nodes? #f)\n(check/quality-level 0)\n(tritet/improve-surface-mesh? #f)\n(tritet/improve-surface-mesh-args (() 10. 0.75))\n(tritet/improve-surface-mesh-args-sv (() 10. 0.75))\n(tritet/improve-surface-mesh-args-ss (() 10. 0.5))\n(tritet/refine-parameters default)\n(tritet/refine-method adv-front)\n(tritet/refine-boundary-sliver-skew 0.9999)\n(tritet/refine-boundary-sliver-skew-sv 0.9999)\n(tritet/refine-boundary-sliver-skew-ss 0.99)\n(tritet/refine-sliver-skew 0.9999)\n(tritet/refine-sliver-skew-sv 0.9999)\n(tritet/refine-sliver-skew-ss 0.99)\n(tritet/refine-target-skew 0.9)\n(tritet/refine-target-skew-sv 0.9)\n(tritet/refine-target-skew-ss 0.85)\n(tritet/refine-target-low-skew 0.85)\n(tritet/refine-target-low-skew-sv 0.85)\n(tritet/refine-target-low-skew-ss 0.8)\n(tritet/refine-target? #t)\n(tritet/report-unmeshed-nodes? #t)\n(tritet/report-unmeshed-faces? #t)\n(tritet/report-max-unmeshed 250)\n(tritet/defaults? #t)\n(tritet/front-multi-stage? #f)\n(tritet/front-refine-args ((0.4 10 #f #f) (0.4 10 0.85 10) (0.4 10 0.75 3)))\n(tritet/front-refine-args-sv ((0.4 10 #f #f) (0.4 10 0.85 10) (0.4 10 0.75 3)))\n(tritet/front-refine-args-ss ((0.25 10 #f #f) (0.25 10 0.8 10) (0.25 10 0.7 3)))\n(tritet/front-refine-args-2 ((0.4 10 "3-2") (((0.4 10 "3-2") (0.85 10 "opt") (#t 0.9 0.8)) ((0.3 10 "3-2") (0.8 10 "opt") (#t 0.9 0.8)) ((0.2 10 "3-2") (0.65 10 "opt") ())) (((0.3 10 "3-2") (0.75 10 "lap")) ((0.2 10 "3-2") (0.7 10 "lap")))))\n(tritet/improve-skew-args-1 (0.9 0.85 10))\n(tritet/improve-skew-args-1-sv (0.9 0.85 10))\n(tritet/improve-skew-args-1-ss (0.9 0.8 10))\n(tritet/improve-mesh-args ((0.2 5) (0.75 5) (0.3 3)))\n(tritet/improve-mesh-args-sv ((0.2 5) (0.75 5) (0.3 3)))\n(tritet/improve-mesh-args-ss ((0.15 5) (0.7 5) (0.2 3)))\n(tritet/sort-boundary-faces? #t)\n(tritet/sort-cells? #t)\n(tritet/refine-boundary-cells? #t)\n(tritet/refine-cells? #t)\n(tritet/refine-levels-2d 1)\n(tritet/refine-levels-3d 2)\n(tritet/refine-cells-2d-args ((0.25 #f 0.45 #f 0.45 0.75)))\n(tritet/refine-cells-3d-args ((0.75 #f 0.85 #f 0.85 0.85) (0.9 #t 0.9 #t 0.9 0.01) (0.7 #t 0.7 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.6 #t 0.6 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.5 #t 0.5 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.4 #t 0.4 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.3 #t 0.3 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.2 #t 0.2 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.1 #t 0.1 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.01 #t 0.01 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01)))\n(tritet/refine-cells-2d-args-sv ((0.25 #f 0.45 #f 0.45 0.75)))\n(tritet/refine-cells-3d-args-sv ((0.75 #f 0.85 #f 0.85 0.85) (0.9 #t 0.9 #t 0.9 0.01) (0.7 #t 0.7 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.6 #t 0.6 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.5 #t 0.5 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.4 #t 0.4 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.3 #t 0.3 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.2 #t 0.2 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.1 #t 0.1 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01) (0.01 #t 0.01 #t 0.75 0.5) (0.9 #t 0.9 #t 0.9 0.01)))\n(tritet/refine-cells-2d-args-ss ((0.2 #f 0.25 #f 0.25 0.75)))\n(tritet/refine-cells-3d-args-ss ((0.6 #f 0.7 #f 0.7 0.85) (0.75 #t 0.75 #t 0.95 0.01) (0.55 #t 0.55 #t 0.6 0.5) (0.85 #t 0.75 #t 0.75 0.01) (0.45 #t 0.45 #t 0.6 0.5) (0.85 #t 0.75 #t 0.75 0.01) (0.45 #t 0.45 #t 0.6 0.5) (0.75 #t 0.75 #t 0.75 0.01) (0.4 #t 0.4 #t 0.6 0.5) (0.75 #t 0.75 #t 0.75 0.01) (0.3 #t 0.3 #t 0.6 0.5) (0.75 #t 0.75 #t 0.75 0.01) (0.2 #t 0.2 #t 0.6 0.5) (0.75 #t 0.75 #t 0.75 0.01) (0.1 #t 0.1 #t 0.6 0.5) (0.75 #t 0.75 #t 0.75 0.01) (0.01 #t 0.01 #t 0.6 0.5) (0.75 #t 0.75 #t 0.75 0.01)))\n(tritet/smu-cells-2d-args ((0.45 15 0. 5)))\n(tritet/smu-cells-3d-args ((0.85 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5)))\n(tritet/smu-cells-2d-args-sv ((0.45 15 0. 5)))\n(tritet/smu-cells-3d-args-sv ((0.85 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5) (0.75 15 0.4 5) (0.9 15 0.4 5)))\n(tritet/smu-cells-2d-args-ss ((0.2 15 0. 5)))\n(tritet/smu-cells-3d-args-ss ((0.7 15 0.3 5) (0.75 15 0.3 5) (0.6 15 0.3 5) (0.75 15 0.3 5) (0.6 15 0.3 5) (0.75 15 0.3 5) (0.6 15 0.3 5) (0.75 15 0.3 5) (0.6 15 0.3 5) (0.75 15 0.3 5) (0.6 15 0.3 5) (0.75 15 0.3 5) (0.6 15 0.3 5) (0.75 15 0.3 5) (0.6 15 0.3 5) (0.75 15 0.3 5) (0.6 15 0.3 5) (0.75 15 0.3 5)))\n(tritet/node-closeness-2d-args ((0.3)))\n(tritet/node-closeness-3d-args ((0.3) (0.3) (0.3) (0.3) (0.3) (0.3) (0.3) (0.3) (0.3) (0.3) (0.3) (0.3) (0.3) (0.3) (0.3) (0.3) (0.3) (0.3)))\n(tritet/smooth-mesh? #t)\n(tritet/swap-faces? #t)\n(tritet/improve-mesh? #t)\n(tritet/remove-sliver? #f)\n(tritet/remove-sliver-args-1 (0.95 0.9 10))\n(tritet/remove-sliver-args-1-sv (0.95 0.9 10))\n(tritet/remove-sliver-args-1-ss (0.9 0.85 10))\n(poly/tet-improve? #f)\n(poly/remove-sliver-args-1 (0.95 0.9 10))\n(poly/remove-sliver-args-1-sv (0.95 0.9 10))\n(poly/remove-sliver-args-1-ss (0.9 0.85 10))\n(part-management/state ())\n(workflow/state (__map__ ("TaskObject1" __map__ ("Arguments" __map__ ("FileName" . "/projectnb2/spine-fe/Tim/TRI/TRI-350_125/Cylinder_6x12/HalfCylinder_33_files/dp0/FFF-1/DM/FFF-1.agdb") ("LengthUnit" . "m") ("NumParts" . 3.)) ("CommandName" . "ImportCadFaceting") ("Errors") ("InactiveTaskList") ("ObjectPath" . "") ("State" . "Up-to-date") ("TaskList") ("TaskType" . "Simple") ("Warnings") ("_name_" . "Import Geometry")) ("TaskObject10" __map__ ("Arguments" __map__) ("CommandName" . "CreatePrismControl") ("Errors") ("InactiveTaskList") ("ObjectPath" . "") ("State" . "Up-to-date") ("TaskList" "TaskObject12") ("TaskType" . "Compound") ("Warnings") ("_name_" . "Add Boundary Layers")) ("TaskObject11" __map__ ("Arguments" __map__ ("PrismPreferences" __map__ ("ShowPrismPreferences" . #f)) ("VolumeFill" . "hexcore") ("VolumeFillControls" __map__ ("BufferLayers" . 1) ("HexMaxCellLength" . 4e-05)) ("VolumeMeshPreferences" __map__ ("CheckSelfProximity" . "no") ("MergeBodyLabels" . "yes") ("ShowVolumeMeshPreferences" . #f))) ("CommandName" . "PrismAndMesh") ("Errors") ("InactiveTaskList") ("ObjectPath" . "") ("State" . "Up-to-date") ("TaskList") ("TaskType" . "Simple") ("Warnings") ("_name_" . "Generate the Volume Mesh")) ("TaskObject12" __map__ ("Arguments" __map__ ("BLControlName" . "smooth-transition_1") ("BLRegionList" "solid" "fluid2" "fluid1") ("BLZoneList" "buffer_bottom" "buffer_int" "buffer_mirror" "fluid2_mirror" "fluid2_top" "fluid2_bottom" "fluid2_interface" "fluid1_bottom" "fluid1_top" "fluid1_mirror" "fluid1_interface") ("CompleteBLRegionList" "solid" "fluid2" "fluid1") ("CompleteBLZoneList" "buffer_bottom" "buffer_int" "buffer_mirror" "fluid2_mirror" "fluid2_top" "fluid2_bottom" "fluid2_interface" "fluid1_bottom" "fluid1_top" "fluid1_mirror" "fluid1_interface") ("NumberOfLayers" . 1)) ("CommandName" . "CreatePrismControl") ("Errors") ("InactiveTaskList") ("ObjectPath" . "") ("State" . "Up-to-date") ("TaskList") ("TaskType" . "Compound Child") ("Warnings") ("_name_" . "smooth-transition_1")) ("TaskObject2" __map__ ("Arguments" __map__) ("CommandName" . "BodyOfInfluence") ("Errors") ("InactiveTaskList") ("ObjectPath" . "") ("State" . "Up-to-date") ("TaskList") ("TaskType" . "Compound") ("Warnings") ("_name_" . "Add Local Sizing")) ("TaskObject3" __map__ ("Arguments" __map__ ("CFDSurfaceMeshControls" __map__ ("MaxSize" . 4e-05) ("MinSize" . 4e-05)) ("ExecuteShareTopology" . "Yes") ("OriginalZones" "fluid1" "fluid2" "solid")) ("CommandName" . "ImportSurfaceMesh") ("Errors") ("InactiveTaskList") ("ObjectPath" . "") ("State" . "Up-to-date") ("TaskList") ("TaskType" . "Simple") ("Warnings") ("_name_" . "Generate the Surface Mesh")) ("TaskObject4" __map__ ("Arguments" __map__ ("CappingRequired" . "No") ("InvokeShareTopology" . "Yes") ("SetupType" . "The geometry consists of only fluid regions with no voids") ("WallToInternal" . "Yes")) ("CommandName" . "GeometrySetup") ("Errors") ("InactiveTaskList" "TaskObject6" "TaskObject8") ("ObjectPath" . "") ("State" . "Up-to-date") ("TaskList" "TaskObject5" "TaskObject7") ("TaskType" . "Conditional") ("Warnings") ("_name_" . "Describe Geometry")) ("TaskObject5" __map__ ("Arguments" __map__) ("CommandName" . "ShareTopology") ("Errors") ("InactiveTaskList") ("ObjectPath" . "") ("State" . "Up-to-date") ("TaskList") ("TaskType" . "Simple") ("Warnings" "The model only have one object. Share Topology might not be needed") ("_name_" . "Apply Share Topology")) ("TaskObject6" __map__ ("Arguments" __map__) ("CommandName" . "CreatePatch") ("Errors") ("InactiveTaskList") ("ObjectPath" . "") ("State" . "Out-of-date") ("TaskList") ("TaskType" . "Compound") ("Warnings") ("_name_" . "Enclose Fluid Regions (Capping)")) ("TaskObject7" __map__ ("Arguments" __map__ ("BoundaryZoneList" "buffer_top" "buffer_ext" "fluid1_interface") ("BoundaryZoneTypeList" "pressure-outlet" "pressure-outlet" "wall") ("OldBoundaryZoneList" "buffer_top" "buffer_ext" "fluid1_interface") ("OldBoundaryZoneTypeList" "wall" "wall" "wall") ("ZoneLocation" "3" "-0.001751271" "-0.0017513435" "0.0032908763" "0" "0.0017513435" "0.0032908763" "buffer_top" "-0.0017513372" "-0.0017513435" "0" "0" "0.0017513435" "0.0032908763" "buffer_ext" "-0.0015512112" "-0.0015513435" "-5.3319156e-11" "9.7300892e-11" "0.0015513435" "0.0032908765" "fluid1_interface")) ("CommandName" . "UpdateBoundary") ("Errors") ("InactiveTaskList") ("ObjectPath" . "") ("State" . "Up-to-date") ("TaskList") ("TaskType" . "Simple") ("Warnings") ("_name_" . "Update Boundaries")) ("TaskObject8" __map__ ("Arguments" __map__) ("CommandName" . "ComputeRegions") ("Errors") ("InactiveTaskList") ("ObjectPath" . "") ("State" . "Out-of-date") ("TaskList") ("TaskType" . "Simple") ("Warnings" "One or more fluid regions could not be detected. Verify the names and types of your regions using the Update Region task.") ("_name_" . "Create Regions")) ("TaskObject9" __map__ ("Arguments" __map__) ("CommandName" . "UpdateRegion") ("Errors") ("InactiveTaskList") ("ObjectPath" . "") ("State" . "Up-to-date") ("TaskList") ("TaskType" . "Simple") ("Warnings") ("_name_" . "Update Regions")) ("Workflow" __map__ ("CurrentTask") ("TaskList" "TaskObject1" "TaskObject2" "TaskObject3" "TaskObject4" "TaskObject9" "TaskObject10" "TaskObject11"))))\n(meshing/state (__map__ ("GlobalSettings" __map__ ("EnableCleanCAD" . #t) ("EnableComplexMeshing" . #f) ("EnableOversetMeshing" . #f) ("FTMRegionData" __map__ ("AllOversetNameList") ("AllOversetSizeList") ("AllOversetTypeList") ("AllOversetVolumeFillList") ("AllRegionFilterCategories") ("AllRegionLeakageSizeList") ("AllRegionLinkedConstructionSurfaceList") ("AllRegionMeshMethodList") ("AllRegionNameList") ("AllRegionOversetComponenList") ("AllRegionSizeList") ("AllRegionSourceList") ("AllRegionTypeList") ("AllRegionVolumeFillList")) ("InitialVersion" . "21.2") ("NormalMode" . #f))))\n(workflow/type "Watertight Geometry")\n(workflow/wb "")\n(poly-hexcore/only-poly-for-selected-volumes ())\n(poly-hexcore/avoid-hanging-nodes (#f "delete-large"))\n(poly-hexcore/keep-core-cells-as-hex? #t)\n(prism/overset-controls ())\n(object/state-when-regions-computed ((fff-1 (15 158047 0 5450 ((18389 8487) (12916 8448) (55 1578) (54 1577) (52 7040) (51 2014) (47 5980) (39 2204) (35 2186) (27 1640) (24 1618) (19 6005) (16 26338) (15 41491) (14 41441))))))\n(tet/region-based-controls ())\n(hexcore/region-based-controls ())\n(hexcore/avoid-hanging-nodes (#f "delete-small"))\n(utl/persistence-map ())))\n'
        thread = settings.create_dataset('Thread Variables',(1,),dtype=dt)    
        head = ['\n(0 "Zone variables:")']
        cellzones = ['(39 (' + str(int(cellId[i])) + ' fluid ' + str(int(cellId[i])) + ')(\n))' for i in range(len(cellId))]
        facezones = ['(39 (' + str(int(faceId[i])) + ' wall ' + str(int(faceId[i])) + ')(\n))' for i in range(len(faceId))]
        edgezones = ['(39 (' + str(int(edgeId[i])) + ' wall ' + str(int(edgeId[i])) + ')(\n))' for i in range(len(edgeId))]
        thread[0] = '\n'.join(head+cellzones+facezones+edgezones)
        
        version = settings.create_dataset('Version',(1,),dtype=dt)
        version[:] = '21.2'    
           
def writeFluentMshFile(Mesh, filename):
    
    if '.msh' not in filename:
        filename += '.msh'
    
    with open(filename, 'w') as f:
        ## Header
        f.write('(1 "Made by Tim")\n\n')
        
        ## Dimensions
        f.write('(2 ' + str(Mesh.nD) +')\n\n')
        
        ## Overall Model
        # Total number of nodes
        f.write('(10 (0 1 ' + format(Mesh.NNode,'x') + ' 0))\n')
        # Total number of edges
        f.write('(11 (0 1 ' + format(Mesh.NEdge,'x') + ' 0))\n')
        # Total number of cells
        f.write('(12 (0 1 ' + format(Mesh.NElem,'x') + ' 0))\n')
        # Total number of faces
        f.write('(13 (0 1 ' + format(Mesh.NFace,'x') + ' 0))\n')
        f.flush()
        ## Create Zone IDs
        # SurfNodes = np.unique(Mesh.SurfConn())
        if len(Mesh.NodeSets) == 0:

            # SurfNodes = set(np.unique([n for elem in Mesh.SurfConn() for n in elem]))
            # IntNodes = set(n for n in range(Mesh.NNode) if n not in SurfNodes)

            SurfNodes = set(np.unique(Mesh.SurfConn))
            IntNodes = set(range(Mesh.NNode)).difference(SurfNodes)

            if len(IntNodes) > 0:
                Mesh.NodeSets = {'boundary':SurfNodes,'interior':IntNodes}
            else:
                Mesh.NodeSets = {'boundary':SurfNodes}
        Mesh.RenumberNodesBySet()
        
        if len(Mesh.FaceSets) == 0:
            Mesh.FaceSets = {'boundary_bcID3':[],'interior_bcID2':[]}
            # tic = time.time()
            # for i in range(Mesh.NFace):
            #     if np.nan in Mesh.FaceElemConn[i]:
            #         Mesh.FaceSets['boundary_bcID3'].append(i)
            #     else:
            #         Mesh.FaceSets['interior_bcID2'].append(i)
            # print(time.time()-tic)
            boundarycheck = np.any(np.isnan(Mesh.FaceElemConn),axis=1)
            Mesh.FaceSets['boundary_bcID3'] = np.where(boundarycheck)[0]
            Mesh.FaceSets['interior_bcID2'] = np.where(~boundarycheck)[0]

        Mesh.RenumberFacesBySet()
        if len(Mesh.ElemSets) == 0:
            Mesh.ElemSets = {'All-fluid':range(Mesh.NElem)}
        
        nodeIDs = range(1,len(Mesh.NodeSets)+1)
        edgeIDs = range(len(Mesh.NodeSets)+1,len(Mesh.NodeSets)+len(Mesh.EdgeSets)+1)
        faceIDs = range(len(Mesh.NodeSets)+len(Mesh.EdgeSets)+1,len(Mesh.NodeSets)+len(Mesh.EdgeSets)+len(Mesh.FaceSets)+1)
        cellIDs = range(len(Mesh.NodeSets)+len(Mesh.EdgeSets)+len(Mesh.FaceSets)+1,len(Mesh.NodeSets)+len(Mesh.EdgeSets)+len(Mesh.FaceSets)+len(Mesh.ElemSets)+1)
        
        
        ## Nodes
        setkeys = list(Mesh.NodeSets.keys())
        for i in range(len(Mesh.NodeSets)):            
            if 'boundary' in setkeys[i]:
                nodetype = 2
            else:
                nodetype = 1
            
            f.write(' '.join(['(10 (' + format(nodeIDs[i],'x'), format(Mesh.NodeSets[setkeys[i]][0]+1,'x'), format(Mesh.NodeSets[setkeys[i]][-1]+1,'x'), str(nodetype), str(Mesh.nD)]) + ') (\n')
            
            for j in Mesh.NodeSets[setkeys[i]]:
                try:
                    f.write(' '.join([str(x) for x in Mesh.NodeCoords[j]]))
                except:
                    merp = 2
                f.write('\n')
            f.write('))\n')
        f.flush() 
        ## Edges
        setkeys = list(Mesh.EdgeSets.keys())
        for i in range(len(Mesh.EdgeSets)):            
            if 'boundary' in setkeys[i]:
                edgetype = 5
            else:
                edgetype = 6
            f.write(' '.join(['(11 (' + format(edgeIDs[i],'x'), format(Mesh.EdgeSets[setkeys[i]][0]+1,'x'), format(Mesh.EdgeSets[setkeys[i]][-1]+1,'x'), str(edgetype)]) + ' 0) (\n')
            for j in Mesh.EdgeSets[setkeys[i]]:
                f.write(' '.join([format(x,'x') for x in Mesh.Edges()[j]]) + ' ')
                f.write('\n')
            f.write('))\n')
        f.flush()
        ## Faces
        setkeys = list(Mesh.FaceSets.keys())
        start = 1
        for i in range(len(Mesh.FaceSets)):            
            bctype = setkeys[i].split('bcID')[-1]
            if all([len(Mesh.Faces[j]) == 2 for j in Mesh.FaceSets[setkeys[i]]]):
                facetype = '2'
            elif all([len(Mesh.Faces[j]) == 3 for j in Mesh.FaceSets[setkeys[i]]]):
                facetype = '3'
            elif all([len(Mesh.Faces[j]) == 4 for j in Mesh.FaceSets[setkeys[i]]]):
                facetype = '4'
            elif all([len(Mesh.Faces[j]) >= 5 for j in Mesh.FaceSets[setkeys[i]]]):
                facetype = '5'
            else:
                facetype = '0'
            end = start + len(Mesh.FaceSets[setkeys[i]])-1
            # f.write(' '.join(['(13 (' + format(faceIDs[i],'x'), format(Mesh.FaceSets[setkeys[i]][0]+1,'x'), format(Mesh.FaceSets[setkeys[i]][-1]+1,'x'), bctype, facetype]) + ') (\n')
            f.write(' '.join(['(13 (' + format(faceIDs[i],'x'), format(start,'x'), format(end,'x'), bctype, facetype]) + ') (\n')
            start = end
            for j in Mesh.FaceSets[setkeys[i]]:
                if facetype == '0':
                    f.write(' '.join([str(len(Mesh.Faces[j]))] + [format(x+1,'x') for x in Mesh.Faces[j]] + [format(int(x)+1,'x') if not np.isnan(x) else format(0,'x') for x in Mesh.FaceElemConn[j]]) + ' ')
                else:
                    f.write(' '.join([format(x+1,'x') for x in Mesh.Faces[j]] + [format(int(x)+1,'x') if not np.isnan(x) else format(0,'x') for x in Mesh.FaceElemConn[j]]) + ' ')
                f.write('\n')
            f.write('))\n')
        f.flush()
        ## Cells
        setkeys = list(Mesh.ElemSets.keys())
        
        for i in range(len(Mesh.ElemSets)):   
            if 'solid' in setkeys[i]:
                zonetype = '17'
            else:
                zonetype = '1'
            
            setNodeConn = [Mesh.NodeConn[elem] for elem in Mesh.ElemSets[setkeys[i]]]
            setFaceConn = [Mesh.FaceConn[elem] for elem in Mesh.ElemSets[setkeys[i]]]
            elemtypes = [0 for j in range(len(setNodeConn))]
            for j in range(len(setNodeConn)):
                if len(setNodeConn[j]) == 3 and len(setFaceConn[j]) == 3:
                    # Triangular
                    elemtypes[j] = 1
                elif len(setNodeConn[j]) == 4 and len(setFaceConn[j]) == 4 and Mesh.nD == 3:
                    # Tetrahedral
                    elemtypes[j] = 2
                elif len(setNodeConn[j]) == 4 and len(setFaceConn[j]) == 4 and Mesh.nD == 2:
                    # Quadrilateral
                    elemtypes[j] = 3
                elif len(setNodeConn[j]) == 8 and len(setFaceConn[j]) == 6:
                    # Hexahedral
                    elemtypes[j] = 4
                elif len(setNodeConn[j]) == 5 and len(setFaceConn[j]) == 5:
                    # Pyramid
                    elemtypes[j] = 5
                elif len(setNodeConn[j]) == 6 and len(setFaceConn[j]) == 5:
                    # Wedge
                    elemtypes[j] = 6
                else:
                    # Polyhedral
                    elemtypes[j] = 7
               
            if len(np.unique(elemtypes)) == 1:
                elemtype = str(elemtypes[0])
                f.write(' '.join(['(12 (' + format(cellIDs[i],'x'), format(Mesh.ElemSets[setkeys[i]][0]+1,'x'), format(Mesh.ElemSets[setkeys[i]][-1]+1,'x'), zonetype, elemtype]) + ')\n')
            else:
                f.write(' '.join(['(12 (' + format(cellIDs[i],'x'), format(Mesh.ElemSets[setkeys[i]][0]+1,'x'), format(Mesh.ElemSets[setkeys[i]][-1]+1,'x'), zonetype, '0']) + ') (\n')
                f.write(' '.join([str(elemtypes[j]) for j in range(len(elemtypes))]))
            f.write(' ))\n')
        f.flush()    
        ## Zone Variables
        f.write('(0 "Zone variables:")\n')
        
        # Cell Zones
        setkeys = list(Mesh.ElemSets.keys())
        for i in range(len(Mesh.ElemSets)):
            if 'solid' in setkeys[i]:
                zonetype = 'solid'
            else:
                zonetype = 'fluid'
            f.write(' '.join(['(39 (' +  str(cellIDs[i]), zonetype, setkeys[i]]) + ')(\n))\n')
        f.flush()   
        # Face Zones
        setkeys = list(Mesh.FaceSets.keys())
        for i in range(len(Mesh.FaceSets)):
            bcid = int(setkeys[i].split('bcID')[-1])
            if bcid == 2:
                zonetype = 'interior'
            elif bcid == 3:
                zonetype = 'wall'
            elif bcid == 4:
                zonetype = 'pressure-inlet'
            elif bcid == 5:
                zonetype = 'pressure-outlet'
            elif bcid == 7:
                zonetype = 'symmetry'
            elif bcid == 8:
                zonetype = 'periodic-shadow'
            elif bcid == 9:
                zonetype = 'pressure-far-field'
            elif bcid == 10:
                zonetype = 'velocity-inlet'
            elif bcid == 12:
                zonetype = 'periodic'
            elif bcid == 14:
                zonetype = 'fan'
            elif bcid == 20:
                zonetype = 'mass-flow-inlet'
            elif bcid == 24:
                zonetype = 'interface'
            elif bcid == 31:
                zonetype = 'parent-face'
            elif bcid == 36:
                zonetype = 'outflow'
            elif bcid == 37:
                zonetype = 'axis'
            else:
                zonetype = 'wall'
            f.write(' '.join(['(39 (' + str(faceIDs[i]), zonetype, setkeys[i]]) + ')(\n))\n')
        f.flush()
        # Edge Zones
        setkeys = list(Mesh.EdgeSets.keys())
        for i in range(len(Mesh.EdgeSets)):            
            if 'boundary' in setkeys[i]:
                zonetype = 'boundary-edge'
            else:
                zonetype = 'interior-edge'
            f.write(' '.join(['(39 (' + str(edgeIDs[i]), zonetype, setkeys[i]]) + ')(\n))\n')
        
        
    
   
    
        
        
    
    
    
    
    
    