# Mesh (better name TBD)


## Package Overview
- `mesh`: Defines the `mesh` class for holding, manipulating, and analyzing meshes
  - The `mesh` class is defined primarily by the attributes `NodeCoords` and `NodeConn` which hold the coordinates and connectivity of the mesh nodes.
  - A variety of properties enable on-demand computation of a variety of mesh related information, which are then cached to prevent recalculation
    - Properties include: `ElemNormals`, `NodeNormals`, `NodeNeighbors`, `Faces`,  `Edges`, `SurfConn`
  - The `NodeData` and  `ElemData` dictionaries can be used to store any (scalar or vector) data related to nodes or elements
  - `read` and `write` methods utilize the [Meshio](https://github.com/nschloe/meshio) package as an easy importing/exporting interface to a variety of mesh file types. If supported by the target file, `write` will include `NodeData` and `ElemData`.
  - `mesh` objects can be unpacked to obtain the underlying `NodeCoords` and `NodeConn` attributes for ease of use with function that take `NodeCoords` and `NodeConn` as inputs
    - For example, the following lines are three equivalent ways of obtaining the face normals of a surface mesh stored in a `mesh` object (`M`):
      ``` python
      ElemNormals = M.ElemNormals
      ElemNormals = utils.CalcFaceNormals(M.NodeCoords,M.NodeConn)
      ElemNormals = utils.CalcFaceNormals(*M)
      ```
  

## Object Oriented Example Useage

``` python
import sys
sys.path.append('<path to parent directory of Mesh>')
import Mesh

# Create a 1x1x1 cube surface mesh with an element size of 0.1:
bounds = [0,1,0,1,0,1]; h = 0.1
M = Mesh.primitives.Box(bounds,h,meshobj=True)

# Print mesh info:
print(M)

# Get node normals - Doing this will calculate and cache the node normal vectors 
NodeNormals = M.NodeNormals

# Write a .vtu file with Node Normals as a Point Array
M.NodeData['Node Normals'] = NodeNormals
M.write('DemoCube.vtu')

```

## Non-Object Oriented Example Useage

``` python
import sys
sys.path.append('<path to parent directory of Mesh>')
import Mesh
import meshio

# Create a 1x1x1 cube surface mesh with an element size of 0.1:
bounds = [0,1,0,1,0,1]; h = 0.1
NodeCoords, NodeConn = Mesh.primitives.Box(bounds,h,meshobj=False)

# Get node normals - Doing this will calculate and cache the node normal vectors 
ElemNormals = Mesh.utils.CalcFaceNormal(NodeCoords,NodeConn)
NodeNeighbors,ElemConn = Mesh.utils.getNodeNeighbors(NodeCoords,NodeConn)
NodeNormals = Mesh.utils.Face2NodeNormal(NodeCoords,NodeConn,ElemConn,ElemNormals)

# Write a .vtu file with Node Normals as a Point Array
m = meshio.Mesh(NodeCoords,[('triangle',NodeConn)], point_data = {'Node Normals':NodeNormals})
m.write('DemoCube.vtu')

```
