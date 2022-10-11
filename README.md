# Mesh (better name TBD)
 
## Object Oriented Example Useage

```
import sys
sys.path.append('<path to parent directory of Mesh>')
import Mesh

# Create a 1x1x1 cube surface mesh with an element size of 0.1:
bounds = [0,1,0,1,0,1]; h = 0.1
M = Mesh.Primitives.Box(bounds,h,meshobj=True)

# Print mesh info:
print(M)

# Get node normals - Doing this will calculate and cache the node normal vectors 
NodeNormals = M.NodeNormals

# Write a .vtu file with Node Normals as a Point Array
M.NodeData['Node Normals'] = NodeNormals
M.write('DemoCube.vtu')

```

## Non-Object Oriented Example Useage

```
import sys
sys.path.append('<path to parent directory of Mesh>')
import Mesh
import meshio

# Create a 1x1x1 cube surface mesh with an element size of 0.1:
bounds = [0,1,0,1,0,1]; h = 0.1
NodeCoords, NodeConn = Mesh.Primitives.Box(bounds,h,meshobj=False)

# Get node normals - Doing this will calculate and cache the node normal vectors 
ElemNormals = Mesh.MeshUtils.CalcFaceNormal(NodeCoords,NodeConn)
NodeNeighbors,ElemConn = Mesh.MeshUtils.getNodeNeighbors(NodeCoords,NodeConn)
NodeNormals = Mesh.MeshUtils.Face2NodeNormal(NodeCoords,NodeConn,ElemConn,ElemNormals)

# Write a .vtu file with Node Normals as a Point Array
m = meshio.Mesh(NodeCoords,[('triangle',NodeConn)], point_data = {'Node Normals':NodeNormals})
m.write('DemoCube.vtu')

```
