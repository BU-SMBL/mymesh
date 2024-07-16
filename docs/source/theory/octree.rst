Octree
======

Octree Definition
-----------------
Octrees are a hierarchical tree structure used for spatial partitioning of
three-dimensional space - they divide space into a system of smaller and 
smaller cubes. The octree starts with a single cube (the "root" of the tree) that 
bounds a region of interest, for example a point cloud or a mesh. This cube is 
then equally divided into eight smaller cubes ("branches" on the tree), and this 
process is continued until a termination criteria is reached. Each "node" in the
tree is linked to its parent and children, enabling the tree to be traversed.  

This partitioning of space allows for efficient searching of the data encoded
by the octree. For example, an octree representation of a surface mesh can be
used when performing :ref:`Ray-Triangle Intersection` tests; rather than testing
the ray against every triangle in the mesh, the ray can be tested for intersection
with an octree node (using a :ref:`Ray-Box Intersection` test) - if the ray
doesn't intersect with the cube, surely it can't intersect any of the triangles
contained within the cube, so a single ray-box intersection test can eliminate
the need for thousands of ray-triangle intersection tests. 

.. graphviz::
    
    graph octree {
    node [shape=point, fontname="source code pro"];
    edge [style=solid];

    a [pos="0.4,0.5!", color=royalblue3, fillcolor=black]; 
    b [pos="1.4,0.5!", color=royalblue3, fillcolor=black];
    c [pos="1.8,0.9!", color=royalblue3, fillcolor=black]; 
    d [pos="0.8,0.9!", color=royalblue3, fillcolor=black]; 

    e [pos="0.4,1.5!", color=royalblue3, fillcolor=black];
    f [pos="1.4,1.5!", color=royalblue3, fillcolor=black]; 
    g [pos="1.8,1.9!", color=royalblue3, fillcolor=black]; 
    h [pos="0.8,1.9!", color=royalblue3, fillcolor=black];

    e0 [pos="0.9, 0.5!", color=royalblue3, fillcolor=white];
    e1 [pos="1.6, 0.7!", color=royalblue3, fillcolor=white];
    e2 [pos="1.3, 0.9!", color=royalblue3, fillcolor=white];
    e3 [pos="0.6, 0.7!", color=royalblue3, fillcolor=white];
    
    e4 [pos="0.4, 1.0!", color=royalblue3, fillcolor=white];
    e5 [pos="1.4, 1.0!" color=royalblue3, fillcolor=white];
    e6 [pos="1.8, 1.4!", color=royalblue3, fillcolor=white];
    e7 [pos="0.8, 1.4!", color=royalblue3, fillcolor=white];

    e8 [pos="0.9, 1.5!", color=royalblue3, fillcolor=white];
    e9 [pos="1.6, 1.7!", color=royalblue3, fillcolor=white];
    e10 [pos="1.3, 1.9!", color=royalblue3, fillcolor=white];
    e11 [pos="0.6, 1.7!", color=royalblue3, fillcolor=white];

    f0 [pos="1.1, 0.7!", color=royalblue3, fillcolor=white];
    f1 [pos="0.9, 1.0!", color=royalblue3, fillcolor=white];
    f2 [pos="1.6, 1.2!", color=royalblue3, fillcolor=white];
    f3 [pos="1.3, 1.4!", color=royalblue3, fillcolor=white];
    f4 [pos="0.6, 1.2!", color=royalblue3, fillcolor=white];
    f5 [pos="1.1, 1.7!", color=royalblue3, fillcolor=white];

    c0 [pos="1.1, 1.2!", color=royalblue3, fillcolor=white];

    a -- b;
    b -- c;
    c -- d;
    d -- a;
    e -- f;
    f -- g;
    g -- h;
    h -- e;
    a -- e;
    b -- f;
    c -- g;
    d -- h;

    e0 -- e2 [style=dotted, color=royalblue3];
    e1 -- e3 [style=dotted, color=royalblue3];
    e8 -- e10 [style=dotted, color=royalblue3];
    e9 -- e11 [style=dotted, color=royalblue3];

    e0 -- e8 [style=dotted, color=royalblue3];
    e1 -- e9 [style=dotted, color=royalblue3];
    e2 -- e10 [style=dotted, color=royalblue3];
    e3 -- e11 [style=dotted, color=royalblue3];

    e4 -- e5 [style=dotted, color=royalblue3];
    e5 -- e6 [style=dotted, color=royalblue3];
    e6 -- e7 [style=dotted, color=royalblue3];
    e7 -- e4 [style=dotted, color=royalblue3];

    f0 -- f5 [style=dotted, color=royalblue3];
    f1 -- f3 [style=dotted, color=royalblue3];
    f2 -- f4 [style=dotted, color=royalblue3];


    labelv0 [label="", pos="0.25,0.5!", shape=none, fontname="Times-Roman"] 
    labelv1 [label="", pos="1.55,0.45!", shape=none, fontname="Times-Roman"]
    labelv2 [label="", pos="1.95,0.9!", shape=none, fontname="Times-Roman"] 
    labelv3 [label="", pos="0.65,1.0!", shape=none, fontname="Times-Roman"] 

    labelv4 [label="", pos="0.25,1.5!", shape=none, fontname="Times-Roman"] 
    labelv5 [label="", pos="1.55,1.45!", shape=none, fontname="Times-Roman"] 
    labelv6 [label="", pos="1.95,1.9!", shape=none, fontname="Times-Roman"] 
    labelv7 [label="", pos="0.65,2.0!", shape=none, fontname="Times-Roman"] 

    }

Generating Octrees
------------------
Octrees can be used to represent various types of data, including point clouds,
voxel meshes, surface meshes, and implicit functions.

Point Cloud Octrees
^^^^^^^^^^^^^^^^^^^
See :func:`~mymesh.octree.Points2Octree`

Perhaps the most straight forward data to represent with an octree is a set of 
points. First, the root node is created to bound the full set of points. It is 
then subdivided into 8 children and the points are checked against the bounds
of each child to determine which points are contained by which node of the 
octree. This process is repeated, with points being passed from parents to the
children containing the points until each octree node contains only one point 
(or until a specified maximum depth is reached). Those nodes are marked as 
"leaf" nodes, while nodes that contain no points are marked as "empty" and no 
longer continue to be subdivided. 

Voxel Octrees
^^^^^^^^^^^^^
See :func:`~mymesh.octree.Voxel2Octree`

Since octrees are based on cubes, it's natural to associate them with voxel 
meshes - voxel meshes can be used to create octrees and vice versa. The creation
of an octree from a voxel mesh is essentially the same as creating an octree
from a point cloud, but the points are the centroids of the voxels. When care
is taken to properly specify the size of the root node, the octree can be 
subdivided until leaf nodes exactly correspond to voxel elements.

Surface Mesh Octrees
^^^^^^^^^^^^^^^^^^^^
See :func:`~mymesh.octree.Surface2Octree`

Implicit Function Octrees
^^^^^^^^^^^^^^^^^^^^^^^^^
See :func:`~mymesh.octree.Function2Octree`

An octree can be constructed to efficiently sample an implicit function,
allowing for higher resolution in complex regions of the surface and lower
resolution in regions with larger features. Several strategies exist for 
creating such an octree, most of which rely on using both the function and 
its gradient to assess the error associated with representing the function
at the current octree level to determine whether or not to subdivide further.

Two popular choices are the Euclidean Distance Error (EDError) :cite:p:`Zhang2003` 
and Quadratic Error Functions (QEF) :cite:p:`Schaefer2005`. 

Euclidean Distance Error
""""""""""""""""""""""""
The Euclidean distance error function measures the error between linear 
interpolation of the vertices of the current octree node and exact 
evaluation of the function at the vertices of the next level of refinement. 



