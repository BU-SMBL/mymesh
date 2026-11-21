.. _connectivity:

Connectivity Representations
============================

.. currentmodule:: mymesh

While meshes are primarily defined as a collection of elements, with each element being defined
by referencing the nodes that comprise it, a variety of data structures can be constructed to
represent connections between different mesh entities, such as nodes, elements, edges, and faces 
which may be useful for performing different mesh operations or analysis. Each connectivity type 
is represented as a 2D data structure, with each row corresponding one entity and containing 
indices associated with another (e.g. each row of ``NodeConn`` corresponds to an element and 
contains indices associated with the list of nodes). Most of these data structures are 
non-rectangular (or "ragged") as the number of connections per entity may vary. The 
connectivities listed in the following table can all be obtained from the the node connectivity
list (``NodeConn``) using either the :class:`mesh` class (see :doc:`mesh_class`) or the 
:mod:`~mymesh.converter` module. 

+------------------------------------------------------------+--------------------------+-------------------------+
| Connectivity Type                                          | Row Correspondence       | Index Association       |
+============================================================+==========================+=========================+
| Node Connectivity (:attr:`~mymesh.mesh.mesh.NodeConn`)     | Element                  | Node                    |
+------------------------------------------------------------+--------------------------+-------------------------+
| Node Neighbors (:attr:`~mymesh.mesh.mesh.NodeNeighbors`)   | Node                     | Node                    |
+------------------------------------------------------------+--------------------------+-------------------------+
| Element Connectivity (:attr:`~mymesh.mesh.mesh.ElemConn`)  | Node                     | Element                 |
+------------------------------------------------------------+--------------------------+-------------------------+
| Element Neighbors (:attr:`~mymesh.mesh.mesh.ElemNeighbors`)| Element                  | Element                 |
+------------------------------------------------------------+--------------------------+-------------------------+
| Edges (:attr:`~mymesh.mesh.mesh.Edges`) [1]_               | Edge                     | Node                    |
+------------------------------------------------------------+--------------------------+-------------------------+
| Edge Connectivity ( :attr:`~mymesh.mesh.mesh.EdgeConn` )   | Element                  | Edge                    |
+------------------------------------------------------------+--------------------------+-------------------------+
| Edge-Element (:attr:`~mymesh.mesh.mesh.EdgeElemConn`)      | Edge                     | Element                 |
+------------------------------------------------------------+--------------------------+-------------------------+
| Faces (:attr:`~mymesh.mesh.mesh.Faces`) [1]_               | Face                     | Node                    |
+------------------------------------------------------------+--------------------------+-------------------------+
| Face Connectivity (:attr:`~mymesh.mesh.mesh.FaceConn`)     | Element                  | Face                    |
+------------------------------------------------------------+--------------------------+-------------------------+
| Face-Element (:attr:`~mymesh.mesh.mesh.FaceElemConn`)      | Face                     | Element                 |
+------------------------------------------------------------+--------------------------+-------------------------+

.. [1] 
    Edges and faces can be represented as "half-edges" and "half-faces" where at 
    shared edges/faces, there multiple two half-edges/faces with each one being 
    associated with only one element and  having a specific orientation. When 
    obtained through the :obj:`mesh` class (:attr:`~mymesh.mesh.mesh.Edges`,  
    :attr:`~mymesh.mesh.mesh.Faces`), they are the "full" rather than the "half"
    representation. Half-edges and half-faces can be obtained through 
    :func:`converter.solid2edges` and :func:`converter.solid2faces` and converted
    to full-edges and full-faces using :func:`converter.edges2unique` and 
    :func:`converter.faces2unique`.


2D Example
----------
The below example shows several connectivity representations of the same mesh consisting, 
of three triangular elements. In each case, the entity corresponding to the first row
is indicated in blue and the associated connections are indicated in red.

::

  NodeCoords = [[0,0,0], [1,.1,0], [.9,.9,0], [-.1,1,0], [-.5,.5,0]]
  NodeConn = [[0, 1, 3], [1, 2, 3], [0, 3, 4]]


.. grid:: 2
    :outline:

    .. grid-item::
      .. graphviz::

        graph tris {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!", color="cornflowerblue"];
        1 [pos="1,0.1!", color="firebrick"]; 
        2 [pos="0.9,0.9!"]; 
        3 [pos="-0.1,1.0!", color="firebrick"]; 
        4 [pos="-.5,.2!", color="firebrick"]

        0 -- 1;
        1 -- 2; 
        1 -- 3;        
        2 -- 3; 
        3 -- 0; 
        3 -- 4;
        4 -- 0;

        node0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro", fontcolor="cornflowerblue"] 
        node1 [label="1", pos="1,-0.05!", shape=none, fontname="source code pro", fontcolor="firebrick"] 
        node2 [label="2", pos="1.0,1.0!", shape=none, fontname="source code pro"] 
        node3 [label="3", pos="-0.1,1.1!", shape=none, fontname="source code pro", fontcolor="firebrick"] 
        node4 [label="4", pos="-0.6,.2!", shape=none, fontname="source code pro", fontcolor="firebrick"] 

        // elem0 [label="0", pos=".3,.3!", shape=none, fontname="source code pro"] 
        // elem1 [label="1", pos=".6,.7!", shape=none, fontname="source code pro"]
        // elem2 [label="2", pos="-.2,.5!", shape=none, fontname="source code pro"]

        }
      
    .. grid-item::

      ::

        NodeNeighbors = [[3, 4, 1], 
                         [0, 2, 3], 
                         [1, 3], 
                         [1, 2, 0, 4], 
                         [3, 0]]

    .. grid-item::
      
      .. graphviz::

        graph tris {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="1,0.1!"]; 
        2 [pos="0.9,0.9!"]; 
        3 [pos="-0.1,1.0!"]; 
        4 [pos="-.5,.2!"]

        0 -- 1;
        1 -- 2; 
        1 -- 3;        
        2 -- 3; 
        3 -- 0; 
        3 -- 4;
        4 -- 0;

        node0 [label=" ", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        node1 [label=" ", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        node2 [label=" ", pos="1.0,1.0!", shape=none, fontname="source code pro"] 
        node3 [label=" ", pos="-0.1,1.1!", shape=none, fontname="source code pro"] 
        node4 [label=" ", pos="-0.6,.2!", shape=none, fontname="source code pro"] 

        elem0 [label="0", pos=".3,.3!", shape=none, fontname="source code pro", fontcolor="cornflowerblue"] 
        elem1 [label="1", pos=".6,.7!", shape=none, fontname="source code pro", fontcolor="firebrick"]
        elem2 [label="2", pos="-.2,.2!", shape=none, fontname="source code pro", fontcolor="firebrick"]

        }

    .. grid-item::

      ::

        ElemNeighbors = [[1, 2], 
                         [0], 
                         [0]]

    .. grid-item::
      
      .. graphviz::

        graph tris {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!", color="cornflowerblue"];
        1 [pos="1,0.1!"]; 
        2 [pos="0.9,0.9!"]; 
        3 [pos="-0.1,1.0!"]; 
        4 [pos="-.5,.2!"]

        0 -- 1;
        1 -- 2; 
        1 -- 3;        
        2 -- 3; 
        3 -- 0; 
        3 -- 4;
        4 -- 0;

        node0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro", fontcolor="cornflowerblue"] 
        node1 [label="1", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        node2 [label="2", pos="1.0,1.0!", shape=none, fontname="source code pro"] 
        node3 [label="3", pos="-0.1,1.1!", shape=none, fontname="source code pro"] 
        node4 [label="4", pos="-0.6,.2!", shape=none, fontname="source code pro"] 

        elem0 [label="0", pos=".3,.3!", shape=none, fontname="source code pro", fontcolor="firebrick"] 
        elem1 [label="1", pos=".6,.7!", shape=none, fontname="source code pro"]
        elem2 [label="2", pos="-.2,.2!", shape=none, fontname="source code pro", fontcolor="firebrick"]

        }

    .. grid-item::

      ::

        ElemConn = [[0, 2], 
                    [0, 1], 
                    [1], 
                    [0, 1, 2], 
                    [2]]

    .. grid-item::
      
      .. graphviz::

        graph tris {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!", color="firebrick"];
        1 [pos="1,0.1!", color="firebrick"]; 
        2 [pos="0.9,0.9!"]; 
        3 [pos="-0.1,1.0!"]; 
        4 [pos="-.5,.2!"]

        0 -- 1 [color="cornflowerblue"];
        1 -- 2; 
        1 -- 3;        
        2 -- 3; 
        3 -- 0; 
        3 -- 4;
        4 -- 0;

        node0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro", fontcolor="firebrick"] 
        node1 [label="1", pos="1,-0.05!", shape=none, fontname="source code pro", fontcolor="firebrick"] 
        node2 [label="2", pos="1.0,1.0!", shape=none, fontname="source code pro"] 
        node3 [label="3", pos="-0.1,1.1!", shape=none, fontname="source code pro"] 
        node4 [label="4", pos="-0.6,.2!", shape=none, fontname="source code pro"] 

        elem0 [label=" ", pos=".3,.3!", shape=none, fontname="source code pro"] 
        elem1 [label=" ", pos=".6,.7!", shape=none, fontname="source code pro"]
        elem2 [label=" ", pos="-.2,.2!", shape=none, fontname="source code pro"]

        }

    .. grid-item::

      ::

        Edges = [[0, 1],
                 [3, 0],
                 [4, 0],
                 [1, 2],
                 [1, 3],
                 [2, 3],
                 [3, 4]]

    .. grid-item::
      
      .. graphviz::

        graph tris {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="1,0.1!"]; 
        2 [pos="0.9,0.9!"]; 
        3 [pos="-0.1,1.0!"]; 
        4 [pos="-.5,.2!"]

        0 -- 1 [color="firebrick"];
        1 -- 2; 
        1 -- 3 [color="firebrick"];        
        2 -- 3; 
        3 -- 0 [color="firebrick"]; 
        3 -- 4;
        4 -- 0;

        node0 [label=" ", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        node1 [label=" ", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        node2 [label=" ", pos="1.0,1.0!", shape=none, fontname="source code pro"] 
        node3 [label=" ", pos="-0.1,1.1!", shape=none, fontname="source code pro"] 
        node4 [label=" ", pos="-0.6,.2!", shape=none, fontname="source code pro"] 

        elem0 [label="0", pos=".3,.3!", shape=none, fontname="source code pro", fontcolor="cornflowerblue"] 
        elem1 [label="1", pos=".6,.7!", shape=none, fontname="source code pro"]
        elem2 [label="2", pos="-.2,.2!", shape=none, fontname="source code pro"]

        }

    .. grid-item::

      ::

        EdgeConn = [[0, 4, 1],
                    [3, 5, 4],
                    [1, 6, 2]]

    .. grid-item::
      
      .. graphviz::

        graph tris {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="1,0.1!"]; 
        2 [pos="0.9,0.9!"]; 
        3 [pos="-0.1,1.0!"]; 
        4 [pos="-.5,.2!"]

        0 -- 1 [color="cornflowerblue"];
        1 -- 2; 
        1 -- 3;        
        2 -- 3; 
        3 -- 0; 
        3 -- 4;
        4 -- 0;

        node0 [label=" ", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        node1 [label=" ", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        node2 [label=" ", pos="1.0,1.0!", shape=none, fontname="source code pro"] 
        node3 [label=" ", pos="-0.1,1.1!", shape=none, fontname="source code pro"] 
        node4 [label=" ", pos="-0.6,.2!", shape=none, fontname="source code pro"] 

        elem0 [label="0", pos=".3,.3!", shape=none, fontname="source code pro", fontcolor="firebrick"] 
        elem1 [label="1", pos=".6,.7!", shape=none, fontname="source code pro"]
        elem2 [label="2", pos="-.2,.2!", shape=none, fontname="source code pro"]

        }

    .. grid-item::

      ::

        EdgeElemConn = [[0], 
                        [0, 2], 
                        [2], 
                        [1], 
                        [0, 1], 
                        [1], 
                        [2]]