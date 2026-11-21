Mesh
====

Standard mesh
-------------
(See also: :class:`~mymesh.mesh.mesh`, :ref:`Working_with_the_mesh_class`)


The :class:`~mymesh.mesh.mesh` class is robust and feature rich, supporting arbitrary, mixed-element meshes, with a variety of methods and on-demand properties.
It's based around two data structures:

1. ``NodeCoords`` - an array containing the coordinates of the nodes, and
2. ``NodeConn`` - an array (or list of lists) containing indices of the nodes that connect to form elements (or the 'node connectivity')

Dictionaries ``NodeData`` and ``ElemData`` can be used to store scalar or vector data alongside the nodes and elements.

Dynamic mesh
------------
(See also: :class:`~mymesh.mesh.dmesh`)

:class:`~mymesh.mesh.dmesh` is a specialized mesh data structure for rapid modifications to a mesh's connectivity, such as during coarsening (:func:`~mymesh.improvement.Contract`) or :mod:`~mymesh.delaunay` triangulation.
It has three key features that enables rapid modifications: amortized :math:`\mathcal{O}(1)` insertion by doubling, swap removal, and a doubly-linked list to track node-element connectivity.

.. note::

    :class:`dmesh` only supports single element type meshes, and has only been fully tested on purely triangular and purely tetrahedral meshes.

Amortized :math:`\mathcal{O}(1)` insertion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In dynamically changing meshes, nodes and/or elements may be repeatedly added and/or removed.
When performing many such operations, the cost of resizing the arrays storing node and element information becomes significant.
To more efficiently handle insertions into arrays, the arrays are maintained with buffer space at the end, and a pointer (e.g. NNode, NElem) that tracks where the end of the meaningful data is. 
Inserting a value into the next available space is a simple and :math:`\mathcal{O}(1)` operation.
Whenever that buffer space runs out, the arrays will still need to be resized. 
Since we don't know how big the array *needs* to be, we can just double the length of the arrays to create plenty of space relative to the size of the existing arrays. 

.. graphviz::

    digraph structs {
        rankdir = "LR"
        node [shape=record];
        
        node [shape=none]; // Use 'none' because the HTML table provides the shape

        nodecoords [pos="0,0!", label=<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                <TR>
                    <TD COLSPAN="3">NodeCoords</TD>
                </TR>
                <TR>
                    <TD>x<sub>0</sub></TD>
                    <TD>y<sub>0</sub></TD>
                    <TD>z<sub>0</sub></TD>
                </TR>
                <TR>
                    <TD>x<sub>1</sub></TD>
                    <TD>y<sub>1</sub></TD>
                    <TD>z<sub>1</sub></TD>
                </TR>
                <TR>
                    <TD>x<sub>2</sub></TD>
                    <TD>y<sub>2</sub></TD>
                    <TD>z<sub>2</sub></TD>
                </TR>
                <TR>
                    <TD>x<sub>3</sub></TD>
                    <TD>y<sub>3</sub></TD>
                    <TD PORT="n3">z<sub>3</sub></TD>
                </TR>
            </TABLE>
        >];
    
        nodecoords2 [pos="3,0!", label=<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                <TR>
                    <TD COLSPAN="3">NodeCoords</TD>
                </TR>
                <TR>
                    <TD>x<sub>0</sub></TD>
                    <TD>y<sub>0</sub></TD>
                    <TD>z<sub>0</sub></TD>
                </TR>
                <TR>
                    <TD>x<sub>1</sub></TD>
                    <TD>y<sub>1</sub></TD>
                    <TD>z<sub>1</sub></TD>
                </TR>
                <TR>
                    <TD>x<sub>2</sub></TD>
                    <TD>y<sub>2</sub></TD>
                    <TD>z<sub>2</sub></TD>
                </TR>
                <TR>
                    <TD>x<sub>3</sub></TD>
                    <TD>y<sub>3</sub></TD>
                    <TD>z<sub>3</sub></TD>
                </TR>
                <TR>
                    <TD>x<sub>4</sub></TD>
                    <TD>y<sub>4</sub></TD>
                    <TD PORT="n4">z<sub>4</sub></TD>
                </TR>
                <TR>
                    <TD BGCOLOR="lightgray">x<sub>1</sub></TD>
                    <TD BGCOLOR="lightgray">y<sub>1</sub></TD>
                    <TD BGCOLOR="lightgray">z<sub>1</sub></TD>
                </TR>
                <TR>
                    <TD BGCOLOR="lightgray">x<sub>2</sub></TD>
                    <TD BGCOLOR="lightgray">y<sub>2</sub></TD>
                    <TD BGCOLOR="lightgray">z<sub>2</sub></TD>
                </TR>
                <TR>
                    <TD BGCOLOR="lightgray">x<sub>3</sub></TD>
                    <TD BGCOLOR="lightgray">y<sub>3</sub></TD>
                    <TD BGCOLOR="lightgray">z<sub>3</sub></TD>
                </TR>
            </TABLE>
        >];
        
    nnode1 [label="NNode", pos="1.2,-.6!"];
    nnode2 [label="NNode", pos="4.2,-.3!"];

    nnode1 -> nodecoords:n3:e;
    nnode2 -> nodecoords2:n4:e;
    }

Swap Removal
^^^^^^^^^^^^

Similar to insertion, removal changes the size of the data, but resizing the arrays whenever an entry is removed would slow things down a lot.
Instead, the item that's to be deleted can just be swapped with the last entry of meaningful data in the array, and the pointers can be moved accordingly.
In addition eliminating the need to resize the arrays when an item is deleted,
it adds more buffer space that can be utilized by future insertion operations. 

While elements are removed the ``NodeConn`` array by swap removal, nodes are removed from the mesh by simply removing the elements that reference them.
While that can cause orphaned nodes that take up space, it prevents the need to modify all connected elements with updated node numbers. 

.. graphviz::
    
    digraph structs {
        rankdir = "LR"
        node [shape=record];
        
        node [shape=none]; // Use 'none' because the HTML table provides the shape

        nodeconn [pos="0,0!", label=<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                <TR>
                    <TD COLSPAN="3">NodeConn</TD>
                </TR>
                <TR>
                    <TD>0</TD>
                    <TD>1</TD>
                    <TD>3</TD>
                </TR>
                <TR>
                    <TD BGCOLOR="red">1</TD>
                    <TD BGCOLOR="red">2</TD>
                    <TD BGCOLOR="red">3</TD>
                </TR>
                <TR>
                    <TD>0</TD>
                    <TD>3</TD>
                    <TD PORT="e3">4</TD>
                </TR>
            </TABLE>
        >];
    
        nodeconn2 [pos="3,0!", label=<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
                <TR>
                    <TD COLSPAN="3">NodeConn</TD>
                </TR>
                <TR>
                    <TD>0</TD>
                    <TD>1</TD>
                    <TD>3</TD>
                </TR>
                <TR>
                    <TD PORT="e2L">0</TD>
                    <TD>3</TD>
                    <TD PORT="e2">4</TD>
                </TR>
                <TR>
                    <TD PORT="e3L" BGCOLOR="lightgray">1</TD>
                    <TD BGCOLOR="lightgray">2</TD>
                    <TD BGCOLOR="lightgray">3</TD>
                </TR>
            </TABLE>
        >];
        
    nelem1 [label="NElem", pos="1.2,-.6!"];
    nelem2 [label="NElem", pos="4.2,-.3!"];

    nelem1 -> nodeconn:e3:e;
    nelem2 -> nodeconn2:e2:e;
    nodeconn2:e3L:w -> nodeconn2:e2L:w [dir=both];
    }

Doubly-linked list for element connectivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's often useful to keep track of all the elements that are connected to a node, however, that number can vary from node to node, meaning the ``ElemConn`` data structure becomes a "ragged" array (or list of lists), which can be difficult and inefficient to work with.


The doubly linked list structure involves five arrays: ``head``, ``next``, ``prev``, ``tail``, and ``elem``.
For node ``i``, ``head[i]`` contains an index ``j`` for ``elem`` that points to the first element connected to node ``i`` (``elem[head[i]]``). 
To get to the next connected element, ``next[j]`` gives the index ``k`` that points to the next element in ``elem``, ``next[k]`` gives the next, and so on until ``next`` contains the ``end`` flag (in this case, ``-1``).
Thus, by traversing the arrays ``head``, ``elem``, and ``next`` we can collect all of the element indices that are connected to a node.
``tail`` and ``prev`` work just liked ``head`` and ``next`` but in reverse.
The value of maintaining ``tail`` and ``prev`` in addition to ``head`` and ``next`` is that they make it easy (and efficient) to insert and remove new entries. 
For example, if a new element is connected to node ``i``, we can insert an entry into ``elem`` at the next available index ``m``, change ``prev[tail[i]]`` from ``end`` to ``m``, and set ``prev[m]`` to ``end``.
The addition of a new connection thus does not depend on how many elements are already connected to the node, as it would if we need to first use ``next`` to traverse from ``head[i]`` to ``tail[i]``

The same insertion method used for the node and element information is used to maintain these arrays.


.. grid:: 2
    :outline:

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
                node3 [label="3", pos="-0.1,1.1!", shape=none, fontname="source code pro", fontcolor="firebrick"] 
                node4 [label="4", pos="-0.6,.2!", shape=none, fontname="source code pro"] 

                elem0 [label="0", pos=".3,.3!", shape=none, fontname="source code pro"] 
                elem1 [label="1", pos=".6,.7!", shape=none, fontname="source code pro"]
                elem2 [label="2", pos="-.2,.2!", shape=none, fontname="source code pro",]
                }

    .. grid-item::

        .. code::

            ElemConn = [[0, 2], 
                        [0, 1], 
                        [1], 
                        [0, 1, 2], 
                        [2]]


.. graphviz::

    digraph structs {
        splines=true;
        node [shape=record];
        
        head_idx [pos="-.1,4.1!", label="<f> node|<f0> 0 |<f1> 1 |<f2> 2 |<f3> 3 |<f4> 4", color=none, fontsize=10];
        head [pos="-.1,3.75!", label="<f> head|<f0> 0 |<f1> 2 |<f2> 4 |<f3> 5 |<f4> 8"];
        elem [pos="1,3!", label="<f> elem|<f0> 0 |<f1> 2 |<f2> 0 | <f3> 1 | <f4> 1 | <f5> 0 |<f6> 1 |<f7> 2 |<f8> 2"];
        next [pos="0.5,2!", label="<f> next|<f0> 1 |<f1> end |<f2> 3 | <f3> end | <f4> end | <f5> 6 |<f6> 7 |<f7> end |<f8> end"];
        prev [pos="2.8,1.3!", label="<f> prev|<f0> end |<f1> 0 |<f2> end | <f3> 2 | <f4> end | <f5> end |<f6> 5 |<f7> 6 |<f8> end"];
        tail [pos="2.2,3.75!", label="<f> tail|<f0> 1 |<f1> 3 |<f2> 4 |<f3> 7 |<f4> 8"];
        tail_idx [pos="2.2,4.1!", label="<f> node|<f0> 0 |<f1> 1 |<f2> 2 |<f3> 3 |<f4> 4", color=none, fontsize=10];

        head:f0:s -> elem:f0:n [color="cornflowerblue"];
        elem:f0:s -> next:f0:n [color="cornflowerblue"];
        next:f0 -> elem:f1 [color="cornflowerblue"];
        elem:f1 -> next:f1 [color="cornflowerblue"];
        
        tail:f3 -> elem:f7 [color="firebrick"];
        elem:f7 -> prev:f7 [color="firebrick"];
        prev:f7 -> elem:f6 [color="firebrick"];
        elem:f6 -> prev:f6 [color="firebrick"];
        prev:f6 -> elem:f5 [color="firebrick"];
        elem:f5 -> prev:f5 [color="firebrick"];
    }
