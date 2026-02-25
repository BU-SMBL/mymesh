2→3 and 3→2 Flips
=================

A convex 5-vertex polyhedron has two valid tetrahedralizations. The first 
(Configuration 1) has two tetrahedra sharing a common face, and the second 
(Configuration 2) has three tetrahedra that all share a common edge. 

.. grid:: 2

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph tet3 {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        d [pos="0.9,0.4!"]; 
        a [pos=".4,0.9!"]; 
        e [pos="0.5,-0.7!"];
        b [pos="0,0!"];
        c [pos=".6,-.2!"]; 

        b -- a [penwidth=1, color="#d08770"];
        c -- a [penwidth=1, color="#d08770"];
        d -- a [penwidth=1, color="#d08770"]; 
        
        b -- c [penwidth=1, color="#5e81ac:#d08770"];
        c -- d [penwidth=1, color="#5e81ac:#d08770"]; 
        d -- b [penwidth=1, color="#d08770:#5e81ac", style=dotted]; 

        b -- e [penwidth=1, color="#5e81ac"];
        c -- e [penwidth=1, color="#5e81ac"];
        d -- e [penwidth=1, color="#5e81ac"]; 

        label0 [label="a", pos="0.3,1.0!", shape=none, fontname="source code pro"] 
        label1 [label="b", pos="-.1,0!",  shape=none, fontname="source code pro"] 
        label2 [label="c", pos=".65,-0.05!", shape=none, fontname="source code pro"] 
        label3 [label="d", pos="1.0,0.4!",  shape=none, fontname="source code pro"] 
        label4 [label="e", pos=".5,-0.8!",  shape=none, fontname="source code pro"] 

        }
      
      Configuration 1 (2 tets)
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph tet3 {
        
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        d [pos="0.9,0.4!"]; 
        a [pos=".4,0.9!"]; 
        e [pos="0.5,-0.7!"];
        b [pos="0,0!"];
        c [pos=".6,-.2!"]; 
        

        b -- a [penwidth=1, color="#d08770:#5e81ac"];
        c -- a [penwidth=1, color="#a3be8c:#5e81ac"];
        d -- a [penwidth=1, color="#a3be8c:#d08770"]; 

        b -- c [penwidth=1, color="#5e81ac"];
        c -- d [penwidth=1, color="#a3be8c"]; 
        d -- b [penwidth=1, color="#d08770", style=dotted]; 

        b -- e [penwidth=1, color="#5e81ac:#d08770"];
        c -- e [penwidth=1, color="#5e81ac:#a3be8c"];
        d -- e [penwidth=1, color="#d08770:#a3be8c"]; 

        a -- e [penwidth=1.5, color="#5e81ac:#d08770:#a3be8c", style=dashed]

        label0 [label="a", pos="0.3,1.0!", shape=none, fontname="source code pro"] 
        label1 [label="b", pos="-.1,0!",  shape=none, fontname="source code pro"] 
        label2 [label="c", pos=".65,-0.05!", shape=none, fontname="source code pro"] 
        label3 [label="d", pos="1.0,0.4!",  shape=none, fontname="source code pro"] 
        label4 [label="e", pos=".5,-0.8!",  shape=none, fontname="source code pro"] 

        }
      
      Configuration 2 (3 tets)

Flipping Procedure
------------------

Every face-connected pair of elements is a candidate for a 2→3 flip, but the
flip will only be valid of those two elements form a convex polyhedron. This
can be checked by verifying that, for each of the 6 outer faces of the 
polyhedron, all the two non-face nodes lie on the same side of the face. 
Similarly, edges connected to three elements are candidates for a 3→2 flip if 
those three elements form a convex, 5-node polyhedron.


Data Structure
--------------
Efficient flipping requires a data structure that can be easily queried and 
modified to facilitate traversing the mesh, finding potentially flippable 
elements, and performing the flip. There are several such data structures
with various trade-offs in terms of memory usage, efficiency, and complexity. 

The data structure used here utilizes three hash tables (python dictionaries) that 
store element, face, and edge connectivity information, in addition to the 
standard node coordinate array. Each table is keyed by the sorted node 
connectivity of its features. Sorting ensures that redundant features aren't 
stored and each element, face, and edge can be unambiguously identified by 
its nodes. This means that at a face where two elements meet, there is a single
face (rather than two half-faces). It also means that the keys of tetrahedra
may not be properly ordered and could be interpreted as inverted elements. For 
this reason, each entry in the element table stores a properly ordered 
(un-inverted) version of the element. 

The element table also stores table keys to the face and edge table entries
of the connected faces and edges (``FaceConn`` and ``EdgeConn``, see 
:ref:`connectivity`). The face and edge tables likewise store the keys to
the element table entries of their connected elements (``FaceElemConn``, 
``EdgeElemConn``). Through this structure, an element's neighbors can be identified
by querying the connected elements of each of the element's faces, and likewise
with edges. To minimize the need for recalculation, the element table also stores
the volume, element quality, and status (whether the element is active in the 
mesh) of each element, and the face table stores the normal vector of each 
face. While elements, faces, and edges may be added to the tables over the 
course of the flipping procedure, they are never removed, only the status
value of the elements is changed. Whether or not a face or edge is active 
in the mesh is inferred from the element table as inactive faces/edges will 
never be reached. As the mesh topology changes, the relevant entries in the 
tables are updated so that the connectivity information accurately represents
the updated mesh.

An example data structure for the tetrahedra depicted above is shown here for
both possible configurations. Note that in a larger mesh, the face and edge 
tables will also contain entries for other neighboring elements.

.. table:: **Element Table**
    :align: center

    +-------------------+--------------------------------------------+------------------------------------------------+
    | Element           | Faces                                      | Edges                                          |
    +===================+============================================+================================================+
    |**Configuration 1**|                                            |                                                |
    +-------------------+--------------------------------------------+------------------------------------------------+
    | (a, b, c, d)      | (a, b, c), (a, c, d), (a, b, d), (b, c, d) | (b, c), (c, d), (b, d), (a, b), (a, c), (a, d) |
    +-------------------+--------------------------------------------+------------------------------------------------+
    | (b, c, d, e)      | (b, c, e), (c, d, e), (b, d, e), (b, c, d) | (b, c), (c, d), (b, d), (b, e), (c, e), (d, e) |
    +-------------------+--------------------------------------------+------------------------------------------------+
    |**Configuration 2**|                                            |                                                |
    +-------------------+--------------------------------------------+------------------------------------------------+
    | (a, b, c, e)      | (a, b, c), (b, c, e), (a, b, e), (a, c, e) | (a, b), (a, c), (b, e), (c, e), (b, c), (a, e) |
    +-------------------+--------------------------------------------+------------------------------------------------+
    | (a, b, d, e)      | (a, b, d), (b, d, e), (a, b, e), (a, d, e) | (a, b), (a, d), (b, e), (d, e), (b, d), (a, e) |
    +-------------------+--------------------------------------------+------------------------------------------------+
    | (a, c, d, e)      | (a, c, d), (c, d, e), (a, c, e), (a, d, e) | (a, c), (a, d), (c, e), (d, e), (c, d), (a, e) |
    +-------------------+--------------------------------------------+------------------------------------------------+


.. table:: **Face Table**
    :align: center

    +---------------+------------------------------------+----------------------------+
    |               | **Configuration 1**                | **Configuration 2**        | 
    +===============+====================================+============================+
    | **Face**      | **Elements**                                                    | 
    +---------------+------------------------------------+----------------------------+
    | (a, b, c)     | (a, b, c, d)                       | (a, b, c, e)               | 
    +---------------+------------------------------------+----------------------------+
    | (a, c, d)     | (a, b, c, d)                       | (a, c, d, e)               | 
    +---------------+------------------------------------+----------------------------+
    | (a, b, d)     | (a, b, c, d)                       | (a, b, d, e)               | 
    +---------------+------------------------------------+----------------------------+
    | (b, c, e)     | (b, c, d, e)                       | (a, b, c, e)               | 
    +---------------+------------------------------------+----------------------------+
    | (c, d, e)     | (b, c, d, e)                       | (a, c, d, e)               | 
    +---------------+------------------------------------+----------------------------+
    | (b, d, e)     | (b, c, d, e)                       | (a, b, d, e)               | 
    +---------------+------------------------------------+----------------------------+
    | (b, c, d)     | (a, b, c, d), (b, c, d, e)         |                            | 
    +---------------+------------------------------------+----------------------------+
    | (a, b, e)     |                                    | (a, b, d, e), (a, b, c, e) | 
    +---------------+------------------------------------+----------------------------+
    | (a, c, e)     |                                    | (a, b, c, e), (a, c, d, e) | 
    +---------------+------------------------------------+----------------------------+
    | (a, d, e)     |                                    | (a, c, d, e), (a, b, d, e) | 
    +---------------+------------------------------------+----------------------------+

.. table:: **Edge Table**
    :align: center

    +------------+----------------------------+------------------------------------------+
    |            | **Configuration 1**        | **Configuration 2**                      | 
    +============+============================+==========================================+
    | **Edge**   | **Elements**                                                          | 
    +------------+----------------------------+------------------------------------------+
    | (a, b)     | (a, b, c, d)               | (a, b, c, e), (a, b, d, e)               | 
    +------------+----------------------------+------------------------------------------+
    | (a, c)     | (a, b, c, d)               | (a, b, c, e), (a, c, d, e)               | 
    +------------+----------------------------+------------------------------------------+
    | (a, d)     | (a, b, c, d)               | (a, c, d, e), (a, b, d, e)               | 
    +------------+----------------------------+------------------------------------------+
    | (b, c)     | (a, b, c, d), (b, c, d, e) | (a, b, c, e)                             | 
    +------------+----------------------------+------------------------------------------+
    | (b, d)     | (a, b, c, d), (b, c, d, e) | (a, b, d, e)                             | 
    +------------+----------------------------+------------------------------------------+
    | (c, d)     | (a, b, c, d), (b, c, d, e) | (a, c, d, e)                             | 
    +------------+----------------------------+------------------------------------------+
    | (b, e)     | (b, c, d, e)               | (a, b, c, e), (a, b, d, e)               | 
    +------------+----------------------------+------------------------------------------+
    | (c, e)     | (b, c, d, e)               | (a, b, c, e), (a, c, d, e)               | 
    +------------+----------------------------+------------------------------------------+
    | (d, e)     | (b, c, d, e)               | (a, c, d, e), (a, b, d, e)               | 
    +------------+----------------------------+------------------------------------------+
    | (a, e)     |                            | (a, b, c, e), (a, b, d, e), (a, c, d, e) | 
    +------------+----------------------------+------------------------------------------+