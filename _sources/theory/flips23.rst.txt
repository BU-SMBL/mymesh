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
