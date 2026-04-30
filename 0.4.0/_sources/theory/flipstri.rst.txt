Triangular Edge Flips
=====================

A convex quadrilateral has two valid triangularizations. 
The edge shared by two adjacent triangles can be flipped to switch between the two triangularizations, so that the one leading to a higher element quality can be chosen.

.. grid:: 2

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph tet3 {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        a [pos="-0.1,-0.1!"]; 
        b [pos="1.,0.!"];
        c [pos="1.2,1.2!"]; 
        d [pos="0.,1.!"]; 

        a -- b [penwidth=1, color="#d08770"];
        b -- c [penwidth=1, color="#d08770"];
        
        a -- c [penwidth=1, color="#d08770:#5e81ac", style=dashed]; 

        c -- d [penwidth=1, color="#5e81ac"];
        d -- a [penwidth=1, color="#5e81ac"];

        label0 [label="a", pos="-0.2,-0.2!", shape=none, fontname="source code pro"] 
        label1 [label="b", pos="1.1,-0.1!", shape=none, fontname="source code pro"] 
        label2 [label="c", pos="1.3,1.3!", shape=none, fontname="source code pro"] 
        label3 [label="d", pos="-0.1,1.1!", shape=none, fontname="source code pro"] 

        }
      
      Configuration 1
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph tet3 {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        a [pos="-0.1,-0.1!"]; 
        b [pos="1.,0.!"];
        c [pos="1.2,1.2!"]; 
        d [pos="0.,1.!"]; 

        c -- d [penwidth=1, color="#d08770"];
        b -- c [penwidth=1, color="#d08770"];
        
        b -- d [penwidth=1, color="#d08770:#5e81ac", style=dashed]; 

        a -- b [penwidth=1, color="#5e81ac"];
        d -- a [penwidth=1, color="#5e81ac"];

        label0 [label="a", pos="-0.2,-0.2!", shape=none, fontname="source code pro"] 
        label1 [label="b", pos="1.1,-0.1!", shape=none, fontname="source code pro"] 
        label2 [label="c", pos="1.3,1.3!", shape=none, fontname="source code pro"] 
        label3 [label="d", pos="-0.1,1.1!", shape=none, fontname="source code pro"] 

        }
      
      Configuration 2 

Flipping Procedure
------------------

Every edge-connected pair of elements is a candidate for a flip.