.. _theory_contour:

Contour
=======

Isosurface contouring methods are used to extract the surface from a field of
values, most often an image or an implicit function evaluated on a grid of 
points. One of the earliest and most famous methods is the Marching Cubes
algorithm by :cite:t:`Lorensen1987`, which inspired a variety of "marching 
element" methods that follow a similar approach. These methods "march" from
element to element, at each one assessing the node values to determine whether
each node is above or below the isovalue threshold in order to determine whether
that node is inside or outside of the surface. Partitions are then created by 
placing new nodes along edges that change from inside to outside. Lookup tables
can then be used to efficiently determine what partitions need to be used based
on which nodes are inside and which nodes are outside. The new nodes can either 
be placed on the midpoint of the edges or interpolation can be used to 
approximate where along the edge the isovalue is.

Marching Squares
----------------
See also :func:`~mymesh.contour.MarchingSquares`

Marching squares operates on square (or quadrilateral) elements. Since each of 
the four nodes can either be inside or outside of the isoline, a :math:`2^4=16`
entry lookup table can be used where a binary lookup index can be obtained based 
on the state of each node (1 = inside, 0 = outside). Marching squares can either 
be used to extract line segments along the boundary or triangles to fill the 
inside of the isoline.

.. graphviz::

  graph bits {  
  node [shape=point, fontname="source code pro"];
  edge [style=solid, penwidth=2];

  v0 [pos="0.,0.!", color="#BF616A", xlabel="0"]; 
  v1 [pos="1.,0.!", color="#EBCB8B", xlabel="1"];
  v2 [pos="1.,1.!", color="#A3BE8C", xlabel="2"]; 
  v3 [pos="0.,1.!", color="#B48EAD", xlabel="3"];

  //v4 [pos="0.5,0.!", color="#5E81AC"];
  //v5 [pos="1.0,0.5!", color="#5E81AC"];
  //v6 [pos="0.5,1.!", color="#5E81AC"];
  //v7 [pos="0.,0.5!", color="#5E81AC"];

  v0 -- v1 [style=dotted];
  v1 -- v2 [style=dotted];
  v2 -- v3 [style=dotted];
  v3 -- v0 [style=dotted];

  struct1 [shape=plaintext, pos="2.5,0.75!",
    label=<
      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" STYLE="ROUNDED">
          <TR>
              <TD BGCOLOR="white" COLOR="white" PORT="f0">Index = </TD>
              <TD BGCOLOR="white" COLOR="#BF616A#B48EAD" PORT="f0">0</TD>
              <TD BGCOLOR="white" COLOR="#EBCB8B" PORT="f1">1</TD>
              <TD BGCOLOR="white" COLOR="#A3BE8C" PORT="f2">2</TD>
              <TD BGCOLOR="white" COLOR="#B48EAD" PORT="f3">3</TD>
          </TR>
      </TABLE>
      >];

  }

Edge Lookup Table

.. grid:: 3 6 6 6
    :outline:

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case0 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        }
      
      Case 0

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case1 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v6 -- v7 [color="#5E81AC"];

        }

      Case 1

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case2 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        
        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v5 -- v6 [color="#5E81AC"];

        }

      Case 2

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case3 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        
        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v5 -- v7 [color="#5E81AC"];

        }

      Case 3

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case4 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v4 -- v5 [color="#5E81AC"];

        }

      Case 4
    
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case5 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v7 -- v4 [color="#5E81AC"];
        v5 -- v6 [color="#5E81AC"];

        }

      Case 5

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case6 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v4 -- v6 [color="#5E81AC"];

        }

      Case 6

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case7 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v4 -- v7 [color="#5E81AC"];

        }

      Case 7

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case8 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        
        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v7 -- v4 [color="#5E81AC"];

        }

      Case 8
    
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case9 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v6 -- v4 [color="#5E81AC"];

        }

      Case 9

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case10 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        
        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v4 -- v5 [color="#5E81AC"];
        v6 -- v7 [color="#5E81AC"];

        }

      Case 10

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case11 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=bllack];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v5 -- v4 [color="#5E81AC"];

        }

      Case 11
    
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case12 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v7 -- v5 [color="#5E81AC"];

        }

      Case 12

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case13 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        
        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v6 -- v5 [color="#5E81AC"];

        }

      Case 13

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case1 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v7 -- v6 [color="#5E81AC"];

        }

      Case 14

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case15 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];


        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        }
      
      Case 15

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph legend {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,1.!", fillcolor=black]; 
        v1 [pos="0.,0.75!", fillcolor=white];
        Inside [shape=plaintext, pos="0.6,1.!"]; 
        Outside [shape=plaintext, pos="0.6,0.75!"];
        blank [pos="0.,0!", color=white];

        }
      

Triangle Lookup Table

.. grid:: 3 6 6 6
    :outline:

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case0 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        }
      
      Case 0

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case1 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        
        v7 -- v6 [color="#5E81AC"];
        v6 -- v3 [color="#5E81AC"];
        v3 -- v7 [color="#5E81AC"];
        }

      Case 1

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case2 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        
        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v5 -- v2 [color="#5E81AC"];
        v2 -- v6 [color="#5E81AC"];
        v6 -- v5 [color="#5E81AC"];

        }

      Case 2

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case3 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        
        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        //v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v7 -- v5 [color="#5E81AC"];
        v5 -- v2 [color="#5E81AC"];
        v2 -- v7 [color="#5E81AC"];
        v2 -- v3 [color="#5E81AC"];
        v3 -- v7 [color="#5E81AC"];
        }

      Case 3

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case4 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v4 -- v1 [color="#5E81AC"];
        v1 -- v5 [color="#5E81AC"];
        v5 -- v4 [color="#5E81AC"]; 

        }

      Case 4
    
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case5 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v4 -- v1 [color="#5E81AC"];
        v1 -- v5 [color="#5E81AC"];
        v5 -- v4 [color="#5E81AC"];
        v5 -- v6 [color="#5E81AC"];
        v6 -- v4 [color="#5E81AC"];
        v6 -- v7 [color="#5E81AC"];
        v7 -- v4 [color="#5E81AC"];
        v6 -- v3 [color="#5E81AC"];
        v3 -- v7 [color="#5E81AC"];

        }

      Case 5

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case6 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        //v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v4 -- v1 [color="#5E81AC"];
        v1 -- v2 [color="#5E81AC"];
        v2 -- v4 [color="#5E81AC"];
        v2 -- v6 [color="#5E81AC"];
        v6 -- v4 [color="#5E81AC"];

        }

      Case 6

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case7 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=white]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        //v1 -- v2 [style=dotted];
        //v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v1 -- v2 [color="#5E81AC"];
        v2 -- v3 [color="#5E81AC"];
        v3 -- v1 [color="#5E81AC"];
        v3 -- v4 [color="#5E81AC"];
        v4 -- v1 [color="#5E81AC"];
        v3 -- v7 [color="#5E81AC"];
        v4 -- v7 [color="#5E81AC"];

        }

      Case 7

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case8 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        
        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v0 -- v4 [color="#5E81AC"];
        v4 -- v7 [color="#5E81AC"];
        v7 -- v0 [color="#5E81AC"];

        }

      Case 8
    
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case9 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        //v3 -- v0 [style=dotted];

        v0 -- v4 [color="#5E81AC"];
        v4 -- v6 [color="#5E81AC"];
        v6 -- v0 [color="#5E81AC"];
        v6 -- v3 [color="#5E81AC"];
        v3 -- v0 [color="#5E81AC"];

        }

      Case 9

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case10 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        
        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v0 -- v4 [color="#5E81AC"];
        v4 -- v7 [color="#5E81AC"];
        v7 -- v0 [color="#5E81AC"];
        v4 -- v5 [color="#5E81AC"];
        v5 -- v7 [color="#5E81AC"];
        v5 -- v6 [color="#5E81AC"];
        v6 -- v7 [color="#5E81AC"];
        v5 -- v2 [color="#5E81AC"];
        v2 -- v6 [color="#5E81AC"];
        }

      Case 10

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case11 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=white];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        //v2 -- v3 [style=dotted];
        //v3 -- v0 [style=dotted];

        v0 -- v2 [color="#5E81AC"];
        v2 -- v3 [color="#5E81AC"];
        v3 -- v0 [color="#5E81AC"];
        v0 -- v4 [color="#5E81AC"];
        v4 -- v5 [color="#5E81AC"];
        v5 -- v0 [color="#5E81AC"];
        v5 -- v2 [color="#5E81AC"];

        }

      Case 11
    
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case12 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        //v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v0 -- v1 [color="#5E81AC"];
        v1 -- v5 [color="#5E81AC"];
        v5 -- v0 [color="#5E81AC"];
        v5 -- v7 [color="#5E81AC"];
        v7 -- v0 [color="#5E81AC"];

        }

      Case 12

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case13 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        
        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=white]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        //v0 -- v1 [style=dotted];
        v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        //v3 -- v0 [style=dotted];

        v0 -- v1 [color="#5E81AC"];
        v1 -- v3 [color="#5E81AC"];
        v3 -- v0 [color="#5E81AC"];
        v1 -- v5 [color="#5E81AC"];
        v5 -- v6 [color="#5E81AC"];
        v6 -- v1 [color="#5E81AC"];
        v6 -- v3 [color="#5E81AC"];
        
        }

      Case 13

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case1 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=white];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        v6 [pos="0.5,1.!", color="#5E81AC"];
        v7 [pos="0.,0.5!", color="#5E81AC"];

        //v0 -- v1 [style=dotted];
        //v1 -- v2 [style=dotted];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];

        v0 -- v1 [color="#5E81AC"];
        v1 -- v2 [color="#5E81AC"];
        v2 -- v0 [color="#5E81AC"];
        v2 -- v6 [color="#5E81AC"];
        v6 -- v0 [color="#5E81AC"];
        v6 -- v7 [color="#5E81AC"];
        v7 -- v0 [color="#5E81AC"];
        }

      Case 14

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case15 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];


        v0 [pos="0.,0.!", fillcolor=black]; 
        v1 [pos="1.,0.!", fillcolor=black];
        v2 [pos="1.,1.!", fillcolor=black]; 
        v3 [pos="0.,1.!", fillcolor=black];
        
        //v4 [pos="0.5,0.!", color="#5E81AC"];
        //v5 [pos="1.0,0.5!", color="#5E81AC"];
        //v6 [pos="0.5,1.!", color="#5E81AC"];
        //v7 [pos="0.,0.5!", color="#5E81AC"];

        v0 -- v1 [color="#5E81AC"];
        v1 -- v2 [color="#5E81AC"];
        v2 -- v3 [color="#5E81AC"];
        v3 -- v0 [color="#5E81AC"];

        v0 -- v2 [color="#5E81AC"];
        

        }
      
      Case 15

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph legend {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,1.!", fillcolor=black]; 
        v1 [pos="0.,0.75!", fillcolor=white];
        Inside [shape=plaintext, pos="0.6,1.!"]; 
        Outside [shape=plaintext, pos="0.6,0.75!"];
        blank [pos="0.,0!", color=white];

        }

Marching Cubes
--------------
:cite:t:`Lorensen1987` 

See also :func:`~mymesh.contour.MarchingCubes`

For a cube, there are :math:`2^8 = 256` possible combinations of vertices
falling inside or outside of the surface, however by rotations and inversion,
this can be reduced to 15 unique triangulations. 

Marching Cubes Lookup Table

.. grid:: 3 6 6 6
    :outline:

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case0 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=white]; 
        v1 [pos="1.4,0.5!", fillcolor=white];
        v2 [pos="1.8,0.9!", fillcolor=white]; 
        v3 [pos="0.8,0.9!", fillcolor=white]; 
        v4 [pos="0.4,1.5!", fillcolor=white];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=white]; 
        v7 [pos="0.8,1.9!", fillcolor=white];
        
        // bottom edges
        //e0 [pos="0.9, 0.5!", color="#5E81AC"];
        //e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        //e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        //e4 [pos="0.4, 1.0!", color="#5E81AC"];
        //e5 [pos="1.4, 1.0!" color="#5E81AC"];
        //e6 [pos="1.8, 1.4!", color="#5E81AC"];
        //e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        //e8 [pos="0.9, 1.5!", color="#5E81AC"];
        //e9 [pos="1.6, 1.7!", color="#5E81AC"];
        //e10 [pos="1.3, 1.9!", color="#5E81AC"];
        //e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        }
      
      Case 0
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case1 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=black]; 
        v1 [pos="1.4,0.5!", fillcolor=white];
        v2 [pos="1.8,0.9!", fillcolor=white]; 
        v3 [pos="0.8,0.9!", fillcolor=white]; 
        v4 [pos="0.4,1.5!", fillcolor=white];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=white]; 
        v7 [pos="0.8,1.9!", fillcolor=white];
        
        // bottom edges
        e0 [pos="0.9, 0.5!", color="#5E81AC"];
        //e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        e4 [pos="0.4, 1.0!", color="#5E81AC"];
        //e5 [pos="1.4, 1.0!" color="#5E81AC"];
        //e6 [pos="1.8, 1.4!", color="#5E81AC"];
        //e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        //e8 [pos="0.9, 1.5!", color="#5E81AC"];
        //e9 [pos="1.6, 1.7!", color="#5E81AC"];
        //e10 [pos="1.3, 1.9!", color="#5E81AC"];
        //e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e0 -- e3 [color="#5E81AC"];
        e3 -- e4 [color="#5E81AC"];
        e4 -- e0 [color="#5E81AC"];

        }
      
      Case 1
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case2 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=black]; 
        v1 [pos="1.4,0.5!", fillcolor=black];
        v2 [pos="1.8,0.9!", fillcolor=white]; 
        v3 [pos="0.8,0.9!", fillcolor=white]; 
        v4 [pos="0.4,1.5!", fillcolor=white];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=white]; 
        v7 [pos="0.8,1.9!", fillcolor=white];
        
        // bottom edges
        //e0 [pos="0.9, 0.5!", color="#5E81AC"];
        e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        e4 [pos="0.4, 1.0!", color="#5E81AC"];
        e5 [pos="1.4, 1.0!" color="#5E81AC"];
        //e6 [pos="1.8, 1.4!", color="#5E81AC"];
        //e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        //e8 [pos="0.9, 1.5!", color="#5E81AC"];
        //e9 [pos="1.6, 1.7!", color="#5E81AC"];
        //e10 [pos="1.3, 1.9!", color="#5E81AC"];
        //e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e1 -- e3 [color="#5E81AC"];
        e3 -- e5 [color="#5E81AC"];
        e5 -- e1 [color="#5E81AC"];
        e3 -- e4 [color="#5E81AC"];
        e4 -- e5 [color="#5E81AC"];

        }
      
      Case 2
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case1 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=black]; 
        v1 [pos="1.4,0.5!", fillcolor=white];
        v2 [pos="1.8,0.9!", fillcolor=white]; 
        v3 [pos="0.8,0.9!", fillcolor=white]; 
        v4 [pos="0.4,1.5!", fillcolor=white];
        v5 [pos="1.4,1.5!", fillcolor=black]; 
        v6 [pos="1.8,1.9!", fillcolor=white]; 
        v7 [pos="0.8,1.9!", fillcolor=white];
        
        // bottom edges
        e0 [pos="0.9, 0.5!", color="#5E81AC"];
        //e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        e4 [pos="0.4, 1.0!", color="#5E81AC"];
        e5 [pos="1.4, 1.0!" color="#5E81AC"];
        //e6 [pos="1.8, 1.4!", color="#5E81AC"];
        //e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        e8 [pos="0.9, 1.5!", color="#5E81AC"];
        e9 [pos="1.6, 1.7!", color="#5E81AC"];
        //e10 [pos="1.3, 1.9!", color="#5E81AC"];
        //e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e0 -- e3 [color="#5E81AC"];
        e3 -- e4 [color="#5E81AC"];
        e4 -- e0 [color="#5E81AC"];

        e5 -- e8 [color="#5E81AC"];
        e8 -- e9 [color="#5E81AC"];
        e9 -- e5 [color="#5E81AC"];

        }
      
      Case 3
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case4 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=black]; 
        v1 [pos="1.4,0.5!", fillcolor=white];
        v2 [pos="1.8,0.9!", fillcolor=white]; 
        v3 [pos="0.8,0.9!", fillcolor=white]; 
        v4 [pos="0.4,1.5!", fillcolor=white];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=black]; 
        v7 [pos="0.8,1.9!", fillcolor=white];
        
        // bottom edges
        e0 [pos="0.9, 0.5!", color="#5E81AC"];
        //e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        e4 [pos="0.4, 1.0!", color="#5E81AC"];
        //e5 [pos="1.4, 1.0!" color="#5E81AC"];
        e6 [pos="1.8, 1.4!", color="#5E81AC"];
        //e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        //e8 [pos="0.9, 1.5!", color="#5E81AC"];
        e9 [pos="1.6, 1.7!", color="#5E81AC"];
        e10 [pos="1.3, 1.9!", color="#5E81AC"];
        //e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e0 -- e3 [color="#5E81AC"];
        e3 -- e4 [color="#5E81AC"];
        e4 -- e0 [color="#5E81AC"];

        e6 -- e9 [color="#5E81AC"];
        e9 -- e10 [color="#5E81AC"];
        e10 -- e6 [color="#5E81AC"];

        }
      
      Case 4
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case5 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=white]; 
        v1 [pos="1.4,0.5!", fillcolor=black];
        v2 [pos="1.8,0.9!", fillcolor=black]; 
        v3 [pos="0.8,0.9!", fillcolor=black]; 
        v4 [pos="0.4,1.5!", fillcolor=white];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=white]; 
        v7 [pos="0.8,1.9!", fillcolor=white];
        
        // bottom edges
        e0 [pos="0.9, 0.5!", color="#5E81AC"];
        //e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        //e4 [pos="0.4, 1.0!", color="#5E81AC"];
        e5 [pos="1.4, 1.0!" color="#5E81AC"];
        e6 [pos="1.8, 1.4!", color="#5E81AC"];
        e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        //e8 [pos="0.9, 1.5!", color="#5E81AC"];
        //e9 [pos="1.6, 1.7!", color="#5E81AC"];
        //e10 [pos="1.3, 1.9!", color="#5E81AC"];
        //e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e0 -- e3 [color="#5E81AC"];
        e3 -- e5 [color="#5E81AC"];
        e5 -- e0 [color="#5E81AC"];
        e5 -- e7 [color="#5E81AC"];
        e5 -- e6 [color="#5E81AC"];
        e6 -- e7 [color="#5E81AC"];
        e3 -- e7 [color="#5E81AC"];
        }
      
      Case 5
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case6 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=black]; 
        v1 [pos="1.4,0.5!", fillcolor=black];
        v2 [pos="1.8,0.9!", fillcolor=white]; 
        v3 [pos="0.8,0.9!", fillcolor=white]; 
        v4 [pos="0.4,1.5!", fillcolor=white];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=black]; 
        v7 [pos="0.8,1.9!", fillcolor=white];
        
        // bottom edges
        //e0 [pos="0.9, 0.5!", color="#5E81AC"];
        e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        e4 [pos="0.4, 1.0!", color="#5E81AC"];
        e5 [pos="1.4, 1.0!" color="#5E81AC"];
        e6 [pos="1.8, 1.4!", color="#5E81AC"];
        //e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        //e8 [pos="0.9, 1.5!", color="#5E81AC"];
        e9 [pos="1.6, 1.7!", color="#5E81AC"];
        e10 [pos="1.3, 1.9!", color="#5E81AC"];
        //e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e3 -- e1 [color="#5E81AC"];
        e1 -- e5 [color="#5E81AC"];
        e5 -- e3 [color="#5E81AC"];
        e4 -- e5 [color="#5E81AC"];
        e4 -- e3 [color="#5E81AC"];
        e6 -- e9 [color="#5E81AC"];
        e9 -- e10 [color="#5E81AC"];
        e10 -- e6 [color="#5E81AC"];
        
        }
      
      Case 6

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case7 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=white]; 
        v1 [pos="1.4,0.5!", fillcolor=black];
        v2 [pos="1.8,0.9!", fillcolor=white]; 
        v3 [pos="0.8,0.9!", fillcolor=white]; 
        v4 [pos="0.4,1.5!", fillcolor=black];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=black]; 
        v7 [pos="0.8,1.9!", fillcolor=white];
        
        // bottom edges
        e0 [pos="0.9, 0.5!", color="#5E81AC"];
        e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        //e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        e4 [pos="0.4, 1.0!", color="#5E81AC"];
        e5 [pos="1.4, 1.0!" color="#5E81AC"];
        e6 [pos="1.8, 1.4!", color="#5E81AC"];
        //e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        e8 [pos="0.9, 1.5!", color="#5E81AC"];
        e9 [pos="1.6, 1.7!", color="#5E81AC"];
        e10 [pos="1.3, 1.9!", color="#5E81AC"];
        e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e0 -- e1 [color="#5E81AC"];
        e1 -- e5 [color="#5E81AC"];
        e5 -- e0 [color="#5E81AC"];
        e4 -- e8 [color="#5E81AC"];
        e8 -- e11 [color="#5E81AC"];
        e11 -- e4 [color="#5E81AC"];
        e6 -- e9 [color="#5E81AC"];
        e9 -- e10 [color="#5E81AC"];
        e10 -- e6 [color="#5E81AC"];

        }
      
      Case 7
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case8 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=black]; 
        v1 [pos="1.4,0.5!", fillcolor=black];
        v2 [pos="1.8,0.9!", fillcolor=black]; 
        v3 [pos="0.8,0.9!", fillcolor=black]; 
        v4 [pos="0.4,1.5!", fillcolor=white];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=white]; 
        v7 [pos="0.8,1.9!", fillcolor=white];
        
        // bottom edges
        //e0 [pos="0.9, 0.5!", color="#5E81AC"];
        //e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        //e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        e4 [pos="0.4, 1.0!", color="#5E81AC"];
        e5 [pos="1.4, 1.0!" color="#5E81AC"];
        e6 [pos="1.8, 1.4!", color="#5E81AC"];
        e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        //e8 [pos="0.9, 1.5!", color="#5E81AC"];
        //e9 [pos="1.6, 1.7!", color="#5E81AC"];
        //e10 [pos="1.3, 1.9!", color="#5E81AC"];
        //e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e4 -- e5 [color="#5E81AC"];
        e5 -- e6 [color="#5E81AC"];
        e6 -- e4 [color="#5E81AC"];
        e4 -- e7 [color="#5E81AC"];
        e7 -- e6 [color="#5E81AC"];

        }
      
      Case 8
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case9 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=black]; 
        v1 [pos="1.4,0.5!", fillcolor=white];
        v2 [pos="1.8,0.9!", fillcolor=black]; 
        v3 [pos="0.8,0.9!", fillcolor=black]; 
        v4 [pos="0.4,1.5!", fillcolor=white];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=white]; 
        v7 [pos="0.8,1.9!", fillcolor=black];
        
        // bottom edges
        e0 [pos="0.9, 0.5!", color="#5E81AC"];
        e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        //e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        e4 [pos="0.4, 1.0!", color="#5E81AC"];
        //e5 [pos="1.4, 1.0!" color="#5E81AC"];
        e6 [pos="1.8, 1.4!", color="#5E81AC"];
        //e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        //e8 [pos="0.9, 1.5!", color="#5E81AC"];
        //e9 [pos="1.6, 1.7!", color="#5E81AC"];
        e10 [pos="1.3, 1.9!", color="#5E81AC"];
        e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e0 -- e1 [color="#5E81AC"];
        e1 -- e6 [color="#5E81AC"];
        e6 -- e0 [color="#5E81AC"];
        e0 -- e10 [color="#5E81AC"];
        e10 -- e6 [color="#5E81AC"];
        e0 -- e4 [color="#5E81AC"];
        e4 -- e11 [color="#5E81AC"];
        e11 -- e10 [color="#5E81AC"];
        e4 -- e10 [color="#5E81AC"];
        }
      
      Case 9
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case10 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=black]; 
        v1 [pos="1.4,0.5!", fillcolor=white];
        v2 [pos="1.8,0.9!", fillcolor=black]; 
        v3 [pos="0.8,0.9!", fillcolor=white]; 
        v4 [pos="0.4,1.5!", fillcolor=black];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=black]; 
        v7 [pos="0.8,1.9!", fillcolor=white];
        
        // bottom edges
        e0 [pos="0.9, 0.5!", color="#5E81AC"];
        e1 [pos="1.6, 0.7!", color="#5E81AC"];
        e2 [pos="1.3, 0.9!", color="#5E81AC"];
        e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        //e4 [pos="0.4, 1.0!", color="#5E81AC"];
        //e5 [pos="1.4, 1.0!" color="#5E81AC"];
        //e6 [pos="1.8, 1.4!", color="#5E81AC"];
        //e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        e8 [pos="0.9, 1.5!", color="#5E81AC"];
        e9 [pos="1.6, 1.7!", color="#5E81AC"];
        e10 [pos="1.3, 1.9!", color="#5E81AC"];
        e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e0 -- e3 [color="#5E81AC"];
        e3 -- e8 [color="#5E81AC"];
        e8 -- e0 [color="#5E81AC"];
        e8 -- e11 [color="#5E81AC"];
        e11 -- e3 [color="#5E81AC"];
        e1 -- e2 [color="#5E81AC"];
        e2 -- e9 [color="#5E81AC"];
        e9 -- e1 [color="#5E81AC"];
        e9 -- e10 [color="#5E81AC"];
        e10 -- e2 [color="#5E81AC"];

        }
      
      Case 10
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case11 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=black]; 
        v1 [pos="1.4,0.5!", fillcolor=white];
        v2 [pos="1.8,0.9!", fillcolor=black]; 
        v3 [pos="0.8,0.9!", fillcolor=black]; 
        v4 [pos="0.4,1.5!", fillcolor=white];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=black]; 
        v7 [pos="0.8,1.9!", fillcolor=white];
        
        // bottom edges
        e0 [pos="0.9, 0.5!", color="#5E81AC"];
        e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        //e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        e4 [pos="0.4, 1.0!", color="#5E81AC"];
        //e5 [pos="1.4, 1.0!" color="#5E81AC"];
        //e6 [pos="1.8, 1.4!", color="#5E81AC"];
        e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        //e8 [pos="0.9, 1.5!", color="#5E81AC"];
        e9 [pos="1.6, 1.7!", color="#5E81AC"];
        e10 [pos="1.3, 1.9!", color="#5E81AC"];
        //e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e0 -- e1 [color="#5E81AC"];
        e1 -- e9 [color="#5E81AC"];
        e9 -- e0 [color="#5E81AC"];
        e0 -- e4 [color="#5E81AC"];
        e4 -- e7 [color="#5E81AC"];
        e7 -- e0 [color="#5E81AC"];
        e7 -- e10 [color="#5E81AC"];
        e10 -- e9 [color="#5E81AC"];
        e9 -- e7 [color="#5E81AC"];

        }
      
      Case 11
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case12 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=white]; 
        v1 [pos="1.4,0.5!", fillcolor=black];
        v2 [pos="1.8,0.9!", fillcolor=black]; 
        v3 [pos="0.8,0.9!", fillcolor=black]; 
        v4 [pos="0.4,1.5!", fillcolor=black];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=white]; 
        v7 [pos="0.8,1.9!", fillcolor=white];
        
        // bottom edges
        e0 [pos="0.9, 0.5!", color="#5E81AC"];
        //e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        e4 [pos="0.4, 1.0!", color="#5E81AC"];
        e5 [pos="1.4, 1.0!" color="#5E81AC"];
        e6 [pos="1.8, 1.4!", color="#5E81AC"];
        e7 [pos="0.8, 1.3!", color="#5E81AC"];

        // top edges
        e8 [pos="0.9, 1.5!", color="#5E81AC"];
        //e9 [pos="1.6, 1.7!", color="#5E81AC"];
        //e10 [pos="1.3, 1.9!", color="#5E81AC"];
        e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e0 -- e3 [color="#5E81AC"];
        e3 -- e5 [color="#5E81AC"];
        e5 -- e6 [color="#5E81AC"];
        e6 -- e7 [color="#5E81AC"];
        e7 -- e3 [color="#5E81AC"];
        e0 -- e5 [color="#5E81AC"];
        e5 -- e7 [color="#5E81AC"];
        e4 -- e11 [color="#5E81AC"];
        e11 -- e8 [color="#5E81AC"];
        e8 -- e4 [color="#5E81AC"];
        }
      
      Case 12
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case13 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=black]; 
        v1 [pos="1.4,0.5!", fillcolor=white];
        v2 [pos="1.8,0.9!", fillcolor=black]; 
        v3 [pos="0.8,0.9!", fillcolor=white]; 
        v4 [pos="0.4,1.5!", fillcolor=white];
        v5 [pos="1.4,1.5!", fillcolor=black]; 
        v6 [pos="1.8,1.9!", fillcolor=white]; 
        v7 [pos="0.8,1.9!", fillcolor=black];
        
        // bottom edges
        e0 [pos="0.9, 0.5!", color="#5E81AC"];
        e1 [pos="1.6, 0.7!", color="#5E81AC"];
        e2 [pos="1.3, 0.9!", color="#5E81AC"];
        e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        e4 [pos="0.4, 1.0!", color="#5E81AC"];
        e5 [pos="1.4, 1.1!" color="#5E81AC"];
        e6 [pos="1.8, 1.4!", color="#5E81AC"];
        e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        e8 [pos="1.0, 1.5!", color="#5E81AC"];
        e9 [pos="1.6, 1.7!", color="#5E81AC"];
        e10 [pos="1.3, 1.9!", color="#5E81AC"];
        e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e0 -- e3 [color="#5E81AC"];
        e3 -- e4 [color="#5E81AC"];
        e4 -- e0 [color="#5E81AC"];
        e1 -- e2 [color="#5E81AC"];
        e2 -- e6 [color="#5E81AC"];
        e6 -- e1 [color="#5E81AC"];
        e5 -- e8 [color="#5E81AC"];
        e8 -- e9 [color="#5E81AC"];
        e9 -- e5 [color="#5E81AC"];
        e7 -- e10 [color="#5E81AC"];
        e10 -- e11 [color="#5E81AC"];
        e11 -- e7 [color="#5E81AC"];

        }
      
      Case 13
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case14 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v0 [pos="0.4,0.5!", fillcolor=white]; 
        v1 [pos="1.4,0.5!", fillcolor=black];
        v2 [pos="1.8,0.9!", fillcolor=black]; 
        v3 [pos="0.8,0.9!", fillcolor=black]; 
        v4 [pos="0.4,1.5!", fillcolor=white];
        v5 [pos="1.4,1.5!", fillcolor=white]; 
        v6 [pos="1.8,1.9!", fillcolor=white]; 
        v7 [pos="0.8,1.9!", fillcolor=black];
        
        // bottom edges
        e0 [pos="0.9, 0.5!", color="#5E81AC"];
        //e1 [pos="1.6, 0.7!", color="#5E81AC"];
        //e2 [pos="1.3, 0.9!", color="#5E81AC"];
        e3 [pos="0.6, 0.7!", color="#5E81AC"];

        // middle edges
        //e4 [pos="0.4, 1.0!", color="#5E81AC"];
        e5 [pos="1.4, 0.75!" color="#5E81AC"];
        e6 [pos="1.8, 1.6!", color="#5E81AC"];
        //e7 [pos="0.8, 1.4!", color="#5E81AC"];

        // top edges
        //e8 [pos="0.9, 1.5!", color="#5E81AC"];
        //e9 [pos="1.6, 1.7!", color="#5E81AC"];
        e10 [pos="1.3, 1.9!", color="#5E81AC"];
        e11 [pos="0.6, 1.7!", color="#5E81AC"];
        
        v0 -- v1 [style=solid];
        v1 -- v2 [style=solid];
        v2 -- v3 [style=dotted];
        v3 -- v0 [style=dotted];
        v0 -- v4 [style=solid];
        v1 -- v5 [style=solid];
        v2 -- v6 [style=solid];
        v3 -- v7 [style=dotted];
        v4 -- v5 [style=solid];
        v5 -- v6 [style=solid];
        v6 -- v7 [style=solid];
        v7 -- v4 [style=solid];

        e0 -- e3 [color="#5E81AC"];
        e3 -- e11 [color="#5E81AC"];
        e11 -- e0 [color="#5E81AC"];
        e0 -- e5 [color="#5E81AC"];
        e5 -- e6 [color="#5E81AC"];
        e6 -- e11 [color="#5E81AC"];
        e11 -- e10 [color="#5E81AC"];
        e10 -- e6 [color="#5E81AC"];
        e0 -- e6 [color="#5E81AC"];

        }
      
      Case 14
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph legend {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,1.!", fillcolor=black]; 
        v1 [pos="0.,0.75!", fillcolor=white];
        Inside [shape=plaintext, pos="0.6,1.!"]; 
        Outside [shape=plaintext, pos="0.6,0.75!"];
        blank [pos="0.,0!", color=white];

        }

Marching Cubes 33
^^^^^^^^^^^^^^^^^
:cite:t:`Chernyaev1995`

See also :func:`~mymesh.contour.MarchingCubes`

One of the main shortcomings of the marching cubes algorithm is that there exist
several ambiguous cases - situations in which there are multiple possible 
triangulations that could be used to separate the inside from the outside of the
surface. For example, the opposite corners of Case 4 could be connected through
the center of the cube. These ambiguities can lead to topological errors and 
non-watertight surfaces. Several methods have been proposed to resolve these 
ambiguities, including an extend lookup table that has 33 entries, rather than
15 :cite:`Chernyaev1995`. Additional tests are used to test whether diagonal
nodes are connected or separated to choose the correct triangulation.


Marching Tetrahedra
-------------------
:cite:t:`Bloomenthal1994`

See also :func:`~mymesh.contour.MarchingTetrahedra`

Another approach to resolve the ambiguities of marching cubes is to partition
tetrahedra, rather than cubes.

Mixed-Surface Lookup Table

.. grid:: 3 6 6 6
    :outline:

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case0 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=white]; 
        v7 [pos="0.6,-0.2!", fillcolor=white];
        v8 [pos="0.9,0.2!", fillcolor=white]; 
        v9 [pos="0.4,0.8!", fillcolor=white];
        
        //v0 [pos="0.3,-0.1!", color="#5E81AC"];
        //v1 [pos="0.75,0.!", color="#5E81AC"];
        //v2 [pos="0.45,0.1!", color="#5E81AC"];
        //v3 [pos="0.5,0.3!", color="#5E81AC"];
        //v4 [pos="0.65,0.5!", color="#5E81AC"];
        //v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];

        }
      
      Case 0

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case1 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=white]; 
        v7 [pos="0.6,-0.2!", fillcolor=white];
        v8 [pos="0.9,0.2!", fillcolor=white]; 
        v9 [pos="0.4,0.8!", fillcolor=black];
        
        //v0 [pos="0.3,-0.1!", color="#5E81AC"];
        //v1 [pos="0.75,0.!", color="#5E81AC"];
        //v2 [pos="0.45,0.1!", color="#5E81AC"];
        v3 [pos="0.5,0.3!", color="#5E81AC"];
        v4 [pos="0.65,0.5!", color="#5E81AC"];
        v5 [pos="0.2,0.4!", color="#5E81AC"];
        

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v3 -- v5 [color="#5E81AC"];
        v5 -- v4 [color="#5E81AC"];
        v4 -- v3 [color="#5E81AC"];

        }
      
      Case 1
    
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case2 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=white]; 
        v7 [pos="0.6,-0.2!", fillcolor=white];
        v8 [pos="0.9,0.2!", fillcolor=black]; 
        v9 [pos="0.4,0.8!", fillcolor=white];
        
        //v0 [pos="0.3,-0.1!", color="#5E81AC"];
        v1 [pos="0.75,0.!", color="#5E81AC"];
        v2 [pos="0.45,0.1!", color="#5E81AC"];
        //v3 [pos="0.5,0.3!", color="#5E81AC"];
        v4 [pos="0.65,0.5!", color="#5E81AC"];
        //v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v4 -- v2 [color="#5E81AC"];
        v2 -- v1 [color="#5E81AC"];
        v1 -- v4 [color="#5E81AC"];

        }
      
      Case 2

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case3 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=white]; 
        v7 [pos="0.6,-0.2!", fillcolor=white];
        v8 [pos="0.9,0.2!", fillcolor=black]; 
        v9 [pos="0.4,0.8!", fillcolor=black];
        
        //v0 [pos="0.3,-0.1!", color="#5E81AC"];
        v1 [pos="0.75,0.!", color="#5E81AC"];
        v2 [pos="0.45,0.1!", color="#5E81AC"];
        v3 [pos="0.5,0.3!", color="#5E81AC"];
        //v4 [pos="0.65,0.5!", color="#5E81AC"];
        v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v1 -- v3 [color="#5E81AC"];
        v3 -- v5 [color="#5E81AC"];
        v5 -- v2 [color="#5E81AC"];
        v2 -- v1 [color="#5E81AC"];

        }
      
      Case 3

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case4 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=white]; 
        v7 [pos="0.6,-0.2!", fillcolor=black];
        v8 [pos="0.9,0.2!", fillcolor=white]; 
        v9 [pos="0.4,0.8!", fillcolor=white];
        
        v0 [pos="0.3,-0.1!", color="#5E81AC"];
        v1 [pos="0.75,0.!", color="#5E81AC"];
        //v2 [pos="0.45,0.1!", color="#5E81AC"];
        v3 [pos="0.5,0.3!", color="#5E81AC"];
        //v4 [pos="0.65,0.5!", color="#5E81AC"];
        //v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v0 -- v3 [color="#5E81AC"];
        v3 -- v1 [color="#5E81AC"];
        v1 -- v0 [color="#5E81AC"];

        }
      
      Case 4

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case5 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=white]; 
        v7 [pos="0.6,-0.2!", fillcolor=black];
        v8 [pos="0.9,0.2!", fillcolor=white]; 
        v9 [pos="0.4,0.8!", fillcolor=black];
        
        v0 [pos="0.3,-0.1!", color="#5E81AC"];
        v1 [pos="0.75,0.!", color="#5E81AC"];
        //v2 [pos="0.45,0.1!", color="#5E81AC"];
        //v3 [pos="0.5,0.3!", color="#5E81AC"];
        v4 [pos="0.65,0.5!", color="#5E81AC"];
        v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v1 -- v0 [color="#5E81AC"];
        v0 -- v5 [color="#5E81AC"];
        v5 -- v4 [color="#5E81AC"];
        v4 -- v1 [color="#5E81AC"];

        }
      
      Case 5
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case6 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=white]; 
        v7 [pos="0.6,-0.2!", fillcolor=black];
        v8 [pos="0.9,0.2!", fillcolor=black]; 
        v9 [pos="0.4,0.8!", fillcolor=white];
        
        v0 [pos="0.3,-0.1!", color="#5E81AC"];
        //v1 [pos="0.75,0.!", color="#5E81AC"];
        v2 [pos="0.45,0.1!", color="#5E81AC"];
        v3 [pos="0.5,0.3!", color="#5E81AC"];
        v4 [pos="0.65,0.5!", color="#5E81AC"];
        //v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v0 -- v3 [color="#5E81AC"];
        v3 -- v4 [color="#5E81AC"];
        v4 -- v2 [color="#5E81AC"];
        v2 -- v0 [color="#5E81AC"];

        }
      
      Case 6
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case7 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=white]; 
        v7 [pos="0.6,-0.2!", fillcolor=black];
        v8 [pos="0.9,0.2!", fillcolor=black]; 
        v9 [pos="0.4,0.8!", fillcolor=black];
        
        v0 [pos="0.3,-0.1!", color="#5E81AC"];
        //v1 [pos="0.75,0.!", color="#5E81AC"];
        v2 [pos="0.45,0.1!", color="#5E81AC"];
        //v3 [pos="0.5,0.3!", color="#5E81AC"];
        //v4 [pos="0.65,0.5!", color="#5E81AC"];
        v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v0 -- v5 [color="#5E81AC"];
        v5 -- v2 [color="#5E81AC"];
        v2 -- v0 [color="#5E81AC"];

        }
      
      Case 7
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case8 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=black]; 
        v7 [pos="0.6,-0.2!", fillcolor=white];
        v8 [pos="0.9,0.2!", fillcolor=white]; 
        v9 [pos="0.4,0.8!", fillcolor=white];
        
        v0 [pos="0.3,-0.1!", color="#5E81AC"];
        //v1 [pos="0.75,0.!", color="#5E81AC"];
        v2 [pos="0.45,0.1!", color="#5E81AC"];
        //v3 [pos="0.5,0.3!", color="#5E81AC"];
        //v4 [pos="0.65,0.5!", color="#5E81AC"];
        v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v0 -- v2 [color="#5E81AC"];
        v2 -- v5 [color="#5E81AC"];
        v5 -- v0 [color="#5E81AC"];

        }
      
      Case 8
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case9 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=black]; 
        v7 [pos="0.6,-0.2!", fillcolor=white];
        v8 [pos="0.9,0.2!", fillcolor=white]; 
        v9 [pos="0.4,0.8!", fillcolor=black];
        
        v0 [pos="0.3,-0.1!", color="#5E81AC"];
        //v1 [pos="0.75,0.!", color="#5E81AC"];
        v2 [pos="0.45,0.1!", color="#5E81AC"];
        v3 [pos="0.5,0.3!", color="#5E81AC"];
        v4 [pos="0.65,0.5!", color="#5E81AC"];
        //v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v0 -- v2 [color="#5E81AC"];
        v2 -- v4 [color="#5E81AC"];
        v4 -- v3 [color="#5E81AC"];
        v3 -- v0 [color="#5E81AC"];

        }
      
      Case 9
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case10 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=black]; 
        v7 [pos="0.6,-0.2!", fillcolor=white];
        v8 [pos="0.9,0.2!", fillcolor=black]; 
        v9 [pos="0.4,0.8!", fillcolor=white];
        
        v0 [pos="0.3,-0.1!", color="#5E81AC"];
        v1 [pos="0.75,0.!", color="#5E81AC"];
        //v2 [pos="0.45,0.1!", color="#5E81AC"];
        //v3 [pos="0.5,0.3!", color="#5E81AC"];
        v4 [pos="0.65,0.5!", color="#5E81AC"];
        v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v0 -- v1 [color="#5E81AC"];
        v1 -- v4 [color="#5E81AC"];
        v4 -- v5 [color="#5E81AC"];
        v5 -- v0 [color="#5E81AC"];

        }
      
      Case 10
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case11 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=black]; 
        v7 [pos="0.6,-0.2!", fillcolor=white];
        v8 [pos="0.9,0.2!", fillcolor=black]; 
        v9 [pos="0.4,0.8!", fillcolor=black];
        
        v0 [pos="0.3,-0.1!", color="#5E81AC"];
        v1 [pos="0.75,0.!", color="#5E81AC"];
        //v2 [pos="0.45,0.1!", color="#5E81AC"];
        v3 [pos="0.5,0.3!", color="#5E81AC"];
        //v4 [pos="0.65,0.5!", color="#5E81AC"];
        //v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v0 -- v1 [color="#5E81AC"];
        v1 -- v3 [color="#5E81AC"];
        v3 -- v0 [color="#5E81AC"];

        }
      
      Case 11
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case12 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=black]; 
        v7 [pos="0.6,-0.2!", fillcolor=black];
        v8 [pos="0.9,0.2!", fillcolor=white]; 
        v9 [pos="0.4,0.8!", fillcolor=white];
        
        //v0 [pos="0.3,-0.1!", color="#5E81AC"];
        v1 [pos="0.75,0.!", color="#5E81AC"];
        v2 [pos="0.45,0.1!", color="#5E81AC"];
        v3 [pos="0.5,0.3!", color="#5E81AC"];
        //v4 [pos="0.65,0.5!", color="#5E81AC"];
        v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v1 -- v2 [color="#5E81AC"];
        v2 -- v5 [color="#5E81AC"];
        v5 -- v3 [color="#5E81AC"];
        v3 -- v1 [color="#5E81AC"];

        }
      
      Case 12
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case6 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=black]; 
        v7 [pos="0.6,-0.2!", fillcolor=black];
        v8 [pos="0.9,0.2!", fillcolor=white]; 
        v9 [pos="0.4,0.8!", fillcolor=black];
        
        //v0 [pos="0.3,-0.1!", color="#5E81AC"];
        v1 [pos="0.75,0.!", color="#5E81AC"];
        v2 [pos="0.45,0.1!", color="#5E81AC"];
        //v3 [pos="0.5,0.3!", color="#5E81AC"];
        v4 [pos="0.65,0.5!", color="#5E81AC"];
        //v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v1 -- v2 [color="#5E81AC"];
        v2 -- v4 [color="#5E81AC"];
        v4 -- v1 [color="#5E81AC"];

        }
      
      Case 13
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case14 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=black]; 
        v7 [pos="0.6,-0.2!", fillcolor=black];
        v8 [pos="0.9,0.2!", fillcolor=black]; 
        v9 [pos="0.4,0.8!", fillcolor=white];
        
        //v0 [pos="0.3,-0.1!", color="#5E81AC"];
        //v1 [pos="0.75,0.!", color="#5E81AC"];
        //v2 [pos="0.45,0.1!", color="#5E81AC"];
        v3 [pos="0.5,0.3!", color="#5E81AC"];
        v4 [pos="0.65,0.5!", color="#5E81AC"];
        v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        
        v3 -- v4 [color="#5E81AC"];
        v4 -- v5 [color="#5E81AC"];
        v5 -- v3 [color="#5E81AC"];

        }
      
      Case 14
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph case15 {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];
        
        v6 [pos="0.,0.!", fillcolor=black]; 
        v7 [pos="0.6,-0.2!", fillcolor=black];
        v8 [pos="0.9,0.2!", fillcolor=black]; 
        v9 [pos="0.4,0.8!", fillcolor=black];
        
        //v0 [pos="0.3,-0.1!", color="#5E81AC"];
        //v1 [pos="0.75,0.!", color="#5E81AC"];
        //v2 [pos="0.45,0.1!", color="#5E81AC"];
        //v3 [pos="0.5,0.3!", color="#5E81AC"];
        //v4 [pos="0.65,0.5!", color="#5E81AC"];
        //v5 [pos="0.2,0.4!", color="#5E81AC"];

        v6 -- v7 [style=solid];
        v7 -- v8 [style=solid];
        v8 -- v6 [style=dotted];
        v6 -- v9 [style=solid];
        v7 -- v9 [style=solid];
        v8 -- v9 [style=solid];
        

        }
      
      Case 15
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph legend {  
        node [shape=point, fontname="source code pro"];
        edge [style=solid, penwidth=2];

        v0 [pos="0.,1.!", fillcolor=black]; 
        v1 [pos="0.,0.75!", fillcolor=white];
        Inside [shape=plaintext, pos="0.6,1.!"]; 
        Outside [shape=plaintext, pos="0.6,0.75!"];
        blank [pos="0.,0!", color=white];

        }

A similar lookup table can be constructed for volume elements (tetrahedra and wedges). Quadrilateral and wedge elements can be decomposed into triangles and tetrahedra, respectively, however care must be taken to ensure the edges of adjacent wedges are partitioned consistently so that they are split along the same diagonal.