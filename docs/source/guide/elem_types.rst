Element Types
=============

MyMesh infers the element type by the length of each element's node connectivity. 
This allows for flexible handling of mixed element meshes, however there is an 
ambiguity between the 2D quadrilateral element and the 3D tetrahedral element. 
By default, MyMesh assumes 4-node elements are tetrahedra in the absence of 
clear indicator that they are quads. When relevant, many functions allow quads 
to be specifically designated. 

+---------------+---------+-----------------+--------------------+
| Element       | Type    | Number of Nodes | Node Connectivity  |
+===============+=========+=================+====================+
| Edge          | Line    | 2               | [0, 1]             |
+---------------+---------+-----------------+--------------------+
| Triangle      | Surface | 3               | [0, 1, 2]          |
+---------------+---------+-----------------+--------------------+
| Quadrilateral | Surface | 4               | [0, 1, 2, 3]       |
+---------------+---------+-----------------+--------------------+
| Tetrahedron   | Volume  | 4               | [0, 1, 2, 3]       |
+---------------+---------+-----------------+--------------------+
| Pyramid       | Volume  | 5               | [0, 1, 2, 3, 4]    |
+---------------+---------+-----------------+--------------------+
| Wedge         | Volume  | 6               | [0, 1, 2, 3, 4, 5] |
+---------------+---------+-----------------+--------------------+
| Hexahedron    | Volume  | 8               |                    |
+---------------+---------+-----------------+--------------------+


.. grid:: 1 2 2 2
    :outline:

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph tri {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="1,0.1!"]; 
        2 [pos="0.5,0.8!"]; 

        0 -- 1; 
        1 -- 2; 
        2 -- 0; 

        label0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        label2 [label="2", pos="0.5,0.9!", shape=none, fontname="source code pro"] 
        }

      Triangle

    .. grid-item::
      :child-align: center
      
      .. graphviz::

        graph quad {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="1,0.1!"]; 
        2 [pos="0.9,0.9!"]; 
        3 [pos="-0.1,1.0!"]; 

        0 -- 1;
        1 -- 2; 
        2 -- 3; 
        3 -- 0; 

        label0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        label2 [label="2", pos="1.0,1.0!", shape=none, fontname="source code pro"] 
        label3 [label="3", pos="-0.1,1.1!", shape=none, fontname="source code pro"] 

        }

      Quadrilateral

.. grid:: 1 2 2 2
    :outline:

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph tet {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="1,0.1!"]; 
        2 [pos="0.9,0.9!"]; 
        3 [pos="-0.1,1.0!"]; 

        0 -- 1;
        1 -- 2; 
        2 -- 0 [style=dotted]; 
        0 -- 3;
        1 -- 3;
        2 -- 3; 

        label0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        label2 [label="2", pos="1.0,1.0!", shape=none, fontname="source code pro"] 
        label3 [label="3", pos="-0.1,1.1!", shape=none, fontname="source code pro"] 

        }
      
      Tetrahedron

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph tet10 {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="1,0.1!"]; 
        2 [pos="0.9,0.9!"]; 
        3 [pos="-0.1,1.0!"]; 

        4 [pos=".5,0.05!"];
        5 [pos=".95,0.5!"];
        6 [pos=".45,0.45!"];
        7 [pos="-.05, 0.5!"];
        8 [pos=".55, 0.46!"];
        9 [pos=".4, 0.95!"];

        0 -- 1;
        1 -- 2; 
        2 -- 0 [style=dotted]; 
        0 -- 3;
        1 -- 3;
        2 -- 3; 

        label0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        label2 [label="2", pos="1.0,1.0!", shape=none, fontname="source code pro"] 
        label3 [label="3", pos="-0.1,1.1!", shape=none, fontname="source code pro"] 

        label4 [label="4", pos=".5,-.075!", shape=none, fontname="source code pro"] 
        label5 [label="5", pos="1.05,.5!", shape=none, fontname="source code pro"] 
        label6 [label="6", pos=".3,.4!", shape=none, fontname="source code pro"] 
        label7 [label="7", pos="-0.15,.5!", shape=none, fontname="source code pro"] 
        label8 [label="8", pos=".65,.45!", shape=none, fontname="source code pro"] 
        label9 [label="9", pos="0.4,1.05!", shape=none, fontname="source code pro"] 
        }
      
      Quadratic Tetrahedron  

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph pyr {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos=".3,0!"];
        1 [pos="0.8,0.3!"]; 
        2 [pos="0.55,0.5!"]; 
        3 [pos="0,0.4!"];
        4 [pos=".4,1!"]

        0 -- 1;
        1 -- 2 [style=dotted]; 
        2 -- 3 [style=dotted]; 
        3 -- 0; 
        0 -- 4;
        1 -- 4;
        2 -- 4 [style=dotted];
        3 -- 4;

        label0 [label="0", pos="0.3,-0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="0.9,0.3!", shape=none, fontname="source code pro"] 
        label2 [label="2", pos="0.55,0.35!", shape=none, fontname="source code pro"] 
        label3 [label="3", pos="-0.1,0.4!", shape=none, fontname="source code pro"] 
        label4 [label="4", pos="0.4,1.1!", shape=none, fontname="source code pro"] 

        }

      Pyramid

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph wdg {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        
        0 [pos="0,0!"];
        1 [pos="1,1!"]; 
        2 [pos="0.1,0.8!"]; 
        3 [pos="0,1.2!"];
        4 [pos="1,2.2!"]; 
        5 [pos=".1,2.0!"]; 


        0 -- 1; 
        1 -- 2 [style=dotted]; 
        2 -- 0 [style=dotted]; 
        3 -- 4; 
        4 -- 5; 
        5 -- 3; 
        0 -- 3;
        1 -- 4;
        2 -- 5 [style=dotted];

        label0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="1.15,1!", shape=none, fontname="source code pro"] 
        label2 [label="2", pos="0.2,0.65!", shape=none, fontname="source code pro"] 

        label3 [label="3", pos="-.1,1.3!", shape=none, fontname="source code pro"] 
        label4 [label="4", pos="1,2.3!", shape=none, fontname="source code pro"] 
        label5 [label="5", pos="0.1,2.1!", shape=none, fontname="source code pro"] 

        }

      Wedge

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph quad {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="1,0.1!"]; 
        2 [pos="1.6,0.6!"]; 
        3 [pos=".6,0.5!"];
        4 [pos="-0.1,1.0!"];
        5 [pos="0.9,0.9!"];  
        6 [pos="1.5,1.4!"]; 
        7 [pos="0.5,1.5!"]; 

        0 -- 4;
        1 -- 5; 
        2 -- 6; 
        3 -- 7 [style=dotted]; 
        4 -- 5;
        5 -- 6;
        6 -- 7;
        7 -- 4;
        0 -- 1;
        1 -- 2;
        2 -- 3 [style=dotted];
        3 -- 0 [style=dotted];

        label0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        label2 [label="2", pos="1.75,0.6!", shape=none, fontname="source code pro"] 
        label3 [label="3", pos=".5,0.6!", shape=none, fontname="source code pro"] 
        label4 [label="4", pos="-0.15,1.1!", shape=none, fontname="source code pro"] 
        label5 [label="5", pos="0.85,1.1!", shape=none, fontname="source code pro"] 
        label6 [label="6", pos="1.6,1.5!", shape=none, fontname="source code pro"] 
        label7 [label="7", pos="0.4,1.6!", shape=none, fontname="source code pro"] 
        }

      Hexahedron





