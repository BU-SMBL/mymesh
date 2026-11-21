Element Types
=============

MyMesh infers the element type by the length of each element's node connectivity,
allowing for flexible handling of mixed element meshes. Some ambiguities exist
between meshes of different Types (``line``, ``surf``, ``vol``), for example
between a 4-node quadrilateral and a 4-node tetrahedron. These ambiguities get 
resolved by the :attr:`mymesh.mesh.Type` attribute or an optional ``Type`` input
for relevant functions (see also :func:`mymesh.utils.identify_type`).

MyMesh primarily uses linear (first order) elements (e.g. 3 node tris, 4 node 
tets, etc.) but also has limited support for quadratic elements (e.g. 6 node 
tris, 10 node tets, etc.). Meshes can be converted to/from quadratic elements 
(see :mod:`mymesh.converter`), but not all functions are set up to handle meshes 
with quadratic elements


+-----------------------------------+--------------------+-----------------+
| Element                           | Type               | Number of Nodes |
+===================================+====================+=================+
| Edge (``edge``)                   | Line (``line``)    | 2               |
+-----------------------------------+--------------------+-----------------+
| Quadratic Edge (``edge3``)        | Line (``line``)    | 3               |
+-----------------------------------+--------------------+-----------------+
| Triangle (``tri``)                | Surface (``surf``) | 3               |
+-----------------------------------+--------------------+-----------------+
| Quadratic Triangle (``tri6``)     | Surface (``surf``) | 6               |
+-----------------------------------+--------------------+-----------------+
| Quadrilateral (``quad``)          | Surface (``surf``) | 4               |
+-----------------------------------+--------------------+-----------------+
|Quadratic Quadrilateral (``quad8``)| Surface (``surf``) | 8               |
+-----------------------------------+--------------------+-----------------+
| Tetrahedron (``tet``)             | Volume (``vol``)   | 4               |
+-----------------------------------+--------------------+-----------------+
| Quadratic Tetrahedron (``tet10``) | Volume (``vol``)   | 10              |
+-----------------------------------+--------------------+-----------------+
| Pyramid (``pyr``)                 | Volume (``vol``)   | 5               |
+-----------------------------------+--------------------+-----------------+
| Quadratic Pyramid (``pyr13``)     | Volume (``vol``)   | 13              |
+-----------------------------------+--------------------+-----------------+
| Wedge (``wdg``)                   | Volume (``vol``)   | 6               |
+-----------------------------------+--------------------+-----------------+
| Quadratic Wedge (``wdg15``)       | Volume  (``vol``)  | 15              |
+-----------------------------------+--------------------+-----------------+
| Hexahedron (``hex``)              | Volume (``vol``)   | 8               |
+-----------------------------------+--------------------+-----------------+
| Quadratic Hexahedron (``hex20``)  | Volume (``vol``)   | 20              |
+-----------------------------------+--------------------+-----------------+


.. grid:: 1 2 2 2
    :outline:

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph edge2 {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="1,0.1!"]; 


        0 -- 1; 

        label0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        }

      Edge

    .. grid-item::
      :child-align: center
      
      .. graphviz::

        graph edge3 {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="0.5,.05!", color="#5E81AC"];
        2 [pos="1,0.1!"]; 


        0 -- 2; 

        label0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="0.5,-0.1!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label2 [label="2", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        }

      Quadratic Edge

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

        graph tri6 {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="1,0.1!"]; 
        2 [pos="0.5,0.8!"]; 
        3 [pos="0.5,0.05!", color="#5E81AC"];
        4 [pos="0.75,0.45!", color="#5E81AC"];
        5 [pos="0.25,0.4!", color="#5E81AC"];

        0 -- 1; 
        1 -- 2; 
        2 -- 0; 

        label0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        label2 [label="2", pos="0.5,0.9!", shape=none, fontname="source code pro"] 
        label3 [label="3", pos="0.5,-0.1!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label4 [label="4", pos="0.8,0.55!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label5 [label="5", pos="0.15,0.4!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        }

      Quadratic Triangle

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
        4 [pos="0.5,0.05!", color="#5E81AC"]; 
        5 [pos="0.95,0.5!", color="#5E81AC"]; 
        6 [pos="0.4,0.95!", color="#5E81AC"]; 
        7 [pos="-0.05,0.5!", color="#5E81AC"]; 

        0 -- 1;
        1 -- 2; 
        2 -- 3; 
        3 -- 0; 

        label0 [label="0", pos="0,-0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="1,-0.05!", shape=none, fontname="source code pro"] 
        label2 [label="2", pos="1.0,1.0!", shape=none, fontname="source code pro"] 
        label3 [label="3", pos="-0.1,1.1!", shape=none, fontname="source code pro"] 
        label4 [label="4", pos="0.5,-0.1!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label5 [label="5", pos="1.1,0.5!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label6 [label="6", pos="0.4,1.1!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label7 [label="7", pos="-0.15,0.5!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 

        }

      Quadratic Quadrilateral

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

        4 [pos=".5,0.05!",color="#5E81AC"];
        5 [pos=".95,0.5!",color="#5E81AC"];
        6 [pos=".45,0.45!",color="#5E81AC"];
        7 [pos="-.05, 0.5!",color="#5E81AC"];
        8 [pos=".55, 0.46!",color="#5E81AC"];
        9 [pos=".4, 0.95!",color="#5E81AC"];

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

        label4 [label="4", pos=".5,-.075!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label5 [label="5", pos="1.05,.5!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label6 [label="6", pos=".3,.4!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label7 [label="7", pos="-0.15,.5!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label8 [label="8", pos=".65,.45!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label9 [label="9", pos="0.4,1.05!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
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
        label2 [label="2", pos="0.45,0.35!", shape=none, fontname="source code pro"] 
        label3 [label="3", pos="-0.1,0.4!", shape=none, fontname="source code pro"] 
        label4 [label="4", pos="0.4,1.1!", shape=none, fontname="source code pro"] 

        }

      Pyramid
    
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph pyr13 {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos=".3,0!"];
        1 [pos="0.8,0.3!"]; 
        2 [pos="0.55,0.5!"]; 
        3 [pos="0,0.4!"];
        4 [pos=".4,1!"];

        5 [pos="0.55,0.15!", color="#5E81AC"];
        6 [pos="0.675,0.4!", color="#5E81AC"];
        7 [pos="0.275,0.45!", color="#5E81AC"];
        8 [pos="0.15,0.2!", color="#5E81AC"];
        9 [pos="0.35,0.5!", color="#5E81AC"];
        10 [pos="0.6,0.65!", color="#5E81AC"];
        11 [pos="0.475,0.75!", color="#5E81AC"];
        12 [pos="0.2,0.7!", color="#5E81AC"];

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
        label2 [label="2", pos="0.45,0.35!", shape=none, fontname="source code pro"] 
        label3 [label="3", pos="-0.1,0.4!", shape=none, fontname="source code pro"] 
        label4 [label="4", pos="0.4,1.1!", shape=none, fontname="source code pro"] 

        label5 [label="5", pos="0.65,0.05!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label6 [label="6", pos="0.6,0.3!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label7 [label="7", pos="0.225,0.55!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label8 [label="8", pos="0.05,0.15!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label9 [label="9", pos="0.425,0.55!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label10 [label="10", pos="0.75,0.65!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label11 [label="11", pos="0.55,0.85!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label12 [label="12", pos="0.05,0.7!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        }

      Quadratic Pyramid

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph wdg {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        
        0 [pos="0.4,0.3!"];
        1 [pos="1,1!"]; 
        2 [pos="0.1,0.8!"]; 
        3 [pos="0.4,1.1!"];
        4 [pos="1,1.8!"]; 
        5 [pos=".1,1.6!"]; 


        0 -- 1; 
        1 -- 2 [style=dotted]; 
        2 -- 0; 
        3 -- 4; 
        4 -- 5; 
        5 -- 3; 
        0 -- 3;
        1 -- 4;
        2 -- 5;

        label0 [label="0", pos="0.3,0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="1.15,1!", shape=none, fontname="source code pro"] 
        label2 [label="2", pos="0.05,0.65!", shape=none, fontname="source code pro"] 

        label3 [label="3", pos="0.4,1.25!", shape=none, fontname="source code pro"] 
        label4 [label="4", pos="1,1.9!", shape=none, fontname="source code pro"] 
        label5 [label="5", pos="0.1,1.7!", shape=none, fontname="source code pro"] 

        }

      Wedge
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph wdg15 {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        
        0 [pos="0.4,0.3!"];
        1 [pos="1,1!"]; 
        2 [pos="0.1,0.8!"]; 
        3 [pos="0.4,1.1!"];
        4 [pos="1,1.8!"]; 
        5 [pos=".1,1.6!"]; 

        6 [pos="0.7,0.65!", color="#5E81AC"];
        7 [pos="0.55,0.9!", color="#5E81AC"];
        8 [pos="0.25,0.55!", color="#5E81AC"];
        9 [pos="0.7,1.45!", color="#5E81AC"];
        10 [pos="0.55,1.7!", color="#5E81AC"];
        11 [pos="0.25,1.35!", color="#5E81AC"];

        12 [pos="0.4,0.7!", color="#5E81AC"];
        13 [pos="1,1.4!", color="#5E81AC"];
        14 [pos="0.1,1.2!", color="#5E81AC"];


        0 -- 1; 
        1 -- 2 [style=dotted]; 
        2 -- 0; 
        3 -- 4; 
        4 -- 5; 
        5 -- 3; 
        0 -- 3;
        1 -- 4;
        2 -- 5;

        label0 [label="0", pos="0.3,0.15!", shape=none, fontname="source code pro"] 
        label1 [label="1", pos="1.15,1!", shape=none, fontname="source code pro"] 
        label2 [label="2", pos="0.05,0.65!", shape=none, fontname="source code pro"] 

        label3 [label="3", pos="0.4,1.25!", shape=none, fontname="source code pro"] 
        label4 [label="4", pos="1,1.9!", shape=none, fontname="source code pro"] 
        label5 [label="5", pos="0.1,1.7!", shape=none, fontname="source code pro"] 

        label6 [label="6", pos="0.85,0.65!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label7 [label="7", pos="0.6,1.0!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label8 [label="8", pos="0.2,0.4!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label9 [label="9", pos="0.775,1.325!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label10 [label="10", pos="0.55,1.85!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label11 [label="11", pos="0.325,1.45!", shape=none, fontname="source code pro", fontcolor="#5E81AC"]

        label12 [label="12", pos="0.3,0.75!", shape=none, fontname="source code pro", fontcolor="#5E81AC"]
        label13 [label="13", pos="1.15,1.4!", shape=none, fontname="source code pro", fontcolor="#5E81AC"]
        label14 [label="14", pos="-.05,1.2!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 

        }

      Quadratic Wedge

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph hex {
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
        label5 [label="5", pos="0.9,1.0!", shape=none, fontname="source code pro"] 
        label6 [label="6", pos="1.6,1.5!", shape=none, fontname="source code pro"] 
        label7 [label="7", pos="0.4,1.6!", shape=none, fontname="source code pro"] 
        }

      Hexahedron
    .. grid-item::
      :child-align: center

      .. graphviz::

        graph hex20 {
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

        8 [pos="0.5,0.05!", color="#5E81AC"];
        9 [pos="1.3,0.35!", color="#5E81AC"];
        10 [pos="1.1,0.55!", color="#5E81AC"];
        11 [pos="0.3,0.25!", color="#5E81AC"];

        12 [pos="0.4,0.95!", color="#5E81AC"];
        13 [pos="1.2,1.15!", color="#5E81AC"];
        14 [pos="1.0,1.45!", color="#5E81AC"];
        15 [pos="0.2,1.25!", color="#5E81AC"];

        16 [pos="-.05,0.5!", color="#5E81AC"];
        17 [pos=".95,0.5!", color="#5E81AC"];
        18 [pos="1.55,1!", color="#5E81AC"];
        19 [pos="0.55,1!", color="#5E81AC"];

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
        label5 [label="5", pos="0.9,1.0!", shape=none, fontname="source code pro"] 
        label6 [label="6", pos="1.6,1.5!", shape=none, fontname="source code pro"] 
        label7 [label="7", pos="0.4,1.6!", shape=none, fontname="source code pro"] 

        label8 [label="8", pos="0.5,-.1!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label9 [label="9", pos="1.45,0.35!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label10 [label="10", pos="1.1,0.65!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label11 [label="11", pos="0.15,0.25!", shape=none, fontname="source code pro", fontcolor="#5E81AC"]

        label12 [label="12", pos="0.35,0.8!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label13 [label="13", pos="1.225,1.!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label14 [label="14", pos="1.0,1.6!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label15 [label="15", pos="0.05,1.25!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 

        label16 [label="16", pos="-.2,0.5!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label17 [label="17", pos=".825,0.375!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label18 [label="18", pos="1.7,1!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        label19 [label="19", pos="0.65,1.1!", shape=none, fontname="source code pro", fontcolor="#5E81AC"] 
        }

      Quadratic Hexahedron




