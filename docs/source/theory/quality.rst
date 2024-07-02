Quality
=======


+---------------------------------------------+---------------+----------------+
| Quality Metric                              | Best Quality  | Worst quality  |
+=============================================+===============+================+
| :func:`~mymesh.quality.AspectRatio`         | 1             | :math:`\infty` |
+---------------------------------------------+---------------+----------------+
| :func:`~mymesh.quality.Skewness`            | 0             | 1              |
+---------------------------------------------+---------------+----------------+
| :func:`~mymesh.quality.Orthogonality`       | 1             | 0              |
+---------------------------------------------+---------------+----------------+
| :func:`~mymesh.quality.OrthogonalQuality`   | 1             | 0              |
+---------------------------------------------+---------------+----------------+

::

    Tri1 = mesh([[0,0,0], [1,0,0], [.5,.866,0]], [[0, 1, 2]])
    Tri2 = mesh([[0,0,0], [.8,.4,0], [.5,.5,0]], [[0, 1, 2]])


.. grid:: 3

    .. grid-item::

        Aspect Ratio 

        .. math:: AR = \frac{l_{max}}{l_{min}}

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph tri {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="1,0!"]; 
        2 [pos="0.5,0.866!"]; 

        0 -- 1; 
        1 -- 2; 
        2 -- 0; 

        }


    .. grid-item::
      :child-align: center

      .. graphviz::

        graph tri {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="0.8,0.4!"]; 
        2 [pos="0.5,0.5!"]; 

        0 -- 1; 
        1 -- 2; 
        2 -- 0; 

        label0 [label = <<I>l</I><SUB>min</SUB>>, pos=".8,0.6!", shape=none, fontname="Times-Italic"] 
        label1 [label= <<I>l</I><SUB>max</SUB>>, pos=".5,0.05!", shape=none, fontname="Times-Italic"] 
        }
      
    .. grid-item::

        Aspect Ratio 

        .. math:: AR = \frac{l_{max}}{l_{min}}

    .. grid-item::
      :child-align: center

      .. graphviz::

        graph tri {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="1,0!"]; 
        2 [pos="0.5,0.866!"]; 

        0 -- 1; 
        1 -- 2; 
        2 -- 0; 

        }


    .. grid-item::
      :child-align: center

      .. graphviz::

        graph tri {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0,0!"];
        1 [pos="0.8,0.4!"]; 
        2 [pos="0.5,0.5!"]; 

        0 -- 1; 
        1 -- 2; 
        2 -- 0; 

        label0 [label = <<I>l</I><SUB>min</SUB>>, pos=".8,0.6!", shape=none, fontname="Times-Italic"] 
        label1 [label= <<I>l</I><SUB>max</SUB>>, pos=".5,0.05!", shape=none, fontname="Times-Italic"] 
        }