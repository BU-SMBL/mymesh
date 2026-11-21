Laplacian Smoothing
===================
See :func:`~mymesh.improvement.LocalLaplacianSmoothing`
:cite:`Cavendish1974`

Laplacian smoothing is a classic smoothing method that works by moving each node
to the center of its :math:`n_i` neighboring nodes. 

.. math::

    \bar{x}'_i = \frac{1}{n_i}\sum_{j=1}^{n_i} \bar{x}_j


.. grid:: 2
    :outline:

    .. grid-item::
      .. graphviz::

        graph original {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0.4,0.3!", color="#bf616a"];
        1 [pos="-1,0.!", color="#5e81ac"]; 
        2 [pos="-0.5,-0.866!", color="#5e81ac"]; 
        3 [pos="0.5,-0.866!", color="#5e81ac"];
        4 [pos="1.0,0.0!", color="#5e81ac"];
        5 [pos="0.5,0.866!", color="#5e81ac"];
        6 [pos="-0.5,0.866!", color="#5e81ac"]; 

        1 -- 2
        2 -- 3
        3 -- 4
        4 -- 5
        5 -- 6
        6 -- 1

        0 -- 1
        0 -- 2
        0 -- 3
        0 -- 4
        0 -- 5
        0 -- 6

        }
      
    .. grid-item::
      .. graphviz::

        graph smoothed {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        0 [pos="0.0,0.0!", color="#bf616a"];
        1 [pos="-1,0.!", color="#5e81ac"]; 
        2 [pos="-0.5,-0.866!", color="#5e81ac"]; 
        3 [pos="0.5,-0.866!", color="#5e81ac"];
        4 [pos="1.0,0.0!", color="#5e81ac"];
        5 [pos="0.5,0.866!", color="#5e81ac"];
        6 [pos="-0.5,0.866!", color="#5e81ac"]; 

        1 -- 2
        2 -- 3
        3 -- 4
        4 -- 5
        5 -- 6
        6 -- 1

        0 -- 1
        0 -- 2
        0 -- 3
        0 -- 4
        0 -- 5
        0 -- 6

        }

Laplacian smoothing is efficient and effective in many applications, however it
has several limitations:

    * Repeated iterations can lead to shrinkage if the boundary isn't fixed 
    * For volumetric elements (e.g. tetrahedra), smoothing can sometimes lead to element inversions or reduced quality
  
These limitations have motivated several variants to the classical smoothing
algorithm.

Tangential Laplacian Smoothing
------------------------------
See :func:`~mymesh.improvement.TangentialLaplacianSmoothing`
:cite:`Ohtake2003`

Tangential Laplacian smoothing mitigates shrinkage by only moving nodes on the
plane tangent to the surface at that point, better preserving the original 
geometry while still smoothing.

The displacement due to local Laplacian smoothing can be calculated as 

.. math::

    \mathbf{U}_i = \frac{1}{n_i}\sum_{j=1}^{n_i} \left(  \bar{x}_j - \bar{x}_i \right)

which can then be projected onto the tangent plane by subtracting the vector
projection of the displacement :math:`\mathbf{U}_i` onto the unit normal vector 
:math:`\hat{n}_i`:

.. math::

  \mathbf{R}_i = \mathbf{U}_i - \left( \mathbf{U}_i \cdot \hat{n}_i \right)\hat{n}_i

The updated node positions are then:

.. math::

  \bar{x}'_i = \bar{x} + \mathbf{R}

Smart Laplacian Smoothing
-------------------------
See :func:`~mymesh.improvement.SmartLaplacianSmoothing`
:cite:`Freitag1997a`

"Smart" Laplacian smoothing follows the same approach as the standard local 
Laplacian smoothing, but vertex movement is only accepted if the quality 
of the connected elements is improved. Different strategies can be employed,
for example only moving a node if the average element quality improves or if 
the worst element quality improves. 