Spring-based Smoothing
======================

Spring-based smoothing methods can be used both to more evenly distribute
nodes and to allow for controlled mesh deformation by treating the mesh
as a network of springs.

Node Spring-based
-----------------
:cite:t:`Blom2000`

See :func:`~mymesh.improvement.NodeSpringSmoothing`

For a node :math:`i` connected by springs to its :math:`n_i` neighbors, the net
force on the node is

.. math::

    \bar{F}_i = \sum_{j=1}^{n_i} k_{ij}(\bar{x}_j - \bar{x}_i) + \bar{F}_i^{applied}

where :math:`k_{ij}` is the stiffness in units of [force/distance] of the spring 
connecting node :math:`i` to its :math:`j^{th}` neighbor, :math:`\bar{x}_i` is 
the coordinates :math:`(x_i, y_i, z_i)` of the :math:`i^{th}`, and 
:math:`\bar{F}_i^{applied}` is an externally applied load.

For a spring network in equilibrium, :math:`\bar{F}_i = 0` and 

.. math::

    \bar{x}_i = \frac{\sum_{j=1}^{n_i} k_{ij} \bar{x}_j + \bar{F}_i^{applied}}{\sum_{j=1}^{n_i} k_{ij}} 

Since the neighboring nodes are also repositioned, this system can be solved 
iteratively as 

.. math::

    \bar{x}_i^{m+1} = \frac{\sum_{j=1}^{n_i} k_{ij} \bar{x}_j^m + \bar{F}_i^{applied}}{\sum_{j=1}^{n_i} k_{ij}} 

until the change between :math:`\bar{x}_i^{m+1}` and :math:`\bar{x}_i^{m+1}` 
becomes sufficiently small. Since achieving equilibrium isn't strictly necessary 
for smoothing, sufficient smoothing can often be achieved in a small number of 
iterations.