Improvement
===========

Node Repositioning
------------------
.. toctree::
    :maxdepth: 1
    :hidden:

    laplaciansmooth
    springsmooth

Node repositioning/mesh smoothing moves the nodes of a mesh with the goal of 
more evenly distributing nodes and/or improving element quality. Throughout 
the process, no nodes or elements are added/removed and the connectivity of the
elements doesn't change.

    * :ref:`Laplacian Smoothing`
    * :ref:`Spring-based Smoothing`

Topological Modifications
-------------------------
.. toctree::
    :maxdepth: 1
    :hidden:

    flips23
    edgecontract

The mesh topology (i.e. connectivity of the elements) can be modified to
improve mesh element quality.
Topological modifications include "flips" where adjacent elements that share a
face or an edge are reconnected in a new configuration, as well as edge 
contraction and splitting, which can be used to coarsen and refine a mesh, 
respectively. 


    * :ref:`2→3 and 3→2 Flips`
    * :ref:`Edge Contraction`


