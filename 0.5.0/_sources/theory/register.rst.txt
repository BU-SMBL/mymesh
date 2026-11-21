.. _theory_register:

Register
========

Registration involves the alignment of one object (the "moving" object) to a reference (or "fixed") object. 


Common instances of registration include point cloud registration and image registration. 
Registration can be used to align different scans/images of an object (e.g. a CT scan and an MRI scan), images of the same object taken at different times, different variations of the same kind of object (e.g. CT scans of bones belonging different individuals), or any other situation where you have two or more objects that can be aligned. 
Registration can be useful for detecting and measuring differences between objects, stitching together overlapping parts of a larger object, and more.

Registration is an optimization process, by which some measure of similarity is maximized (or some measure of distance is minimized). 
As such, it requires an objective function (a way to score the similarity/distance), a parameter space (a set of parameters that can be used to move the object), and an algorithm by which the parameter space can be explored in order to find the optimal set of parameters for the chosen objective function.

Registration strategies will often consider one "fixed" object which is held in place while the other "moving" object is transformed to line up with the fixed object.


Types of registration
---------------------

There are a few ways to classify different approaches to registration.

Classification by modality
^^^^^^^^^^^^^^^^^^^^^^^^^^
Registration strategies can be classified by what types of data are being aligned.

- Image registration: The alignment of two or more images (see also: :func:`~mymesh.register.Image2Image`)

- Point cloud registration: The alignment of two or more sets of points (see also: :func:`~mymesh.register.Point2Point`)

Classification by transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Registration strategies can also be classified by what type of transformation (i.e. parameter space) is used to align the objects.

- Rigid registration: The object is moved by translations and rotations, but not deformed.

- Similarity registration: The object is moved by translations and rotations, and can be uniformly scaled.

- Affine registration: The object is moved by translations and rotations, can be scaled in each direction, and can undergo shearing deformations. 

- Elastic registration: The object can be deformed during the alignment process (note: currently not available in mymesh). 

Iterative Closest Point (ICP)
-----------------------------
Reference: :cite:t:`Besl1992`

The iterative closest point algorithm is a classic and popular method for rigid registration of point sets. 

The ICP algorithm involves the repetition of two key steps:
1. Identifying the "closest point" correspondences; and
2. Minimizing the distance between corresponding points.


Minimizing the distance between corresponding points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Reference: :cite:t:`Arun1987`

Once pairs of corresponding points have been identified, the squared error between a set of :math:`N` fixed points (:math:`p^f = \{p_i^f\}`) and a corresponding set of :math:`N` moving points (:math:`p^m = \{p_i^m\}`) is written:

.. math::
    
    \Sigma^2 = \sum_{i=1}^{N} ||p_i^f - (\mathbf{R} p_i^m + \mathbf{T})||^2

where :math:`\mathbf{R}` and :math:`\mathbf{T}` are rotation and translation matrices, respectively.
The objective is to find :math:`\mathbf{R}, \ \mathbf{T}` that minimize the squared error: :math:`\underset{\mathbf{R}, \ \mathbf{T}}{\min}\Sigma^2`. 
Before attempting to minimize this squared error, we can simplify the problem by separating translation and rotation. 
At the least squares minimum of :math:`\Sigma^2`, the centroid of the moving points (:math:`c(p^m) = \frac{1}{N}\sum_{i=1}^{N}p_i^m`) will be equal to that of the fixed points (:math:`c(p^f) = \frac{1}{N}\sum_{i=1}^{N}p_i^f`) :cite:p:`Arun1987`, meaning the rotation can first be solved by centering both sets of points at the origin by subtracting the centroids from the points:

.. math::

    q_i^f =  p_i^f - c(p^f), \\
    q_i^m = p_i^m - c(p^m),

simplifying the squared error to 

.. math::
    
    \Sigma^2 = \sum_{i=1}^{N} ||q_i^f - (\mathbf{R} q_i^m)||^2.

The translation can subsequently be determined as 

.. math::

    \mathbf{T} = c(p^f) - \mathbf{R}c(p^m).

The rotation can be found by performing `singular value decomposition <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_ of the cross covariance matrix (see :cite:t:`Arun1987` for derivation)

.. math::
    
    H = \sum_{i=1}^{N} q_i^m (q_i^f)^t = \mathbf{U \Sigma V}^t.

The rotation :math:`\mathbf{R}` is then

.. math::

    \mathbf{R} = \mathbf{V} \mathbf{U}^t.


.. graphviz::

    graph pq {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        a [pos="0.0, 0.0!"];
        b [pos="1.0, 0.0!"];
        c [pos="1.0, 1.0!"];
        d [pos="0.0, 1.0!"];

        e [pos="0.25,  0.25!", shape=square, width=0.05, color="#5e81ac", label=""];
        f [pos="1.23, 0.08!", shape=square, width=0.05, color="#5e81ac", label=""];
        g [pos="1.41, 1.06!", shape=square, width=0.05, color="#5e81ac", label=""];
        h [pos="0.42, 1.23!", shape=square, width=0.05, color="#5e81ac", label=""];

        a -- b;
        b -- c;
        c -- d;
        d -- a; 

        e -- f [color="#5e81ac"];
        f -- g [color="#5e81ac"];
        g -- h [color="#5e81ac"];
        h -- e [color="#5e81ac"];

        a -- e [style=dotted, color="#4c566a"];
        b -- f [style=dotted, color="#4c566a"];
        c -- g [style=dotted, color="#4c566a"];
        d -- h [style=dotted, color="#4c566a"];

        labelf [label=<<I>p<SUP>f</SUP></I>>, pos="-0.2,-0.2!", shape=none, fontname="Times-Roman"] 
        labelm [label=<<I>p<SUP>m</SUP></I>>, pos="1.5,1.2!", shape=none, fontname="Times-Roman", fontcolor="#5e81ac"] 
    }
