.. _theory_curvature:

Curvature
=========

Curvature Metrics
-----------------

Principal Curvatures
^^^^^^^^^^^^^^^^^^^^

The principal curvatures are the maximum (:math:`\kappa_1`) and minimum 
(:math:`\kappa_2`) curvatures of a surface at a particular point. When both 
principal curvatures are positive, the surface is convex at that point, when
they're both negative, the surface is concave at that point, and when they 
have opposite signs the surface is hyperbolic (saddle-shaped) at that point.

The principal curvatures can be calculated from the mean (:math:`H`) and 
Gaussian (:math:`K`) curvatures as 

.. math::
    \kappa_{1,2} = H \pm \sqrt{(H^2 - K)}

Mean Curvature
^^^^^^^^^^^^^^

See :func:`~mymesh.curvature.MeanCurvature`

Mean curvature is the average of the two principal curvatures

.. math::
    H = \frac{1}{2}(\kappa_1 + \kappa_2)

where :math:`\kappa_1, \kappa_2` are the maximum and minimum principal curvatures, 
respectively.

Gaussian Curvature
^^^^^^^^^^^^^^^^^^

See :func:`~mymesh.curvature.GaussianCurvature`

Gaussian curvature of a surface is unchanged when a surface is bent without
stretching (e.g. bending a plane into a cylinder) and is thus considered
to be independent of the embedding of a surface, making it an "intinsic" 
geometric property (see `Gauss's theorema egregium <https://en.wikipedia.org/wiki/Gaussian_curvature#Theorema_egregium>`_).

Gaussian curvature (:math:`K`) is calculated as

.. math::
    K = \kappa_1 \kappa_2

where :math:`\kappa_1, \kappa_2` are the maximum and minimum principal curvatures, 
respectively.

.. _theory_shape-index:

Shape Index
^^^^^^^^^^^

See :func:`~mymesh.curvature.ShapeIndex`

Shape index :cite:p:`Koenderink1992a` describes the directionality of curvature, 
independent of scale. It captures the intuitive notion that a sphere is 
recognizable as a sphere regardless of how big it is. Shape index is a 
unitless measure that ranges from -1 (concave spherical cup) to 1 (convex 
spherical cap). :math:`\pm` 0.5 correspond to cylindrical curvatures and 0 
corresponds to a saddle. 

Shape index (:math:`s`) is calculated as

.. math::
    s = \frac{2}{\pi} \arctan \frac{\kappa_2 + \kappa_1}{\kappa_2 - \kappa_1}

where :math:`\kappa_1, \kappa_2` are the maximum and minimum principal curvatures, 
respectively.

.. _theory_curvedness:

Curvedness
^^^^^^^^^^

See :func:`~mymesh.curvature.Curvedness`

Curvedness :cite:p:`Koenderink1992a` complements shape index by capturing the 
magnitude of curvature, independent of direction. It is a non-negative measure
of the scale of a surface's curvature. For a sphere, curvedness is equal to
the principal curvatures. 

Curvedness (:math:`c`) is calculated as 

.. math::
    c = \sqrt{\frac{\kappa_1^2 + \kappa_2^2}{2}}

where :math:`\kappa_1, \kappa_2` are the maximum and minimum principal curvatures, 
respectively.


Conventions
^^^^^^^^^^^

Some measures of curvature are coordinate system and/or reference frame 
independent, while others are not. For example, a sphere may be considered
concave or convex depending on whether it's being viewed from the inside
or the outside. When measuring the curvature of surfaces, it is thus important
to define a consistent convention when calculating and discussing curvature.

While different conventions can be used, the most common, and the one 
adopted here, is that surface normal vectors point outward from the surface.
Thus a sphere with normal vectors pointed towards the center would have
negative principal curvatures and be considered concave, while if the normal
vectors point outwards, the principal curvatures would be positive and the 
surface would be convex. The surface normals follow from the ordering of nodes
in the surface elements of a mesh, and thus it is important to be mindful of 
element orientation when measuring curvature.

For function- and image-based curvatures, the inside of a surface is considered
to be less than and greater than the threshold isosurface value, respectively.
For implicit functions, this follows from the convention that the inside of an implicit function, while for images this is based on the assumption that foreground 
objects are bright (as is the case in CT scans of bone).

Interpretation
^^^^^^^^^^^^^^

+---------------------+--------------------------------------+-----------------------------------------+-----------------------+
| Classification      | Principal Curvatures                 | Gaussian (:math:`K`) & Mean (:math:`H`) | Shape Index           |
+=====================+======================================+=========================================+=======================+
| Convex, Elliptical  | :math:`\kappa_{1,2} > 0`             | :math:`K > 0, H > 0`                    |:math:`s > 0.5`        |
+---------------------+--------------------------------------+-----------------------------------------+-----------------------+
| Concave, Elliptical | :math:`\kappa_{1,2} < 0`             | :math:`K > 0, H < 0`                    |:math:`s < -0.5`       |
+---------------------+--------------------------------------+-----------------------------------------+-----------------------+
| Hyperbolic (Saddle) | :math:`\kappa_1 > 0, \kappa_2 < 0`   | :math:`K < 0`                           |:math:`-0.5 < s < 0.5` |
+---------------------+--------------------------------------+-----------------------------------------+-----------------------+

Calculating Curvature
---------------------

Mesh-Based Curvature
^^^^^^^^^^^^^^^^^^^^

Mesh-based curvature calculation relies only on the mesh itself to estimate 
local curvature. This relies on looking at neighborhoods of points around a 
point of interest, most commonly the "one-ring" neighborhood of nodes directly
connected to the point of interest. The biggest advantage to this class of 
methods is that only the mesh and mesh-based features (such as normal vectors)
are required. The disadvantage is that they can be inaccurate and have an 
inherent dependence on mesh and mesh quality, with low quality elements 
resulting in spurious results. 

Quadratic Surface Fitting
"""""""""""""""""""""""""

Reference: :cite:t:`Goldfeather2004`

Quadratic surface fitting (see :func:`~mymesh.curvature.QuadFit`) locally fits a quadratic
surface :math:`z = f(x,y) = \frac{A}{2}x^2 + Bxy + \frac{C}{2}y^2` to a 
local neighborhood of nodes. This can be accomplished by examining this 
neighborhood in a local coordinate system, in which the node of interest is 
positioned at :math:`(0, 0, 0)` and the normal vector to that node is oriented
in the :math:`[0, 0, 1]` direction. This allows for convenient fitting to the 
quadratic surface equation, leading to a system of equations:

.. math::

    \left.
    \begin{align}
    z_0 &= \frac{A}{2}x_0^2 + Bx_0y_0 + \frac{C}{2}y_0^2 \\
    z_1 &= \frac{A}{2}x_1^2 + Bx_1y_1 + \frac{C}{2}y_1^2 \\
    \vdots & \\ 
    z_n &= \frac{A}{2}x_n^2 + Bx_ny_n + \frac{C}{2}y_n^2 
    \end{align}\
    \right\} = 
    \underbrace{\begin{bmatrix}
    \vdots & \vdots & \vdots \\
    \frac{1}{2}x_i^2 & x_iy_i & \frac{1}{2}y_i^2 \\
    \vdots & \vdots & \vdots 
    \end{bmatrix}}_{\mathbf{A}}
    \underbrace{\begin{bmatrix}
    A \\ B \\ C
    \end{bmatrix}}_{\mathbf{x}}
    = 
    \underbrace{\begin{bmatrix}
    \vdots \\ z_i \\ \vdots
    \end{bmatrix}}_{\mathbf{b}}

for all :math:`n` nodes in the neighborhood (including the point of interest).
For anything but the corner of an open surface, this system of equations will be 
be overdetermined and can be solved by least squares 
(:math:`\mathbf{A}^\intercal\mathbf{A} \mathbf{x} = \mathbf{A}^\intercal \mathbf{b}`).

The parameters :math:`A, B,` and :math:`C` are the components of the Weingarten matrix

.. math::

    W = \begin{bmatrix} A & B \\ B & C \end{bmatrix}

the eigenvalues of which are the principal curvatures.

.. _theory-cubic-surface-fitting:

Cubic Surface Fitting
"""""""""""""""""""""

Reference: :cite:t:`Goldfeather2004`

Cubic surface fitting (see :func:`~mymesh.curvature.CubicFit`) locally fits a cubic
surface :math:`z = f(x,y) = \frac{A}{2}x^2 + Bxy + \frac{C}{2}y^2 + Dx^3 + Ex^2y + Fxy^2 + Gy^3` 
to a local neighborhood of nodes. This can be accomplished by examining this 
neighborhood in a local coordinate system, in which the node of interest is 
positioned at :math:`(0, 0, 0)` and the normal vector to that node is oriented
in the :math:`[0, 0, 1]` direction. This allows for convenient fitting to the 
cubic surface equation, leading to a system of equations:

.. math::

    \begin{align}
    z_0 &= \frac{A}{2}x_0^2 + Bx_0y_0 + \frac{C}{2}y_0^2 + Dx_0^3 + Ex_0^2y_0 + Fx_0y_0^2 + Gy_0^3 \\
    z_1 &= \frac{A}{2}x_1^2 + Bx_1y_1 + \frac{C}{2}y_1^2 + Dx_1^3 + Ex_1^2y_1 + Fx_1y_1^2 + Gy_1^3 \\
    \vdots & \\ 
    z_n &= \frac{A}{2}x_n^2 + Bx_ny_n + \frac{C}{2}y_n^2 + Dx_n^3 + Ex_n^2y_n + Fx_ny_n^2 + Gy_n^3
    \end{align}

In many cases, there may not be enough nodes in the neighborhood to solve for 
the seven unknowns from these equations alone, therefore we can use additional
information from the node normal vectors to obtain two additional equations per
node. 

The surface normal at a point :math:`i` in the local reference frame can be 
written as

.. math::

    n(x,y) = 
    \begin{bmatrix}
        -\frac{\partial z_i}{\partial x} \\ 
        -\frac{\partial z_i}{\partial y} \\ 
        1
    \end{bmatrix} =
    \begin{bmatrix}
        -(Ax + By + 3Dx^2 + 2Exy + Fy^2) \\
        -(Bx + Cy + Ex^2 + 2Fxy + 3Gy^2) \\
        1
    \end{bmatrix}

A unit normal :math:`\hat{n}(x,y) = \begin{bmatrix} a_i & b_i & c_i \end{bmatrix}^T`
can be divided by :math:`-c_i` such that :math:`n(x,y) = -\hat{n}(x,y)/c_i`, which
gives

.. math::

    \begin{align}
    Ax + By + 3Dx^2 + 2Exy + Fy^2 = -a_i/c_i, \\
    Bx + Cy + Ex^2 + 2Fxy + 3Gy^2 = -b_i/c_i.
    \end{align}

Now using each of the three equations for each point, a linear system can be 
constructed

.. math::

    \underbrace{\begin{bmatrix} 
    \vdots           & \vdots &           \vdots & \vdots &   \vdots &   \vdots & \vdots \\
    \frac{1}{2}x_i^2 & x_iy_i & \frac{1}{2}y_i^2 & x_i^3  & x_i^2y_i & x_iy_i^2 & y_i^3 \\
    x_i              & y_i    & 0                & 3x_i^2 & 2x_iy _i &  y_i^2   & 0 \\
    0                & x_i    & y_i              & 0      & x_i^2    & 2x_iy_i  & 3y_i^2 \\
    \vdots           & \vdots &           \vdots & \vdots &   \vdots &   \vdots & \vdots \\
    \end{bmatrix}}_{\mathbf{A}}
    \underbrace{\begin{bmatrix} A \\ B \\ C \\ D \\ E \\ F \\ G \end{bmatrix}}_{\mathbf{x}}
    = 
    \underbrace{\begin{bmatrix} \vdots \\  z_i \\ \frac{a_i}{c_i} \\ \frac{b_i}{c_i} \\ \vdots \end{bmatrix}}_{\mathbf{b}} 

This system of equations will be can be solved by least squares 
(:math:`\mathbf{A}^\intercal\mathbf{A} \mathbf{x} = \mathbf{A}^\intercal \mathbf{b}`).

The parameters :math:`A, B,` and :math:`C` are the components of the Weingarten matrix

.. math::

    W = \begin{bmatrix} A & B \\ B & C \end{bmatrix}

the eigenvalues of which are the principal curvatures.

Analytical Curvature
^^^^^^^^^^^^^^^^^^^^

Reference: :cite:t:`Goldman2005`



