Rays
====

Ray-Shape Intersection Tests
----------------------------
Ray-Shape intersection tests determine whether rays intersect with a shape
and where those intersections occur.

Ray-Triangle Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^

*The Möller-Trumbore Intersection Test* :cite:p:`Moller2005`

See :func:`~mymesh.rays.RayTriangleIntersection`, 
:func:`~mymesh.rays.RayTrianglesIntersection`, 
:func:`~mymesh.rays.RaysTrianglesIntersection`.

The Möller-Trumbore test efficiently determines whether a ray, 
:math:`R(t) = O + tD` with origin :math:`O` and direction :math:`D` intersects 
the triangle with :math:`V_0, V_1, V_2`. The test ultimately relies on 
determining whether the barycentric coordinates :math:`(u,v)` of the projection 
of the ray onto the plane of the triangle (:math:`T(u,v)`) fall within the 
triangle (:math:`u,v \geq 0` and :math:`u+v\leq 1`), with checks along the way 
to ensure that only as much computation as is necessary is performed.

.. graphviz::

    graph raytri {
    node [shape=point, fontname="source code pro"];
    edge [style=solid];

    a [pos="1.0,0.6!"]; 
    b [pos="1.8,0.5!"];
    c [pos="0.8,2.0!"]; 

    o [pos="0.5,0.5!"];
    r [pos="2.0,2.0!", style="invis"]
    i [pos="1.2,1.2!", shape="circle", height=".1!", width=".1", label=""]

    a -- b;
    b -- c;
    c -- a;
    o -- r [dir="forward", arrowhead="normal"];

    labelv0 [label=<V<SUB>0</SUB>>, pos="1.0,0.45!", shape=none, fontname="Times-Roman"] 
    labelv1 [label=<V<SUB>1</SUB>>, pos="1.8,0.35!", shape=none, fontname="Times-Roman"] 
    labelv2 [label=<V<SUB>2</SUB>>, pos="0.6,2.0!", shape=none, fontname="Times-Roman"] 
    label1 [label="O", pos=".4,.4!",  shape=none, fontname="Times-Roman"] 

    }

The intersection point (:math:`T(u,v)`) between :math:`R(t)` and the triangle is
computed by solving :math:`R(t) = T(u,v)` or 

.. math::

    O + tD = (1-u-v)V_0+uV_1 + vV_2

The algorithm begins by computing the edge vectors :math:`E_1=V_1-V_0` and 
:math:`E_2 = V_2 - V_0`  followed by the calculation of the determinant of the
:math:`3\times3` matrix 

.. math:: 
    
    \det\left(\begin{bmatrix} r_0 & r_1 & r_2 \\ e_{2,0} & e_{2,1} & e_{2,2} \\ e_{1,0} & e_{1,1} & e_{1,2} \end{bmatrix} \right) = E_1 \cdot (R \times E_2) = det. 
    
If this determinant is zero, then the ray lies in the plane of the triangle and 
the test concludes that there is no intersection. The small parameter 
:math:`\epsilon` is used to determine if the determinant is sufficiently 
near-zero, with :math:`\epsilon=10^{-6}` used in the original paper.  The 
barycentric coordinate :math:`u` is first calculated and then checked for 
admissibility (:math:`0 < u < 1`) followed by calculation of :math:`v` and checking 
that :math:`0<v` and :math:`u+v  < 1`. Finally the parameter :math:`t` of the
intersection point can be calculated as 

.. math::
    
    t_i=\frac{1}{det}(E_2 \cdot ((O-V_0) \times E_1))

For a unidirectional intersection test (only in the positive direction of 
:math:`R(t)`), :math:`t` must be positive. For a bidirectional test, the value 
of :math:`t` is inconsequential for determining if the intersection occurs. The 
intersection point is then

.. math:: 
    
    R(t_i) = O + t_i D

Ray-Box Intersection
^^^^^^^^^^^^^^^^^^^^
:cite:t:`Williams2005`

See :func:`~mymesh.rays.RayBoxIntersection`, 
:func:`~mymesh.rays.RayBoxesIntersection`.

This test determines whether a ray, 
:math:`R(t) = O + tD = \begin{bmatrix}R_x & R_y & R_z \end{bmatrix}` with origin 
:math:`O` and direction :math:`D` intersects an axis-aligned box with bounds 
:math:`(x_{min},x_{max}),(y_{min},y_{max}),(z_{min},z_{max})`. The test works by 
finding the values of the parameter :math:`t` that correspond to the points 
where the ray reaches each bound of the box. For example, 
:math:`t_{xmin} = (x_{min} - O_x)/R_x`, :math:`t_{xmax} = (x_{max} - O_x)/R_x`
correspond to the points where the ray crosses the lower and upper x axis bounds
of the box. The order of these calculations is determined based on the sign of 
each component of the ray vector such that :math:`t_{xmin} \leq t_{xmax}` (i.e. 
if :math:`R_x<0`, :math:`t_{xmin}=(x_{max} - O_x)/R_x`). 

.. graphviz::

    graph raytri {
    node [shape=point, fontname="source code pro"];
    edge [style=solid];

    a [pos="0.4,0.5!"]; 
    b [pos="0.4,1.5!"];
    c [pos="1.4,1.5!"]; 
    d [pos="1.4,0.5!"]; 
    e [pos="0.8,0.9!"]; 
    f [pos="0.8,1.9!"];
    g [pos="1.8,1.9!"]; 
    h [pos="1.8,0.9!"]; 

    o [pos="0.7,0.2!"];
    r [pos="1.5,2.5!", style="invis"]
    i [pos="1.24,1.75!", shape="circle", height=".1!", width=".1", label=""]
    i2 [pos="0.875,0.7!", shape="circle", height=".1!", width=".1", label=""]

    a -- b;
    b -- c;
    c -- d;
    d -- a;
    e -- f;
    f -- g;
    g -- h;
    h -- e;
    a -- e;
    b -- f;
    c -- g;
    d -- h;

    o -- r [dir="forward", arrowhead="normal"];

    labelv0 [label=<V<SUB>0</SUB>>, pos="0.25,0.5!", shape=none, fontname="Times-Roman"] 
    labelv1 [label=<V<SUB>1</SUB>>, pos="1.55,0.45!", shape=none, fontname="Times-Roman"]
    labelv2 [label=<V<SUB>2</SUB>>, pos="1.95,0.9!", shape=none, fontname="Times-Roman"] 
    labelv3 [label=<V<SUB>3</SUB>>, pos="0.65,1.0!", shape=none, fontname="Times-Roman"] 

    labelv4 [label=<V<SUB>4</SUB>>, pos="0.25,1.5!", shape=none, fontname="Times-Roman"] 
    labelv5 [label=<V<SUB>5</SUB>>, pos="1.55,1.45!", shape=none, fontname="Times-Roman"] 
    labelv6 [label=<V<SUB>6</SUB>>, pos="1.95,1.9!", shape=none, fontname="Times-Roman"] 
    labelv7 [label=<V<SUB>7</SUB>>, pos="0.65,2.0!", shape=none, fontname="Times-Roman"] 

    label1 [label="O", pos=".6,.2!",  shape=none, fontname="Times-Roman"] 

    }

For the ray to intersect the box, the limit-intersection parameters 
(:math:`t_{xmin}, t_{xmax}, t_{ymin},...`) for each axis must be consistent with
each other. If, for example, :math:`t_{ymin} > t_{xmax}`, that means that ray 
intersects with the first :math:`y` bound of the box *after* it has crossed both 
:math:`x` bounds and could not intersect with the box itself. If, instead, 
:math:`t_{xmin} \leq t_{ymax}` and :math:`t_{ymin} \leq t_{xmax}` then there may 
be an intersection and the test can proceed to checking the :math:`z` limits. If 
:math:`\max{(t_{xmin},t_{ymin})} \leq t_{zmax}` and 
:math:`t_{zmin} \leq \min{(t_{xmax},t_{ymax})}`, then there is a section of the 
ray that falls between the bounds of the box on all three axes, so the ray must
intersect with the box. 

Ray-Segment Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^

Plane-Shape Intersection Tests
------------------------------
Plane-Triangle Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plane-Box Intersection
^^^^^^^^^^^^^^^^^^^^^^

Shape-Shape Intersection Tests
------------------------------

Triangle-Triangle Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Triangle-Box Intersection
-------------------------

Segment-Segment Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^