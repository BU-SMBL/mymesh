.. _theory-rays:
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
:math:`O` and direction :math:`D=\begin{bmatrix}D_x & D_y & D_z \end{bmatrix}` intersects an axis-aligned box with bounds 
:math:`(x_{min},x_{max}),(y_{min},y_{max}),(z_{min},z_{max})`. The test works by 
finding the values of the parameter :math:`t` that correspond to the points 
where the ray reaches each bound of the box. For example, 
:math:`t_{xmin} = (x_{min} - O_x)/D_x`, :math:`t_{xmax} = (x_{max} - O_x)/D_x`
correspond to the points where the ray crosses the lower and upper x axis bounds
of the box. The order of these calculations is determined based on the sign of 
each component of the ray vector such that :math:`t_{xmin} \leq t_{xmax}` (i.e. 
if :math:`D_x<0`, :math:`t_{xmin}=(x_{max} - O_x)/D_x`). 

.. graphviz::

    graph raybox {
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

For unidirectional intersections (intersections only in the positive direction
of the ray from the point), at least one of the limit-intersection parameters
must be positive.

Ray-Segment Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^
See :func:`~mymesh.rays.RaySegmentIntersection`, :func:`~mymesh.rays.RaySegmentsIntersection`  

The ray-segment intersection test checks whether a ray, :math:`R(t) = O + tD = \begin{bmatrix}R_x & R_y & R_z \end{bmatrix}` with origin :math:`O` and direction :math:`D`, 
intersects with a line segment, :math:`\bar{p_1 p_2}`. 

First, to test of the ray is coplanar with the line segment, a test point along
the ray can be obtained as :math:`p = O + (1)D`. To test if the four points
:math:`p_1, p_2, O, p` are coplanar, the scalar triple product of three vectors,
:math:`\bar{a} = p_2 - p_1`, :math:`\bar{b} = p - O`, :math:`\bar{c} = O - p_1` can 
be obtained. If the scalar triple product is 0 the points are coplanar, otherwise 
there is no intersection. (The scalar triple product can be interpreted as giving 
the volume of a parallelepiped defined by the vectors 
:math:`\bar{a}, \bar{b}, \bar{c}`).

If the ray and line segment are coplanar, the parametric form of the line 
:math:`\bar{p_1 p_2}` can be written as :math:`p1 + s \bar{a}`. The point of 
intersection in terms of the parametric terms :math:`s, t` can be obtained as

:math:`s = \frac{(c \times b) \cdot (a \times b)}{(a \times b)^2}`
:math:`t = \frac{(c \times a) \cdot (a \times b)}{(a \times b)^2}`

If :math:`0 \leq t`, the point of intersection is in the positive direction of 
the ray, and if :math:`s \in [0,1]` then the point of intersection is on the 
line segment. If both conditions are met, the ray and segment intersect.


.. graphviz::

    graph rayseg {
    node [shape=point, fontname="source code pro"];
    edge [style=solid];

    p1 [pos="-1.0,1.0!"]; 
    p2 [pos="1.0,-1.0!"];

    o [pos="-1,-1!"];
    r [pos="-.5,-.5!", style="invis"]
    i [pos="0,0!", shape="circle", height=".1!", width=".1", label=""]

    p1 -- p2;
    o -- r [dir="forward", arrowhead="normal"];
    
    labelO [label="O", pos="-1.1,-1.1!", shape=none, fontname="Times-Roman"] 
    labelR [label="R", pos="-.5,-.4!", shape=none, fontname="Times-Roman"] 
    labelp1 [label=<P<SUB>1</SUB>>, pos="-1.2,1.1!", shape=none, fontname="Times-Roman"]
    labelp2 [label=<P<SUB>2</SUB>>, pos="1.2,-1.1!", shape=none, fontname="Times-Roman"]
    }

Plane-Shape Intersection Tests
------------------------------
Plane-Triangle Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The plane-triangle intersection test checks whether a plane intersects with an triangle by calculating the signed distance between each vertex of the triangle and the plane. If all vertices of the triangle are on the same side of the plane, then there is no intersection. 

The signed distance between a vertex :math:`v` a plane defined by a point :math:`p` and a normal vector :math:`\hat{n}` is 

Plane-Box Intersection
^^^^^^^^^^^^^^^^^^^^^^
The plane-box intersection test checks whether a plane intersects with an axis-aligned box by calculating the signed distance between each vertex of the box and the plane. If all vertices of the box are on the same side of the plane, then there is no intersection. 

The signed distance between a vertex :math:`v` a plane defined by a point :math:`p` and a normal vector :math:`\hat{n}` is 

.. math::
    
    d = \hat{n} \cdot v - \hat{n} \cdot p

.. graphviz::

    graph raybox {
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

    p1 [pos="0.3,0.8!", style="invis"];
    p2 [pos="0.9,0.4!", style="invis"];
    p3 [pos="0.9,1.6!", style="invis"];
    p4 [pos="0.3,2.0!", style="invis"];

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

    p1 -- p2 [style="dotted"]
    p2 -- p3 [style="dotted"]
    p3 -- p4 [style="dotted"]
    p4 -- p1 [style="dotted"]

    labelv0 [label=<V<SUB>0</SUB>>, pos="0.25,0.5!", shape=none, fontname="Times-Roman"] 
    labelv1 [label=<V<SUB>1</SUB>>, pos="1.55,0.45!", shape=none, fontname="Times-Roman"]
    labelv2 [label=<V<SUB>2</SUB>>, pos="1.95,0.9!", shape=none, fontname="Times-Roman"] 
    labelv3 [label=<V<SUB>3</SUB>>, pos="0.65,1.0!", shape=none, fontname="Times-Roman"] 

    labelv4 [label=<V<SUB>4</SUB>>, pos="0.25,1.5!", shape=none, fontname="Times-Roman"] 
    labelv5 [label=<V<SUB>5</SUB>>, pos="1.55,1.45!", shape=none, fontname="Times-Roman"] 
    labelv6 [label=<V<SUB>6</SUB>>, pos="1.95,1.9!", shape=none, fontname="Times-Roman"] 
    labelv7 [label=<V<SUB>7</SUB>>, pos="0.65,2.0!", shape=none, fontname="Times-Roman"] 

    }

Shape-Shape Intersection Tests
------------------------------

Triangle-Triangle Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Triangle-Box Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^

Segment-Segment Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See :func:`~mymesh.rays.SegmentSegmentIntersection`, :func:`~mymesh.rays.SegmentsSegmentsIntersection` 

The segment-segment intersection test works very similarly to the 
ray-segment intersection test. The key difference is that both parametric variables (:math:`s, t`) must be in the interval :math:`[0,1]`.

.. graphviz::

    graph segseg {
    node [shape=point, fontname="source code pro"];
    edge [style=solid];

    p1 [pos="-1.0,1.0!"]; 
    p2 [pos="1.0,-1.0!"];
    p3 [pos="-1,-1!"];
    p4 [pos="1,1!"];
    i [pos="0,0!", shape="circle", height=".1!", width=".1", label=""]

    p1 -- p2;
    p3 -- p4;
    
    labelp1 [label=<P<SUB>1</SUB>>, pos="-1.2,1.1!", shape=none, fontname="Times-Roman"]
    labelp2 [label=<P<SUB>2</SUB>>, pos="1.2,-1.1!", shape=none, fontname="Times-Roman"]
    labelp3 [label=<P<SUB>3</SUB>>, pos="-1.2,-1.1!", shape=none, fontname="Times-Roman"]
    labelp4 [label=<P<SUB>4</SUB>>, pos="1.2,1.1!", shape=none, fontname="Times-Roman"]
    }


Point Inclusion Tests
---------------------

Point in Boundary
^^^^^^^^^^^^^^^^^
See :func:`~mymesh.rays.PointInBoundary`

To determine whether a point is contained within a closed boundary, the 
intersections between the boundary and an arbitrary coplanar infinite ray originating at 
the point can be counted. For a point inside the boundary, the ray will intersect 
the boundary an odd number of times, while if it's outside the boundary it will 
intersect the ray an even number of times, regardless of how intricate the boundary
is and how many intersections there are, as long as the boundary is closed and 
not self-intersecting. This is known as the *even-odd rule* or the *parity rule*
:cite:p:`Hormann2001`.


.. graphviz::

    digraph cluster_0 {
    node [shape=point];
    splines=false
    a [pos="0,0!", width=0.01];
    b [pos="0.1,0.5!", width=0.01];
    c [pos="0.2,0.7!", width=0.01];
    d [pos="0.3,0.8!", width=0.01];
    e [pos="0.4,0.4!", width=0.01];
    f [pos="0.5,0.3!", width=0.01];
    g [pos="0.6,0.4!", width=0.01];
    h [pos="0.7,0.7!", width=0.01];
    i [pos="0.8,0.8!", width=0.01];
    j [pos="0.9,0.2!", width=0.01];
    k [pos="0.7,-0.1!", width=0.01];
    l [pos="0.5,-0.2!", width=0.01];
    m [pos="0.2,-0.15!", width=0.01];
    p1 [pos="0.5,0!"];
    r1 [pos="0.9,-0.3!",style=invis];
    p2 [pos="0.5,0.6!",fillcolor=none]
    r2 [pos="1.2,0.7!",style=invis];
    p3 [pos="0.7,0.5!"];
    r3 [pos="-0.2,0.6!",style=invis];
    
    a -> b -> c -> d -> e -> f -> g -> h -> i -> j -> k -> l -> m -> a [arrowhead=false];
    
    p1 -> r1;
    p2 -> r2;
    p3 -> r3;
    }

Point in Surface
^^^^^^^^^^^^^^^^
See :func:`~mymesh.rays.PointInSurf`

To determine whether a point is contained within a closed surface, the 
intersections between the surface and an arbitrary infinite ray originating at 
the point can be counted. For a point inside the surface, the ray will intersect 
the surface an odd number of times, while if it's outside the surface it will 
intersect the ray an even number of times, regardless of how intricate the surface
is and how many intersections there are, as long as the surface is closed and 
not self-intersecting. This is known as the *even-odd rule* or the *parity rule*
:cite:p:`Hormann2001`.


    