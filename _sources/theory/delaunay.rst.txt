Delaunay Triangulation
======================

Bowyer-Watson
-------------
See :func:`~mymesh.delaunay.BowyerWatson2d`, :func:`~mymesh.delaunay.BowyerWatson3d`
 
:cite:t:`Bowyer1981`, :cite:t:`Watson1981`

The Bowyer-Watson algorithm was developed independently and concurrently by 
Adrian Bowyer :cite:p:`Bowyer1981` and David Watson :cite:p:`Watson1981` (both
published their papers in the same issue of *The Computer Journal*). The 
algorithm is an incremental approach to n-dimensional Delaunay triangulation 
where one point is inserted at a time, pre-existing triangles whose 
circumcircles contain the new point a removed, and the cavity is retriangulated.

The MyMesh implementation of the Bowyer-Watson algorithm will be described here
thoroughly for two dimensional triangulation, then the relevant changes needed
to extend the algorithm to three dimensional tetrahedralization will be 
discussed.

Bowyer-Watson: Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to start inserting points, an initial bounding triangle is necessary.
This "super-triangle" must be big enough to contain all of the points that 
are to be triangulated. In theory, the super-triangle should also be big enough 
that its vertices are not contained within the circumcircles of any of the 
triangles that are to be created, however, in practice it can be difficult to 
know ahead of time how big that should be. Alternatively, the super-triangle
can simply be big enough to contain the points and the vertices can be 
treated as infinitely far away without, which is discussed (ADD LINK). 

The choice of super-triangle is non-unique, but to ensure it bounds all points,
it can be defined such that its incircle is centered on the average of the set 
of points with radius being the largest distance from the center to a point


.. graphviz::

    graph supertriangle {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        a [pos="1.0, 0.6!"];
        b [pos="0.8, 1.0!"];
        c [pos="0.1, 0.9!"];
        d [pos="0.0, .45!"];
        e [pos="0.5, 0.1!"];
        f [pos="0.4, 0.6!"];

        circle [pos="0.4667,0.60833!", shape=circle, width=1.0668, label="", color="#5E81AC"]

        s1 [pos="0.4667, -0.4585!", color="#d08770"];
        s2 [pos="1.3905, 1.1417!", color="#d08770"];
        s3 [pos="-0.4572, 1.1417!", color="#d08770"];

        s1 -- s2 [penwidth=1, color="#d08770"];
        s2 -- s3 [penwidth=1, color="#d08770"];
        s3 -- s1 [penwidth=1, color="#d08770"]; 
    }

The data structure for constructing the triangulation consists of two tables 
(:code:`dict`) - an element table and an edge table. The element table, keyed by
the node numbers that define the element, contain the oriented "half-edges" 
connected to each element and the edge table is keyed by edges and contains 
the element key connected to that edge. Each half-edge is connected to only one 
element, it's "twin" has opposite numbering and is connected to a neighboring
element, providing a structure allowing for efficient mesh traversal.

The initial data structure for the super-triangle with vertices 
(:math:`p_{1^*}`, :math:`p_{2^*}`, :math:`p_{3^*}`) is:

+-----------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| Elements                                            | Edges                                                                                                          |
+=====================================================+================================================================================================================+
| (:math:`p_{1^*}`, :math:`p_{2^*}`, :math:`p_{3^*}`) | (:math:`p_{1^*}`, :math:`p_{2^*}`), (:math:`p_{2^*}`, :math:`p_{3^*}`), (:math:`p_{3^*}`, :math:`p_{1^*}`)     |
+-----------------------------------------------------+----------------------------------------------------------------------------------------------------------------+

+-------------------------------------+-------------------------------------------------------+
| Edges                               | Element                                               |
+=====================================+=======================================================+
| (:math:`p_{1^*}`, :math:`p_{2^*}`)  | (:math:`p_{1^*}`, :math:`p_{2^*}`, :math:`p_{3^*}`)   |
+-------------------------------------+-------------------------------------------------------+
| (:math:`p_{2^*}`, :math:`p_{3^*}`)  | (:math:`p_{1^*}`, :math:`p_{2^*}`, :math:`p_{3^*}`)   |
+-------------------------------------+-------------------------------------------------------+
| (:math:`p_{3^*}`, :math:`p_{2^*}`)  | (:math:`p_{1^*}`, :math:`p_{2^*}`, :math:`p_{3^*}`)   |
+-------------------------------------+-------------------------------------------------------+

Bowyer-Watson: Point Insertion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 6

    .. grid-item::
        .. graphviz::

            graph supertriangle {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];

                s1 [pos="0.4667, -0.4585!", color="#d08770"];
                s2 [pos="1.3905, 1.1417!", color="#d08770"];
                s3 [pos="-0.4572, 1.1417!", color="#d08770"];

                s1 -- s2 [penwidth=1, color="#d08770"];
                s2 -- s3 [penwidth=1, color="#d08770"];
                s3 -- s1 [penwidth=1, color="#d08770"]; 
                
                s1 -- a  [style="dotted"]
                s2 -- a  [style="dotted"]
                s3 -- a  [style="dotted"]
            }
    .. grid-item::
        .. graphviz::

            graph supertriangle {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];

                s1 [pos="0.4667, -0.4585!", color="#d08770"];
                s2 [pos="1.3905, 1.1417!", color="#d08770"];
                s3 [pos="-0.4572, 1.1417!", color="#d08770"];

                s1 -- s2 [penwidth=1, color="#d08770"];
                s2 -- s3 [penwidth=1, color="#d08770"];
                s3 -- s1 [penwidth=1, color="#d08770"]; 

                s1 -- a [style="dotted"]
                s2 -- a [style="dotted"]
                s3 -- a [style="dotted"]
                s2 -- b [style="dotted"]
                s3 -- b [style="dotted"]
                a -- b
            }
    .. grid-item::
        .. graphviz::

            graph supertriangle {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];
                c [pos="0.1, 0.9!"];

                s1 [pos="0.4667, -0.4585!", color="#d08770"];
                s2 [pos="1.3905, 1.1417!", color="#d08770"];
                s3 [pos="-0.4572, 1.1417!", color="#d08770"];

                s1 -- s2 [penwidth=1, color="#d08770"];
                s2 -- s3 [penwidth=1, color="#d08770"];
                s3 -- s1 [penwidth=1, color="#d08770"]; 

                s1 -- a [style="dotted"]
                s2 -- a [style="dotted"]
                s2 -- b [style="dotted"]
                s3 -- b [style="dotted"]
                s1 -- c [style="dotted"]
                s3 -- c [style="dotted"]
                a -- b
                b -- c
                c -- a
            }
    .. grid-item::
        .. graphviz::

            graph supertriangle {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];
                c [pos="0.1, 0.9!"];
                d [pos="0.0, .45!"];

                s1 [pos="0.4667, -0.4585!", color="#d08770"];
                s2 [pos="1.3905, 1.1417!", color="#d08770"];
                s3 [pos="-0.4572, 1.1417!", color="#d08770"];

                s1 -- s2 [penwidth=1, color="#d08770"];
                s2 -- s3 [penwidth=1, color="#d08770"];
                s3 -- s1 [penwidth=1, color="#d08770"];

                s1 -- a [style="dotted"]
                s2 -- a [style="dotted"]
                s2 -- b [style="dotted"]
                s3 -- b [style="dotted"]
                s3 -- c [style="dotted"]
                s3 -- d [style="dotted"]
                s1 -- d [style="dotted"]
                a -- b
                b -- c
                c -- a
                d -- c 
                d -- a
            }
    .. grid-item::
        .. graphviz::

            graph supertriangle {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];
                c [pos="0.1, 0.9!"];
                d [pos="0.0, .45!"];
                e [pos="0.5, 0.1!"];

                s1 [pos="0.4667, -0.4585!", color="#d08770"];
                s2 [pos="1.3905, 1.1417!", color="#d08770"];
                s3 [pos="-0.4572, 1.1417!", color="#d08770"];

                s1 -- s2 [penwidth=1, color="#d08770"];
                s2 -- s3 [penwidth=1, color="#d08770"];
                s3 -- s1 [penwidth=1, color="#d08770"]; 

                s1 -- a [style="dotted"]
                s2 -- a [style="dotted"]
                s2 -- b [style="dotted"]
                s3 -- b [style="dotted"]
                s3 -- c [style="dotted"]
                s3 -- d [style="dotted"]
                s1 -- d [style="dotted"]
                s1 -- e [style="dotted"]
                a -- b
                b -- c
                d -- c 
                d -- e
                e -- a
                e -- b
                e -- c
                
            }
    .. grid-item::
        .. graphviz::

            graph supertriangle {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];
                c [pos="0.1, 0.9!"];
                d [pos="0.0, .45!"];
                e [pos="0.5, 0.1!"];
                f [pos="0.4, 0.6!"];

                s1 [pos="0.4667, -0.4585!", color="#d08770"];
                s2 [pos="1.3905, 1.1417!", color="#d08770"];
                s3 [pos="-0.4572, 1.1417!", color="#d08770"];

                s1 -- s2 [penwidth=1, color="#d08770"];
                s2 -- s3 [penwidth=1, color="#d08770"];
                s3 -- s1 [penwidth=1, color="#d08770"]; 

                s1 -- a [style="dotted"]
                s2 -- a [style="dotted"]
                s2 -- b [style="dotted"]
                s3 -- b [style="dotted"]
                s3 -- c [style="dotted"]
                s3 -- d [style="dotted"]
                s1 -- d [style="dotted"]
                s1 -- e [style="dotted"]
                a -- b
                b -- c
                c -- d
                d -- e
                e -- a
                a -- f
                b -- f
                c -- f
                d -- f
                e -- f
                
            }

The basic point insertion procedure is for adding point :math:`p_i` is as 
follows:

    1. **Point Location**: Locate the existing triangle :math:`t_p` that contains 
       :math:`p_i`.
    2. **Cavity Formation**: Identify the set of triangles :math:`\{t_j\}` whose 
       circumcircles contain :math:`p_i` and remove them from the triangulation, 
       leaving behind a  star-shaped cavity.
    3. **Retriangulation**: Re-triangulate the cavity by connecting each vertex on 
       the cavity boundary to the newly inserted point


Point Location
""""""""""""""
CITE

To start the point insertion process, it's necessary to find all triangles in 
the existing triangulation the need to be removed (those whose circumcircles
contain the new point). These triangles will all be connected, and the triangle
(:math:`t_p`) that contains the point is surely among them, so it's useful to 
start by identifying that triangle. Points can be tested for inclusion in a
triangle by calculating the barycentric coordinates 
(:func:`~mymesh.utils.BaryTri`).

To find the triangle that contains :math:`p_i` without having to test every
triangle, a walking algorithm can be used to move from a randomly selected
starting triangle towards :math:`t_p`. If all of the barycentric 
coordinates of :math:`p_i` in a triangle :math:`t_j` (:math:`\alpha_0`, 
:math:`\beta_0`, :math:`\gamma_0`) are positive, then :math:`p_i` is in 
:math:`t_j` and the algorithm terminates. Otherwise, the most negative 
barycentric coordinate indicates the vertex of :math:`t_j` furthest from 
:math:`p_i`, so taking a step into the triangle connect to :math:`t_j` across the 
edge opposite that vertex will be a step towards :math:`p_i`. Repeating this 
process will create a path that leads directly to identifying the triangle 
:math:`t_p` tha contains :math:`p_i`

.. graphviz::

    graph supertriangle {
        node [shape=point, fontname="source code pro"];
        edge [style=solid];

        a [pos="1.0, 0.6!"];
        b [pos="0.8, 1.0!"];
        c [pos="0.1, 0.9!"];
        d [pos="0.0, .45!"];
        e [pos="0.5, 0.1!"];
        f [pos="0.4, 0.6!", color="#5E81AC"];

        s1 [pos="0.4667, -0.4585!", color="#A3BE8C"];
        s2 [pos="1.3905, 1.1417!", color="#A3BE8C"];
        s3 [pos="-0.4572, 1.1417!", color="#A3BE8C"];

        s1 -- s2 [penwidth=1, color="#A3BE8C"];
        s2 -- s3 [penwidth=1, color="#A3BE8C"];
        s3 -- s1 [penwidth=1, color="#A3BE8C"]; 

        s1 -- a [style="dotted"]
        s2 -- a [style="dotted"]
        s2 -- b [style="dotted"]
        s3 -- b [style="dotted"]
        s3 -- c [style="dotted"]
        s3 -- d [style="dotted"]
        s1 -- d [style="dotted"]
        s1 -- e [style="dotted"]
        a -- b
        b -- c [color="#000000:#BF616A"]
        d -- c 
        d -- e
        e -- a
        e -- b [color="#000000:#BF616A"]
        c -- e [color="#000000:#BF616A"]

        p1 [pos="1.0635,0.9139!", width=0]
        p2 [pos="0.7667,0.5667!", width=0]
        p1 -- p2
        p2 -- f
        
    }

Cavity Formation
""""""""""""""""
Having found the triangle :math:`t_p` that contains the point :math:`p_i`, it is next
necessary to identify all other triangles whose circumcircles contain :math:`p_i`.
It can be shown that these triangles are all connected and when removed form a
"star-shaped"[#f1]_ cavity. Starting from :math:`t_p`, the cavity can be 
expanded outwards, checking each adjacent triangle. If the triangle is invalid
(i.e. its circumcircle contains :math:`p_i`), its marked for deletion and its 
neighbors are added to the queue to be checked. When moving from an invalid
triangle across an edge into a valid triangle, that edge is marked as a boundary
of the cavity. 

As noted previously, it's important to ensure that the vertices of the 
super-triangle don't interfere with the triangulation of the interior points.
Rather than placing the vertices of the super triangle infinitely far away, 
testing of triangles that contain super-triangle vertices can be handled as
special cases when building the cavity:

    1. If crossing an edge *not* touching the super-triangle into a triangle 
       that *is* touching the super-triangle, mark that edge as a boundary 
       of the cavity treat that triangle as if it doesn't have the point in 
       it's circumcircle.



.. [#f1] 
    A polygon being star-shaped means that there is at least one point in the polygon 
    that is "visible" from each vertex, i.e. lines can be drawn from the vertices to
    the point without intersecting the boundaries of the polygon.

Circumcircle Test
"""""""""""""""""
Testing whether or not a point is in a triangle's circumcircle is an essential
part of most Delaunay triangulation algorithms, and the efficiency of the test
is key to the efficiency of the overall algorithm. A popular test to determine
if a point :math:`p` is in a triangle :math:`t` with counter-clockwise vertices 
:math:`a, b, c` involves solving the determinant:

.. math::

    v = \det
    \begin{pmatrix} 
        a_x & a_y & a_x^2 + a_y^2 & 1 \\
        b_x & b_y & b_x^2 + b_y^2 & 1 \\
        c_x & c_y & c_x^2 + c_y^2 & 1 \\
        p_x & p_y & p_x^2 + p_y^2 & 1 \\
    \end{pmatrix} 

If :math:`v > 0`, then :math:`p` is in :math:`t`. 

This test can be geometrically interpreted as a projection of the four points 
:math:`a, b, c, p` onto the paraboloid :math:`z=x^2+y^2` and calculation of the
signed volume of the tetrahedron formed by :math:`a, b, c, p`. If :math:`p` lies
exactly on the circumcircle of :math:`t`, the four points will be coplanar on
the paraboloid and the signed volume :math:`v = 0`, while :math:`p` inside the 
circumcircle will lead to a tetrahedron with a positive signed volume. 



.. grid:: 2

    .. grid-item::
        .. graphviz::

            graph supertriangle {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];
                c [pos="0.1, 0.9!"];
                d [pos="0.0, .45!"];
                e [pos="0.5, 0.1!"];
                f [pos="0.4, 0.6!", color="#5E81AC"];

                s1 [pos="0.4667, -0.4585!", color="#A3BE8C"];
                s2 [pos="1.3905, 1.1417!", color="#A3BE8C"];
                s3 [pos="-0.4572, 1.1417!", color="#A3BE8C"];

                s1 -- s2 [penwidth=1, color="#A3BE8C"];
                s2 -- s3 [penwidth=1, color="#A3BE8C"];
                s3 -- s1 [penwidth=1, color="#A3BE8C"]; 

                s1 -- a [style="dotted"]
                s2 -- a [style="dotted"]
                s2 -- b [style="dotted"]
                s3 -- b [style="dotted"]
                s3 -- c [style="dotted"]
                s3 -- d [style="dotted"]
                s1 -- d [style="dotted"]
                s1 -- e [style="dotted"]
                a -- b [color="#000000:#BF616A"]
                b -- c [color="#000000:#BF616A"]
                c -- d [color="#000000:#BF616A"]
                d -- e [color="#000000:#BF616A"]
                e -- a [color="#000000:#BF616A"]
                e -- b [color="#BF616A"]
                e -- c [color="#BF616A"]

                c1 [pos="0.5, 0.6!", shape=circle, width=1, label="", color="#BF616A"]
                c2 [pos="0.4654, 0.5827!", shape=circle, width=0.96786, label="", color="#BF616A"]

            }
    
    .. grid-item::
        .. graphviz::

            graph supertriangle {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];
                c [pos="0.1, 0.9!"];
                d [pos="0.0, .45!"];
                e [pos="0.5, 0.1!"];
                f [pos="0.4, 0.6!", color="#5E81AC"];

                s1 [pos="0.4667, -0.4585!", color="#A3BE8C"];
                s2 [pos="1.3905, 1.1417!", color="#A3BE8C"];
                s3 [pos="-0.4572, 1.1417!", color="#A3BE8C"];

                s1 -- s2 [penwidth=1, color="#A3BE8C"];
                s2 -- s3 [penwidth=1, color="#A3BE8C"];
                s3 -- s1 [penwidth=1, color="#A3BE8C"]; 

                s1 -- a [style="dotted"]
                s2 -- a [style="dotted"]
                s2 -- b [style="dotted"]
                s3 -- b [style="dotted"]
                s3 -- c [style="dotted"]
                s3 -- d [style="dotted"]
                s1 -- d [style="dotted"]
                s1 -- e [style="dotted"]
                a -- b [color="#000000:#5E81AC"]
                b -- c [color="#000000:#5E81AC"]
                c -- d [color="#000000:#5E81AC"]
                d -- e [color="#000000:#5E81AC"]
                e -- a [color="#000000:#5E81AC"]


            }

Retriangulation
"""""""""""""""

Once the cavity has been formed and the edges at the boundary of the cavity 
identified, the vertices of the cavity can be simply connected to the inserted
point to retriangulate the cavity. Due to the oriented nature of the half-edges
in the data structure, the triangles can be formed in a way that ensures that 
the points of every triangle are ordered counter clockwise. 



.. grid:: 2

    .. grid-item::
        .. graphviz::

            graph supertriangle {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];
                c [pos="0.1, 0.9!"];
                d [pos="0.0, .45!"];
                e [pos="0.5, 0.1!"];
                f [pos="0.4, 0.6!", color="#5E81AC"];

                s1 [pos="0.4667, -0.4585!", color="#A3BE8C"];
                s2 [pos="1.3905, 1.1417!", color="#A3BE8C"];
                s3 [pos="-0.4572, 1.1417!", color="#A3BE8C"];

                s1 -- s2 [penwidth=1, color="#A3BE8C"];
                s2 -- s3 [penwidth=1, color="#A3BE8C"];
                s3 -- s1 [penwidth=1, color="#A3BE8C"]; 

                s1 -- a [style="dotted"]
                s2 -- a [style="dotted"]
                s2 -- b [style="dotted"]
                s3 -- b [style="dotted"]
                s3 -- c [style="dotted"]
                s3 -- d [style="dotted"]
                s1 -- d [style="dotted"]
                s1 -- e [style="dotted"]
                a -- b [color="#000000:#5E81AC"]
                b -- c [color="#000000:#5E81AC"]
                c -- d [color="#000000:#5E81AC"]
                d -- e [color="#000000:#5E81AC"]
                e -- a [color="#000000:#5E81AC"]

                a -- f [color="#5E81AC"]
                b -- f [color="#5E81AC"]
                c -- f [color="#5E81AC"]
                d -- f [color="#5E81AC"]
                e -- f [color="#5E81AC"]
            }

    .. grid-item::
        .. graphviz::

            graph supertriangle {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];
                c [pos="0.1, 0.9!"];
                d [pos="0.0, .45!"];
                e [pos="0.5, 0.1!"];
                f [pos="0.4, 0.6!"];

                s1 [pos="0.4667, -0.4585!", color=white];
                s2 [pos="1.3905, 1.1417!", color=white];
                s3 [pos="-0.4572, 1.1417!", color=white];

                a -- b
                b -- c 
                c -- d 
                d -- e 
                e -- a 

                a -- f
                b -- f
                c -- f
                d -- f 
                e -- f 
            }