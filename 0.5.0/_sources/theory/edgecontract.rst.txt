Edge Contraction
================
See also :func:`~mymesh.improvement.Contract`

Reference: :cite:t:`Faraj2016`


Valid contraction:

.. grid:: 2

    .. grid-item::
        .. graphviz::

            graph initial {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];
                c [pos="0.1, 0.9!"];
                d [pos="0.0, .45!"];
                e [pos="0.65, 0.1!"];
                f [pos="0.3, 0.6!"];
                g [pos="0.5, 0.6!"];

                s1 [pos="0.4667, -0.4585!", color=white];
                s2 [pos="1.3905, 1.1417!", color=white];
                s3 [pos="-0.4572, 1.1417!", color=white];

                a -- b
                b -- c 
                c -- d 
                d -- e 
                e -- a 

                a -- g
                b -- g
                e -- g
                g -- f [dir=forward, color="#bf616a", arrowsize=0.5]
                b -- f
                c -- f
                d -- f 
                e -- f 
            }

    .. grid-item::
        .. graphviz::

            graph initial {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];
                c [pos="0.1, 0.9!"];
                d [pos="0.0, .45!"];
                e [pos="0.65, 0.1!"];
                f [pos="0.3, 0.6!"];
                g [pos="0.3, 0.6!"];

                s1 [pos="0.4667, -0.4585!", color=white];
                s2 [pos="1.3905, 1.1417!", color=white];
                s3 [pos="-0.4572, 1.1417!", color=white];

                a -- b
                b -- c 
                c -- d 
                d -- e 
                e -- a 

                a -- g
                b -- g
                e -- g
                f -- g [color="#bf616a", arrowsize=0.5]
                b -- f
                c -- f
                d -- f 
                e -- f 
            }



Invalid contraction:

.. grid:: 2

    .. grid-item::
        .. graphviz::

            graph initial {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];
                c [pos="0.1, 0.9!"];
                d [pos="0.0, .45!"];
                e [pos="0.65, 0.1!"];
                f [pos="0.3, 0.6!"];
                g [pos="0.5, 0.6!"];

                s1 [pos="0.4667, -0.4585!", color=white];
                s2 [pos="1.3905, 1.1417!", color=white];
                s3 [pos="-0.4572, 1.1417!", color=white];

                a -- b
                b -- c 
                c -- d 
                d -- e 
                e -- a 

                a -- g
                b -- g
                e -- g
                g -- f 
                b -- f
                c -- f
                d -- f 
                e -- f [dir=back, color="#bf616a", arrowsize=0.5]
            }

    .. grid-item::
        .. graphviz::

            graph initial {
                node [shape=point, fontname="source code pro"];
                edge [style=solid];

                a [pos="1.0, 0.6!"];
                b [pos="0.8, 1.0!"];
                c [pos="0.1, 0.9!"];
                d [pos="0.0, .45!"];
                e [pos="0.65, 0.1!"];
                f [pos="0.65, 0.1!"];
                g [pos="0.5, 0.6!"];

                s1 [pos="0.4667, -0.4585!", color=white];
                s2 [pos="1.3905, 1.1417!", color=white];
                s3 [pos="-0.4572, 1.1417!", color=white];

                a -- b
                b -- c 
                c -- d 
                d -- e 
                e -- a 

                a -- g
                b -- g [color="#b48ead"]
                e -- g
                f -- g [color="#b48ead"]
                b -- f [color="#b48ead"]
                c -- f
                d -- f 
                e -- f 
            }