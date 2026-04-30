Implicit Meshing
================

Mesh generation based on implicit functions is a major focus of MyMesh. It
offers a way to generate surface and volume meshes of objects defined by 
mathematical functions. 

What is an implicit function?
-----------------------------
In three dimensions, implicit functions represent surfaces with the form:

.. math::
    f(x,y,z) = 0

For example, a sphere defined by :math:`x^2+y^2+z^2=r^2` can be represented 
implicitly by the function :math:`f(x,y,z) = x^2+y^2+z^2 - r^2 = 0`. Any point
satisfying the condition :math:`f(x,y,z) = 0` is considered to be on the surface,
while points where :math:`f(x,y,z) < 0` are considered to be inside the surface
and points where :math:`f(x,y,z) > 0` are considered to be outside the surface.
This convention is assumed by default throughout the implicit meshing functions
of MyMesh (others may adopt the opposite convention elsewhere). 

Defining implicit functions
---------------------------
Implicit functions being defined for use with MyMesh should generally be defined
as functions of three variables (:code:`x`, :code:`y`, :code:`z`) and accept 
vectorizable inputs (if :code:`x`, :code:`y`, :code:`z` are scalars, 
:code:`func(x, y, z)` should return a scalar, if :code:`x`, :code:`y`, :code:`z` 
are numpy arrays, :code:`func(x, y, z)` should return an array). For example, the 
`"surface of genus 2" example from wikipedia <https://en.wikipedia.org/wiki/Implicit_surface>`_ 
can be defined as follows:

.. plot::
    :context: reset

    def example_func(x,y,z):
        f = 2*y*(y**2 - 3*x**2)*(1-z**2) + (x**2 + y**2)**2 - (9*z**2 - 1)*(1-z**2)
        return f

Several built-in implicit function are available in :mod:`mymesh.implicit`, 
including a sphere, box, and torus. For example, the implicit function of 
a unit sphere (center = (0,0,0), radius=1) can be obtained from: 
:code:`func = implicit.sphere([0,0,0], 1)`

More complex geometries can be obtained by combining multiple functions using
union, intersection, and difference operations (see :ref:`Implicit CSG` for 
more details).

Meshing Implicit Functions
--------------------------

Surface Meshing


.. plot::
    :context: close-figs
    :nofigs: 

    m = implicit.SurfaceMesh(example_func, [-2,2,-2,2,-2,2], .05)

.. grid:: 2

    .. grid-item::
        .. plot::
            :context: close-figs
            :include-source: False

            m.plot(bgcolor='w', view='iso')

    .. grid-item::
        .. plot::
            :context: close-figs
            :include-source: False

            m.Clip().plot(bgcolor='w', view='iso')

Volume Meshing



.. plot::
    :context: close-figs
    :nofigs: 

    m = implicit.VoxelMesh(example_func, [-2,2,-2,2,-2,2], .1)

.. grid:: 2

    .. grid-item::
        .. plot::
            :context: close-figs
            :include-source: False

            m.plot(bgcolor='w', view='iso', show_edges=True)

    .. grid-item::
        .. plot::
            :context: close-figs
            :include-source: False

            m.Clip().plot(bgcolor='w', view='iso', show_edges=True)

.. plot::
    :context: close-figs
    :nofigs: 

    m = implicit.TetMesh(example_func, [-2,2,-2,2,-2,2], .05)

.. grid:: 2

    .. grid-item::
        .. plot::
            :context: close-figs
            :include-source: False

            m.plot(bgcolor='w', view='iso')

    .. grid-item::
        .. plot::
            :context: close-figs
            :include-source: False

            m.Clip().plot(bgcolor='w', view='iso')