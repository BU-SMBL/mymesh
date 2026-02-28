Curvature
=========

Accuracy
--------

The following benchmark compares the accuracy of the principal curvatures for three meshes (a sphere, a sphere with improved quality mesh, and a torus) as computed by three :mod:`mymesh` methods (:func:`~mymesh.curvature.QuadFit`, :func:`~mymesh.curvature.CubicFit`, :func:`~mymesh.curvature.AnalyticalCurvature`) and the methods of two other libraries: VTK (via pyvista) and trimesh. 
For the unit sphere, both principal curvatures are theoretically equal to 1, and for the torus with minor radius 1, the maximum principal curvature is theoretically equal to 1 while the minimum principal curvature varies spatially.

The results highlight two points: 

1. the sensitivity of mesh-based curvature to the quality of the mesh (as evidenced by the more accurate results for the smoothed sphere mesh), and 
2. that the methods implemented in :mod:`mymesh` are comparable or superior to the alternatives that were tested (at least in these test cases). 

The best performing method in all cases was the implicit function-based curvature (:func:`~mymesh.curvature.AnalyticalCurvature`), however this is only possible when an implicit function representation of the surface exists. 
The cubic surface fitting method (:func:`~mymesh.curvature.CubicFit`, :cite:`Goldfeather2004`) is generally superior to other mesh-based approaches, but may be more susceptible to errors due to the low-quality elements. 

See also: :ref:`theory_curvature`, :mod:`~mymesh.curvature`, :ref:`Curvature Analysis`. 

.. plot::
    :context: close-figs

    import mymesh
    from mymesh import curvature, implicit
    import numpy as np
    import matplotlib.pyplot as plt
    import pyvista as pv
    import trimesh

    def test_curvature(M, func=None):
        # Function to test different methods of curvature calculation

        # mymesh - quadratic fitting
        k1_quadratic, k2_quadratic = curvature.QuadFit(M.NodeCoords, 
                                                M.NodeConn, 
                                                M.NodeNeighbors, 
                                                M.NodeNormals)

        # mymesh - cubic fitting
        k1_cubic, k2_cubic = curvature.CubicFit(M.NodeCoords, 
                                                M.NodeConn, 
                                                M.NodeNeighbors, 
                                                M.NodeNormals)
        

        # vtk (via pyvista)
        k1_vtk = M.to_pyvista().extract_surface().curvature(curv_type='maximum')
        k2_vtk = M.to_pyvista().extract_surface().curvature(curv_type='minimum')

        # trimesh
        r = 0.05
        M_trimesh = trimesh.base.Trimesh(M.NodeCoords, M.NodeConn)
        K_trimesh = trimesh.curvature.discrete_gaussian_curvature_measure(M_trimesh, M_trimesh.vertices, r) / trimesh.curvature.sphere_ball_intersection(1, r)
        H_trimesh = trimesh.curvature.discrete_mean_curvature_measure (M_trimesh, M_trimesh.vertices, r) / trimesh.curvature.sphere_ball_intersection(1, r)
        k1_trimesh = H_trimesh + np.sqrt(np.maximum(H_trimesh**2-K_trimesh,0))
        k2_trimesh = H_trimesh - np.sqrt(np.maximum(H_trimesh**2-K_trimesh,0))

        
        if func is not None:
            # mymesh - implicit
            k1_implicit, k2_implicit, K_implicit, H_implicit = curvature.AnalyticalCurvature(func, M.NodeCoords)
            
            return (k1_quadratic, k2_quadratic), \
                    (k1_cubic, k2_cubic), \
                    (k1_vtk, k2_vtk), \
                    (k1_trimesh, k2_trimesh), \
                    (k1_implicit, k2_implicit)

        return (k1_quadratic, k2_quadratic), \
                    (k1_cubic, k2_cubic), \
                    (k1_vtk, k2_vtk), \
                    (k1_trimesh, k2_trimesh)

.. plot::
    :include-source: False
    :show-source-link: True
    :context: close-figs

    func = implicit.sphere([0,0,0], 1)
    Sphere = implicit.SurfaceMesh(func, [-1,1,-1,1,-1,1], 0.1)
    Sphere.verbose=False

    (k1_quadratic_S, k2_quadratic_S), \
    (k1_cubic_S, k2_cubic_S), \
    (k1_vtk_S, k2_vtk_S), \
    (k1_trimesh_S, k2_trimesh_S), \
    (k1_implicit_S, k2_implicit_S) = test_curvature(Sphere, func)

    SmoothSphere = mymesh.improvement.Flip(Sphere, strategy='valence')
    SmoothSphere = implicit.SurfaceNodeOptimization(SmoothSphere, implicit.sphere([0,0,0], 1), 0.1, iterate=10)
    SmoothSphere.verbose=False
    
    (k1_quadratic_Smooth, k2_quadratic_Smooth), \
    (k1_cubic_Smooth, k2_cubic_Smooth), \
    (k1_vtk_Smooth, k2_vtk_Smooth), \
    (k1_trimesh_Smooth, k2_trimesh_Smooth), \
    (k1_implicit_Smooth, k2_implicit_Smooth) = test_curvature(SmoothSphere, func)

    func = implicit.torus([0,0,0], 2, 1)
    Torus = implicit.SurfaceMesh(func, [-3,3,-3,3,-3,3], 0.2)
    Torus = mymesh.improvement.Flip(Torus, strategy='valence')
    SmoothTorus = implicit.SurfaceNodeOptimization(Torus, func, 0.1, iterate=10)
    SmoothTorus.verbose=False
    
    (k1_quadratic_T, k2_quadratic_T), \
    (k1_cubic_T, k2_cubic_T), \
    (k1_vtk_T, k2_vtk_T), \
    (k1_trimesh_T, k2_trimesh_T), \
    (k1_implicit_T, k2_implicit_T) = test_curvature(SmoothTorus, func)

    # Plot meshes
    mymesh.visualize.Subplot([Sphere, SmoothSphere, SmoothTorus], (1,3), titles=['Sphere', 'Smooth Sphere', 'Torus'], figsize=(18,5), show_edges=True)

    # Plot curvatures
    plt.subplots(1, 3, figsize=(18,5))
    plt.subplot(1, 3, 1)
    
    plt.boxplot([k1_quadratic_S[~np.isnan(k1_quadratic_S)], 
                 k1_cubic_S[~np.isnan(k1_cubic_S)], 
                 k1_implicit_S, k1_vtk_S, k1_trimesh_S],
                positions=[.75,1.75,2.75,3.75,4.75], widths=0.3,
                medianprops=dict(color='blue'), label='max principal'
                )
    plt.boxplot([k2_quadratic_S[~np.isnan(k2_quadratic_S)], 
                 k2_cubic_S[~np.isnan(k2_cubic_S)], 
                 k2_implicit_S, k2_vtk_S, k1_trimesh_S],
                positions=[1.25,2.25,3.25,4.25,5.25], widths=0.3,
                medianprops=dict(color='orange'), label='min principal')
    plt.xticks(ticks=[1,2,3,4,5],
            labels=['mymesh\nquad-fit','mymesh\ncubic-fit','mymesh\nimplicit','vtk','trimesh'])
    plt.axhline(1, linestyle=':', color='black')
    plt.ylim(0,3)
    plt.ylabel('Principal curvature (1/length)')
    plt.legend()
    plt.title('Sphere')
    
    plt.subplot(1, 3, 2)
    plt.boxplot([k1_quadratic_Smooth[~np.isnan(k1_quadratic_Smooth)], 
                 k1_cubic_Smooth[~np.isnan(k1_cubic_Smooth)], 
                 k1_implicit_Smooth, k1_vtk_Smooth, k1_trimesh_Smooth],
                positions=[.75,1.75,2.75,3.75,4.75], widths=0.3,
                medianprops=dict(color='blue'), label='k1')
    plt.boxplot([k2_quadratic_Smooth[~np.isnan(k2_quadratic_Smooth)],
                 k2_cubic_Smooth[~np.isnan(k2_cubic_Smooth)], 
                 k2_implicit_Smooth, k2_vtk_Smooth, k2_trimesh_Smooth],
                positions=[1.25,2.25,3.25,4.25,5.25], widths=0.3,
                medianprops=dict(color='orange'), label='k2')
    plt.xticks(ticks=[1,2,3,4,5],
            labels=['mymesh\nquad-fit','mymesh\ncubic-fit','mymesh\nimplicit','vtk','trimesh'])
    plt.axhline(1, linestyle=':', color='black')
    plt.ylim(0,3)
    plt.title('Smooth Sphere')
    
    plt.subplot(1, 3, 3)
    plt.boxplot([k1_quadratic_T[~np.isnan(k1_quadratic_T)], 
                 k1_cubic_T[~np.isnan(k1_cubic_T)], 
                 k1_implicit_T, k1_vtk_T, k1_trimesh_T],
                positions=[1,2,3,4,5], widths=0.4,
                medianprops=dict(color='blue'), label='k1')
    plt.xticks(ticks=[1,2,3,4,5],
            labels=['mymesh\nquad-fit','mymesh\ncubic-fit','mymesh\nimplicit','vtk','trimesh'])
    plt.axhline(1, linestyle=':', color='black')
    plt.ylim(0,3)
    plt.title('Smooth Torus')
    plt.show()
