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