Delaunay
========

Delaunay triangulation is a well studied problem, and there are several well-established and commonly used programs for achieving Delaunay triangulations and tetrahedralizations, with and without constraints (constrained Delaunay triangulation, CDT).

The current implementations in :mod:`mymesh` don't yet meet or exceed the performance or functionality of other available options (though this is an area of active development), so using well established implementations is recommended:

- :code:`qhull` (available via :class:`scipy.spatial.Delaunay`, or :func:`mymesh.delaunay.SciPy`) for n dimensional Delaunay triangulations
- :code:`triangle` for 2 dimensional constrained/unconstrained Delaunay triangulations
- :code:`TetGen` for 3 dimensional constrained/unconstrained Delaunay triangulations
- :code:`gmsh`
- :code:`CGAL`

Here you can see a comparison between the performance of :mod:`mymesh` and some alternatives for Delaunay triangulation and other related methods.

See also: :ref:`Delaunay Triangulation`, :mod:`mymesh.delaunay`


2D Delaunay Triangulation
-------------------------

.. plot::
    :include-source: False
    :show-source-link: True

    import sys, os, time, subprocess
    import scipy
    import numpy as np
    import matplotlib.pyplot as plt

    def run_no_numba(npoints, seed):
        code = f"""
    import os, subprocess, time
    import numpy as np
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    from mymesh import delaunay
    # pre-run in case of any first time delays
    delaunay.BowyerWatson2d(np.random.rand(10,2))
    np.random.seed({seed})
    points = np.random.rand({npoints}, 2)
    tic = time.time()
    delaunay.BowyerWatson2d(points)
    t = time.time()-tic
    print(t)
    """
        result = subprocess.run([sys.executable, "-c", code], capture_output=True)
        t = float(result.stdout)

        return t


    from mymesh import delaunay
    npoints = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=int)

    bowyer_watson_avg = []
    bowyer_watson_std = []
    bowyer_watson_no_numba_avg = []
    bowyer_watson_no_numba_std = []
    scipy_avg = []
    scipy_std = []
    triangle_avg = []
    triangle_std = []

    # pre-run for any first time pre-compiling
    points = np.random.rand(10,2)
    delaunay.BowyerWatson2d(points)
    scipy.spatial.Delaunay(points)
    delaunay.Triangle(points)

    N = 3
    for i,n in enumerate(npoints):
        # BowyerWatson
        if i < 2 or bowyer_watson_avg[-1] < 5:
            reps = []
            for j in range(N):
                np.random.seed(j)
                points = np.random.rand(n, 2)
                tic = time.time()
                delaunay.BowyerWatson2d(points)
                reps.append(time.time()-tic)

            bowyer_watson_avg.append(np.mean(reps))
            bowyer_watson_std.append(np.std(reps))

        # BowyerWatson - No Numba
        if i < 2 or bowyer_watson_no_numba_avg[-1] < 5:
            reps = []
            for j in range(N):
                t = run_no_numba(n, j)
                reps.append(t)
            bowyer_watson_no_numba_avg.append(np.mean(reps)/N)
            bowyer_watson_no_numba_std.append(np.std(reps)/N)

        # SciPy
        if i < 2 or scipy_avg[-1] < 5:
            reps = []
            for j in range(N):
                np.random.seed(j)
                points = np.random.rand(n, 2)
                tic = time.time()
                scipy.spatial.Delaunay(points)
                reps.append(time.time()-tic)
            scipy_avg.append(np.mean(reps))
            scipy_std.append(np.std(reps))

        # Triangle
        if i < 2 or triangle_avg[-1] < 5:
            reps = []
            for j in range(N):
                np.random.seed(j)
                points = np.random.rand(n, 2)
                tic = time.time()
                delaunay.Triangle(points)
                reps.append(time.time()-tic)
            triangle_avg.append(np.mean(reps))
            triangle_std.append(np.std(reps))

    # Plot
    plt.errorbar(npoints[:len(bowyer_watson_avg)], bowyer_watson_avg, yerr=bowyer_watson_std, color='red')
    plt.errorbar(npoints[:len(bowyer_watson_no_numba_avg)], bowyer_watson_no_numba_avg, yerr=bowyer_watson_no_numba_std, color='pink')
    plt.errorbar(npoints[:len(scipy_avg)], scipy_avg, yerr=scipy_std, color='green')
    plt.errorbar(npoints[:len(triangle_avg)], triangle_avg, yerr=triangle_std, color='purple')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['mymesh', 'mymesh (no numba)', 'scipy (qhull)', 'Triangle'])
    plt.xlabel('# of points')
    plt.ylabel('Time (s)')
    plt.title('2D Delaunay Triangulation')
    plt.grid()
    plt.show()

2D Constrained Delaunay Triangulation
-------------------------------------

.. plot::
    :include-source: False
    :show-source-link: True

    import sys, os, time, subprocess
    import scipy
    import numpy as np
    import matplotlib.pyplot as plt

    def run_no_numba(npoints, seed):
        code = f"""
    import os, subprocess, time
    import numpy as np
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    from mymesh import delaunay
    # pre-run in case of any first time delays
    delaunay.BowyerWatson2d(np.random.rand(10,2), Constraints=np.random.choice(np.arange(0, 10, dtype=int), size=(3,2), replace=False))

    np.random.seed({seed})
    points = np.random.rand({npoints}, 2)
    constraints = np.random.choice(np.arange(0, len(points), dtype=int), size=(3,2),replace=False)
    tic = time.time()
    delaunay.BowyerWatson2d(points, Constraints=constraints)
    t = time.time()-tic
    print(t)
    """
        result = subprocess.run([sys.executable, "-c", code], capture_output=True)
        t = float(result.stdout)

        return t


    from mymesh import delaunay
    npoints = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=int)

    bowyer_watson_avg = []
    bowyer_watson_std = []
    bowyer_watson_no_numba_avg = []
    bowyer_watson_no_numba_std = []
    triangle_avg = []
    triangle_std = []

    # pre-run for any first time pre-compiling
    points = np.random.rand(10,2)
    constraints = np.random.choice(np.arange(0, len(points), dtype=int), size=(5,2),replace=False)
    delaunay.BowyerWatson2d(points, Constraints=constraints)
    delaunay.Triangle(points, Constraints=constraints)
    #
    N = 3
    for i,n in enumerate(npoints):
        # BowyerWatson
        if i < 2 or bowyer_watson_avg[-1] < 5:
            reps = []
            for j in range(N):
                np.random.seed(j)
                points = np.random.rand(n, 2)
                constraints = np.random.choice(np.arange(0, len(points), dtype=int), size=(5,2),replace=False)
                tic = time.time()
                delaunay.BowyerWatson2d(points, Constraints=constraints)
                reps.append(time.time()-tic)

            bowyer_watson_avg.append(np.mean(reps))
            bowyer_watson_std.append(np.std(reps))

        # BowyerWatson - No Numba
        if i < 2 or bowyer_watson_no_numba_avg[-1] < 5:
            reps = []
            for j in range(N):
                t = run_no_numba(n, j)
                reps.append(t)
            bowyer_watson_no_numba_avg.append(np.mean(reps)/N)
            bowyer_watson_no_numba_std.append(np.std(reps)/N)


        # Triangle
        if i < 2 or triangle_avg[-1] < 5:
            reps = []
            for j in range(N):
                np.random.seed(j)
                points = np.random.rand(n, 2)
                constraints = np.random.choice(np.arange(0, len(points), dtype=int), size=(5,2),replace=False)
                tic = time.time()
                delaunay.Triangle(points, Constraints=constraints)
                reps.append(time.time()-tic)
            triangle_avg.append(np.mean(reps))
            triangle_std.append(np.std(reps))

    # Plot
    plt.errorbar(npoints[:len(bowyer_watson_avg)], bowyer_watson_avg, yerr=bowyer_watson_std, color='red')
    plt.errorbar(npoints[:len(bowyer_watson_no_numba_avg)], bowyer_watson_no_numba_avg, yerr=bowyer_watson_no_numba_std, color='pink')
    plt.errorbar(npoints[:len(triangle_avg)], triangle_avg, yerr=triangle_std, color='purple')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['mymesh', 'mymesh (no numba)', 'Triangle'])
    plt.xlabel('# of points')
    plt.ylabel('Time (s)')
    plt.title('2D Constrained Delaunay Triangulation')
    plt.grid()
    plt.show()



2D Convex Hull
--------------

.. plot::
    :include-source: False
    :show-source-link: True

    import sys, os, time, subprocess
    import scipy
    import numpy as np
    import matplotlib.pyplot as plt

    def run_no_numba(npoints, seed):
        code = f"""
    import os, subprocess, time
    import numpy as np
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    from mymesh import delaunay
    # pre-run in case of any first time delays
    delaunay.ConvexHull(np.random.rand(10,2), method='GiftWrapping')
    np.random.seed({seed})
    points = np.random.rand({npoints}, 2)
    tic = time.time()
    delaunay.ConvexHull(points, method='GiftWrapping')
    t = time.time()-tic
    print(t)
    """
        result = subprocess.run([sys.executable, "-c", code], capture_output=True)
        t = float(result.stdout)

        return t


    from mymesh import delaunay
    npoints = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=int)

    gift_wrapping_avg = []
    gift_wrapping_std = []
    gift_wrapping_no_numba_avg = []
    gift_wrapping_no_numba_std = []
    quickhull_avg = []
    quickhull_std = []
    scipy_avg = []
    scipy_std = []

    # pre-run for any first time pre-compiling
    points = np.random.rand(10,2)
    delaunay.ConvexHull(points, method='GiftWrapping')
    delaunay.ConvexHull(points, method='QuickHull')
    delaunay.ConvexHull(points, method='scipy')

    N = 3
    for i,n in enumerate(npoints):
        # QuickHull
        if i < 2 or quickhull_avg[-1] < 5:
            reps = []
            for j in range(N):
                np.random.seed(j)
                points = np.random.rand(n, 2)
                tic = time.time()
                delaunay.ConvexHull(points, method='QuickHull')
                reps.append(time.time()-tic)

            quickhull_avg.append(np.mean(reps))
            quickhull_std.append(np.std(reps))
        # GiftWrapping
        if i < 2 or gift_wrapping_avg[-1] < 5:
            reps = []
            for j in range(N):
                np.random.seed(j)
                points = np.random.rand(n, 2)
                tic = time.time()
                delaunay.ConvexHull(points, method='GiftWrapping')
                reps.append(time.time()-tic)

            gift_wrapping_avg.append(np.mean(reps))
            gift_wrapping_std.append(np.std(reps))

        # GiftWrapping - No Numba
        if i < 2 or gift_wrapping_no_numba_avg[-1] < 5:
            reps = []
            for j in range(N):
                t = run_no_numba(n, j)
                reps.append(t)
            gift_wrapping_no_numba_avg.append(np.mean(reps)/N)
            gift_wrapping_no_numba_std.append(np.std(reps)/N)

        # SciPy
        if i < 2 or scipy_avg[-1] < 5:
            reps = []
            for j in range(N):
                np.random.seed(j)
                points = np.random.rand(n, 2)
                tic = time.time()
                delaunay.ConvexHull(points, method='scipy')
                reps.append(time.time()-tic)
            scipy_avg.append(np.mean(reps))
            scipy_std.append(np.std(reps))


    # Plot
    plt.errorbar(npoints[:len(quickhull_avg)], quickhull_avg,
                            yerr=quickhull_std, color='blue')
    plt.errorbar(npoints[:len(gift_wrapping_avg)], gift_wrapping_avg, 
                            yerr=gift_wrapping_std, color='red')
    plt.errorbar(npoints[:len(gift_wrapping_no_numba_avg)], 
                    gift_wrapping_no_numba_avg, 
                    yerr=gift_wrapping_no_numba_std, color='pink')
    plt.errorbar(npoints[:len(scipy_avg)], scipy_avg, 
                yerr=scipy_std, color='green')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['mymesh QuickHull', 'mymesh GiftWrapping', 'mymesh GiftWrapping (no numba)', 'scipy (qhull)'])
    plt.xlabel('# of points')
    plt.ylabel('Time (s)')
    plt.title('2D Convex Hull')
    plt.grid()
    plt.show()