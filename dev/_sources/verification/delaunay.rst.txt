Delaunay
========

Delaunay triangulation is a well studied problem, and there are several well-established and commonly used programs for achieving Delaunay triangulations and tetrahedralizations, with and without constraints (constrained Delaunay triangulation, CDT).

The current implementations in :mod:`mymesh` don't yet meet or exceed the performance or functionality of other available options (though this is an area of active development), so using well established implementations is recommended:

- :code:`qhull` (available via :func:`scipy.spatial.Delaunay`, or :func:`mymesh.delaunay.SciPy`) for n dimensional Delaunay triangulations
- :code:`triangle` for 2 dimensional constrained/unconstrained Delaunay triangulations
- :code:`TetGen` for 3 dimensional constrained/unconstrained Delaunay triangulations
- :code:`gmsh`
- :code:`CGAL`

Here you can see a comparison between the performance of :mod:`mymesh` and some alternatives 

See also: :ref:`Delaunay Triangulation`, :mod:`mymesh.delaunay`

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
    delaunay.BowyerWatson2d2(np.random.rand(10,2))
    np.random.seed({seed})
    points = np.random.rand({npoints}, 2)
    tic = time.time()
    delaunay.BowyerWatson2d2(points)
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
    bowyer_watson2_avg = []
    bowyer_watson2_std = []
    bowyer_watson2_no_numba_avg = []
    bowyer_watson2_no_numba_std = []
    scipy_avg = []
    scipy_std = []
    triangle_avg = []
    triangle_std = []

    # pre-run for any first time pre-compiling
    points = np.random.rand(10,2)
    delaunay.BowyerWatson2d(points)
    delaunay.BowyerWatson2d2(points)
    scipy.spatial.Delaunay(points)
    delaunay.Triangle(points)

    N = 3
    for i,n in enumerate(npoints):
        points = np.random.rand(n, 2)
        
        # BowyerWatson 
        '''
        if i < 2 or bowyer_watson_avg[-1] < 1:
            reps = []
            for j in range(N):
                np.random.seed(j)
                points = np.random.rand(n, 2)
                tic = time.time()
                delaunay.BowyerWatson2d(points)
                reps.append(time.time()-tic)
            
            bowyer_watson_avg.append(np.mean(reps))
            bowyer_watson_std.append(np.std(reps))
        '''

        # BowyerWatson2
        if i < 2 or bowyer_watson2_avg[-1] < 5:
            reps = []
            for j in range(N):
                np.random.seed(j)
                points = np.random.rand(n, 2)
                tic = time.time()
                delaunay.BowyerWatson2d2(points)
                reps.append(time.time()-tic)
            
            bowyer_watson2_avg.append(np.mean(reps))
            bowyer_watson2_std.append(np.std(reps))

        # BowyerWatson2 - No Numba
        if i < 2 or bowyer_watson2_avg[-1] < 5:
            reps = []
            for j in range(N):
                t = run_no_numba(n, j)
                reps.append(t)
            bowyer_watson2_no_numba_avg.append(np.mean(reps)/N)
            bowyer_watson2_no_numba_std.append(np.std(reps)/N)

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
    #plt.errorbar(npoints, bowyer_watson_avg, yerr=bowyer_watson_std, color='blue')
    plt.errorbar(npoints[:len(bowyer_watson2_avg)], bowyer_watson2_avg, yerr=bowyer_watson2_std, color='red')
    plt.errorbar(npoints[:len(bowyer_watson2_no_numba_avg)], bowyer_watson2_no_numba_avg, yerr=bowyer_watson2_no_numba_std, color='pink')
    plt.errorbar(npoints[:len(scipy_avg)], scipy_avg, yerr=scipy_std, color='green')
    plt.errorbar(npoints[:len(triangle_avg)], triangle_avg, yerr=triangle_std, color='purple')
    plt.xscale('log')
    plt.yscale('log')
    # plt.legend(['mymesh', 'scipy (qhull)', 'Triangle'])
    plt.legend(['mymesh', 'mymesh (no numba)', 'scipy (qhull)', 'Triangle'])
    plt.xlabel('# of points')
    plt.ylabel('Time (s)')
    plt.title('2D Delaunay Triangulation')
    plt.grid()
    plt.show()