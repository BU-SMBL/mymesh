Tree
====


K-Nearest Neighbor Search
-------------------------

2D Point Data
#############

.. code::

    points = np.random.rand(2,3)
    x = np.random.rand(2)
    k = 3

    # quadtree build
    quadtree = tree.Points2Quadtree(points)

    # quadtree search
    qtree_out = quadtree.query_knn(x, k=k)

    # kdtree build
    kdtree = scipy.spatial.KDTree(points)

    # kdtree search
    kdtree_out = kdtree.query(x, k=k)
    kdtree_out = (np.atleast_1d(kdtree_out[0]), np.atleast_2d(points[kdtree_out[1]]))

    # brute-force
    d = scipy.spatial.distance.cdist(points, np.atleast_2d(x)).flatten()
    indices = np.argpartition(d, k)[:k]
    distances = d[indices]
    brute_out = (distances, points[indices])

.. plot::
    :include-source: False
    :show-source-link: True

    import time
    import scipy
    import numpy as np
    from mymesh import tree

    npoints = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6], dtype=int)

    kdtree_build_time = []
    kdtree_search_time = []
    qtree_build_time = []
    qtree_search_time = []
    brute_search_time = []

    x = np.random.rand(2)
    k = 3

    # pre-run for any first time pre-compiling
    points = np.random.rand(10,2)
    quadtree = tree.Points2Quadtree(points)
    qtree_out = quadtree.query_knn(x, k=k)
    kdtree = scipy.spatial.KDTree(points)
    kdtree_out = kdtree.query(x, k=k)
    d = scipy.spatial.distance.cdist(points, np.atleast_2d(x)).flatten()

    
    for n in npoints:
        points = np.random.rand(n, 2)
        
        # quadtree build
        tic = time.time()
        quadtree = tree.Points2Quadtree(points)
        qtree_build_time.append(time.time()-tic)

        # quadtree search
        tic = time.time()
        qtree_out = quadtree.query_knn(x, k=k)
        qtree_search_time.append(time.time()-tic)

        # kdtree build
        tic = time.time()
        kdtree = scipy.spatial.KDTree(points)
        kdtree_build_time.append(time.time()-tic)

        # kdtree search
        tic = time.time()
        kdtree_out = kdtree.query(x, k=k)
        kdtree_out = (np.atleast_1d(kdtree_out[0]), np.atleast_2d(points[kdtree_out[1]]))
        kdtree_search_time.append(time.time()-tic)

        # brute-force
        tic = time.time()
        d = scipy.spatial.distance.cdist(points, np.atleast_2d(x)).flatten()
        indices = np.argpartition(d, k)[:k]
        distances = d[indices]
        brute_out = (distances, points[indices])
        brute_search_time.append(time.time()-tic)

    # Plot 
    plt.loglog(npoints, qtree_build_time, color='blue', linestyle='dashed')
    plt.loglog(npoints, qtree_search_time, color='blue', linestyle='dotted')
    plt.loglog(npoints, np.add(qtree_build_time,qtree_search_time), color='blue',  linestyle='solid')

    plt.loglog(npoints, kdtree_build_time, color='green', linestyle='dashed')
    plt.loglog(npoints, kdtree_search_time, color='green', linestyle='dotted')
    plt.loglog(npoints, np.add(kdtree_build_time, kdtree_search_time), color='green',  linestyle='solid')

    plt.loglog(npoints, brute_search_time, color='black', linewidth=3)

    plt.legend([
        'Quadtree (Build)',
        'Quadtree (Search)',
        'Quadtree (Total)',
        'Scipy KDTree (Build)',
        'Scipy KDTree (Search)',
        'Scipy KDTree (Total)',
        'Brute-Force'
        ])


    plt.ylabel('Time (s)')
    plt.xlabel('# of Points')
    plt.show()


3D Point Data
#############

.. code::

    points = np.random.rand(n,3)
    x = np.random.rand(3)
    k = 3

    # octree build
    octree = tree.Points2Octree(points)

    # octree search
    otree_out = octree.query_knn(x, k=k)

    # kdtree build
    kdtree = scipy.spatial.KDTree(points)

    # kdtree search
    kdtree_out = kdtree.query(x, k=k)
    kdtree_out = (np.atleast_1d(kdtree_out[0]), np.atleast_2d(points[kdtree_out[1]]))

    # brute-force
    d = scipy.spatial.distance.cdist(points, np.atleast_2d(x)).flatten()
    indices = np.argpartition(d, k)[:k]
    distances = d[indices]
    brute_out = (distances, points[indices])

.. plot::
    :include-source: False
    :show-source-link: True

    import time
    import scipy
    import numpy as np
    from mymesh import tree

    npoints = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6], dtype=int)

    kdtree_build_time = []
    kdtree_search_time = []
    otree_build_time = []
    otree_search_time = []
    brute_search_time = []

    x = np.random.rand(3)
    k = 3

    # pre-run for any first time pre-compiling
    points = np.random.rand(10,3)
    octree = tree.Points2Octree(points)
    otree_out = octree.query_knn(x, k=k)
    kdtree = scipy.spatial.KDTree(points)
    kdtree_out = kdtree.query(x, k=k)
    d = scipy.spatial.distance.cdist(points, np.atleast_2d(x)).flatten()

    
    for n in npoints:
        points = np.random.rand(n, 3)
        
        # octree build
        tic = time.time()
        octree = tree.Points2Octree(points)
        otree_build_time.append(time.time()-tic)

        # quadtree search
        tic = time.time()
        otree_out = octree.query_knn(x, k=k)
        otree_search_time.append(time.time()-tic)

        # kdtree build
        tic = time.time()
        kdtree = scipy.spatial.KDTree(points)
        kdtree_build_time.append(time.time()-tic)

        # kdtree search
        tic = time.time()
        kdtree_out = kdtree.query(x, k=k)
        kdtree_out = (np.atleast_1d(kdtree_out[0]), np.atleast_2d(points[kdtree_out[1]]))
        kdtree_search_time.append(time.time()-tic)

        # brute-force
        tic = time.time()
        d = scipy.spatial.distance.cdist(points, np.atleast_2d(x)).flatten()
        indices = np.argpartition(d, k)[:k]
        distances = d[indices]
        brute_out = (distances, points[indices])
        brute_search_time.append(time.time()-tic)

    # Plot 
    plt.loglog(npoints, otree_build_time, color='blue', linestyle='dashed')
    plt.loglog(npoints, otree_search_time, color='blue', linestyle='dotted')
    plt.loglog(npoints, np.add(otree_build_time,otree_search_time), color='blue',  linestyle='solid')

    plt.loglog(npoints, kdtree_build_time, color='green', linestyle='dashed')
    plt.loglog(npoints, kdtree_search_time, color='green', linestyle='dotted')
    plt.loglog(npoints, np.add(kdtree_build_time, kdtree_search_time), color='green',  linestyle='solid')

    plt.loglog(npoints, brute_search_time, color='black', linewidth=3)

    plt.legend([
        'Octree (Build)',
        'Octree (Search)',
        'Octree (Total)',
        'Scipy KDTree (Build)',
        'Scipy KDTree (Search)',
        'Scipy KDTree (Total)',
        'Brute-Force'
        ])


    plt.ylabel('Time (s)')
    plt.xlabel('# of Points')
    plt.show()
