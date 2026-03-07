Tree
====


K-Nearest Neighbor Search
-------------------------

Spatial tree structures can dramatically speed up nearest neighbor searching compared to brute-force approaches that require distances between all points to be calculated, however, there is an added up-front cost of constructing the tree.
If only a single search needs to be performed, the brute-force approach may be the most efficient option, but if performing multiple searches, or other operations that can utilize the tree structure, performance can be enhanced significantly.

Here the performance of multiple tree structures are compared to a brute-force search.
A quadtree (:class:`~mymesh.tree.QuadtreeNode`),  KD tree (:class:`~mymesh.tree.KDtreeNode`), and SciPy's implementation of a :class:`scipy.spatial.KDTree` are compared to a brute force search for 2D point data.
For 3D point data, an octree (:class:`~mymesh.tree.OctreeNode`), is used in place of the quadtree.
Note that the KD tree can be used for any k-dimensional data.

2D Point Data
#############

.. code::
    
    d = 2       # Dimensions
    k = 3       # Number of nearest neighbors
    points = np.random.rand(n,d)
    x = np.random.rand(n)

    # quadtree build
    quadtree = tree.Points2Quadtree(points)

    # quadtree search
    qtree_out = quadtree.query_knn(x, k=k)

    # kdtree build
    kdtree = tree.Points2KDtree(points)

    # kdtree search
    kdtree_out = kdtree.query_knn(x, k=k)

    # scipy kdtree build
    sci_kdtree = scipy.spatial.KDTree(points)

    # scipy kdtree search
    sci_kdtree_out = sci_kdtree.query(x, k=k)
    sci_kdtree_out = (np.atleast_1d(sci_kdtree_out[0]), np.atleast_2d(points[sci_kdtree_out[1]]))

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

    sci_kdtree_build_time = []
    sci_kdtree_search_time = []
    kdtree_build_time = []
    kdtree_search_time = []
    qtree_build_time = []
    qtree_search_time = []
    brute_search_time = []
    
    k = 3
    d = 2
    x = np.random.rand(d)

    # pre-run for any first time pre-compiling
    points = np.random.rand(10,d)
    quadtree = tree.Points2Quadtree(points)
    qtree_out = quadtree.query_knn(x, k=k)
    sci_kdtree = scipy.spatial.KDTree(points)
    sci_kdtree_out = sci_kdtree.query(x, k=k)
    dist = scipy.spatial.distance.cdist(points, np.atleast_2d(x)).flatten()

    
    for n in npoints:
        points = np.random.rand(n, d)
        
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
        kdtree = tree.Points2KDtree(points)
        kdtree_build_time.append(time.time()-tic)

        # kdtree search
        tic = time.time()
        kdtree_out = kdtree.query_knn(x, k=k)
        kdtree_search_time.append(time.time()-tic)

        # scipy kdtree build
        tic = time.time()
        sci_kdtree = scipy.spatial.KDTree(points)
        sci_kdtree_build_time.append(time.time()-tic)

        # scipy kdtree search
        tic = time.time()
        sci_kdtree_out = sci_kdtree.query(x, k=k)
        sci_kdtree_out = (np.atleast_1d(sci_kdtree_out[0]), np.atleast_2d(points[sci_kdtree_out[1]]))
        sci_kdtree_search_time.append(time.time()-tic)

        # brute-force
        tic = time.time()
        dist = scipy.spatial.distance.cdist(points, np.atleast_2d(x)).flatten()
        indices = np.argpartition(dist, k)[:k]
        distances = dist[indices]
        brute_out = (distances, points[indices])
        brute_search_time.append(time.time()-tic)

    # Plot 
    plt.loglog(npoints, qtree_build_time, color='blue', linestyle='dashed')
    plt.loglog(npoints, qtree_search_time, color='blue', linestyle='dotted')
    plt.loglog(npoints, np.add(qtree_build_time,qtree_search_time), color='blue',  linestyle='solid')

    plt.loglog(npoints, kdtree_build_time, color='red', linestyle='dashed')
    plt.loglog(npoints, kdtree_search_time, color='red', linestyle='dotted')
    plt.loglog(npoints, np.add(kdtree_build_time, kdtree_search_time), color='red',  linestyle='solid')

    plt.loglog(npoints, sci_kdtree_build_time, color='green', linestyle='dashed')
    plt.loglog(npoints, sci_kdtree_search_time, color='green', linestyle='dotted')
    plt.loglog(npoints, np.add(sci_kdtree_build_time, sci_kdtree_search_time), color='green',  linestyle='solid')

    plt.loglog(npoints, brute_search_time, color='black', linewidth=3)

    plt.legend([
        'Quadtree (Build)',
        'Quadtree (Search)',
        'Quadtree (Total)',
        'KDTree (Build)',
        'KDTree (Search)',
        'KDTree (Total)',
        'Scipy KDTree (Build)',
        'Scipy KDTree (Search)',
        'Scipy KDTree (Total)',
        'Brute-Force'
        ],
        fontsize='x-small')


    plt.ylabel('Time (s)')
    plt.xlabel('# of Points')
    plt.show()


3D Point Data
#############

.. code::

    d = 3       # Dimensions
    k = 3       # Number of nearest neighbors
    points = np.random.rand(n,d)
    x = np.random.rand(n)

    # octree build
    octree = tree.Points2Octree(points)

    # octree search
    otree_out = octree.query_knn(x, k=k)

    # kdtree build
    kdtree = tree.Points2KDtree(points)

    # kdtree search
    kdtree_out = kdtree.query_knn(x, k=k)

    # scipy kdtree build
    sci_kdtree = scipy.spatial.KDTree(points)

    # scipy kdtree search
    sci_kdtree_out = sci_kdtree.query(x, k=k)
    sci_kdtree_out = (np.atleast_1d(sci_kdtree_out[0]), np.atleast_2d(points[sci_kdtree_out[1]]))

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

    sci_kdtree_build_time = []
    sci_kdtree_search_time = []
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
    dist = scipy.spatial.distance.cdist(points, np.atleast_2d(x)).flatten()

    
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
        kdtree = tree.Points2KDtree(points)
        kdtree_build_time.append(time.time()-tic)

        # kdtree search
        tic = time.time()
        kdtree_out = kdtree.query_knn(x, k=k)
        kdtree_search_time.append(time.time()-tic)

        # scipy kdtree build
        tic = time.time()
        sci_kdtree = scipy.spatial.KDTree(points)
        sci_kdtree_build_time.append(time.time()-tic)

        # scipy kdtree search
        tic = time.time()
        sci_kdtree_out = sci_kdtree.query(x, k=k)
        sci_kdtree_out = (np.atleast_1d(sci_kdtree_out[0]), np.atleast_2d(points[sci_kdtree_out[1]]))
        sci_kdtree_search_time.append(time.time()-tic)

        # brute-force
        tic = time.time()
        dist = scipy.spatial.distance.cdist(points, np.atleast_2d(x)).flatten()
        indices = np.argpartition(dist, k)[:k]
        distances = dist[indices]
        brute_out = (distances, points[indices])
        brute_search_time.append(time.time()-tic)

    # Plot 
    plt.loglog(npoints, otree_build_time, color='blue', linestyle='dashed')
    plt.loglog(npoints, otree_search_time, color='blue', linestyle='dotted')
    plt.loglog(npoints, np.add(otree_build_time,otree_search_time), color='blue',  linestyle='solid')

    plt.loglog(npoints, kdtree_build_time, color='red', linestyle='dashed')
    plt.loglog(npoints, kdtree_search_time, color='red', linestyle='dotted')
    plt.loglog(npoints, np.add(kdtree_build_time, kdtree_search_time), color='red',  linestyle='solid')

    plt.loglog(npoints, sci_kdtree_build_time, color='green', linestyle='dashed')
    plt.loglog(npoints, sci_kdtree_search_time, color='green', linestyle='dotted')
    plt.loglog(npoints, np.add(sci_kdtree_build_time, sci_kdtree_search_time), color='green',  linestyle='solid')


    plt.loglog(npoints, brute_search_time, color='black', linewidth=3)

    plt.legend([
        'Octree (Build)',
        'Octree (Search)',
        'Octree (Total)',
        'KDTree (Build)',
        'KDTree (Search)',
        'KDTree (Total)',
        'Scipy KDTree (Build)',
        'Scipy KDTree (Search)',
        'Scipy KDTree (Total)',
        'Brute-Force'
        ],
        fontsize='x-small')


    plt.ylabel('Time (s)')
    plt.xlabel('# of Points')
    plt.show()
