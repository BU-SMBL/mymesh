Contour
=======

Contouring methods are used to extract surfaces from scalar fields (e.g. images or implicit functions). 
MyMesh has several different contouring methods with different trade-offs and capabilities.
Here, the efficiency of these methods are compared against each other and against contouring methods provided in `PyVista <https://docs.pyvista.org/>`_ (VTK) :cite:`Sullivan2019`. 

See also: :ref:`theory_contour`, :mod:`mymesh.contour`, :ref:`contouring`

3D Surface Contouring
---------------------

Contouring of a unit sphere in a uniform grid

.. plot::
    :include-source: False
    :show-source-link: True

    import sys, os, time, subprocess
    import scipy
    import numpy as np
    import matplotlib.pyplot as plt
    import pyvista as pv

    from mymesh import primitives, implicit, contour
    voxel_size = np.array([0.1, 0.1/2, 0.1/4, 0.1/8])#, 0.1/16])

    mc_avg = []
    mc_std = []
    mc_img_avg = []
    mc_img_std = []
    mc33_avg = []
    mc33_std = []
    mt_avg = []
    mt_std = []
    pvmc_avg = []
    pvmc_std = []
    pvfe_avg = []
    pvfe_std = []
    # pre-run for any first time pre-compiling
    h = 0.25
    grid = primitives.Grid([-1,1,-1,1,-1,1], h)
    grid.NodeData['f'] = implicit.sphere([0,0,0],1)(*grid.NodeCoords.T)
    tetgrid = primitives.Grid([-1,1,-1,1,-1,1], h, ElemType='tet')
    tetgrid.NodeData['f'] = implicit.wrapfunc(implicit.gyroid)(*tetgrid.NodeCoords.T)
    img = grid.NodeData['f'].reshape(np.repeat(int(np.round((grid.NNode)**(1/3))),3), order='F')
    pvgrid = pv.ImageData(
        dimensions=img.shape,
        spacing=(h,h,h),
        origin=(0,0,0),
    )
    contour.MarchingCubes(grid.NodeCoords, grid.NodeConn, grid.NodeData['f'], method='original')
    contour.MarchingCubesImage(img, method='original')
    contour.MarchingCubes(grid.NodeCoords, grid.NodeConn, grid.NodeData['f'], method='33')
    contour.MarchingTetrahedra(grid.NodeCoords, grid.NodeConn, grid.NodeData['f'])
    pvgrid.contour([0], grid.NodeData['f'], method='marching_cubes') 
    pvgrid.contour([0], grid.NodeData['f'], method='flying_edges')

    N = 3
    npoints = []
    for i,h in enumerate(voxel_size):

        grid = primitives.Grid([-1,1,-1,1,-1,1], h)
        grid.NodeData['f'] = implicit.sphere([0,0,0],1)(*grid.NodeCoords.T)
        tetgrid = primitives.Grid([-1,1,-1,1,-1,1], h, ElemType='tet')
        tetgrid.NodeData['f'] = implicit.wrapfunc(implicit.gyroid)(*tetgrid.NodeCoords.T)
        img = grid.NodeData['f'].reshape(np.repeat(int(np.round((grid.NNode)**(1/3))),3), order='F')
        pvgrid = pv.ImageData(
            dimensions=img.shape,
            spacing=(h,h,h),
            origin=(0,0,0),
        )

        npoints.append(grid.NNode)
        # Marching Cubes
        if i < 2 or mc_avg[-1] < 5:
            reps = []
            for j in range(N):
                tic = time.time()
                contour.MarchingCubes(grid.NodeCoords, grid.NodeConn, grid.NodeData['f'], method='original')
                reps.append(time.time()-tic)

            mc_avg.append(np.mean(reps))
            mc_std.append(np.std(reps))
        # Marching Cubes Image
        if i < 2 or mc_img_avg[-1] < 5:
            reps = []
            for j in range(N):
                tic = time.time()
                contour.MarchingCubesImage(img, method='original')
                reps.append(time.time()-tic)

            mc_img_avg.append(np.mean(reps))
            mc_img_std.append(np.std(reps))
        # Marching Cubes 33
        if i < 2 or mc33_avg[-1] < 5:
            reps = []
            for j in range(N):
                tic = time.time()
                contour.MarchingCubes(grid.NodeCoords, grid.NodeConn, grid.NodeData['f'], method='33')
                reps.append(time.time()-tic)

            mc33_avg.append(np.mean(reps))
            mc33_std.append(np.std(reps))
        # Marching Tetrahedra
        if i < 2 or mt_avg[-1] < 5:
            reps = []
            for j in range(N):
                tic = time.time()
                contour.MarchingTetrahedra(tetgrid.NodeCoords, tetgrid.NodeConn, tetgrid.NodeData['f'])
                reps.append(time.time()-tic)

            mt_avg.append(np.mean(reps))
            mt_std.append(np.std(reps))
        # pyvista marching cubes
        if i < 2 or pvmc_avg[-1] < 5:
            reps = []
            for j in range(N):
                tic = time.time()
                pvgrid.contour([0], grid.NodeData['f'], method='marching_cubes') 
                reps.append(time.time()-tic)

            pvmc_avg.append(np.mean(reps))
            pvmc_std.append(np.std(reps))
        # pyvista flying edges
        if i < 2 or pvfe_avg[-1] < 5:
            reps = []
            for j in range(N):
                tic = time.time()
                pvgrid.contour([0], grid.NodeData['f'], method='flying_edges') 
                reps.append(time.time()-tic)

            pvfe_avg.append(np.mean(reps))
            pvfe_std.append(np.std(reps))


    # Plot
    plt.errorbar(npoints[:len(mc_avg)], mc_avg, yerr=mc_std, color='#bf616a')
    plt.errorbar(npoints[:len(mc_img_avg)], mc_img_avg, yerr=mc_img_std, color='#5e81ac')
    plt.errorbar(npoints[:len(mc33_avg)], mc33_avg, yerr=mc_img_std, color='#ebcb8b')
    plt.errorbar(npoints[:len(mt_avg)], mt_avg, yerr=mt_std, color='#a3be8c')
    plt.errorbar(npoints[:len(pvmc_avg)], pvmc_avg, yerr=pvmc_std, color='#b48ead', linestyle='dotted')
    plt.errorbar(npoints[:len(pvfe_avg)], pvfe_avg, yerr=pvfe_std, color='#2e3440', linestyle='dotted')

    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['marching cubes', 'marching cubes (image)', 'marching cubes 33', 'marching tetrahedra', 'marching cubes (pyvista)', 'flying edges (pyvista)'])
    plt.xlabel('# of points in grid')
    plt.ylabel('Time (s)')
    plt.title('Surface Contouring')
    plt.grid()
    plt.show()

.. Note::
    Marching tetrahedra operates on six times as many elements for the same number of points due to the cube-to-tetrahedra decomposition (which isn't included in the measurement of computational time).

System Information
------------------

.. jupyter-execute::
    :hide-code:

    import platform, importlib, psutil, cpuinfo
    
    uname = platform.uname()
    cpu_info = cpuinfo.get_cpu_info()
    svmem = psutil.virtual_memory()

    print("*"*10, "System Information", "*"*10)
    print(f"System: {uname.system}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Numpy Version: {importlib.metadata.version('numpy')}")
    print(f"CPU: {cpu_info['brand_raw']}")  
    print("CPU Physical Cores:", psutil.cpu_count(logical=False))
    print(f"Available Memory: {svmem.available/(1024**3):.1f}/{svmem.total/(1024**3):.1f} GB")
    print("*"*40)