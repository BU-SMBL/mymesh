# -*- coding: utf-8 -*-
# Created on Fri Oct 14 13:16:00 2024
# @author: toj
"""
Tools for registering or aligning point clouds, meshes, and images.

.. currentmodule:: mymesh.register

Registration
============
.. autosummary::
    :toctree: submodules/

    AxisAlignPoints
    AxisAlignImage
    Point2Point
    Mesh2Mesh
    Image2Image
    ICP

Transformation
==============
.. autosummary::
    :toctree: submodules/

    transform_points
    transform_image
    rotation2d
    rotation
    translation2d
    translation
    rigid2d
    rigid
    similarity2d
    similarity
    affine2d
    affine
    T2d
    R2d
    S2d
    Sh2d
    T3d
    R3d
    S3d
    Sh3d

Similarity Metrics
==================
.. autosummary::
    :toctree: submodules/

    dice
    jaccard
    mutual_information
    hausdorff
    closest_point_MSE
    symmetric_closest_point_MSE

Optimization
============
.. autosummary::
    :toctree: submodules/

    optimize

Visualization
=============
.. autosummary::
    :toctree: submodules/

    ImageOverlay

"""

import numpy as np
import scipy
import sys, os, copy, warnings
from mymesh import utils

def AxisAlignPoints(points, axis_order=[2,1,0], center=None, return_transformed=True, return_transform=False):
    """
    Align an point cloud to the x, y, z axes. This works by identifying
    the minimum volume bounding box (see :func:~`mymesh.utils.MVBB`) and 
    aligning that box to the principal axes, so point clouds representing 
    rounded objects with ambiguous orientation may be oriented
    seemingly-arbitrarily. The center of the object (defined as the centroid
    of the MVBB) will be preserved in the alignment unless a different center
    is specified.

    Parameters
    ----------
    points : array_like
        Array of point coordinates (shape=(n,3))
    axis_order : array_like, optional
        Orientation of the aligned object in terms of the lengths of each side
        of the object, by default [0,1,2]. The first axis will correspond to the
        shortest side of the object and the last index to the longest side. For 
        example, with [0,1,2], the longest side will be aligned with the z (2) 
        axis, and the shortest will be aligned with the x (0) axis. Must be a 
        combination of 0, 1, and 2.
    center : array_like or NoneType, optional
        If provided, coordinates `[x,y,z]` of where to place to place the center
        of the bounding box of the object after transformation. If `None`, the 
        center of the oriented points will be the center of the original points,
        by default None.
    return_transform : bool, optional
        Option to return the transformation matrix as well as the transformed
        point cloud, by default False

    Returns
    -------
    transformed : np.ndarray
        Array of point coordinates transformed to be aligned to the axes
    transform : np.ndarray, optional
        Affine transformation matrix (shape=(4,4)) to transform `points` to 
        `transformed` (`transformed=(transform@points.T).T`). Only returned if
        `return_transform = True`.
    """    
    assert ValueError(len(axis_order) == 3 and np.array_equal([0,1,2], np.sort(axis_order))), 'axis_order must contain only 0, 1, and 2.'

    mvbb, mat = utils.MVBB(points, return_matrix=True)
    
    # Modify rotation to specified axis_order
    mvbb_t = transform_points(mvbb, mat)
    side_lengths = np.max(mvbb_t,axis=0) - np.min(mvbb_t,axis=0)
    current_order = np.argsort(side_lengths)
    if not np.all(current_order == axis_order):
        idx = np.argsort(np.argsort(current_order)[np.argsort(axis_order)])
        perpendicular_transform = np.eye(3)[idx]
        if np.linalg.det(perpendicular_transform) < 0:
            # If the determinant is negative, it would cause a reflection, inverting a column fixes this
            perpendicular_transform[:,0] = -1*perpendicular_transform[:,0]
        mat = perpendicular_transform@mat
        mvbb_t = transform_points(mvbb, mat)
    
    # Restore center after rotation
    original_center = np.mean(points,axis=0)
    transformed_center = transform_points(np.atleast_2d(original_center), mat)[0]
    if center is None:
        center = original_center
    else:
        assert isinstance(center, (tuple, list, np.ndarray)) and len(center) == 3, 'If provided, center must be be a three element list or array.'
        center = np.asarray(center)
    center_shift = center - transformed_center
    
    transform = np.eye(4)
    transform[:3,:3] = mat
    transform[:3,3] = center_shift
    
    if return_transformed:
        transformed = transform_points(points, transform)
        if return_transform:
            return transformed, transform
        else:
            return transformed
    elif return_transform:
        return transform
    return

def AxisAlignImage(img, axis_order=[2,1,0], threshold=None, center='image', scale=1, interpolation_order=1, transform_args=None, return_transformed=True, return_transform=False):
    """
    Align an object in an image to the x, y, z axes. This works by identifying
    the minimum volume bounding box (see :func:~`mymesh.utils.MVBB`) and 
    aligning that box to the principal axes, so objects with rounded objects 
    with ambiguous orientation may be oriented seemingly-arbitrarily. 

    Parameters
    ----------
    img : array_like
        3 dimensional image array of the image
    axis_order : array_like, optional
        Orientation of the aligned image in terms of the lengths of each side
        of the object, by default [0,1,2]. The first axis will correspond to the
        shortest side of the object and the last index to the longest side. For 
        example, with [0,1,2], the longest side will be aligned with the z (2) 
        axis, and the shortest will be aligned with the x (0) axis. Must be a 
        combination of 0, 1, and 2.
        Threshold value used to binarize the image for identification of the
        object. If the image is already binarized, this is not necessary, by 
        default None.
    center : str, array_like, or NoneType, optional
        Location of the center of the object after axis alignment, by default
        'image'. Options are:

        - 'image': Centers the object at the center of the image
        - 'object': Keeps the center of the object in place
        - 
            `[x,y,z]`: A three element list or array specifies the location, in
            voxels, of where to place to place the center of the bounding box 
            of the object after transformation. 

    scale : float, optional
        Scale factor used to resample the image to either reduce (scale < 1) 
        or increase (scale > 1) the size/resolution of the image used for
        alignment, by default 1. The returned image will still be of the 
        original resolution.
    interpolation_order : int, optional
        Interpolation order used in image scaling and transformation (see 
        `scipy.ndimage.zoom <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html#scipy.ndimage.zoom>`_)
        and scaling (if used). Must be an integer in the range 0-5. Lower order
        is more efficient at the cost of quality. By default, 1. 
    transform_args : dict, optional
        Optional input arguments passed to `scipy.ndimage.affine_transform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html>`_, by 
        default `dict(mode='constant', order=interpolation_order)`.
    return_transform : bool, optional
        Option to return the transformation matrix as well as the transformed
        point cloud, by default False
    

    Returns
    -------
    transformed : np.ndarray
        Array of point coordinates transformed to be aligned to the axes
    transform : np.ndarray, optional
        Affine transformation matrix (shape=(4,4)) to transform `img` to 
        `transformed`. Only returned if
        `return_transform = True`.
    """
    assert ValueError(len(axis_order) == 3 and np.array_equal([0,1,2], np.sort(axis_order))), 'axis_order must contain only 0, 1, and 2.'
    img = np.asarray(img)
    
    # Process transform_args input
    if transform_args is None:
        transform_args = dict(mode='constant', order=interpolation_order)
    else:
        if 'mode' not in transform_args:
            transform_args['mode'] = 'constant'
        if 'order' not in transform_args:
            transform_args['order'] = interpolation_order
            
    if scale != 1:
        transform_args['order'] = interpolation_order
        imgS = scipy.ndimage.zoom(img, scale, order=interpolation_order)
    else:
        imgS = img
    # Process threshold input
    if threshold is None:
        # This option intended for already binarized images (flexible enough to accomodate different types of binarization, e.g. True/False, 1/0, 255/0, ...)
        # If the image is not binary, this will assume the midpoint of the range of values as the threshold
        threshold = (np.max(img) + np.min(img))/2
    binarized = imgS > threshold
    
    
    
    # Process center input
    if type(center) is str:
        if center == 'image':
            center = np.array(np.shape(img))/2
        elif center == 'center':
            center = None
    else:
        assert isinstance(center, (tuple, list, np.ndarray)) and len(center) == 3, 'If provided as coordinates, center must be be a three element list or array.'

    
    points = np.column_stack(np.where(binarized)).astype(np.float64)/scale # Dividing by scale puts points back to the original coordinate system (`center` uses the original image so is consistent with this coordinate system)

    axis_order = np.asarray(axis_order)[[2,1,0]] # Flip axes to correspond to image axis order (z,y,x)
    transform = AxisAlignPoints(points, axis_order=axis_order, center=center, return_transformed=False, return_transform=True)

    # maxs = np.max(transformed_points,axis=0)
    # mins = np.min(transformed_points,axis=0)
    # if np.any(maxs > np.shape(img)) or np.any(mins < 0):
    #     warnings.warn('Some of the object is being moved out of frame. Consider padding the image, adjusting center, or changing the axis_order.')
    
    if return_transformed:
        transformed = transform_image(img, transform, options=transform_args)
        if return_transform:
            return transformed, transform
        else:
            return transformed
    elif return_transform:
        return transform
    return

def Point2Point(points1, points2, T0=None, bounds=None, transform='rigid', metric='icp', 
    method='icp', decimation=1, transform_args={}, optimizer_args=None, verbose=True):
    """
    Point cloud-to-point cloud alignment. points2 will be aligned to points1.

    Parameters
    ----------
    points1 : array_like
        Fixed points that points2 will be registered to
    points2 : array_like
        Moving points that will be registered to points1
    T0 : array_like or NoneType, optional
        Initial transformation to apply to points2, by default, None. This can 
        serve as an 'initial guess' of the alignment.
    bounds : array_like or NoneType, optional
        Optimization bounds, formatted as [(min,max),...] for each parameter.
        If None, bounds are selected that should cover most possible 
        transformations, by default None. Not used by all optimizers.
    transform : str, optional
        Transformation model, by default 'rigid'.

        - 'rigid': translations and rotations
        - 'similarity: translations, rotations, and uniform scaling
        - 'affine': translations, rotations, triaxial scaling, shearing

    metric : str, optional
        Similarity metric to compare the two point clouds, by default 'closest_point_MSE'
    method : str, optional
        Optimization method, by default 'direct'. See 
        :func:`~mymesh.register.optimize` for details.
    decimation : float, optional
        Scalar factor in the range (0,1] used to reduce the size of the point set,
        by default 1. ``decimation = 1`` uses the full point sets, numbers less 
        than one will reduce the size of both point sets by that factor by 
        randomly selecting a set of points to use. For example ``decimation = 0.5``
        will use only half of the points of each set. A random seed of 0 is used
        for repeatable results. Note that if verbose=True, the final score
        will be reported for the full point set, not the decimated point set
        used during optimization.
    transform_args : dict, optional
        Additional arguments for the chosen transformation model, by default {}.
        See :func:`~mymesh.register.rigid`, :func:`~mymesh.register.similarity`,
        or :func:`~mymesh.register.affine`.
    optimizer_args : dict, optional
        Additional arguments for the chosen optimizer, by default None. See 
        :func:`~mymesh.register.optimize` for details.
    verbose : bool, optional
        Verbosity, by default True. If True, iteration progress will be printed.

    Returns
    -------
    new_points : np.ndarray
        Transformed coordinate array of points2 registered to points1.
    T : np.ndarray
        Affine transformation matrix (shape=(4,4)) to transform points2 to new_points.
        :code:`new_points = transform_points(points2, T)`
    """
    
    if T0 is not None:
        points2T0 = transform_points(points2, T0)
    else:
        points2T0 = points2
        T0 = np.eye(np.shape(points2)[1] + 1)
    if 0 < decimation <= 1:
        rng = np.random.default_rng(0)
        idx1 = rng.choice(len(points1), size=int(len(points1)*decimation), replace=False)
        idx2 = rng.choice(len(points2), size=int(len(points2)*decimation), replace=False)
        ref_points = points1[idx1]
        moving_points = points2T0[idx2]
    else:
        raise ValueError(f'decimation must be a scalar value in the range (0,1], not {str(decimation):s}.')

    if metric.lower() == 'icp' or method.lower() == 'icp':
        verbose=False

    center = np.mean(points1,axis=0)
    if transform.lower() == 'rigid':
        nparam = 6
        transformation = lambda x : rigid(x, center=center)
        x0 = np.zeros(nparam)
        if bounds is None:
            eps = 1e-10
            bounds = [
                sorted([np.min(points2[:,0]) - np.max(points1[:,0])-eps, np.max(points2[:,0]) - np.min(points1[:,0])+eps]),
                sorted([np.min(points2[:,1]) - np.max(points1[:,1])-eps, np.max(points2[:,1]) - np.min(points1[:,1])+eps]),
                sorted([np.min(points2[:,2]) - np.max(points1[:,2])-eps, np.max(points2[:,2]) - np.min(points1[:,2])+eps]),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi)
            ]
            
        if verbose:
            print('iter.||      score||       tx |       ty |       tz |    alpha |     beta |    gamma ')
            print('-----||-----------||----------|----------|----------|----------|----------|----------')          
    elif transform.lower() == 'similarity':
        nparam = 7
        transformation = lambda x : similarity(x, center=center)
        x0 = np.zeros(nparam)
        x0[6] = 1
        if bounds is None:
            eps = 1e-10
            xbounds = sorted([np.min(points2[:,0]) - np.max(points1[:,0])-eps, np.max(points2[:,0]) - np.min(points1[:,0])+eps])
            ybounds = sorted([np.min(points2[:,1]) - np.max(points1[:,1])-eps, np.max(points2[:,1]) - np.min(points1[:,1])+eps])
            zbounds = sorted([np.min(points2[:,2]) - np.max(points1[:,2])-eps, np.max(points2[:,2]) - np.min(points1[:,2])+eps])
            bounds = [
                xbounds,
                ybounds,
                zbounds,
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (0.9, 1.1),
            ]
        if verbose:
            print('iter.||      score||       tx |       ty |       tz |    alpha |     beta |    gamma |        s |')
            print('-----||-----------||----------|----------|----------|----------|----------|----------|----------|')
    elif transform.lower() == 'affine':
        nparam = 15
        transformation = lambda x : affine(x, center=center)
        x0 = np.zeros(nparam)
        x0[6:9] = 1
        if bounds is None:
            eps = 1e-10
            xbounds = sorted([np.min(points2[:,0]) - np.max(points1[:,0])-eps, np.max(points2[:,0]) - np.min(points1[:,0])+eps])
            ybounds = sorted([np.min(points2[:,1]) - np.max(points1[:,1])-eps, np.max(points2[:,1]) - np.min(points1[:,1])+eps])
            zbounds = sorted([np.min(points2[:,2]) - np.max(points1[:,2])-eps, np.max(points2[:,2]) - np.min(points1[:,2])+eps])
            bounds = [
                xbounds,
                ybounds,
                zbounds,
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (0.9, 1.1),
                (0.9, 1.1),
                (0.9, 1.1),
                np.divide(xbounds,10),
                np.divide(xbounds,10),
                np.divide(ybounds,10),
                np.divide(ybounds,10),
                np.divide(zbounds,10),
                np.divide(zbounds,10),
            ]
        if verbose:
            print('iter.||      score||       tx |       ty |       tz |    alpha |     beta |    gamma |       sx |       sy |       sz |    shxy |')
            print('-----||-----------||----------|----------|----------|----------|----------|----------|----------|----------|----------|----------')
    elif transform.lower() == 'rigid2d':
        nparam = 3
        transformation = lambda x : rigid2d(x, center=center)
        x0 = np.zeros(nparam)
        if bounds is None:
            eps = 1e-10
            bounds = [
                sorted([np.min(points2[:,0]) - np.max(points1[:,0])-eps, np.max(points2[:,0]) - np.min(points1[:,0])+eps]),
                sorted([np.min(points2[:,1]) - np.max(points1[:,1])-eps, np.max(points2[:,1]) - np.min(points1[:,1])+eps]),
                (-np.pi, np.pi)
            ]
            
        if verbose:
            print('iter.|| score||       tx |       ty |    theta ')
            print('-----||------||----------|----------|----------')           
    elif transform.lower() == 'similarity2d':
        nparam = 4
        transformation = lambda x : similarity2d(x, center=center)
        x0 = np.zeros(nparam)
        x0[3] = 1
        if verbose:
            print('iter.|| score||       tx |       ty |    theta |        s ')
            print('-----||------||----------|----------|----------|----------')
    elif transform.lower() == 'affine2d':
        nparam = 7
        transformation = lambda x : affine2d(x, center=center)
        x0 = np.zeros(nparam)
        x0[3:5] = 1
        if verbose:
            print('iter.||      score||       tx |       ty |    theta |       s1 |       s2 |     sh01 |     sh10 ')
            print('-----||-----------||----------|----------|----------|----------|----------|----------|----------')
    else:
        raise ValueError('Invalid transform model. Must be one of: "rigid", "similarity" or "affine".')
        
    assert len(x0) == nparam, f"The provided parameters for x0 don't match the transformation model ({transform:s})."

    if metric.lower() == 'symmetric_closest_point_mse':
        tree1 = scipy.spatial.KDTree(ref_points) 
        # Note: can't precommute tree for the moving points
        obj = lambda p1, p2 : symmetric_closest_point_MSE(p1, p2, tree1=tree1)     
    elif metric.lower() == 'hausdorff':
        obj = hausdorff
    elif metric.lower() == 'closest_point_mse':
        tree1 = scipy.spatial.KDTree(ref_points)
        obj = lambda p1, p2 : closest_point_MSE(p1, p2, tree1=tree1)
    elif metric.lower() == 'icp':
        pass
    else:
        raise ValueError(f'Similarity metric f"{metric:s}" is not supported for Point2Point registration.')
    
    if metric.lower() == 'icp' or method.lower() == 'icp':
        _, T = ICP(ref_points, moving_points, T0=T0)
        new_points = transform_points(points2, T)
    else:
        def objective(x):
            objective.k += 1
            if verbose: print('{:5d}'.format(objective.k),end='')
            T = transformation(x)
            pointsT = transform_points(moving_points)

            f = obj(points1, pointsT)
            if verbose: 
                print(f'||{f:11.4f}|',end='')
                print(('|{:10.4f}'*len(x)).format(*x))
            return f
        objective.k = 0
        x,f = optimize(objective, method, x0, bounds, optimizer_args=optimizer_args)
        
        T = transformation(x) @ T0
        # pointsT = (T@np.column_stack([points2, np.ones(len(points2))]).T).T
        # new_points = pointsT[:,:-1]
        new_points = transform_points(points2, T)
        f = obj(points1, new_points)
        if verbose: 
            print('-----||-----------|', end='')
            for i in x:
                print('|----------',end = '')
            print('')
            print(f'final||{f:11.4f}|',end='')
            print(('|{:10.4f}'*len(x)).format(*x))

    return new_points, T

def Mesh2Mesh(M1, M2, T0=None, bounds=None, transform='rigid', metric='icp', 
    method='icp', decimation=1, transform_args={}, optimizer_args=None, verbose=True):
    """
    Mesh-to-mesh registration. M2 will be aligned to M1.

    Parameters
    ----------
    M1 : mymesh.mesh
        Fixed mesh that M2 will be aligned to
    M2 : mymesh.mesh
        Moving mesh that will be aligned to M1
    T0 : array_like or NoneType, optional
        Initial transformation to apply to M2, by default, None. This can 
        serve as an 'initial guess' of the alignment.
    bounds : array_like or NoneType, optional
        Optimization bounds, formatted as [(min,max),...] for each parameter.
        If None, bounds are selected that should cover most possible 
        transformations, by default None. Not used by all optimizers.
    transform : str, optional
        Transformation model, by default 'rigid'.

        - 'rigid': translations and rotations
        - 'similarity: translations, rotations, and uniform scaling
        - 'affine': translations, rotations, triaxial scaling, shearing

    metric : str, optional
        Similarity metric to compare the two point clouds, by default 'closest_point_MSE'
    method : str, optional
        Optimization method, by default 'direct'. See 
        :func:`~mymesh.register.optimize` for details.
    decimation : float, optional
        Scalar factor in the range (0,1] used to reduce the size of the point set,
        by default 1. ``decimation = 1`` uses the full point sets, numbers less 
        than one will reduce the size of both point sets by that factor by 
        randomly selecting a set of points to use. For example ``decimation = 0.5``
        will use only half of the points of each set. A random seed of 0 is used
        for repeatable results. Note that if verbose=True, the final score
        will be reported for the full point set, not the decimated point set
        used during optimization.
    transform_args : dict, optional
        Additional arguments for the chosen transformation model, by default {}.
        See :func:`~mymesh.register.rigid`, :func:`~mymesh.register.similarity`,
        or :func:`~mymesh.register.affine`.
    optimizer_args : dict, optional
        Additional arguments for the chosen optimizer, by default None. See 
        :func:`~mymesh.register.optimize` for details.
    verbose : bool, optional
        Verbosity, by default True. If True, iteration progress will be printed.


    Returns
    -------
    Mnew : mymesh.mesh
        Transformed transformed mesh registered to M1.
    T : np.ndarray
        Affine transformation matrix (shape=(4,4)) to transform M2 to Mnew.
        :code:`Mnew.NodeCoords = transform_points(M.NodeCoords, T)`

    Examples
    --------

    .. plot::
        :context: close-figs

        import mymesh
        from mymesh import register
        import numpy as np

        # Load two versions of the stanford bunny (fine and coarsened)
        m1 = mymesh.demo_mesh('bunny') 
        m2 = mymesh.demo_mesh('bunny-res2')

        # Perform an arbitrary rotation to the copy
        R = register.rotation([np.pi/6, -np.pi/6, np.pi/6])
        m2.NodeCoords = register.transform_points(m2.NodeCoords, R)

        # Align the two meshes using the iterative closest point (ICP) algorithm
        m_aligned, T = register.Mesh2Mesh(m1, m2, method='icp')
    
    
    .. grid:: 2

        .. grid-item::

            .. plot::
                :context: close-figs
                :include-source: False

                mcopy = m1.copy()
                mcopy.NodeData['label'] = np.zeros(mcopy.NNode)
                m2.NodeData['label'] = np.ones(m2.NNode)
                mcopy.merge(m2)

                mcopy.plot(scalars='label', show_colorbar=False, view='xy')

        .. grid-item::

            .. plot::
                :context: close-figs
                :include-source: False

                mcopy = m1.copy()
                mcopy.NodeData['label'] = np.zeros(mcopy.NNode)
                m_aligned.NodeData['label'] = np.ones(m_aligned.NNode)
                mcopy.merge(m_aligned)

                mcopy.plot(scalars='label', show_colorbar=False, view='xy')
        

    """


    points1 = M1.NodeCoords
    points2 = M2.NodeCoords

    new_points2, T = Point2Point(points1, points2, T0=T0, bounds=bounds, transform=transform, metric=metric, method=method, decimation=decimation, transform_args=transform_args, optimizer_args=optimizer_args, verbose=verbose)
    Mnew = M2.copy()
    Mnew.NodeCoords = new_points2
    return Mnew, T

def Image2Image(img1, img2, T0=None, bounds=None, center='image', transform='rigid', metric='dice', 
        method='direct', scale=1, interpolation_order=1, threshold=None, transform_args=None, 
        optimizer_args=None, decimation=1, verbose=True, point_method='boundary'):
    """
    Image registration for 2D or 3D images. :code:`img2` will be registered to :code:`img1`

    Parameters
    ----------
    img1 : array_like
        Image array of the fixed image. Two or three dimensional numpy array of image data
    img2 : array_like
        image array of the moving image. Two or three dimensional numpy array of image data
    T0 : array_like or NoneType, optional
        Initial transformation to apply to img2, by default, None. This can 
        serve as an 'initial guess' of the alignment.
    bounds : array_like or NoneType, optional
        Optimization bounds, formatted as [(min,max),...] for each parameter.
        If None, bounds are selected that should cover most possible 
        transformations, by default None. Not used by all optimizers.
    center : str, array_like, optional
        Location of the center of rotation of the image. This will be used as the 
        "center" input to the transformation model (e.g. :func:`rigid`, :func:`affine`).
        If "center" is given in `transform_args`, that value will be used instead.

        - 'image': Rotation about the center of the image, :code:`np.shape(img)/2`.
        - 
            `[x,y,z]` or `[x,y]`: A tow or three element list or array 
            specifies the location, in voxels/pixels, of where to place to 
            place the center of rotation 

    transform : str, optional
        Transformation model, by default 'rigid'. If 2d images are given, the
        transformation model will automatically be modified to the two dimensional
        variant (e.g. 'rigid' -> 'rigid2d')

        - 'translation': translations

        - 'rotation': rotations

        - 'rigid': translations and rotations

        - 'similarity: translations, rotations, and uniform scaling

        - 'affine': translations, rotations, triaxial scaling, shearing

        - 'translation2d': translations in two dimensions

        - 'rotation2d': rotations in two dimensions

        - 'rigid2': translations and rotations in two dimensions

        - 'similarity: translations, rotations, and uniform scaling in two dimensions

        - 'affine': translations, rotations, triaxial scaling, shearing in two dimensions

    metric : str, optional
        Similarity metric to compare the two point clouds, by default 'hausdorff'
    method : str, optional
        Optimization method, by default 'direct'. See 
        :func:`~mymesh.register.optimize` for details.
    scale : float, optional
        Scale factor used to resample the image to either reduce (scale < 1) 
        or increase (scale > 1) the size/resolution of the image, by default
        1. The returned image will still be of the original resolution.
    interpolation_order : int, optional
        Interpolation order used in image transformation (see 
        `scipy.ndimage.affine_transform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html#affine-transform>`_)
        and scaling (if used). Must be an integer in the range 0-5. Lower order
        is more efficient at the cost of quality. By default, 1. 
    threshold : NoneType, float, or tuple, optional
        Threshold value(s) to binarize the images. Images are binarized by `img > threshold`.
        If given as a float or scalar value, this threshold value is applied to both images.
        If given as a two-element tuple (or array_like), the first value is applied to `img1`
        and the second value is paplied to `img2`. If None, the image is assumed to already
        be binarized (or doesn't require binarization, depending on which similarity metric
        is chosen). Images can be binarized arbitrarily, consisting of `True`/`False`, `1`/`0`,
        `255`/`0`, etc. If the image is not already binarized and no threshold is given, the 
        threshold value will be the midpoint of the range of values for each image (which may
        not give the intended result). By default, None.
    transform_args : dict, optional
        Optional input arguments passed to `scipy.ndimage.affine_transform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html>`_, by 
        default `dict(mode='constant', order=interpolation_order)`.
    optimizer_args : dict, optional
        Additional arguments for the chosen optimizer, by default None. See 
        :func:`~mymesh.register.optimize` for details.
    verbose : bool, optional
        Verbosity, by default True. If True, iteration progress will be printed.
    point_method : str, optional
        Method for converting image to points if using a point-based method,
        by default 'boundary'.

        - 'boundary' : The voxels of the boundaries of the thresholeded image wil be converted to points
        - 'binary' : All thresholded voxels will be converted to points
        - 'skeleton' : The morphological skeleton of the thresholded image will be converted to points

    Returns
    -------
    new_img : np.ndarray
        Transformed image array of img2 registered to img1.
    x : np.ndarray
        Transformation parameters for the transformation that registers img2
        to img1.

    Examples
    --------

    .. plot::
        :context: close-figs
        :nofigs:

        import mymesh
        from mymesh import register
        import numpy as np
        import matplotlib.pyplot as plt

        # Load two the CT scan of the Stanford Bunny
        img1 = mymesh.demo_image('bunny') 

        thresh = 100

        # Perform an arbitrary rotation to the copy
        R = register.rotation([np.pi/6, -np.pi/6, np.pi/6], 
                                center=np.array(img1.shape)/2)
        img2 = register.transform_image(img1, R)

        # Align the two meshes using the iterative closest point (ICP) algorithm
        img_aligned, T = register.Image2Image(img1, img2, method='icp', threshold=thresh)

    .. grid:: 2

        .. grid-item::

            .. plot::
                :context: close-figs
                :include-source: False

                import matplotlib.pyplot as plt

                overlay = register.ImageOverlay(img1, img2, threshold=thresh)
                plt.imshow(overlay[:,250], cmap='inferno')
                plt.axis('off')

        .. grid-item::

            .. plot::
                :context: close-figs
                :include-source: False

                overlay = register.ImageOverlay(img1, img_aligned, threshold=thresh)
                plt.imshow(overlay[:,250], cmap='inferno')
                plt.axis('off')
    
    .. note::
        
        An overlay image can be displayed using :func:`ImageOverlay`, e.g.

        .. code::

            import matplotlib.pyplot as plt
            overlay = register.ImageOverlay(img1, img_aligned, threshold=thresh)
            plt.imshow(overlay[:,250], cmap='inferno')

    """

    # Assess dimensionality
    if len(np.shape(img2)) == 3:
        nD = 3
    elif len(np.shape(img2)) == 2:
        nD = 2
        if '2d' not in transform:
            transform += '2d'
    else:
        raise ValueError('Image must be either two or three dimensional.')

    if method.lower() == 'icp' or metric.lower() == 'icp':
        method = 'icp'
        metric = 'icp'

    # Process threshold input
    if threshold is None:
        # This option intended for already binarized images (flexible enough to accomodate different types of binarization, e.g. True/False, 1/0, 255/0, ...)
        # If the image is not binary, this will assume the midpoint of the range of values as the threshold
        threshold1 = (np.max(img1) + np.min(img1))/2
        threshold2 = (np.max(img2) + np.min(img2))/2
    elif isinstance(threshold, (list, tuple, np.ndarray)):
        assert len(threshold) == 2, 'threshold must be defined as a single value or a list/tuple/array of two values (one for each image).'
        threshold1, threshold2 = threshold
    else:
        threshold1 = threshold2 = threshold
        
    # Process T0 input
    if T0 is not None:
        img2T0 = transform_image(img2, T0, options=dict(order=interpolation_order))
    else:
        # Initialize by centering img2 on img1
        center_diff = np.subtract(scipy.ndimage.center_of_mass(img1 > threshold1), scipy.ndimage.center_of_mass(img2 > threshold2))
        if nD == 3:
            T0 = translation(center_diff)
        else:
            T0 = translation2d(center_diff)
        img2T0 = transform_image(img2, T0, options=dict(order=interpolation_order))    
    
    # Process scale input
    if scale != 1:
        moving_img = scipy.ndimage.zoom(img2T0, scale, order=interpolation_order)
        fixed_img =  scipy.ndimage.zoom(img1, scale, order=interpolation_order)
    else:
        moving_img = img2T0
        fixed_img = img1
    
    # Process metric input
    point_based = False
    grayscale = False
    if metric.lower() == 'mutual_information' or metric.lower() == 'MI':
        obj = mutual_information
        grayscale = True
    elif metric.lower() == 'dice':
        obj = lambda img1, img2 : -dice(img1 > threshold1, img2 > threshold2)
    elif metric.lower() == 'symmetric_closest_point_mse':
        point_based = True
        
        # tree1 = scipy.spatial.KDTree(points1) 
        # Note: can't precommute tree for the moving points
        # obj = lambda p1, p2 : symmetric_closest_point_MSE(p1, p2, tree1=tree1)
    elif metric.lower() == 'icp':
        point_based = True
        verbose = False

    else:
        raise ValueError(f'Similarity metric f"{metric:s}" is not supported for Image2Image3d registration.')

    if point_based:
        obj = None
        if threshold is not None:
            binarized1 = fixed_img > threshold1
            binarized2 = moving_img > threshold2
        else:
            binarized1 = fixed_img
            binarized2 = moving_img

        if point_method.lower() == 'binary':
            points1 = np.column_stack(np.where(binarized1))
            points2 = np.column_stack(np.where(binarized2))
        elif point_method.lower() == 'boundary':
            boundary1 = binarized1 & ~scipy.ndimage.binary_erosion(binarized1)
            boundary2 = binarized2 & ~scipy.ndimage.binary_erosion(binarized2)

            points1 = np.column_stack(np.where(boundary1))
            points2 = np.column_stack(np.where(boundary2))

        elif point_method.lower() == 'skeleton':
            try:
                from skimage.morphology import skeletonize
            except:
                raise ImportError('scikit-image is required for skeletonization. Install with: `pip install scikit-image`.')
            
            skel1 = skeletonize(binarized1)
            skel2 = skeletonize(binarized2)

            points1 = np.column_stack(np.where(skel1))
            points2 = np.column_stack(np.where(skel2))


    # Process transform_args input
    if transform_args is None:
        transform_args = dict(mode='constant', order=interpolation_order)



    # Process center input
    if type(center) is str and center.lower() == 'image':
        center = np.array(np.shape(fixed_img))/2
    else:
        center = np.asarray(center) * scale

    # Process transform input
    if transform.lower() == 'rigid':
        nparam = 6
        transformation = lambda x : rigid(x, center=center)
        x0 = np.zeros(nparam)
        if bounds is None:
            bounds = [
                (-0.25*np.shape(fixed_img)[0],0.25*np.shape(fixed_img)[0]),
                (-0.25*np.shape(fixed_img)[1],0.25*np.shape(fixed_img)[1]),
                (-0.25*np.shape(fixed_img)[2],0.25*np.shape(fixed_img)[2]),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi)
            ]
            
        if verbose:
            print('iter.||      score||       tx |       ty |       tz |    alpha |     beta |    gamma ')
            print('-----||-----------||----------|----------|----------|----------|----------|----------')   
    elif transform.lower() == 'translation':
        nparam = 3
        transformation = lambda x : translation(x, center=center)
        x0 = np.zeros(nparam)
        if bounds is None:
            bounds = [
                (-0.25*np.shape(fixed_img)[0],0.25*np.shape(fixed_img)[0]),
                (-0.25*np.shape(fixed_img)[1],0.25*np.shape(fixed_img)[1]),
                (-0.25*np.shape(fixed_img)[2],0.25*np.shape(fixed_img)[2]),
            ]
            
        if verbose:
            print('iter.||      score||       tx |       ty |       tz ')
            print('-----||-----------||----------|----------|----------')   
    elif transform.lower() == 'rotation':
        nparam = 3
        transformation = lambda x : rotation(x, center=center)
        x0 = np.zeros(nparam)
        if bounds is None:
            bounds = [
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi)
            ]
            
        if verbose:
            print('iter.||      score||    alpha |     beta |    gamma ')
            print('-----||-----------||----------|----------|----------') 
    elif transform.lower() == 'similarity':
        nparam = 7
        transformation = lambda x : similarity(x, center=center)
        x0 = np.zeros(nparam)
        x0[6] = 1
        if bounds is None:
            bounds = [
                (-0.25*np.shape(fixed_img)[0],0.25*np.shape(fixed_img)[0]),
                (-0.25*np.shape(fixed_img)[1],0.25*np.shape(fixed_img)[1]),
                (-0.25*np.shape(fixed_img)[2],0.25*np.shape(fixed_img)[2]),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (0.9, 1.1),
            ]
        if verbose:
            print('iter.||      score||       tx |       ty |       tz |    alpha |     beta |    gamma |        s |')
            print('-----||-----------||----------|----------|----------|----------|----------|----------|----------|')
    elif transform.lower() == 'affine':
        nparam = 15
        transformation = lambda x : affine(x, center=center)
        x0 = np.zeros(nparam)
        x0[6:9] = 1
        if bounds is None:
            bounds = [
                (-0.25*np.shape(fixed_img)[0],0.25*np.shape(fixed_img)[0]),
                (-0.25*np.shape(fixed_img)[1],0.25*np.shape(fixed_img)[1]),
                (-0.25*np.shape(fixed_img)[2],0.25*np.shape(fixed_img)[2]),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (0.9, 1.1),
                (0.9, 1.1),
                (0.9, 1.1),
                np.divide((0.25*np.shape(fixed_img)[0],0.75*np.shape(fixed_img)[0]),10),
                np.divide((0.25*np.shape(fixed_img)[0],0.75*np.shape(fixed_img)[0]),10),
                np.divide((0.25*np.shape(fixed_img)[1],0.75*np.shape(fixed_img)[1]),10),
                np.divide((0.25*np.shape(fixed_img)[1],0.75*np.shape(fixed_img)[1]),10),
                np.divide((0.25*np.shape(fixed_img)[2],0.75*np.shape(fixed_img)[2]),10),
                np.divide((0.25*np.shape(fixed_img)[2],0.75*np.shape(fixed_img)[2]),10),
            ]
        if verbose:
            print('iter.||      score||       tx |       ty |       tz |    alpha |     beta |    gamma |        s |')
            print('-----||-----------||----------|----------|----------|----------|----------|----------|----------|')
    elif transform.lower() == 'rigid2d':
        nparam = 3
        transformation = lambda x : rigid2d(x, center=center)
        x0 = np.zeros(nparam)
        if bounds is None:
            bounds = [
                (-0.25*np.shape(fixed_img)[0],0.25*np.shape(fixed_img)[0]),
                (-0.25*np.shape(fixed_img)[1],0.25*np.shape(fixed_img)[1]),
                (-np.pi, np.pi),
            ]
            
        if verbose:
            print('iter.||      score||       tx |       ty |    theta ')
            print('-----||-----------||----------|----------|----------')   
    elif transform.lower() == 'translation2d':
        nparam = 2
        transformation = lambda x : translation2d(x, center=center)
        x0 = np.zeros(nparam)
        if bounds is None:
            bounds = [
                (-0.25*np.shape(fixed_img)[0],0.25*np.shape(fixed_img)[0]),
                (-0.25*np.shape(fixed_img)[1],0.25*np.shape(fixed_img)[1]),
            ]
            
        if verbose:
            print('iter.||      score||       tx |       ty ')
            print('-----||-----------||----------|----------')   
    elif transform.lower() == 'rotation2d':
        nparam = 1
        transformation = lambda x : rotation(x, center=center)
        x0 = np.zeros(nparam)
        if bounds is None:
            bounds = [
                (-np.pi, np.pi)
            ]
            
        if verbose:
            print('iter.||      score||    theta ')
            print('-----||-----------||----------') 
    elif transform.lower() == 'similarity2d':
        nparam = 4
        transformation = lambda x : similarity2d(x, center=center)
        x0 = np.zeros(nparam)
        x0[3] = 1
        if bounds is None:
            bounds = [
                (-0.25*np.shape(fixed_img)[0],0.25*np.shape(fixed_img)[0]),
                (-0.25*np.shape(fixed_img)[1],0.25*np.shape(fixed_img)[1]),
                (-np.pi, np.pi),
                (0.9, 1.1),
            ]
        if verbose:
            print('iter.||      score||       tx |       ty |    theta |        s ')
            print('-----||-----------||----------|----------|----------|----------')
    elif transform.lower() == 'affine2d':
        nparam = 15
        transformation = lambda x : affine2d(x, center=center)
        x0 = np.zeros(nparam)
        x0[6:9] = 1
        if bounds is None:
            bounds = [
                (-0.25*np.shape(fixed_img)[0],0.25*np.shape(fixed_img)[0]),
                (-0.25*np.shape(fixed_img)[1],0.25*np.shape(fixed_img)[1]),
                (-np.pi, np.pi),
                (0.9, 1.1),
                (0.9, 1.1),
                np.divide((0.25*np.shape(fixed_img)[0],0.75*np.shape(fixed_img)[0]),10),
                np.divide((0.25*np.shape(fixed_img)[1],0.75*np.shape(fixed_img)[1]),10),
            ]
        if verbose:
            print('iter.||      score||       tx |       ty |    theta |       s1 |       s2 |     sh01 |     sh10 ')
            print('-----||-----------||----------|----------|----------|----------|----------|----------|----------')        
    else:
        raise ValueError(f'Transformation model "{transform:s}" is not supported for Image2Image registration.')       
    
    if point_based:
        _, T1 = Point2Point(points1, points2, T0=None, bounds=bounds, transform=transform, metric=metric, method=method, transform_args=transform_args, decimation=decimation, optimizer_args=optimizer_args, verbose=verbose)
        # T = T1 @ T0
        # new_img = transform_image(img2, T, options=transform_args)
    else:
        if np.shape(img1) != np.shape(img2):
            raise Exception('Images must be the same size for image-based image-to-image registration.')
        def objective(x):
            objective.k += 1
            if verbose: 
                print('{:5d}'.format(objective.k),end='')
            T = transformation(x)
            
            imgT = transform_image(moving_img, T, options=transform_args)

            f = obj(fixed_img, imgT)
            if verbose: 
                print(f'||{f:11.4f}|',end='')
                print(('|{:10.4f}'*len(x)).format(*x))
            return f
        objective.k = 0
        x,f = optimize(objective, method, x0, bounds, optimizer_args=optimizer_args)
        
        T1 = transformation(x)
    if scale != 1:
        # Account for scaling so that the transformation matrix refers to the original image size
        D = np.eye(len(T0))
        D[:nD,:nD] =  np.diag([scale]*nD)
        T1 = np.linalg.inv(D) @ T1 @ D
    
    T = T1 @ T0 # include T0 so that img is transformed first by T0, then by the new transform
    new_img = transform_image(img2, T, options=transform_args)
    if obj is not None:
        f = obj(img1, new_img)
        if verbose: 
            print('-----||-----------|', end='')
            for i in x:
                print('|----------',end = '')
            print('')
            print(f'final||{f:11.4f}|',end='')
            print(('|{:10.4f}'*len(x)).format(*x))

    return new_img, T

def Mesh2Image3d():
    return

def Image2Mesh3d(img, M, h=1, threshold=None, scale=1, decimation=1, center_mesh=False, interpolation_order=3, x0=None, bounds=None, transform='rigid', metric='icp', method='icp', transform_args={}, optimizer_args=None, verbose=True):

    if scale != 1:
        img =  scipy.ndimage.zoom(img, scale, order=interpolation_order)
    mesh_points = np.copy(M.NodeCoords)
    if center_mesh:
        mesh_points = np.copy(M.NodeCoords)
        mesh_points[:,0] += -(np.max(mesh_points[:,0]) + np.min(mesh_points[:,0]))/2 + np.shape(img)[2]*h/scale/2
        mesh_points[:,1] += -(np.max(mesh_points[:,1]) + np.min(mesh_points[:,1]))/2 + np.shape(img)[1]*h/scale/2
        mesh_points[:,2] += -(np.max(mesh_points[:,2]) + np.min(mesh_points[:,2]))/2 + np.shape(img)[0]*h/scale/2
    if 0 < decimation <= 1:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(mesh_points), size=int(len(mesh_points)*decimation), replace=False)
        mesh_points = mesh_points[idx]
    else:
        raise ValueError(f'decimation must be a scalar value in the range (0,1], not {str(decimation):s}.')

    if threshold is not None:
        binarized = img > threshold
    else:
        binarized = img
    
    img_points = np.fliplr(np.column_stack(np.where(binarized)) * h/scale) # flipping to make image coordinate system (z,y,x) match mesh coordinate system

    new_points, T = Point2Point(mesh_points, img_points, x0=x0, bounds=bounds, transform=transform, metric=metric, method=method, transform_args=transform_args, optimizer_args=optimizer_args, verbose=verbose)

    P = np.array([[0., 0., 1., 0.],
       [0., 1., 0., 0.],
       [1., 0., 0., 0.],
       [0., 0., 0., 1.]])
    T2 = P@T@P  # Reorder transformations to the image coordinate system
    T2[:3,3] *= scale/h # Scale translations back to units of voxels

    new_image = transform_image(img, T)

    return new_image, T

def ICP(points1, points2, T0=None, tol=1e-8, maxIter=100):
    """
    Iterative Closest Point (ICP) algorithm for registering two point sets

    Parameters
    ----------
    points1 : array_like
        Fixed points that points2 will be registered to
    points2 : array_like
        Moving points that will be registered to points1

    Returns
    -------
    new_points : np.ndarray
        Transformed coordinate array of points2 registered to points1.
    T : np.ndarray
        Affine transformation matrix (shape=(4,4)) to transform points2 to new_points.
        :code:`new_points = transform_points(points2, T)`
    """

    if T0 is not None:
        points2T0 = transform_points(points2, T0)
    else:
        points2T0 = points2
        T0 = np.eye(np.shape(points2)[1] + 1)
    
    center_of_mass1 = np.mean(points1, axis=0)
    center_of_mass2 = np.mean(points2T0, axis=0)

    # center both point clouds
    fixed_points = points1 - center_of_mass1
    moving_points = points2T0 - (center_of_mass2)

    tree1 = scipy.spatial.KDTree(fixed_points)

    if np.shape(moving_points)[1] == 3:
        T = np.eye(4)
        # T[:3,3] = -center_of_mass2
        I = np.eye(3)
        Z = np.zeros(3)
        O = np.array([1])
    elif np.shape(moving_points)[1] == 2:
        T = np.eye(3)
        # T[:2,2] = -center_of_mass2
        I = np.eye(2)
        Z = np.zeros(2)
        O = np.array([1])
    else:
        raise ValueError('Points must be either 2D or 3D')

    # Convergence checks when the rotation and translation for the current iteration are close to zero
    convergence = lambda R : np.allclose(R, I, rtol=0, atol=tol)
    
    R = np.zeros_like(I)
    # t = np.zeros_like(Z)

    Rtotal = np.eye(len(R))

    i = 0
    while not convergence(R) and i < maxIter:
        # Distances for each pt in moving_points to the closest point in fixed_points
        distances21, idx1 = tree1.query(moving_points, workers=-1) 
        
        # Cross covariance matrix
        H = moving_points.T @ fixed_points[idx1] 

        # Singular Value Decomposition to determine rotation matrix
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Update moving points
        moving_points = (R @ moving_points.T).T # + t
        
        # Accumulate transformations
        Rtotal = R @ Rtotal

        i += 1
    
    if i == maxIter:
        warnings.warn('ICP did not converge to the specified tolerance within the maximum number of iterations.', RuntimeWarning)
    # Perform final translation to center and include T0
    
    
    T[:Rtotal.shape[0],:Rtotal.shape[1]] = Rtotal
    # move center of points to origin, perform rotation, then move to center of fixed points
    Tfinal = translation(center_of_mass1)@T@translation(-1*center_of_mass2) @ T0
    
    # T = np.block([[Rtotal, (center_of_mass1)[:,None]], [Z, O]]) @ np.block([[I, (-center_of_mass2)[:,None]], [Z, O]]) @ T0

    new_points = transform_points(points2, Tfinal)
    return new_points, Tfinal

### Transformations
def T2d(t0,t1):
    r"""
    Generate a translation matrix

    .. math::

        \mathbf{T} = \begin{bmatrix}
            1 & 0 & t_0 \\
            0 & 1 & t_1 \\
            0 & 0 & 1
            \end{bmatrix}

    Parameters
    ----------
    t0 : float
        Translation in the 0 axis (spatial y)
    t1 : float
        Translation in the 1 axis (spatial x)

    Returns
    -------
    t : np.ndarray
        3x3 translation matrix
    """    
    t = np.array([[1,0,t0],
                [0,1,t1],
                [0,0,1]])
    return t
    
def R2d(theta,center):
    r"""
    Generate a rotation matrix for a rotation about a point
    
    .. math::

        \mathbf{R} = \begin{bmatrix}
        cos(\theta) & -\sin(\theta) & 0 \\
        \sin(\theta) & \cos(\theta) & 0 \\
        0 & 0 & 1
        \end{bmatrix}

    
    For a center other than the origin, :math:`\begin{bmatrix}c_0, c_1 \end{bmatrix}`:

    :math:`\mathbf{R_c} = \mathbf{T}(\begin{bmatrix}c_0, c_1\end{bmatrix}) \mathbf{R}} \mathbf{T}(\begin{bmatrix}-c_0, -c_1 \end{bmatrix})`

    Parameters
    ----------
    theta : float
        Rotation, in radians
    center : list or np.ndarray
        Reference point for the rotation
    
    Returns
    -------
    r : np.ndarray
        3x3 Rotation matrix
    """
    
    if type(center)==list:
        center = np.array(center)
    T1 = T2d(*center)
    
    T2 = T2d(*-center)
    Rtheta = np.array([[np.cos(theta),-np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [            0,             0, 1]])
    
    r = np.linalg.multi_dot([T1,Rtheta,T2]) 
    return r

def S2d(s0,s1,reference=np.array([0,0])):
    r"""
    Generate a scaling matrix

    .. math::

        \mathbf{S} = \begin{bmatrix}
        s_0 & 0 & 0 \\
        0 & s_1 & 0 \\
        0 & 0 & 1
        \end{bmatrix}

    Parameters
    ----------
    s0 : float
        Scale factor in the 0 axis (spatial y)
    s1 : float
        Scale factor in the 1 axis (spatial x)

    Returns
    -------
    s : np.ndarray
        3x3 scaling matrix
    """    
    s = np.array([[1/s0,0,0],
                  [0,1/s1,0],
                  [0,0,1]])

    T1 = T2d(*reference)
    T2 = T2d(*-np.asarray(reference))
    s = np.linalg.multi_dot([T1,s,T2]) 

    return s

def Sh2d(sh01,sh10,reference=np.array([0,0])):
    r"""
    Generate a shearing matrix
    ref:https://www.mathworks.com/help/images/matrix-representation-of-geometric-transformations.html

    .. math::

        \mathbf{Sh} = \begin{bmatrix}
                1 & sh_{01} & 0 \\
                sh_{01} & 1 & 0 \\
                0 & 0 & 1
            \end{bmatrix}

    Parameters
    ----------
    sh01 : float
        Shear in the x-y direction
    sh10 : float
        Shear in the y-x direction

    Returns
    -------
    sh : np.ndarray
        3x3 scaling matrix
    """
    
    sh = np.array([[1,      sh10,  0],
                    [sh01,      1, 0],
                    [0,         0, 1]])
    T1 = T2d(*reference)
    T2 = T2d(*-np.asarray(reference))
    sh = np.linalg.multi_dot([T1,sh,T2]) 
    return sh

def T3d(t0,t1,t2):
    r"""
    Generate a translation matrix

    .. math::

        \mathbf{T} = \begin{bmatrix}
            1 & 0 & 0 & t_0 \\
            0 & 1 & 0 & t_1 \\
            0 & 0 & 1 & t_2 \\
            0 & 0 & 0 & 1
            \end{bmatrix}

    Parameters
    ----------
    t0 : float
        Translation in the x axis 
    t1 : float
        Translation in the y axis
    t2 : float
        Translation in the z axis 

    Returns
    -------
    t : np.ndarray
        4x4 translation matrix
    """    
    t = np.array([[1,0,0,t0],
                [0,1,0,t1],
                [0,0,1,t2],
                [0,0,0,1]])
    return t
    
def R3d(alpha,beta,gamma,center=np.array([0,0,0]),rotation_order=[0,1,2],rotation_mode='cartesian'):
    r"""
    Generates a rotation matrix for a rotation about a point

    .. math::

        \mathbf{R_0} = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & \cos(\alpha) & -\sin(\alpha) & 0 \\
        0 & \sin(\alpha) & \cos(\alpha) & 0 \\
        0 & 0 & 0 & 1
        \end{bmatrix}

        \mathbf{R_1} = \begin{bmatrix}
        \cos(\beta) & 0 & \sin(\beta) & 0 \\
        0 & 1 & 0 & 0 \\
        -\sin(\beta) & 0 & \cos(\beta) & 0 \\
        0 & 0 & 0 & 1
        \end{bmatrix}

        \mathbf{R_2} = \begin{bmatrix}
        \cos(\gamma) & -\sin(\gamma) & 0 & 0 \\
        \sin(\gamma) & \cos(\gamma) & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1
        \end{bmatrix}

        \mathbf{R_{012}} = \mathbf{R_2} \mathbf{R_1} \mathbf{R_0}

    For a center other than the origin, :math:`\begin{bmatrix}c_0, c_1, c_2 \end{bmatrix}`:

    :math:`\mathbf{R_c} = \mathbf{T}(\begin{bmatrix}c_0, c_1, c_2 \end{bmatrix}) \mathbf{R_{012}} \mathbf{T}(\begin{bmatrix}-c_0, -c_1, -c_2 \end{bmatrix})`
    
    Parameters
    ----------
    alpha : float
        Rotation about the x, in radians
    beta : float
        Rotation about the y, in radians
    gamma : float
        Rotation about the z, in radians
    center : array_like, optional
        Reference point for the rotation, by default np.array([0,0,0]).
    rotation_order : array_like, optional
        Order to perform rotations about the x (0), y (1), and z (2) axes,  by
        default [0,1,2]

    Returns
    -------
    r : np.ndarray
        4x4 Rotation matrix
    """
    if type(center)==list:
        center = np.array(center)
    T1 = T3d(*center)
    
    T2 = T3d(*-center)
    if rotation_mode == 'cartesian':
        Rx = np.array([[1,             0,             0, 0],
                    [0, np.cos(alpha),-np.sin(alpha), 0],
                    [0, np.sin(alpha), np.cos(alpha), 0],
                    [0,             0,             0, 1]])
        
        Ry = np.array([[ np.cos(beta),  0, np.sin(beta), 0],
                    [            0,  1,            0, 0],
                    [-np.sin(beta),  0, np.cos(beta), 0],
                    [            0,  0,            0, 1]])
        
        Rz = np.array([[np.cos(gamma),-np.sin(gamma),0,0],
                    [np.sin(gamma), np.cos(gamma),0,0],
                    [            0,             0,1,0],
                    [            0,             0,0,1]])
        R = [Rx, Ry, Rz]
        R = np.linalg.multi_dot([R[rotation_order[2]], R[rotation_order[1]], R[rotation_order[0]]])
    else:
        raise NotImplementedError('Rotation modes other than "cartesian" not yet implemented.')
    # elif rotation_mode == 'euler':
    #     c1 = np.cos(alpha); c2 = np.cos(beta); c3 = np.cos(gamma);
    #     s1 = np.sin(alpha); s2 = np.sin(beta); s3 = np.sin(gamma);

    #     R = np.array([
    #             [c2 , -c3*s2, s2*s3, 0],
    #             [c1*s2, c1*c2*c3-s1*s3, -c3*s1 - c1*c2*s3, 0],
    #             [s1*s2, c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, 0],
    #             [0, 0, 0, 1]
    #         ])

    r = np.linalg.multi_dot([T1,R,T2]) 
    return r

def S3d(s0,s1,s2,reference=np.array([0,0,0])):
    r"""
    Generate a 3d scaling matrix

    .. math::

        \mathbf{S} = \begin{bmatrix}
        s_0 & 0 & 0 & 0 \\
        0 & s_1 & 0 & 0 \\
        0 & 0 & s_2 & 0 \\
        0 & 0 & 0 & 1
        \end{bmatrix}

    Parameters
    ----------
    s0 : float
        Scale factor in the x axis
    s1 : float
        Scale factor in the y axis
    s2 : float
        Scale factor in the z axis

    Returns
    -------
    s : np.ndarray
        4x4 scaling matrix
    """    
    s = np.array([[1/s0,0,0,0],
                  [0,1/s1,0,0],
                  [0,0,1/s2,0],
                  [0,0,0,1]])

    T1 = T3d(*reference)
    T2 = T3d(*-np.asarray(reference))
    s = np.linalg.multi_dot([T1,s,T2]) 

    return s

def Sh3d(sh01,sh10,sh02,sh20,sh12,sh21,reference=np.array([0,0,0])):
    r"""
    Generates a shearing matrix
    ref:https://www.mathworks.com/help/images/matrix-representation-of-geometric-transformations.html

    .. math::

        \mathbf{Sh} = \begin{bmatrix}
                1 & sh_{01} & sh_{20} & 0 \\
                sh_{01} & 1 & sh_{21} & 0 \\
                sh_{02} & sh_{12} & 1 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}

    Parameters
    ----------
    sh01 : float
        Shear in the xy direction
    sh10 : float
        Shear in the yx direction
    sh02 : float
        Shear in the xz direction
    sh20 : float
        Shear in the zx direction
    sh12 : float
        Shear in the yz direction
    sh21 : float
        Shear in the zy direction
    reference : array_like
        Center to chear about, by default np.array([0,0,0])

    Returns
    -------
    sh : np.ndarray
        4x4 scaling matrix
    """
    
    sh = np.array([[1,     sh10, sh20, 0],
                    [sh01,    1, sh21, 0],
                    [sh02, sh12,    1, 0],
                    [0,       0,    0, 1]])
    T1 = T3d(*reference)
    T2 = T3d(*-np.asarray(reference))
    sh = np.linalg.multi_dot([T1,sh,T2]) 
    return sh

def rotation(x, center=np.array([0,0,0]), rotation_order=[0,1,2], rotation_mode='cartesian', image=False):
    """
    Rigid rotation in 3D.

    Parameters
    ----------
    x : list
        3 item list, containing the x, y, and z rotations
        [alpha, beta, gamma], where angles are specified in radians.
    center : list or np.ndarary, optional
        Reference point for the rotation
    rotation_order : array_like, optional
        Order to perform rotations about the x (0), y (1), and z (2) axes,  by
        default [0,1,2]

    Returns
    -------
    A : np.ndarray
        Affine rotation matrix (shape=(4,4))
    """    
    if image:
        [gamma,beta,alpha] = x
    else:
        [alpha,beta,gamma] = x
    
    r = R3d(alpha,beta,gamma,np.asarray(center),rotation_order=rotation_order,rotation_mode=rotation_mode)

    A = r
    
    return A

def rotation2d(x, center=np.array([0,0])):
    """
    Rigid rotation in 2D.

    Parameters
    ----------
    x : array_like
        1 item list, containing the rotation
        [theta], where angles are specified in radians.
    center : list or np.ndarary, optional
        Reference point for the rotation

    Returns
    -------
    A : np.ndarray
        Affine rotation matrix (shape=(3,3))
    """    
    [theta] = x
    
    r = R2d(theta,np.asarray(center))

    A = r
    
    return A

def translation(x, image=False):
    """
    Rigid transformation consisting of translation and rotation in 3D.

    Parameters
    ----------
    x : list
        3 item list, containing the x, y, and z translations 
        [t0, t1, t2], where displacements are specified in pixels.

    Returns
    -------
    A : np.ndarray
        Affine translation matrix (shape=(4,4))
    """    
    if image:
        [t2,t1,t0] = x
    else:
        [t0,t1,t2] = x
    
    t = T3d(t0,t1,t2)

    A = t
    
    return A

def translation2d(x, image=False):
    """
    Rigid transformation consisting of translation and rotation in 2D.

    Parameters
    ----------
    x : list
        2 item list, containing the x, and y translations 
        [t0, t1], where displacements are specified in pixels.

    Returns
    -------
    A : np.ndarray
        Affine translation matrix (shape=(3,3))
    """    
    if image:
        [t1,t0] = x
    else:
        [t0,t1] = x
    
    t = T2d(t0,t1)

    A = t
    
    return A

def rigid2d(x, center=np.array([0,0]), image=False):
    """
    Rigid transformation consisting of translation and rotation in 2D.

    Parameters
    ----------
    I : np.ndarray
        numpy array containing the image data
    x : list
        6 item list, containing the x, y, and z translations and rotations
        [t0, t1, t2, alpha, beta, gamma], where angles are specified in radians
        and displacements are specified in pixels.
    center : list or np.ndarary, optional
        Reference point for the rotation

    Returns
    -------
    I2 : np.ndarray
        numpy array containing the transformed image data
    """    
    if image:
        [t1,t0,theta] = x
    else:
        [t0,t1,theta] = x
    
    t = T2d(t0,t1)
    r = R2d(theta,np.asarray(center))

    A = t@r
    
    return A

def rigid(x, center=np.array([0,0,0]), rotation_order=[0,1,2], rotation_mode='cartesian', image=False):
    """
    Rigid transformation consisting of translation and rotation in 3D.

    Parameters
    ----------
    x : list
        6 item list, containing the x, y, and z translations and rotations
        [t0, t1, t2, alpha, beta, gamma], where angles are specified in radians
        and displacements are specified in pixels.
    center : list or np.ndarary, optional
        Reference point for the rotation

    Returns
    -------
    A : np.ndarray
        Affine transformation matrix (shape=(4,4))
    """    
    if image:
        [t2,t1,t0,gamma,beta,alpha] = x
    else:
        [t0,t1,t2,alpha,beta,gamma] = x
    
    t = T3d(t0,t1,t2)
    r = R3d(alpha,beta,gamma,np.asarray(center),rotation_order=rotation_order,rotation_mode=rotation_mode)

    A = t@r
    
    return A

def similarity2d(x, center=None, rotation_order=[0,1], rotation_mode='cartesian', image=False):

    if image:
        [t1,t0,theta,s0] = x
    else:
        [t0,t1,t2,theta,s0] = x
    t = T2d(t0,t1)
    r = R2d(alpha,beta,gamma,np.asarray(center),rotation_order=rotation_order,rotation_mode=rotation_mode)
    s = S2d(s0, s0, reference=center)
    A = s@t@r
    return A

def similarity(x, center=None, rotation_order=[0,1,2], rotation_mode='cartesian', image=False):

    if image:
        [t2,t1,t0,gamma,beta,alpha,s0] = x
    else:
        [t0,t1,t2,alpha,beta,gamma,s0] = x
    t = T3d(t0,t1,t2)
    r = R3d(alpha,beta,gamma,np.asarray(center),rotation_order=rotation_order,rotation_mode=rotation_mode)
    s = S3d(s0, s0, s0, reference=center)
    A = s@t@r
    return A

def affine2d(x, center=np.array([0,0]), image=False):

    if image:
        [t1,t0,theta,s1,s0,sh10,sh01] = x
    else:
        [t0,t1,theta,s0,s1,sh01,sh10] = x
    t = T2d(t0,t1)
    r = R2d(theta,np.asarray(center))
    sh = Sh2d(sh01,sh10,reference=center)
    s = S2d(s0, s1, reference=center)
    A = s@sh@t@r

    return A

def affine(x, center=np.array([0,0,0]), rotation_order=[0,1,2], rotation_mode='cartesian', image=False):

    if image:
        [t2,t1,t0,gamma,beta,alpha,s2,s1,s0,sh21,sh12,sh20,sh02,sh10,sh01] = x
    else:
        [t0,t1,t2,alpha,beta,gamma,s0,s1,s2,sh01,sh10,sh02,sh20,sh12,sh21] = x
    t = T3d(t0,t1,t2)
    r = R3d(alpha,beta,gamma,np.asarray(center),rotation_order=rotation_order,rotation_mode=rotation_mode)
    sh = Sh3d(sh01,sh10,sh02,sh20,sh12,sh21,reference=center)
    s = S3d(s0, s1, s2, reference=center)
    A = s@sh@t@r

    return A

def transform_points(points, T):
    """
    Apply transformation matrix to an array of points.

    Parameters
    ----------
    points : array_like
        Array of point coordinates (shape=(n,3)).
    T : array_like
        Transformation matrix (either (2,2), (3,3), or (4,4)).

    Returns
    -------
    new_points : np.ndarray
        Transformed point coordinates

    """    
    
    if np.shape(points)[1] == 2:
        TwoD = True
    elif np.shape(points)[1] == 3:
        TwoD = False
    else:
        raise ValueError('Points must be 2D or 3D, with shape=(n,2) or (n,3).')
        
    if np.shape(T) == (4,4):
        # Affine matrix for 3D
        if TwoD:
            raise ValueError('Transformation matrix for 3D point set must have shape (3,3), (4,4).')
        pointsT = (T@np.column_stack([points, np.ones(len(points))]).T).T
        new_points = pointsT[:,:-1]
    elif np.shape(T) == (3,3):
        if TwoD:
            # Affine matrix for 2D
            pointsT = (T@np.column_stack([points, np.ones(len(points))]).T).T
            new_points = pointsT[:,:-1]
        else:
            # Non-affine matrix for 3D
            new_points = (T@np.asarray(points).T).T
    elif np.shape(T) == (2,2):
        # Non-affine matrix for 2D
        if not TwoD:
            raise ValueError('Transformation matrix for 2D point set must have shape (2,2), (3,3).')
        new_points = (T@np.asarray(points).T).T
    else:
        raise ValueError('Transformation matrix must have shape (2,2), (3,3), or (4,4).')
        
    return new_points

def transform_image(image, T, options=dict()):
    """
    Apply transformation matrix to an image. This is essentially an interface
    to `scipy.ndimage.affine_transform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html>`_
    but takes into account the need to invert the transformation matrix for 
    consistency between the "pull" resampling performed by `affine_transform`
    and the "push" transformations used to transform points. The `options` input
    allows for inputting any of the keyword arguments used by `affine_transform`.

    Parameters
    ----------
    image : array_like
        Image array
    T : array_like
        Transformation matrix (either 3x3 or 4x4).
    options : dict, optional
        Options to be used by `scipy.ndimage.affine_transform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html>`_. If none are
        provide, all defaults will be used, by default dict(). Common options 
        that may be used are `mode` to allow wrapping ('grid-wrap') or mirroring 
        ('mirror') to change what happens when the contents of the image are
        moved beyond the bounds of the image, and `order` which changes the 
        interpolation order of the transformation (the default is 3, 
        transformations can be performed more efficiently by reducing to 1).

    Returns
    -------
    new_image : np.ndarray
        Transformed image array.
    """    
    new_image = scipy.ndimage.affine_transform(image, np.linalg.inv(T), **options)

    return new_image
### Similarity Metrics
def dice(u, v):
    """
    Dice-Sorensen coefficient for measuring the similarity between two binary images or arrays :cite:p:`Dice1945` :cite:p:`Sorensen1948` 

    Parameters
    ----------
    u : np.ndarray(dtype=bool)
        Array of binary data (can be any shape)
    v : np.ndarray(dtype=bool)
        Array of binary data (can be any shape)

    Returns
    -------
    D : float
        Dice-Sorensen coefficient in the range [0, 1]. 1 = identical, 0 = no overlap.
    """    
    TP = np.sum(u & v)
    FP = np.sum(u & np.logical_not(v))
    FN = np.sum(np.logical_not(u) & v)
    D = 2*TP/(2*TP + FP + FN)
    return D

def jaccard(u, v):
    """
    Jaccard index for measuring the similarity between two binary images or arrays

    Parameters
    ----------
    u : np.ndarray(dtype=bool)
        Array of binary data (can be any shape)
    v : np.ndarray(dtype=bool)
        Array of binary data (can be any shape)

    Returns
    -------
    J : float
        Jaccard index in the range [0, 1]. 1 = identical, 0 = no overlap.
    """ 
    TP = np.sum(u & v)
    FP = np.sum(u & np.logical_not(v))
    FN = np.sum(np.logical_not(u) & v)
    J = TP/(TP + FP + FN)
    return J

def mutual_information(img1, img2):
    
    data1 = img1.flatten()
    data2 = img2.flatten()
    
    # Data masking to disregard empty pixels that appear due to transformation
    data1 = data1[data2>0]
    data2 = data2[data2>0]

    bins = 100
    hist1, edges1 = np.histogram(data1, bins=bins, range=(0,255))
    P1 = hist1/np.sum(hist1) # probability
    H1 = -np.sum(P1[P1>0] * np.log2(P1[P1>0])) # Entropy ( >0 prevents log of 0)

    hist2, edges2 = np.histogram(data2, bins=bins, range=(0,255))
    P2 = hist2/np.sum(hist2)
    H2 = -np.sum(P2[P2>0] * np.log2(P2[P2>0]))

    hist12, xedges, yedges = np.histogram2d(data1, data2, bins=bins, range=((0,255),(0,255)))
    P12 = hist12/np.sum(hist12)
    H12 = -np.sum(P12[P12>0] * np.log2(P12[P12>0]))

    MI = H1 + H2 - H12
    return -MI

def hausdorff(points1, points2):
    """
    Directed hausdorff distance between two sets of points. The Hausdorff distance is the
    largest closest-point distance between two point sets. This is a non-symmetric measure,
    i.e. hausdorff(points1, points2) != hausdorff(points2, points1).
    This is a wrapper to scipy.spatial.distance.directed_hausdorff

    Parameters
    ----------
    points1 : array_like
        Coordinates of the first point set
    points2 : array_like
        Coordinates of the second point set

    Returns
    -------
    d, float
        Hausdorff distance
    """    
    d, i, j = scipy.spatial.distance.directed_hausdorff(points1, points2)

    return d

def closest_point_MSE(points1, points2, tree1=None):
    """
    Mean squared error of the closest point distances between two point sets.
    This is a non-symmetric measure, 
    i.e. closest_point_MSE(points1, points2) != closest_point_MSE(points2, points1).

    Parameters
    ----------
    points1 : array_like
        Coordinates of the first point set
    points2 : array_like
        Coordinates of the second point set
    tree1 : scipy.spatial.KDTree, optional
        Optional pre-computed KDTree structure of points1, by default None

    Returns
    -------
    MSE : float
        Mean squared error of closest points
    """
    if tree1 is None:
        tree1 = scipy.spatial.KDTree(points1)
    distances, paired_indices = tree1.query(points2)
    MSE = np.sum(distances**2)/len(points2)

    return MSE

def symmetric_closest_point_MSE(points1, points2, tree1=None, tree2=None):
    """
    Mean squared error of the closest point distances between two point sets.
    Unlike :func:`closest_point_MSE`, this is a symmetric measure, 
    i.e. closest_point_MSE(points1, points2) == closest_point_MSE(points2, points1).

    Parameters
    ----------
    points1 : array_like
        Coordinates of the first point set
    points2 : array_like
        Coordinates of the second point set
    tree1 : scipy.spatial.KDTree, optional
        Optional pre-computed KDTree structure of points1, by default None.
        (`tree1 = scipy.spatial.KDTree(points1)`)
    tree2 : scipy.spatial.KDTree, optional
        Optional pre-computed KDTree structure of points2, by default None
        (`tree2 = scipy.spatial.KDTree(points2)`)

    Returns
    -------
    MSE : float
        Mean squared error of closest points
    """

    if tree1 is None:
        tree1 = scipy.spatial.KDTree(points1)
    if tree2 is None:
        tree2 = scipy.spatial.KDTree(points2)
    
    distances1, _ = tree1.query(points2)
    distances2, _ = tree2.query(points1)
    distances = np.append(distances1, distances2)
    MSE = np.sum(distances**2)/len(distances)
    
    return MSE

## Feature Detection
def intrinsic_shape_signatures(points, r=None, tree=None, weighted=False):

    if tree is None:
        tree = scipy.spatial.KDTree(points)

    d, i = tree.query(points,2)
    nearest_distances = d[:,1] # Ignoring the zero distance to self
    r = np.mean(nearest_distances) * 2
    query = tree.query_ball_tree(tree, r)

    # if weighted:
    #     weights = [np.linalg.norm(points[i] - points[query[i]], axis=1) for i in range(len(points))]

    # else:
    COV = []
    for i in range(len(points)):
        diff = points[query[i]] - points[i]
        COV.append((diff.T@diff)/len(diff))

    COV = np.array(COV)
    eigvals, eigvecs = np.linalg.eig(COV)
    eigsort = np.argsort(eigvals,axis=1)[:, ::-1] # sort descending
    eigvals = np.take_along_axis(np.real(eigvals), eigsort, 1)
    eigvecs = np.take_along_axis(np.real(eigvecs), eigsort[:,None,:], 2)

    gamma10 = eigvals[:,1]/eigvals[:,0]
    gamma21 = eigvals[:,2]/eigvals[:,1]

    thresh10 = thresh21 = 0.975

    salience = eigvals[:,2].copy()
    salience[(eigvals[:,1]/eigvals[:,0] >= thresh10) | (eigvals[:,2]/eigvals[:,1] >= thresh21)] = 0

    

    return

### Optimization
def optimize(objective, method, x0=None, bounds=None, optimizer_args=None):
    """
    Optimization interface for registration. This function interfaces with
    optimizers from scipy.optimize and pdfo.

    Parameters
    ----------
    objective : callable
        Objective function that takes a single input, `x`.

    method : str
        Optimization algorithm. For the most part, these are direct interfaces 
        to either scipy.optimize or pdfo, however some predefined options are
        chosen for certain methods.

        scipy global optimizers:
        ------------------------
        These are global optimization methods that see the global minimum
        of the objective function within specified bounds. The `bounds`
        input is required for all of these methods.

        - `'direct'`: Uses the DIRECT algorithm through :func:`scipy.optimize.direct`.

        - 
            '`directl'`: Uses the locally-biased version DIRECT algorthim through :func:`scipy.optimize.direct`.
            This is equivalent to using `method='direct'` with `optimizer_args=dict(locally_biased=True)`.

        - 
            `'differential_evolution'`: Uses the differential evolution algorithm through :func:`scipy.optimize.differential_evolution`

        - 
            `'brute'`: Uses a brute force approach, evaluating the function at every point within a multidimensional grid
            using :func:`scipy.optimize.brute`. It's recommeded to use the 'Ns' option to specify the number of points 
            to sample along each axis (`optimizer_args=dict(Ns=n)`), the default value of 20 may be too high for many
            registration applications for large datasets. The total number of function evaluations is `Ns**len(x)`.
            By default, the optional 'finish' input, which performs local optimization following the conclusion of the 
            brute force search, is turned off, but can be reactivated with `optimizer_args=dict(finish=True)`

        scipy local optimizers:
        -----------------------
        All minimizers available through :func:`scipy.optimize.minimize` are available. One exception is that, if pdfo
        is installed, it will be used instead of scipy if `method='cobyla'`.  If `method='scipy'` is given, the default
        optimizer will be chosen by scipy based on the given problem (depends on the presence of bounds or constraints).

        pdfo local optimizers:
        ----------------------
        pdfo, or "Powell's derivative free optimizers" are a group of algorithms developed by M. J. D. Powell for 
        gradient/derivative free optimization. pdfo offers a scipy-like interface to Powell's algorithms.

        - `'uobyqa'`:  Unconstrained Optimization BY Quadratic Approximation
        
        - `'newuoa'`: NEW Unconstrained Optimization Algorithm
        
        - `'bobyqa'`: Bounded Optimization BY Quadratic Approximation
        
        - `'lincoa'`: LINear Constrained Optimization Algorithm
        
        - `'cobyla'`: Constrained Optimization BY Linear Approximation


    x0 : array_like, optional
        Initial guess for the optimization, by default None. This is required for local, but not global
        methods.
    bounds : array_like, list of tuples, optional
        List of bounds for each parameter, e.g. `[(-1, 1), (-1, 1), ...]`, by default None
        Bounds are required for some optimizers, particularly the global methods.
    optimizer_args : dict, optional
        Additional input arguments to the chosen method, by default None. 
        See available options in the documentation of scipy or pdfo. 

        Example (`method='nelder-mead'`):
            `optimizer_args = dict(maxiter=100, fatol=1e-3)`

        Note that the optional arguments or methods differ. Some have an `options` input, 
        which must be defined within `optimizer_args`
        Example (`method='powell'`):
            `optimizer_args = dict(options=dict(maxiter=100))`

    Returns
    -------
    x : np.ndarray
        Optimized parameters
    f : float
        Value of the objective function at the identified optimal parameters

    """    
    try:
        import pdfo
        pdfo_avail = True
    except:
        pdfo_avail = False
    # default optimizer settings for selected optimizers
    if optimizer_args is None:
        if method.lower() == 'direct':
            optimizer_args = dict(locally_biased=False)
        else:
            optimizer_args = {}
    
    # Scipy global optimizers
    if method.lower() == 'direct' or method.lower() == 'directl':
        if bounds is None:
            raise ValueError('bounds are required for the "direct" optimizer.')
        if method.lower() == 'directl':
            optimizer_args['locally_biased'] = True
        res = scipy.optimize.direct(objective, bounds, **optimizer_args)
    elif method.lower() == 'differential_evolution':
        if bounds is None:
            raise ValueError('bounds are required for the "differential_evolution" optimizer.')
        res = scipy.optimize.differential_evolution(objective, bounds, **optimizer_args)
    elif method.lower() == 'brute':
        if bounds is None:
            raise ValueError('bounds are required for the "brute" optimizer.')
        if 'finish' not in optimizer_args:
            optimizer_args['finish'] = None
        optimizer_args['full_output'] = True # this is required for output processing
        x, f, _, _= scipy.optimize.brute(objective, bounds, **optimizer_args)
        res = dict(x=x, fun=f, success=True) # create a result dict to work with the syntax of the other optimizers
        
    # Scipy local optimizers
    elif method.lower() in ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                'l-bfgs-b', 'tnc', 'cobyqa', 'slsqp', 'trust-constr', 
                'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov', 'scipy'] or (not pdfo_avail and method.lower()=='cobyla'):
        if x0 is None:
            raise ValueError(f'x0 is required for the {method:s} optimizer')
        if bounds is not None and 'bounds' not in optimizer_args:
            optimizer_args['bounds'] = bounds
        if method.lower() == 'scipy':
            # Use default selection
            method = None
        res = scipy.optimize.minimize(objective, x0, method=method, **optimizer_args)
    
    # Powell derivative-free optimizers (via pdfo)
    elif method.lower() in ['uobyqa', 'newuoa', 'bobyqa', 'lincoa', 'cobyla', 'pdfo']:
        if not pdfo_avail:
            raise ImportError('For optimization with {method:s}, pdfo is required: pip install pdfo')
            
        if x0 is None:
            raise ValueError(f'x0 is required for the {method:s} optimizer')
        
        if method.lower() in ['uobyqa', 'newuoa']:
            bounds = None
        if bounds is not None and 'bounds' not in optimizer_args:
            optimizer_args['bounds'] = bounds
        
        if method.lower() == 'pdfo':
            # Use default selection
            method = None
            
        res = pdfo.pdfo(objective, x0, method=method, **optimizer_args)
            
    else:
        raise ValueError(f'Method "{method:s}" is not supported.')
    if not res['success']:
        message = res["message"]
        warnings.warn(f'Optimization was not successful. \nOptimizer exited with message "{message:s}".', category=RuntimeWarning)
    
    x = res['x']
    f = res['fun']
    return x, f
        
def ImageOverlay(img1, img2, threshold=None):
    """
    Generate an overlay image. The overlay image will have the same shape as 
    :code:`img1` and :code:`img2` and will have the following values:

    - 0 : ~(img1 > threshold) & ~(img2 > threshold)
    
    - 1 : (img1 > threshold) & ~(img2 > threshold)

    - 2 : ~(img1 > threshold) & (img2 > threshold)

    - 3 : (img1 > threshold) & (img2 > threshold)


    Parameters
    ----------
    img1 : array_like
        Image array of the fixed image. Two or three dimensional numpy array of image data
    img2 : array_like
        image array of the moving image. Two or three dimensional numpy array of image data
    threshold : NoneType, float, or tuple, optional
            Threshold value(s) to binarize the images. Images are binarized by `img > threshold`.
            If given as a float or scalar value, this threshold value is applied to both images.
            If given as a two-element tuple (or array_like), the first value is applied to `img1`
            and the second value is paplied to `img2`. If None, the image is assumed to already
            be binarized (or doesn't require binarization, depending on which similarity metric
            is chosen). Images can be binarized arbitrarily, consisting of `True`/`False`, `1`/`0`,
            `255`/`0`, etc. If the image is not already binarized and no threshold is given, the 
            threshold value will be the midpoint of the range of values for each image (which may
            not give the intended result). By default, None.

    Returns
    -------
    overlay : np.ndarray
        Image array of the overlay
    """    

    assert np.shape(img1) == np.shape(img2), 'Images must be the same size.'
    if isinstance(threshold, (list, tuple, np.ndarray)):
        assert len(threshold) == 2, 'threshold must be defined as a single value or a list/tuple/array of two values (one for each image).'
        threshold1, threshold2 = threshold
    else:
        threshold1 = threshold2 = threshold
    
    if threshold1 is not None:
        bw1 = img1 > threshold1
    else:
        bw1 = img1
        
    if threshold2 is not None:
        bw2 = img2 > threshold2
    else:
        bw2 = img2
    
    overlay = np.zeros_like(bw1, dtype=np.int32)
    overlay[bw1] = 1
    overlay[bw2] = 2
    overlay[bw1 & bw2] = 3
    return overlay
   
