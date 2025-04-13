# -*- coding: utf-8 -*-
# Created on Fri Oct 14 13:16:00 2024
# @author: toj
"""
Tools for registering or aligning point clouds, meshes, and images.

There are three core components to a registration algorithm: the transformation
model, the similarity metric (or objective function), and the optimization 
strategy. 

.. graphviz::

    graph registration {
    node [shape=box, style=rounded];
    edge [style=solid];

    Object1 [pos="0,1!"];
    Object2 [pos="0,0!"]; 

    subgraph register {
        color = "black";
        style = rounded;

        Transform [pos="0,2!"]
        Comparison [pos="0,1!"]
        Update [pos="0,0!"]
    }

    }


.. currentmodule:: mymesh.contour

Registration
============
.. autosummary::
    :toctree: submodules/

    Point2Point
    Image2Image3d
    Image2Image2d

Transformation
==============
.. autosummary::
    :toctree: submodules/

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
    mutual_information
    hausdorff
    closest_point_MSE

Optimization
============
.. autosummary::
    :toctree: submodules/

    optimize

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
        mat = perpendicular_transform@mat
        mvbb_t = transform_points(mvbb, mat)
    
    # Restore center after rotation
    if center is None:
        center = np.mean(mvbb,axis=0)
    else:
        assert isinstance(center, (tuple, list, np.ndarray)) and len(center) == 3, 'If provided, center must be be a three element list or array.'
        center = np.asarray(center)
    center_shift = center - np.mean(mvbb_t,axis=0)
    
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

def AxisAlignImage(img, axis_order=[2,1,0], threshold=None, center='image', transform_options=dict(), return_transformed=True, return_transform=False):
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
        - `[x,y,z]`: A three element list or array specifies the location, in
        voxels, of where to place to place the center of the bounding box of the 
        object after transformation. 
    transform_options : dict, optional
        Optional input arguments passed to `scipy.ndimage.affine_transform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html>`_, by 
        default dict().
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
    if threshold is not None:
        # TODO: Should have other thresholding options
        binarized = img > threshold 
    else:
        if img.dtype is np.dtype(bool) or np.all(np.isin(img, [np.min(img), np.max(img)])):
            binarized = img
            raise ValueError('For non-binary images, a threshold must be given.')

    if type(center) is str:
        if center == 'image':
            center = np.array(np.shape(img))/2
        elif center == 'center':
            center = None
    else:
        assert isinstance(center, (tuple, list, np.ndarray)) and len(center) == 3, 'If provided as coordinates, center must be be a three element list or array.'

    
    points = np.column_stack(np.where(binarized))

    axis_order = np.asarray(axis_order)[[2,1,0]] # Flip axes to correspond to image axis order (z,y,x)
    transform = AxisAlignPoints(points, axis_order=axis_order, center=center, return_transformed=False, return_transform=True)

    # maxs = np.max(transformed_points,axis=0)
    # mins = np.min(transformed_points,axis=0)
    # if np.any(maxs > np.shape(img)) or np.any(mins < 0):
    #     warnings.warn('Some of the object is being moved out of frame. Consider padding the image, adjusting center, or changing the axis_order.')
    
    if return_transformed:
        transformed = transform_image(img, transform, options=transform_options)
        if return_transform:
            return transformed, transform
        else:
            return transformed
    elif return_transform:
        return transform
    return


def Point2Point(points1, points2, x0=None, bounds=None, transform='rigid', metric='symmetric_closest_point_MSE', 
    method='direct', decimation=1, transform_args={}, optimizer_args=None, verbose=True):
    """
    Point cloud-to-point cloud alignment. points2 will be aligned to points1.

    .. Note::
        If a two dimensional transform model is used on a three dimensional point cloud, ???

    Parameters
    ----------
    points1 : array_like
        Fixed points
    points2 : array_like
        Moving points 
    x0 : array_like or NoneType, optional
        Initial guess of transformation parameters. This array should be 
        consistent with the selected transformation model (``transform``). If
        None, the initial guess will be 0s for all parameters other than scaling
        and 1s for scaling parameters, by default None. Not used by all
        optimizers.
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
    x : np.ndarray
        Transformation parameters for the transformation that registers points2
        to points1.

    """
    if 'center' not in transform_args:
        transform_args['center'] = np.mean(points1,axis=0)
    if transform.lower() == 'rigid':
        nparam = 6
        transformation = lambda x : rigid(x, **transform_args)
        if x0 is None:
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
        transformation = lambda x : similarity(x, **transform_args)
        if x0 is None:
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
        transformation = lambda x : affine(x, **transform_args)
        if x0 is None:
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
        transformation = lambda x : rigid2d(x, **transform_args)
        if x0 is None:
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
        transformation = lambda x : similarity2d(x, **transform_args)
        if x0 is None:
            x0 = np.zeros(nparam)
            x0[3] = 1
        if verbose:
            print('iter.|| score||       tx |       ty |    theta |        s ')
            print('-----||------||----------|----------|----------|----------')
    elif transform.lower() == 'affine2d':
        nparam = 7
        transformation = lambda x : affine2d(x, **transform_args)
        if x0 is None:
            x0 = np.zeros(nparam)
            x0[3:5] = 1
        if verbose:
            print('iter.||      score||       tx |       ty |    theta |       s1 |       s2 |     sh01 |     sh10 ')
            print('-----||-----------||----------|----------|----------|----------|----------|----------|----------')
    else:
        raise ValueError('Invalid transform model. Must be one of: "rigid", "similarity" or "affine".')
        
    assert len(x0) == nparam, f"The provided parameters for x0 don't match the transformation model ({transform:s})."

    if 0 < decimation <= 1:
        rng = np.random.default_rng(0)
        idx1 = rng.choice(len(points1), size=int(len(points1)*decimation), replace=False)
        idx2 = rng.choice(len(points2), size=int(len(points2)*decimation), replace=False)
        ref_points = points1[idx1]
        moving_points = points2[idx2]
        moving_points = np.column_stack([moving_points, np.ones(len(moving_points))])
    else:
        raise ValueError(f'decimation must be a scalar value in the range (0,1], not {str(decimation):s}.')


    if metric.lower() == 'symmetric_closest_point_mse':
        tree1 = scipy.spatial.KDTree(ref_points) 
        # Note: can't precommute tree for the moving points
        obj = lambda p1, p2 : symmetric_closest_point_MSE(p1, p2, tree1=tree1)     
    elif metric.lower() == 'hausdorff':
        obj = hausdorff
    elif metric.lower() == 'closest_point_mse':
        tree1 = scipy.spatial.KDTree(ref_points)
        obj = lambda p1, p2 : closest_point_MSE(p1, p2, tree1=tree1)
    else:
        raise ValueError(f'Similarity metric f"{metric:s}" is not supported for Point2Point registration.')
    
    def objective(x):
        objective.k += 1
        if verbose: print('{:5d}'.format(objective.k),end='')
        T = transformation(x)
        pointsT = (T@moving_points.T).T

        f = obj(points1, pointsT[:,:-1])
        if verbose: 
            print(f'||{f:11.4f}|',end='')
            print(('|{:10.4f}'*len(x)).format(*x))
        return f
    objective.k = 0
    x,f = optimize(objective, method, x0, bounds, optimizer_args=optimizer_args)
    
    T = transformation(x)
    pointsT = (T@np.column_stack([points2, np.ones(len(points2))]).T).T
    new_points = pointsT[:,:-1]
    f = obj(points1, new_points)
    if verbose: 
        print('-----||-----------|', end='')
        for i in x:
            print('|----------',end = '')
        print('')
        print(f'final||{f:11.4f}|',end='')
        print(('|{:10.4f}'*len(x)).format(*x))

    return new_points, T

def Mesh2Mesh(M1, M2, x0=None, bounds=None, transform='rigid', metric='symmetric_closest_point_MSE', 
    method='direct', decimation=1, transform_args={}, optimizer_args=None, verbose=True):

    points1 = M1.NodeCoords
    points2 = M2.NodeCoords

    new_points2, T = Point2Point(points1, points2, x0=x0, bounds=bounds, transform=transform, metric=metric, method=method, decimation=decimation, transform_args=transform_args, optimizer_args=optimizer_args, verbose=verbose)
    Mnew = M2.copy()
    Mnew.NodeCoords = new_points2
    return Mnew, T

def Image2Image3d(img1, img2, x0=None, bounds=None, transform='rigid', metric='dice', 
        method='direct', scalefactor=1, interpolation_order=3, threshold=None, transform_args={}, optimizer_args=None, decimation=1, verbose=True):
        """
        3-dimensional image registration. img2 will be registered to img1

        Parameters
        ----------
        img1 : array_like
            3 dimensional image array of the fixed image
        img2 : array_like
            3 dimensional image array of the moving image
        x0 : array_like or NoneType, optional
            Initial guess of transformation parameters. This array should be 
            consistent with the selected transformation model (``transform``). If
            None, the initial guess will be 0s for all parameters other than scaling
            and 1s for scaling parameters, by default None. Not used by all
            optimizers.
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
            Similarity metric to compare the two point clouds, by default 'hausdorff'
        method : str, optional
            Optimization method, by default 'direct'. See 
            :func:`~mymesh.register.optimize` for details.
        scalefactor : float, optional
            Scalar factor in the range (0,1] used to reduce the size of the point set,
            by default 1.
        interpolation_order : int, optional
            Interpolation order used in image transformation (see 
            `scipy.ndimage.affine_transform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html#affine-transform>`_)
            and scaling (if used). Must be an integer in the range 0-5. Lower order
            is more efficient at the cost of quality. By default, 3. 
        threshold : NoneType, float, or tuple, optional


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
        new_img : np.ndarray
            Transformed image array of img2 registered to img1.
        x : np.ndarray
            Transformation parameters for the transformation that registers img2
            to img1.

        """
        if scalefactor != 1:
            moving_img = scipy.ndimage.zoom(img2, scalefactor, order=interpolation_order)
            fixed_img =  scipy.ndimage.zoom(img1, scalefactor, order=interpolation_order)
        else:
            moving_img = img2
            fixed_img = img1
            
        if transform.lower() == 'rigid':
            nparam = 6
            transformation = lambda x : rigid(x, **transform_args)
            if x0 is None:
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
        elif transform.lower() == 'similarity':
            nparam = 7
            transformation = lambda x : similarity(x, **transform_args)
            if x0 is None:
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
            transformation = lambda x : affine(x, **transform_args)
            if x0 is None:
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
                print('iter.||      score||       tx |       ty |       tz |    alpha |     beta |    gamma |       sx |       sy |       sz |    shxy |')
                print('-----||-----------||----------|----------|----------|----------|----------|----------|----------|----------|----------|----------')
        else:
            raise ValueError(f'Similarity metric f"{metric:s}" is not supported for Image2Image3d registration.')       
            
        point_based = False
        if metric.lower() == 'mutual_information' or metric.lower() == 'MI':
            obj = mutual_information
        elif metric.lower() == 'dice':
            obj = lambda img1, img2 : -dice(img1 > threshold, img2 > threshold)
        elif metric.lower() == 'symmetric_closest_point_mse':
            if threshold is not None:
                binarized1 = fixed_img > threshold
                binarized2 = moving_img > threshold
            else:
                binarized1 = fixed_img
                binarized2 = moving_img
            point_based = True
            points1 = np.column_stack(np.where(binarized1))
            points2 = np.column_stack(np.where(binarized2))
            tree1 = scipy.spatial.KDTree(points1) 
            # Note: can't precommute tree for the moving points
            obj = lambda p1, p2 : symmetric_closest_point_MSE(p1, p2, tree1=tree1)
        else:
            raise ValueError(f'Similarity metric f"{metric:s}" is not supported for Image2Image3d registration.')

        if point_based:
            _, T = Point2Point(points1, points2, x0=x0, bounds=bounds, transform=transform, metric=metric, method=method, transform_args=transform_args, decimation=decimation, optimizer_args=optimizer_args, verbose=verbose)
            new_img = transform_image(img2, T)
        else:
            if np.shape(img1) != np.shape(img2):
               raise Exception('Images must be the same size for image-based image-to-image registration.')
            def objective(x):
                objective.k += 1
                if verbose: 
                    print('{:5d}'.format(objective.k),end='')
                T = transformation(x)
                
                imgT = transform_image(moving_img, T)

                f = obj(fixed_img, imgT)
                if verbose: 
                    print(f'||{f:11.4f}|',end='')
                    print(('|{:10.4f}'*len(x)).format(*x))
                return f
            objective.k = 0
            x,f = optimize(objective, method, x0, bounds, optimizer_args=optimizer_args)
            
            T = transformation(x)
            new_img = transform_image(img2, T)
            f = obj(img1, new_img)
            if verbose: 
                print('-----||------|', end='')
                for i in x:
                    print('|----------',end = '')
                print('')
                print(f'final||{f:11.4f}|',end='')
                print(('|{:10.4f}'*len(x)).format(*x))

        return new_img, T

def Image2Image2d(img1, img2, x0=None, bounds=None, transform='rigid', metric='mutual_information', 
    method='direct', scalefactor=1, interpolation_order=3, threshold=None, transform_args={}, optimizer_args=None, verbose=True):
    """
    2-dimensional image registration. img2 will be registered to img1

    Parameters
    ----------
    img1 : array_like
        2 dimensional image array of the fixed image
    img2 : array_like
        2 dimensional image array of the moving image
    x0 : array_like or NoneType, optional
        Initial guess of transformation parameters. This array should be 
        consistent with the selected transformation model (``transform``). If
        None, the initial guess will be 0s for all parameters other than scaling
        and 1s for scaling parameters, by default None. Not used by all
        optimizers.
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
        Similarity metric to compare the two point clouds, by default 'hausdorff'
    method : str, optional
        Optimization method, by default 'direct'. See 
        :func:`~mymesh.register.optimize` for details.
    scalefactor : float, optional
        Scalar factor in the range (0,1] used to reduce the size of the point set,
        by default 1. 
    interpolation_order : int, optional
        Interpolation order used in image transformation (see 
        `scipy.ndimage.affine_transform <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html#affine-transform>`_)
        and scaling (if used). Must be an integer in the range 0-5. Lower order
        is more efficient at the cost of quality. By default, 3. 
    threshold : NoneType, float, or tuple, optional


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
    new_img : np.ndarray
        Transformed image array of img2 registered to img1.
    x : np.ndarray
        Transformation parameters for the transformation that registers img2
        to img1.

    """
    if transform.lower() == 'rigid':
        nparam = 3
        transformation = lambda x : rigid2d(x, image=True, center=np.array(img1.shape)/2, **transform_args)
        x0 = np.zeros(nparam)
        if bounds is None:
            pass
            # bounds = [
            #     sorted([np.min(points2[:,0]) - np.max(points1[:,0]), np.max(points2[:,0]) - np.min(points1[:,0])]),
            #     sorted([np.min(points2[:,1]) - np.max(points1[:,1]), np.max(points2[:,1]) - np.min(points1[:,1])]),
            #     sorted([np.min(points2[:,2]) - np.max(points1[:,2]), np.max(points2[:,2]) - np.min(points1[:,2])]),
            #     (-np.pi, np.pi)
            # ]
            
        if verbose:
            print('iter.|| score||       tx |       ty |    theta ')
            print('-----||------||----------|----------|----------')
            
    elif transform.lower() == 'similarity':
        pass
        # nparam = 7
        # transformation = lambda x : similarity(x, **transform_args)
        # x0 = np.zeros(nparam)
        # x0[6] = 1
    elif transform.lower() == 'affine':
        pass
        # nparam = 15
        # transformation = lambda x : affine(x, **transform_args)
        # x0 = np.zeros(nparam)
        # x0[6:9] = 1
    else:
        raise ValueError('Invalid transform model. Must be one of: "rigid", "similarity" or "affine".')
        
    assert len(x0) == nparam, f"The provided parameters for x0 don't match the transformation model ({transform:s})."
    
    if metric.lower() == 'mutual_information' or metric.lower() == 'MI':
        obj = mutual_information
    elif metric.lower() == 'dice':
        obj = dice
    else:
        raise ValueError(f'Similarity metric f"{metric:s}" is not supported for Image2Image2d registration.')

    if scalefactor != 1:
        moving_img = scipy.ndimage.zoom(img2, scalefactor, order=interpolation_order)
        fixed_img =  scipy.ndimage.zoom(img1, scalefactor, order=interpolation_order)
    else:
        moving_img = img2
        fixed_img = img1

    
    def objective(x):
        objective.k += 1
        if verbose: 
            print('{:5d}'.format(objective.k),end='')
        T = np.linalg.inv(transformation(x))
        imgT = scipy.ndimage.affine_transform(moving_img, T, order=interpolation_order)

        f = obj(fixed_img, imgT)
        if verbose: 
            print(f'||{f:.4f}|',end='')
            print(('|{:10.4f}'*len(x)).format(*x))
        return f
    objective.k = 0
    x,f = optimize(objective, method, x0, bounds, optimizer_args=optimizer_args)
    
    T = np.linalg.inv(transformation(x))
    new_img = scipy.ndimage.affine_transform(img2, T, order=interpolation_order)
    f = obj(img1, new_img)
    if verbose: 
        print('-----||------|', end='')
        for i in x:
            print('|----------',end = '')
        print('')
        print(f'final||{f:.4f}|',end='')
        print(('|{:10.4f}'*len(x)).format(*x))

    return new_img, T

def Mesh2Image3d():
    return

def Image2Mesh3d(img, M, h=1, threshold=None, scalefactor=1, decimation=1, center_mesh=False, interpolation_order=3, x0=None, bounds=None, transform='rigid', metric='symmetric_closest_point_MSE', method='direct', transform_args={}, optimizer_args=None, verbose=True):

    if scalefactor != 1:
        img =  scipy.ndimage.zoom(img, scalefactor, order=interpolation_order)
    mesh_points = np.copy(M.NodeCoords)
    if center_mesh:
        mesh_points = np.copy(M.NodeCoords)
        mesh_points[:,0] += -(np.max(mesh_points[:,0]) + np.min(mesh_points[:,0]))/2 + np.shape(img)[2]*h/scalefactor/2
        mesh_points[:,1] += -(np.max(mesh_points[:,1]) + np.min(mesh_points[:,1]))/2 + np.shape(img)[1]*h/scalefactor/2
        mesh_points[:,2] += -(np.max(mesh_points[:,2]) + np.min(mesh_points[:,2]))/2 + np.shape(img)[0]*h/scalefactor/2
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
    
    img_points = np.fliplr(np.column_stack(np.where(binarized)) * h/scalefactor) # flipping to make image coordinate system (z,y,x) match mesh coordinate system

    new_points, T = Point2Point(mesh_points, img_points, x0=x0, bounds=bounds, transform=transform, metric=metric, method=method, transform_args=transform_args, optimizer_args=optimizer_args, verbose=verbose)

    P = np.array([[0., 0., 1., 0.],
       [0., 1., 0., 0.],
       [1., 0., 0., 0.],
       [0., 0., 0., 1.]])
    T2 = P@T@P  # Reorder transformations to the image coordinate system
    T2[:3,3] *= scalefactor/h # Scale translations back to units of voxels

    new_image = transform_image(img, T)

    return new_image, T

### Transformations
def T2d(t0,t1,h=1):
    """
    T Generates a translation matrix

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
    t = np.array([[1,0,t0/h],
                [0,1,t1/h],
                [0,0,1]])
    return t
    
def R2d(theta,center):
    """
    R Generates a rotation matrix for a rotation about a point
    
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
    """
    S Generates a scaling matrix

    Parameters
    ----------
    s0 : float
        Scale factor in the 0 axis (spatial y)
    s1 : float
        Scale factor in the 1 axis (spatial x)

    Returns
    -------
    s : np.ndarray
        4x4 scaling matrix
    """    
    s = np.array([[1/s0,0,0],
                  [0,1/s1,0],
                  [0,0,1]])

    T1 = T2d(*reference)
    T2 = T2d(*-np.asarray(reference))
    s = np.linalg.multi_dot([T1,s,T2]) 

    return s

def Sh2d(sh01,sh10,reference=np.array([0,0])):
    """
    Sh Generates a shearing matrix
    ref:https://www.mathworks.com/help/images/matrix-representation-of-geometric-transformations.html
    Parameters
    ----------
    sx : float
        S in the x direction
    sy : float
        S in the y direction
    sz : float
        S in the z direction

    Returns
    -------
    s : np.ndarray
        4x4 scaling matrix
    """
    
    sh = np.array([[1,      sh10,  0],
                    [sh01,      1, 0],
                    [0,         0, 1]])
    T1 = T2d(*reference)
    T2 = T2d(*-np.asarray(reference))
    sh = np.linalg.multi_dot([T1,sh,T2]) 
    return sh

def T3d(t0,t1,t2,h=1):
    """
    T Generates a translation matrix

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
    
def R3d(alpha,beta,gamma,center,rotation_order=[0,1,2],rotation_mode='cartesian'):
    """
    R Generates a rotation matrix for a rotation about a point

    Parameters
    ----------
    alpha : float
        Rotation about the x, in radians
    beta : float
        Rotation about the y, in radians
    gamma : float
        Rotation about the z, in radians
    center : list or np.ndarray
        Reference point for the rotation

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
    """
    S Generates a scaling matrix

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
    """
    Sh Generates a shearing matrix
    ref:https://www.mathworks.com/help/images/matrix-representation-of-geometric-transformations.html
    Parameters
    ----------
    sx : float
        S in the x direction
    sy : float
        S in the y direction
    sz : float
        S in the z direction

    Returns
    -------
    s : np.ndarray
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
        [t2,t1,t0,gamma,beta,alpha] = x
    else:
        [t0,t1,t2,alpha,beta,gamma] = x
    
    t = T3d(t0,t1,t2)
    r = R3d(alpha,beta,gamma,np.asarray(center),rotation_order=rotation_order,rotation_mode=rotation_mode)

    A = t@r
    
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
        Transformation matrix (either 3x3 or 4x4).

    Returns
    -------
    new_points : np.ndarray
        Transformed point coordinates

    """    
    if np.shape(T) == (4,4):
        # Affine matrix
        pointsT = (T@np.column_stack([points, np.ones(len(points))]).T).T
        new_points = pointsT[:,:-1]
    elif np.shape(T) == (3,3):
        # non-affine transfomration matirx
        new_points = (T@np.asarray(points).T).T
    else:
        raise Exception("I haven't set up handling of any other cases yet.")
        
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
        that may be used are `mode` to allow wrapping ('wrap') or mirroring 
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
    TP = np.sum(u & v)
    FP = np.sum(u & np.logical_not(v))
    FN = np.sum(np.logical_not(u) & v)
    D = 2*TP/(2*TP + FP + FN)
    return D

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

    d, i, j = scipy.spatial.distance.directed_hausdorff(points1, points2)

    return d

def closest_point_MSE(points1, points2, tree1=None):

    if tree1 is None:
        tree1 = scipy.spatial.KDTree(points1)
    distances, paired_indices = tree1.query(points2)
    MSE = np.sum(distances**2)/len(points2)

    return MSE

def symmetric_closest_point_MSE(points1, points2, tree1=None, tree2=None):
    
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

    # default optimizer settings for selected optimizers
    if optimizer_args is None:
        if method.lower() == 'direct':
            optimizer_args = dict(locally_biased=False)
        else:
            optimizer_args = {}

    if method.lower() == 'direct' or method.lower() == 'directl':
        if bounds is None:
            raise ValueError('bounds are required for the "direct" optimizer.')
        if method.lower() == 'directl':
            optimizer_args['locally_biased'] = True
        res = scipy.optimize.direct(objective, bounds, **optimizer_args)
    elif method.lower() in ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                'l-bfgs-b', 'tnc', 'cobyla', 'cobyqa', 'slsqp', 'trust-constr', 
                'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']:
        if x0 is None:
            raise ValueError(f'x0 is required for the {method:s} optimizer')
        if bounds is not None and 'bounds' not in optimizer_args:
            optimizer_args['bounds'] = bounds
        res = scipy.optimize.minimize(objective, x0, method=method, **optimizer_args)
    else:
        raise ValueError(f'Method "{method:s}" is not supported.')
    if not res['success']:
        message = res["message"]
        warnings.warn(f'Optimization was not successful. \nOptimizer exited with message "{message:s}".', category=RuntimeWarning)
    
    x = res['x']
    f = res['fun']
    return x, f
    
