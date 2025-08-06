# -*- coding: utf-8 -*-
# Created on Sun Feb 25 14:07:16 2024
# @author: toj
"""
Mesh visualization and plotting

:mod:`mymesh.visualize` is still experimental and may not work as expected
on all systems or in all development environments. For more stable and
full-featured mesh visualization, a mesh (``M``) can be converted to a PyVista
mesh for visualization:

.. code-block::

    import pyvista as pv
    pv_mesh = pv.wrap(M.mymesh2meshio())
    pv_mesh.plot()

Visualization
=============
.. autosummary::
    :toctree: submodules/

    View
    Subplot

Visualization Utilities
=======================
.. autosummary::
    :toctree: submodules/

    FaceColor
    ParseColor
    GetTheme
    set_vispy_backend

"""

#%%
import numpy as np
import io, re, warnings
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    warnings.warn('Matplotlib is used for mesh visualization. If needed, install with `pip install matplotlib`.')

from . import converter, utils, mesh

def View(M, interactive=True, bgcolor=None,
    color=None, face_alpha=1, color_convert=None, 
    clim=None, theme='default', scalar_preference='nodes',
    view='iso', scalars=None,
    show_edges=False, show_faces=True, show_points=False, point_size=2,
    line_width=1, line_color=None, 
    return_image=False, hide=False, shading='flat'):
    """
    Visualize a mesh.
    View uses vispy for visualization.

    Parameters
    ----------
    M : mymesh.mesh
        Mesh object to be viewed
    interactive : bool, optional
        View mesh in an interactive window, allowing for panning, rotation, etc.
        By default True.
    bgcolor : _type_, optional
        _description_, by default None
    color : str, array_like, or NoneType, optional
        Color of the surface, specified as a named color (see :func:`ParseColor`,
        hex color code, or RGBA list/tuple/array). If None, the color will
        be chosen based on the `theme`, by default None.
    face_alpha : Float, optional
        Alpha value controlling face opacity - 0 = transparent, 1 = opaque, by 
        default 1.
    color_convert : str or NoneType, optional
        Convert the colors to simulate different types of color vision. Available 
        options are "deuteranomaly", "protanomaly", "tritanomaly", or "grayscale",
        by default None. Color conversion is performed using 
        `Colorspacious <https://github.com/njsmith/colorspacious>`_
    clim : tuple or NoneType, optional
        Color range bounds if displaying scalar data, by default None.
    theme : str, optional
        Select a theme (see :func:`GetTheme`), by default 'default'.
    scalar_preference : str, optional
        In the even that `scalars` is specified by a string that references
        an entry in both M.NodeData and M.ElemData, this will decide which 
        is chosen. Specified as either 'nodes' or 'elements', by default 'nodes'.
    view : str, optional
        Named orientation for viewing the mesh, by default 'iso'

        Options are:
        
        - "iso" or "isometric" : isometric view
        - "dimetric" : dimetric view
        - "trimetric" : trimetric view
        - "xy" : View of the x-y plane with +x to the right and +y up
        - "xz" : View of the x-z plane with +x to the right and +z ups

    scalars : str, array_like, or None, optional
        Scalar values to color the mesh by. If specified by a string, the string
        should be an entry in either M.NodeData or M.ElemData. If the string is 
        present in both, `scalar_preference` will be used determine whether node
        or element data will be displayed (node by default). If specified by an
        array_like (numpy ndarray or list), the length of the array_like most be
        equal to either the number of nodes or the number of elements in the 
        mesh. If None, the mesh will be given a solid color following `color`.  
        By default None.
    show_edges : bool, optional
        Show lines denoting element edges. For a wireframe view, set 
        `show_edges=True`, `show_faces=False`, by default False.
    show_faces : bool, optional
        Show element faces in the mesh. For a wireframe view, set 
        `show_edges=True`, `show_faces=False`, by default True
    line_width : int, optional
        Width of lines shown if show_edges=True, by default 1
    line_color : None, optional
        Color of edges shown if show_edges=True, by default None.
        If None, color will be selected based on theme.
    point_size : float, optional
        Size of points, if show_points=True, by default 2.
    return_image : bool, optional
        If true, image array of the plot will be returned, by default False
    hide : bool, optional
        If true, plot will not be shown, by default False
    shading : str, optional
        Shading mode, by default 'flat'
        Options are 'flat', 'smooth', None

    Returns
    -------
    img_data : np.ndarray, optional
        Image array of the plot. Only returned if return_image=True.

    """    
    try:
        import vispy
        from vispy import app, scene
        from vispy.io import read_mesh, load_data_file
        from vispy.scene.visuals import Mesh as vispymesh
        from vispy.scene.visuals import Line, Markers
        from vispy.scene import transforms
        from vispy.visuals.filters import ShadingFilter, WireframeFilter, FacePickingFilter
    except:
        raise ImportError('vispy is needed for visualization. Install with: pip install vispy')
    try:
        from PIL import Image
    except:
        raise ImportError('PIL needed. Install with: pip install pillow')
    # determine execution environment
    try:
        import IPython
        ip = IPython.get_ipython()
        if ip is None:
            # IPython is installed, but not active
            ipython = False
        else:
            ipython = True
    except:
        ipython = False

    # set backend
    if hide:
        interactive = False
    # if ipython and interactive:
    #     chosen = set_vispy_backend('jupyter_rfb')
    #     if chosen != 'jupyter_rfb':
    #         warnings.warn(f'jupyter_rfb is needed for interactive visualization in IPython. \nInstall with: pip install jupyter_rfb. \nFalling back to {chosen:s} backend.')
    # else:
    #     chosen = set_vispy_backend()
    chosen = set_vispy_backend('jupyter_rfb')
    
    # Set Theme (Need to handle this better)
    theme = GetTheme(theme, scalars)
    if color is None:
        color = theme[0]
    if bgcolor is None:
        bgcolor = theme[1]
    if line_color is None:
        line_color = theme[2]
    
    # Create canvas
    canvas = scene.SceneCanvas(keys='interactive', bgcolor=ParseColor(bgcolor), title='MyMesh Viewer',show=interactive)

    # Set view mode
    viewmode='arcball'
    canvasview = canvas.central_widget.add_view()
    canvasview.camera = viewmode
    
    # Set up mesh
    vertices = np.asarray(M.NodeCoords)# - np.mean(M.NodeCoords,axis=0) # Centering mesh in window
    if vertices.shape[1] == 2:
        vertices = np.hstack([vertices, np.zeros((len(vertices),1))])
    if M.Type == 'vol':
        SurfConn, ids = converter.solid2surface(*M, return_SurfElem=True)
        _,faces, inv = converter.surf2tris(M.NodeCoords, SurfConn, return_inv=True)
        # TODO: Need to properly use ids and tri_ids to transfer element data to the surface
    else:
        _,faces, inv = converter.surf2tris(M.NodeCoords, M.SurfConn, return_inv=True)

    # Process scalars
    if scalars is not None: 
        if type(scalars) is str:
            if scalar_preference.lower() == 'nodes':
                if scalars in M.NodeData.keys():
                    scalars = M.NodeData[scalars]
                elif scalars in M.ElemData.keys():
                    scalars = M.ElemData[scalars]
                else:
                    raise ValueError(f'Scalar {scalars:s} not present in mesh.')
            elif scalar_preference.lower() == 'elements':
                if scalars in M.ElemData.keys():
                    scalars = M.ElemData[scalars]
                elif scalars in M.NodeData.keys():
                    scalars = M.NodeData[scalars]
                else:
                    raise ValueError(f'Scalar {scalars:s} not present in mesh.')
            else:
                raise ValueError('scalar_preference must be "nodes" or "elements"')

        if clim is None:
            clim = (np.nanmin(scalars), np.nanmax(scalars))
        # TODO: Might want a clipping option
        color_scalars = matplotlib.colors.Normalize(clim[0], clim[1], clip=False)(scalars)

        if len(scalars) == M.NElem:
            if M.Type == 'vol':
                scalars = np.asarray(scalars)[ids[inv]]
                color_scalars = color_scalars[ids[inv]]
            else:
                scalars = np.asarray(scalars)[inv]
                color_scalars = color_scalars[inv]
            face_colors = FaceColor(len(faces), color, face_alpha, scalars=color_scalars, color_convert=color_convert)

            vertex_colors = None
        elif len(scalars) == len(vertices):
            vertex_colors = FaceColor(len(vertices), color, face_alpha, scalars=color_scalars, color_convert=color_convert)
            face_colors = None
        
    else:
        face_colors = FaceColor(len(faces), color, face_alpha, color_convert=color_convert)
        vertex_colors = None

    vsmesh = vispymesh(np.asarray(vertices), np.asarray(faces), face_colors=face_colors, vertex_colors=vertex_colors)
    
    # Set edges
    if show_edges and show_faces:
        wireframe_enabled = True
        wireframe_only = False
        faces_only = False
        enabled = True
    elif show_edges and not show_faces:
        wireframe_enabled = True
        wireframe_only = True
        faces_only = False
    elif not show_edges and show_faces:
        wireframe_enabled = False
        wireframe_only = False
        faces_only = True
    else:
        wireframe_enabled = True
        wireframe_only = True
        faces_only = False
        line_width = 0
    

    if wireframe_enabled:
        wireframe = Line(pos=M.NodeCoords, connect=M.Surface.Edges, width=line_width, color=ParseColor(line_color))

    # Set initial position
    canvasview.camera.depth_value = 1e3

    vsmesh.transform = transforms.MatrixTransform()
    if view is None:
        pass
        # vsmesh.transform.rotate(120, (1, 0, 0))
        # vsmesh.transform.rotate(-30, (0, 0, 1))
    elif view == 'iso' or view == 'isometric':
        vsmesh.transform.rotate(45, (0, 0, 1))     
        vsmesh.transform.rotate(35.264, (1, 0, 0))
    elif view == 'dimetric':
        vsmesh.transform.rotate(45, (0, 0, 1))      # 45 degrees around X-axis
        vsmesh.transform.rotate(20.705, (1, 0, 0))
    elif view == 'trimetric':
        vsmesh.transform.rotate(60, (0, 0, 1))      # 60 degrees around X-axis
        vsmesh.transform.rotate(30, (1, 0, 0))
    elif view == 'xy':
        vsmesh.transform.rotate(90, (1, 0, 0))
    elif view == '-xy':
        vsmesh.transform.rotate(90, (1, 0, 0))
        vsmesh.transform.rotate(180, (0, 0, 1))
    elif view == 'xz':
        pass
    elif view == '-xz':
        vsmesh.transform.rotate(180, (0, 0, 1))
    elif view == 'x-z':
        vsmesh.transform.rotate(180, (1, 0, 0))
    elif view == '-x-z':
        vsmesh.transform.rotate(180, (0, 1, 0))
    elif view == 'yz':
        vsmesh.transform.rotate(-90, (0, 0, 1))
    elif view == '-yz':
        vsmesh.transform.rotate(90, (0, 0, 1))
    
    if wireframe_enabled:
        vsmesh.set_gl_state(polygon_offset_fill=True,
                               polygon_offset=(1, 1), depth_test=True)
        wireframe.transform = vsmesh.transform
        # wireframe.set_gl_state(depth_test=True)
        canvasview.add(wireframe)
    if not wireframe_only:
        canvasview.add(vsmesh)
    if show_points:
        if vertex_colors is None:
            vertex_colors = 'black'
        points = Markers(pos=vertices, edge_width=0, symbol='o', face_color=vertex_colors, size=point_size, edge_color=vertex_colors)
        points.transform = vsmesh.transform
        canvasview.add(points)
    
    
    # Set shading/lighting
    shading_filter = ShadingFilter()
    vsmesh.attach(shading_filter)
    shading_filter.shading = shading

    def attach_headlight(canvasview):
        light_dir = (0, 1, 0, 0.001)
        shading_filter.light_dir = light_dir[:3]
        initial_light_dir = canvasview.camera.transform.imap(light_dir)

        @canvasview.scene.transform.changed.connect
        def on_transform_change(event):
            transform = canvasview.camera.transform
            shading_filter.light_dir = transform.map(initial_light_dir)[:3]

    attach_headlight(canvasview)
    
    AABB = utils.AABB(vertices)
    aabb =  np.matmul(np.linalg.inv(vsmesh.transform.matrix), np.hstack([AABB, np.zeros((8, 1))]).T).T
    mins = np.min(aabb, axis=0)
    maxs = np.max(aabb, axis=0)
    diffs = maxs-mins
    canvasview.camera.set_range((mins[0]+0.85*diffs[0], maxs[0]-0.85*diffs[0]), 
                                (mins[1]+0.85*diffs[1], maxs[1]-0.85*diffs[1]), 
                                (mins[2]+0.85*diffs[2], maxs[2]-0.85*diffs[2]))
    
    # Render
    if chosen == 'jupyter_rfb' and interactive:
        return canvas
    elif interactive:
        app.run()

    if chosen == 'jupyter_rfb':
        img_data = canvas._backend.snapshot().data
    else:
        img_data = canvas.render().copy(order='C')
    image = Image.fromarray(img_data)

    if ipython and not hide:
        IPython.display.display(image)

    if return_image:
        return img_data

def Subplot(meshes, shape, show=True, return_fig=False, figsize=None, titles=None, **kwargs):

    # Plotting:
    fig, axes = plt.subplots(shape[0], shape[1], figsize=figsize)
    if titles is None:
        titles = ['' for i in range(len(meshes))]
    for m, ax, title in zip(meshes, axes.ravel(), titles):
        subfig, subax = m.plot(show=False,return_fig=True,**kwargs)
        ax.imshow(subax.get_images()[0].get_array())
        ax.set_title(title)
        ax.set_axis_off()
        plt.close(subfig)

    if show:
        plt.show()
    if return_fig:
        return fig, ax

def FaceColor(NFaces, color, face_alpha, scalars=None, color_convert=None):
    
    if scalars is None:
        if type(color) is str:
            color = ParseColor(color, face_alpha)
            face_colors = np.tile(color,(NFaces,1))
        elif isinstance(color, (list, tuple, np.ndarray)):
            assert len(np.shape(color)) == 1 and np.shape(color)[0] == 4
            face_colors = np.tile(color,(NFaces,1))
    else:
        if type(color) is str:
            color = ParseColor(color, face_alpha)
            face_colors = color(scalars, face_alpha)

    if color_convert is not None:
        try:
            from colorspacious import cspace_convert
        except:
            raise ImportError("The colorspacious package is required for color conversion. To install: pip install colorspacious.")
        if type(color_convert) is tuple or type(color_convert) is list:
            severity = color_convert[1]
            color_convert = color_convert[0]
        else:
            severity = 50
        if type(color_convert) is str:
            if color_convert in ['deuteranomaly', 'protanomaly', 'tritanomaly']:
                cvd_space = {"name"    : "sRGB1+CVD",
                            "cvd_type" : "deuteranomaly",
                            "severity" : severity}
                face_colors = cspace_convert(face_colors[:,:3], cvd_space, "sRGB1")
            elif color_convert in ['grayscale','greyscale']:
                face_colors_JCh = cspace_convert(face_colors[:,:3], "sRGB1", "JCh")
                face_colors_JCh[:, 1] = 0
                face_colors[:,:3] = cspace_convert(face_colors_JCh, "JCh", "sRGB1")

    return face_colors

def ParseColor(color, alpha=1, ):

    if type(color) is str:
        matplotlib_colors = list(matplotlib.colors.BASE_COLORS.keys())+\
                            list(matplotlib.colors.CSS4_COLORS.keys())+\
                            list(matplotlib.colors.TABLEAU_COLORS.keys())+\
                            list(matplotlib.colors.XKCD_COLORS.keys())
        # Single color
        if color in matplotlib_colors:
            # Check matplotlib colors
            color = matplotlib.colors.to_rgba(color, alpha)
        
        elif re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color):
            # Check if hex color code
            hexstr = color[1:]
            if len(hexstr) == 3:
                hexstr = ''.join([digit*2 for digit in hexstr])
            color = tuple(int(hexstr[i]+hexstr[i+1],16)/255 for i in range(0, 6, 2)) + (alpha,)

        # Colormaps
        elif color in plt.colormaps():
            color = plt.get_cmap(color)
    
    elif isinstance(color, (list, tuple, np.ndarray)):
        # Single color
        if len(np.shape(color)) == 1 and (np.shape(color)[0] in (3,4)):
            color = tuple(color)
            if len(color) == 3:
                color += (alpha,)
        
        # Colormaps
        else:
            pass

    return color

def GetTheme(theme, scalars):
    if theme == 'default':
        bgcolor = 'white'
        if scalars is None:
            color = 'white'
        else:
            color = 'coolwarm'
        linecolor = 'black'
    
    if theme == 'nord':
        bgcolor = '#2E3440'
        if scalars is None:
            color = 'white'
        else:
            color = 'cividis'
        linecolor = 'black'
    return color, bgcolor, linecolor

def set_vispy_backend(preference='glfw'):
    """
    Set the backend for VisPy. Can only be set once.

    Parameters
    ----------
    preference : str, optional
        Preferred vispy backend, by default 'PyGlet'. If not available, an 
        alternative will be attempted.

    Returns
    -------
    chosen : str
        The name of the backend that was selected

    """    
    try:
        import vispy
    except:
        raise ImportError('vispy is needed for visualization. Install with: pip install vispy')

    options = ['pyqt6', 'glfw', 'pyside6', 'pyqt5', 'pyqt4', 'pyglet', 'pyside', 'pyside2', 'sdl2', 'osmesa', 'jupyter_rfb']

    preference = preference.lower()
    if preference in options:
        options.remove(preference)
        options.insert(0, preference)
    else:
        warnings.warn(f'VisPy backend must be one of {str(options):s}. {preference:s} is not supported.')

    success = False
    chosen = None
    for backend in options:
        try:
            vispy.use(app=backend)
            success = True
            chosen = backend
            break
        except Exception as e:
            if 'Can only select a backend once, already using' in str(e):
                success = True
                chosen = str(e).split('[')[1][1:-3]
                break

    if not success:
        raise ImportError('A valid vispy backend must be installed. Glfw is recommended: pip install glfw')

    return chosen
    

    