# -*- coding: utf-8 -*-
# Created on Sun Feb 25 14:07:16 2024
# @author: toj
"""
Mesh visualization and plotting

:mod:`mymesh.visualize` is in the early stages of development. For more 
full-featured mesh visualization, a mesh (``M``) can be converted to a PyVista
mesh:

.. code-block::

    import pyvista as pv
    pvmesh = pv.wrap(M.mymesh2meshio())

"""

#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io, re, warnings

from . import converter, utils, mesh

def View(M, interactive=True, bgcolor=None,
    color=None, face_alpha=1, color_convert=None, 
    clim=None, theme='default', scalar_preference='nodes',
    view=None, scalars=None,
    show_edges=False, show_faces=True, line_width=1, line_color=None,
    show_colorbar=None, colorbar_args={}, return_image=False, hide=False,  
    shading='flat'):
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
    view : _type_, optional
        _description_, by default None
    scalars : _type_, optional
        _description_, by default None
    show_edges : bool, optional
        _description_, by default False
    show_faces : bool, optional
        _description_, by default True
    line_width : int, optional
        _description_, by default 1
    line_color : _type_, optional
        _description_, by default None
    show_colorbar : _type_, optional
        _description_, by default None
    colorbar_args : dict, optional
        _description_, by default {}
    return_image : bool, optional
        _description_, by default False
    hide : bool, optional
        _description_, by default False
    shading : str, optional
        _description_, by default 'flat'
        Options are 'flat', 'smooth', None

    Returns
    -------
    _type_
        _description_

    """    
    ###
    # Shading: 'flat', 'smooth', None
    # viewmode: 'arcball', 'fly', 'turntable'
    # color_convert: "deuteranomaly", "protanomaly", "tritanomaly"

    ###

    try:
        import vispy
        from vispy import app, scene
        from vispy.io import read_mesh, load_data_file
        from vispy.scene.visuals import Mesh as vispymesh
        from vispy.scene.visuals import Line
        from vispy.scene import transforms
        from vispy.visuals.filters import ShadingFilter, WireframeFilter, FacePickingFilter
    except:
        raise ImportError('vispy is needed for visualization. Install with: pip install vispy')

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
    canvas = scene.SceneCanvas(keys='interactive', bgcolor=ParseColor(bgcolor), title='MyMesh Viewer',show=interactive,resizable=False)

    # Set view mode
    viewmode='arcball'
    canvasview = canvas.central_widget.add_view()
    canvasview.camera = viewmode
    
    # Set up mesh
    vertices = np.asarray(M.NodeCoords)# - np.mean(M.NodeCoords,axis=0) # Centering mesh in window
    if M.Type == 'vol':
        SurfConn, ids = converter.solid2surface(*M, return_SurfElem=True)
        _,faces, inv = converter.surf2tris(M.NodeCoords, SurfConn, return_inv=True)
        # TODO: Need to properly use ids and tri_ids to transfer element data to the surface
    else:
        _,faces, inv = converter.surf2tris(M.NodeCoords, M.SurfConn, return_inv=True)

    # Process scalars
    if scalars is not None: 
        if show_colorbar is None:
            show_colorbar = True
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
        show_colorbar = False
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
        vsmesh.transform.rotate(120, (1, 0, 0))
        vsmesh.transform.rotate(-30, (0, 0, 1))
    elif view == 'xy':
        vsmesh.transform.rotate(90, (1, 0, 0))
    
    if wireframe_enabled:
        vsmesh.set_gl_state(polygon_offset_fill=True,
                               polygon_offset=(1, 1), depth_test=True)
        wireframe.transform = vsmesh.transform
        # wireframe.set_gl_state(depth_test=True)
        canvasview.add(wireframe)
    if not wireframe_only:
        canvasview.add(vsmesh)
    
    
    # Set shading/lighting
    shading_filter = ShadingFilter()
    vsmesh.attach(shading_filter)
    shading_filter.shading = shading

    def attach_headlight(canvasview):
        light_dir = (0, 1, 0, 0)
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
    canvasview.camera.set_range((mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2]))
    
    # Render
    if ipython and interactive:
        return canvas
    elif interactive:
        app.run()

    if return_image:
        try:
            from PIL import Image
        except:
            raise ImportError('PIL needed. Install with: pip install pillow')

        img_data = canvas.render().copy(order='C')
        image = Image.fromarray(img_data)

    if ipython and not hide:
        IPython.display.display(image)

    if return_image:
        return img_data

def FaceColor(NFaces, color, face_alpha, scalars=None, color_convert=None):
    
    if scalars is None:
        if type(color) is str:
            color = ParseColor(color, face_alpha)
            face_colors = np.tile(color,(NFaces,1))
        elif isinstance(color, [list, tuple, np.ndarray]):
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
        bgcolor = '#2E3440'
        if scalars is None:
            color = 'white'
        else:
            color = 'cividis'
        linecolor = 'black'
    return color, bgcolor, linecolor

def set_vispy_backend(preference='PyQt6'):
    try:
        import vispy
    except:
        raise ImportError('vispy is needed for visualization. Install with: pip install vispy')

    options = ['PyGlet', 'PyQt6', 'PyQt5', 'PyQt4', 'PySide', 'PySide2', 
                'PySide6', 'Glfw', 'SDL2', 'osmesa', 'jupyter_rfb']

    if preference in options:
        options.remove(preference)
        options.insert(0, preference)

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
        raise ImportError('A valid vispy backend must be installed. PyQt6 is recommended: pip install pyqt6')

    return chosen
    

    