"""
Mesh visualization and plotting

:mod:`mymesh.visualize` is in the early stages of development. For more 
full-featured mesh visualization, a mesh (``M``) can be converted to a PyVista
mesh:

.. code-block::

    import pyvista as pv
    pvmesh = pv.wrap(M.mymesh2meshio())

Created on Sun Feb 25 14:07:16 2024

@author: toj
"""

#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io, re, warnings

from . import converter, utils

def View(M, interactive=True,  
    shading='flat', bgcolor=None,
    color=None, face_alpha=1, color_convert=None, 
    clim=None, theme='default', scalar_preference='nodes',
    view=None, scalars=None,
    show_edges=False, show_faces=True, line_width=1,
    show_colorbar=None, colorbar_args={}, return_image=False, hide=False):
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
    if ipython and interactive:
        chosen = set_vispy_backend('jupyter_rfb')
        if chosen != 'jupyter_rfb':
            warnings.warn(f'jupyter_rfb is needed for interactive visualization in IPython. \nInstall with: pip install jupyter_rfb. \nFalling back to {chosen:s} backend.')
    else:
        chosen = set_vispy_backend()


    # Set Theme (Need to handle this better)
    if bgcolor is None:
        _, bgcolor = GetTheme(theme, scalars)
    if color is None:
        color, _ = GetTheme(theme, scalars)

    # Set up mesh
    vertices = np.asarray(M.NodeCoords)# - np.mean(M.NodeCoords,axis=0) # Centering mesh in window
    _,faces = converter.surf2tris(M.NodeCoords, M.SurfConn)

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

        if len(scalars) == len(faces):
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

    # Create canvas
    canvas = scene.SceneCanvas(keys='interactive', bgcolor=ParseColor(bgcolor), title='MyMesh Viewer',show=interactive,resizable=False)

    # Set view mode
    viewmode='arcball'
    canvasview = canvas.central_widget.add_view()
    canvasview.camera = viewmode
    # Set initial position
    # canvasview.camera.transform = transforms.MatrixTransform()
    # print(canvasview.camera.transform)
    # canvasview.camera.transform.rotate(90, (0, 1, 1))
    canvasview.camera.depth_value = 1e3

    vsmesh.transform = transforms.MatrixTransform()
    if view is None:
        vsmesh.transform.rotate(120, (1, 0, 0))
        vsmesh.transform.rotate(-30, (0, 0, 1))
    elif view == 'xy':
        vsmesh.transform.rotate(90, (1, 0, 0))
    canvasview.add(vsmesh)

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
    wireframe_filter = WireframeFilter(enabled=wireframe_enabled, wireframe_only=wireframe_only, faces_only=faces_only,width=line_width)
    vsmesh.attach(wireframe_filter)

    # Add colorbar
    if show_colorbar:
        pass
        # viewbox = vispy.scene.ViewBox(camera=vispy.scene.cameras.BaseCamera())
        # if 'cmap' not in colorbar_args.keys():
        #     colorbar_args['cmap'] = color
        # if 'orientation' not in colorbar_args.keys():
        #     colorbar_args['orientation'] = 'right'

        # colorbar = vispy.scene.ColorBarWidget(**colorbar_args)
        # colorbar.transform = transforms.MatrixTransform()
        # colorbar.transform.rotate(90, (1, 0, 0))
        # colorbar.transform.rotate(-45, (0, 0, 1))

        # viewbox.add(colorbar)
        # canvasview.add(viewbox)
    
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
    return color, bgcolor

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
    

    