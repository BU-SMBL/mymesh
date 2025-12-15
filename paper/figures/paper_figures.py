#%%
import mymesh
from mymesh import *
import numpy as np
import pyvista as pv
#%% Image -> Mesh
threshold = 100
bunny_img = mymesh.demo_image('bunny')[::-1,:,:] # flip zstack for visualization purposes
voxelsize = (0.337891, 0.337891, 0.5) # mm
#%%
bunny_voxel = image.VoxelMesh(bunny_img, voxelsize, scalefactor=0.75)
half_voxel = bunny_voxel.Clip(normal=[0,1,0])
bunny_voxel2 = image.VoxelMesh(bunny_img, voxelsize, threshold, scalefactor=0.2)
bunny_surf = image.SurfaceMesh(bunny_img, voxelsize, threshold, scalefactor=0.5)
bunny_tet = image.TetMesh(bunny_img, voxelsize, threshold, scalefactor=0.5, interpolation='linear')
half_tet = bunny_tet.Clip(pt=[0,88,0], normal=[0,1,0])

#%%
plotter = pv.Plotter()
plotter.add_mesh(half_voxel.to_pyvista(), cmap='gray', show_scalar_bar=False)
plotter.view_vector([0,-1,0])
camera1 = plotter.camera
plotter.show()
plotter.screenshot('bunny_img.png', scale=2)

plotter = pv.Plotter()
plotter.add_mesh(bunny_surf.to_pyvista(), color='white', opacity=1)
plotter.view_vector([-0.4,-1.0,-0.2])
camera1 = plotter.camera
plotter.show()
plotter.screenshot('bunny_surf.png', scale=2)

plotter = pv.Plotter()
plotter.add_mesh(half_tet.to_pyvista(), color='white', opacity=1)
plotter.view_vector([-0.4,-1.0,-0.2])
camera1 = plotter.camera.copy()
plotter.show()
plotter.screenshot('bunny_tet.png', scale=2)

plotter = pv.Plotter()
plotter.add_mesh(half_tet.to_pyvista(), color='white', show_edges=True, line_width=4)
plotter.view_vector([-0.4,-1.0,-0.2])
plotter.camera_position = [(65.49504667850935, 67.33632944253326, 136.62277759300397),
 (70.97935402738158, 81.04709781471367, 139.3649312674401),
 (0.0, 0.0, 1.0)]
plotter.show()
plotter.screenshot('bunny_tet_zoom.png', scale=2)


plotter = pv.Plotter()
plotter.add_mesh(bunny_voxel2.to_pyvista(), color='white', opacity=1)
plotter.view_vector([-0.4,-1.0,-0.2])
plotter.camera = camera1
plotter.show()
plotter.screenshot('bunny_voxel.png', scale=2)

# %% function -> mesh

func = implicit.tpms('S')
voxel = implicit.VoxelMesh(func, [0,0.5,0,1,0,1], 0.01, mode='notrim')
surf = implicit.SurfaceMesh(func, [0.5,1,0,1,0,1], 0.01)

plotter = pv.Plotter()
plotter.add_mesh(voxel.to_pyvista(), show_scalar_bar=False, cmap='coolwarm', )
plotter.add_mesh(surf.to_pyvista(), color='white')

plotter.show()
plotter.screenshot('implicit_s.png', scale=2)

# %% Implicit CSG
func1 = implicit.box(-.9,.9,-.9,.9,-.9,.9)
func2 = implicit.sphere([0,0,0],1.1)
cube = implicit.TetMesh(func1, [-1,1,-1,1,-1,1], .05)
diff = implicit.TetMesh(func2, [-1,1,-1,1,-1,1], .05, background=cube, threshold_direction=1)
plotter = pv.Plotter()
plotter.add_mesh(diff.to_pyvista(), color='white')
plotter.show()
plotter.screenshot('csg_box.png', scale=2)

# %% Miscellaneous 
bunny_surf.NodeData = bunny_surf.getCurvature(nRings=3)

bunny_coarse = improvement.Contract(bunny_surf, 8)

bunny_tet.NodeData['f'] = implicit.thicken(implicit.wrapfunc(implicit.tpms('S', 20)),1)(*bunny_tet.NodeCoords.T)
bunny_tetS = bunny_tet.Contour('f', 0, threshold_direction=-1)

#%%

plotter = pv.Plotter()
plotter.add_mesh(bunny_coarse.to_pyvista(), color='white', opacity=1)
plotter.view_vector([-0.4,-1.0,-0.2])
plotter.show()
plotter.screenshot('bunny_coarse.png', scale=2)


plotter = pv.Plotter()
plotter.add_mesh(bunny_surf.to_pyvista(), scalars='Mean Curvature', 
                 clim=(-.2,.2),
                 scalar_bar_args=dict(fmt='%.1f',label_font_size=25, title_font_size=25, title='Mean Curvature (1/mm)'))
plotter.view_vector([-0.4,-1.0,-0.2])
plotter.show()
plotter.screenshot('bunny_curvature.png', scale=2)


plotter = pv.Plotter()
plotter.add_mesh(bunny_tetS.to_pyvista(), color='white')
plotter.view_vector([-0.4,-1.0,-0.2])
plotter.show()
plotter.screenshot('bunny_S.png', scale=2)

root = tree.Surface2Octree(*bunny_surf,maxdepth=6)
voxel = mesh(*tree.Octree2Voxel(root,sparse=False))
edges = mesh(voxel.NodeCoords, voxel.Edges)

plotter = pv.Plotter()
plotter.add_mesh(bunny_surf.to_pyvista(), color='white')
plotter.add_mesh(edges.to_pyvista(), color='black', opacity=1)
plotter.view_vector([-0.4,-1.0,-0.2])
plotter.show()

# %%
