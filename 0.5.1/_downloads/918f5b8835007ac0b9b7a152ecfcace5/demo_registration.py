"""
Registration
============

This example illustrates registration of meshes and images using the Stanford
Bunny.

"""
#%%
import mymesh
from mymesh import register
import numpy as np
import matplotlib.pyplot as plt

#%% Load example meshes
bunny1 = mymesh.demo_mesh('bunny') 
bunny2 = mymesh.demo_mesh('bunny-res2')

# Perform an arbitrary rotation to bunny2 so that they are mis-aligned
bunny2 = bunny2.Transform([np.pi/6, -np.pi/6, np.pi/6], transformation='rotation', InPlace=True)

# Plot overlap of bunny1 and bunny2
def overlap_plot(m1, m2):
    m1.NodeData['label'] = np.zeros(m1.NNode)
    m2.NodeData['label'] = np.ones(m2.NNode)
    overlap = m1.copy()
    overlap.merge(m2)
    overlap.plot(scalars='label', show_colorbar=False, view='xy')
overlap_plot(bunny1, bunny2)

#%% Load example image
bunny_img1 = mymesh.demo_image('bunny') 
threshold = 100
voxelsize = (0.337891, 0.337891, 0.5) # mm

# Perform an arbitrary rotation to make a mis-aligned copy the image
R = register.rotation([np.pi/6, -np.pi/6, np.pi/6], 
                        center=np.array(bunny_img1.shape)/2)
bunny_img2 = register.transform_image(bunny_img1, R)

# Plot overlap of bunny1_img and bunny2_img 
# (purple=img1, orange=img2, pale yellow=overlap)
overlay = register.ImageOverlay(bunny_img1, bunny_img2, threshold=threshold)
plt.imshow(overlay[:,250], cmap='inferno')
plt.axis('off')
#%%
# Mesh-to-Mesh Registration
# -------------------------
# 
# Align the two meshes using the iterative closest point (ICP) algorithm
bunny_aligned, T = register.Mesh2Mesh(bunny1, bunny2)

# Plot 
overlap_plot(bunny1, bunny_aligned)

#%%
# Image-to-Image Registration
# ---------------------------
#

# Align the two images using the iterative closest point (ICP) algorithm
img_aligned, T = register.Image2Image(bunny_img1, bunny_img2, 
                                      method='icp', threshold=threshold,
                                      scale=0.5)
# Plot overlap
# (purple=img1, orange=img2, pale yellow=overlap)
overlay = register.ImageOverlay(bunny_img1, img_aligned, threshold=threshold)
plt.imshow(overlay[:,250], cmap='inferno')
plt.axis('off')
# %%
