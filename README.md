![](resources/mymesh_logo.png)

MyMesh is a general purpose toolbox for generating, manipulating, and analyzing 
meshes for finite element, finite difference, or finite volume simulations. It 
has particular focuses on implicit function and image-based mesh generation.

MyMesh was originally developed in support of the Ph.D. research of Tim 
Josephson in Elise Morgan’s Skeletal Mechanobiology and Biomechanics Lab at 
Boston University.

# Getting Started
See the full documentation at: https://bu-smbl.github.io/mymesh/

## Installing from the Python Pacakge Index (PyPI):
```
pip install mymesh[all]
```

## Installing from source:
Download/clone the repository, then run
```
pip install -e <path>/mymesh[all]
```
with `<path>` replaced with the file path to the mymesh root directory.

to install with only the minimum required dependencies and skip optional 
dependencies, omit `[all]`. 