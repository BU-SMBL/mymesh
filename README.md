![](resources/mymesh_logo.png)


[![PyPI - Version](https://img.shields.io/pypi/v/mymesh)](https://pypi.org/project/mymesh/)
[![status](https://joss.theoj.org/papers/325341363198c2405d4b7e78722bd9f1/status.svg)](https://joss.theoj.org/papers/325341363198c2405d4b7e78722bd9f1)
[![Static Badge](https://img.shields.io/badge/archive%20-%20Zenodo%20-%20%20%231A90DE?link=https%3A%2F%2Fzenodo.org%2Frecords%2F17511909)](https://zenodo.org/records/17511909)
[![Static Badge](https://img.shields.io/badge/license%20-%20MIT%20-%20%20%23750014)](https://github.com/BU-SMBL/mymesh?tab=MIT-1-ov-file#readme)

A mesh is a discrete representation that subdivides a geometry or computational domain into a collection of points (nodes) connected by simple shapes (elements).
Meshes are used for a variety of purposes, including simulations (e.g. finite element, finite volume, and finite difference methods), visualization and computer graphics, image analysis, and additive manufacturing.
`mymesh` is a general purpose set of tools for generating, manipulating, and analyzing meshes. 
`mymesh` is particularly focused on implicit function and image-based meshing, with other functionality including:

- geometric and curvature analysis,
- intersection and inclusion tests (e.g. ray-surface intersection and point-in-surface tests),
- mesh boolean operations (intersection, union, difference),
- sweep construction methods (extrusions, revolutions),
- point set, mesh, and image registration,
- mesh quality evaluation and improvement,
- mesh type conversion (e.g. volume to surface, hexahedral or mixed-element to tetrahedral, first-order elements to second-order elements).

`mymesh` was originally developed in support of research within the Skeletal Mechanobiology and Biomechanics Lab at Boston University. 

# Getting Started
For more details, see the [full documentation](https://bu-smbl.github.io/mymesh/)

## Installing from the [Python Package Index (PyPI)](https://pypi.org/project/mymesh/)
```
pip install mymesh[all]
```

To install only the minimum required dependencies, omit `[all]`.

## Installing from source:
Download/clone the repository, then run
```
pip install <path>/mymesh
```
with `<path>` replaced with the file path to the mymesh root directory.

# Development

## AI Usage Disclosure
Generative AI was not used to write the documentation or the functionality of `mymesh`. 
Initial development of `mymesh` began in the summer of 2021, before the release of OpenAI's ChatGPT (Nov. 30, 2022) and the widespread proliferation of powerful generative AI chatbots.
While generative AI was never used to generate code for `mymesh`, it was in some instances consulted alongside other resources (e.g. scientific literature, StackExchange).
Generative AI has been used in the following ways throughout the development of 
`mymesh`:

- as a resource for some mesh-specific and general-purpose programming concepts, such as methods for improving efficiency of certain operations,
- assistance in setting up packaging infrastructure (e.g. pyproject.toml, GitHub workflows),
- conceptualization of test cases for some unit tests.
  
# Attribution
If you've found MyMesh useful in your work, please cite it by referring to the paper in the [Journal of Open Source Software](https://doi.org/10.21105/joss.10003). 
You can also cite specific versions by referring to the [Zenodo archive](https://zenodo.org/records/17511909).

>Josephson & Morgan, (2026). MyMesh: General purpose, implicit, and image-based meshing in Python. Journal of Open Source Software, 11(122), 10003, https://doi.org/10.21105/joss.10003