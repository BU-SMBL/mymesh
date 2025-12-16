![](resources/mymesh_logo.png)

[![PyPI - Version](https://img.shields.io/pypi/v/mymesh)](https://pypi.org/project/mymesh/)
![Static Badge](https://img.shields.io/badge/doi%20-%20Zenodo%20-%20%20%231A90DE?link=https%3A%2F%2Fzenodo.org%2Frecords%2F17511909)


A mesh is a discrete representation of a geometry or computational domain where space is subdivided it into a collection of points (nodes) connected by simple shapes (elements).
Meshes are used for a variety of purposes, including simulations (e.g. finite element, finite volume, and finite difference methods), visualization & computer graphics, image analysis, and additive manufacturing.
`mymesh` is a general purpose set of tools for generating, manipulating, and analyzing meshes. `mymesh` is particularly focused on implicit function and image-based meshing, with other functionality including:

- geometric and curvature analysis,
- intersection and inclusion tests (e.g. ray-surface intersection and point-in-surface tests)
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
pip install -e <path>/mymesh
```
with `<path>` replaced with the file path to the mymesh root directory.

# Development

## Note on the usage of generative AI
MyMesh was and will continue to be developed by humans. Initial development of
MyMesh began in the summer of 2021, before the release of OpenAI's ChatGPT 
(Nov. 30, 2022) and the widespread proliferation of powerful generative AI 
chatbots. Since the release of ChatGPT, Claude (Anthropic), Gemini (Google), and
others, I have at times explored their capabilities by asking them meshing
questions, receiving a mix of helpful and unhelpful responses. While generative
AI was never used to generate the code for MyMesh, it was in some instances 
consulted alongside other resources (e.g. StackExchange) for recommendations
on how to improve efficiency of certain processes.
Generative AI has been used in the following ways throughout the development of 
MyMesh:
- As a consultant for understanding concepts, alongside academic literature.
- As a resource for general-purpose programming concepts, such as methods for improving efficiency of certain operations.
- Assistance in setting up packaging infrastructure (e.g. pyproject.toml, github workflows).
- Assistance in the creation of test cases for some unit tests.
  