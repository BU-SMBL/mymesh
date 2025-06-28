![](resources/mymesh_logo.png)

MyMesh is a general purpose toolbox for generating, manipulating, and analyzing 
meshes for finite element, finite difference, or finite volume simulations. It 
has particular focuses on implicit function and image-based mesh generation.

MyMesh was originally developed in support of the Ph.D. research of Tim 
Josephson in Elise Morgan’s Skeletal Mechanobiology and Biomechanics Lab at 
Boston University.

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

## Usage of generative AI
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
    - Generation of test cases for some unit tests.
  