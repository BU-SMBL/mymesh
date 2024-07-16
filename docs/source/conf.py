# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
import matplotlib
import matplotlib.pyplot as plt
import mymesh

project = 'MyMesh'
copyright = '2023, Timothy O. Josephson'
author = 'Timothy O. Josephson'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.todo', 'sphinx.ext.mathjax', 'sphinx.ext.ifconfig', 'sphinx.ext.viewcode', 'sphinx.ext.githubpages', 'sphinx.ext.napoleon', 'sphinx.ext.autosectionlabel', 'sphinx.ext.autosummary', 'matplotlib.sphinxext.plot_directive','sphinx_design','sphinx.ext.graphviz','sphinx_copybutton','sphinxcontrib.bibtex','jupyter_sphinx']
autodoc_mock_imports = []
autodoc_member_order = 'bysource'
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['TetGen.py']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_title = f"{project} v{release} Manual"
html_theme =  'pydata_sphinx_theme' #'sphinx_rtd_theme' #
html_static_path = ['_static']
html_logo = '_static/mymesh_logo.svg'
html_css_files = ['css/mymesh.css']
html_theme_options = dict(collapse_navigation=True, navigation_depth=1)
html_context = {
   "default_mode": "light"
}
# Plotting options
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = ['png']
plot_pre_code = '''
import numpy as np
from matplotlib import pyplot as plt
from mymesh import *
visualize.set_vispy_backend(preference='PyQt6')
'''

graphviz_dot = r"C:\Program Files\Graphviz\bin\neato.exe"

copybutton_prompt_text = ">>> "

bibtex_bibfiles = ['references.bib']
