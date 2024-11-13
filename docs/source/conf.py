# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys, re
import matplotlib
import matplotlib.pyplot as plt
import mymesh

project = 'MyMesh'
copyright = '2023, Timothy O. Josephson'
author = 'Timothy O. Josephson'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.todo', 'sphinx.ext.mathjax', 'sphinx.ext.ifconfig', 'sphinx.ext.viewcode', 'sphinx.ext.githubpages', 'sphinx.ext.napoleon', 'sphinx.ext.autosectionlabel', 'sphinx.ext.autosummary', 'matplotlib.sphinxext.plot_directive','sphinx_design','sphinx.ext.graphviz','sphinx_copybutton','sphinxcontrib.bibtex','jupyter_sphinx','sphinx_gallery.gen_gallery']
autodoc_mock_imports = []
autodoc_member_order = 'bysource'
autosummary_generate = True

templates_path = ['_templates']

sphinx_gallery_conf = {
     'filename_pattern': re.escape(os.sep) + 'demo_',
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'examples',  # path to where to save gallery generated output
     'download_all_examples': False,
     'remove_config_comments': True,
     'capture_repr': (),
}  

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_title = f"{project} v{release} Manual"
html_theme =  'pydata_sphinx_theme' #'sphinx_rtd_theme' #
html_static_path = ['_static']
html_logo = '_static/mymesh_logo.svg'
html_css_files = ['css/mymesh.css']
html_theme_options = dict(collapse_navigation=True, 
                           navigation_depth=1,
                           icon_links= [
                              {
                                 # Label for this link
                                 "name": "GitHub",
                                 # URL where the link will redirect
                                 "url": "https://github.com/BU-SMBL/mymesh",  # required
                                 # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
                                 "icon": "fa-brands fa-square-github",
                                 # The type of image to be used (see below for details)
                                 "type": "fontawesome",
                              }]
                        )
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

# graphviz_dot = r"C:\Program Files\Graphviz\bin\neato.exe"
graphviz_dot = r"neato"
copybutton_prompt_text = ">>> "

bibtex_bibfiles = ['references.bib']
