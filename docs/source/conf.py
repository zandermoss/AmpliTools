import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('..'))

project = 'AmpliTools'
copyright = '2023, Alexander "Zander" Moss'
author = 'Alexander "Zander" Moss'
release = '1.0'

extensions = ['sphinx_rtd_theme',
              'sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.githubpages'
              ]

templates_path = ['_templates']
exclude_patterns = ["setup"]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
