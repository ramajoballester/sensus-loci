# Configuration file for the Sphinx documentation builder.

import os
import sys
from importlib.metadata import metadata
from datetime import datetime

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information

project = 'sensus-loci'
project_slug = project.lower()
author = metadata(project.lower())['Author-email'].split('<')[0][:-1]
version = metadata(project.lower())['Version']
release = version
copyright = f'{datetime.now().year}, {author}'
# release = '0.1'
# version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'numpydoc',
    'sphinx.ext.autodoc.typehints',
    'myst_parser',      # For markdown support
    'sphinx.ext.autosectionlabel',
    'sphinx_tabs.tabs',
    'sphinx.ext.viewcode',
    'sphinx_markdown_tables',
    'sphinx.ext.napoleon'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']
templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
# html_theme = 'pytorch_sphinx_theme'


html_theme_options = {
    'prev_next_buttons_location': None,
}

# github_url = 'https://github.com/ramajoballester/sensus-loci'
html_baseurl = 'https://sensus-loci.readthedocs.io/en/latest/'
html_favicon = '../images/favicon.ico'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Configure autodoc to generate documentation for both class and module docstrings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    # 'no-show-inheritance': True,
}

# add_module_names = False