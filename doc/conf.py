# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

# pylint: skip-file

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import datalab_kernel

# -- Project information -----------------------------------------------------

project = "DataLab-Kernel"
author = "DataLab Platform Developers"
copyright = "2025, DataLab Platform Developers"
release = datalab_kernel.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "myst_nb",  # Markdown and notebook support
    "sphinx_design",
    "sphinx_copybutton",
]

# MyST-NB configuration
nb_execution_mode = "off"  # Don't re-execute notebooks during build
nb_execution_timeout = 180  # Timeout for notebook execution (if enabled)

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress some warnings
suppress_warnings = ["config.cache"]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = project
html_logo = "_static/DataLab-Kernel-Banner.svg"
html_show_sourcelink = False

html_theme_options = {
    "show_toc_level": 2,
    "github_url": "https://github.com/DataLab-Platform/DataLab-Kernel/",
    "logo": {
        "text": f"v{datalab_kernel.__version__}",
    },
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/datalab-kernel",
            "icon": "_static/pypi.svg",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
        {
            "name": "DataLab",
            "url": "https://datalab-platform.com",
            "icon": "_static/DataLab.svg",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
    ],
}
html_static_path = ["_static"]

# -- Options for LaTeX output ------------------------------------------------
latex_logo = "_static/DataLab-Kernel-Banner.svg"

# -- Options for sphinx-intl package -----------------------------------------
locale_dirs = ["locale/"]
gettext_compact = False
gettext_location = False

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
}

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sigima": ("https://sigima.readthedocs.io/en/latest/", None),
    "datalab": ("https://datalab-platform.com/en/", None),
    "ipykernel": ("https://ipykernel.readthedocs.io/en/stable/", None),
}
