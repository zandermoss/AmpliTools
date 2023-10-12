==========
AmpliTools
==========
**AmpliTools** provides a framework for symbolic calculation of scattering amplitudes
in scalar quantum field theories with arbitrary interactions and mass spectra. Kinematic
manipulations are carried out in a basis of Mandelstam invariants, so *AmpliTools* calculations
are valid in arbitrary spacetime dimension.

*AmpliTools* supports:

* Automatic Feynman diagram generation and isomorphism reduction using `NetworkX <https://networkx.org/>`_ and `nautypy <https://zandermoss.github.io/nautypy/>`_.
* Construction and manipulation of symbolic amplitude expressions from Feynman diagrams using `SymPy <https://www.sympy.org/>`_.
* Flavor tensor algebra, symmetry group actions, and index canonization.
* Reduction of symbolic amplitudes to minimal kinematic bases.
* User interaction through the `Jupyter Notebook <https://jupyter.org/>`_, allowing symbolic input with a relatively clean syntax, typeset symbolic output with LaTeX, and visualization of Feynman diagrams. 

*AmpliTools* was developed to carry out specific calculations in scalar field theories
(see https://arxiv.org/abs/2012.13076). Although it was designed with generalization to
arbitrary spins in mind, it currently supports only scalar theories.

Usage of ``AmpliTools`` is demonstrated in an example notebook, ``examples/scalar_example.ipynb``.

Reference documentation is available at https://zandermoss.github.io/AmpliTools/

Installation
============
* Source code is available at https://github.com/zandermoss/AmpliTools/

* ``nautypy`` can be built from source available on `github <https://github.com/zandermoss/nautypy/>`_

Requirements
------------
* In addition to ``nautypy``, ``AmpliTools`` requires::

    hashable_containers (packaged with nautypy)
    sympy
    networkx
    jupyter
    matplotlib

Quick Start
-----------
* Run ``install.sh``

* Or::

    python3 setup.py build
    pip install -r requirements.txt .

Examples
========
* Running ``jupyter lab`` from the ``examples/`` directory will expose an example notebook
  ``examples/scalar_example.ipynb``. 

* This notebook demonstrates *AmpliTools* by computing symbolic expressions for the four and
  five point tree amplitudes of a renormalizable scalar theory. The notebook is annotated in
  an attempt to explain the roles of the most important components of the software, and to
  exhibit various features of the interface.

Documentation
=============
* Docs built with `Sphinx <https://www.sphinx-doc.org/>`_.

Building Docs
-------------
* Automatically with ``install.sh``.
* Manually::

      cd docs/
      make html

* ``index.html`` can then be found in docs/build/html/
