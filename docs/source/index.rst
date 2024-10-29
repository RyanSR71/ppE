Welcome to WaveformUncertainty's documentation!
===================================

Package version:
|version|

**WaveformUncertainty** is a Python library that gives the infrastructure necessary for sampling over waveform uncertainty during parameter estimation runs. This infrastructure includes: easy generation of waveform model differences, the parameterization of waveform model differences for efficient storage, a model for waveform uncertainty priors, a waveform model that includes waveform uncertainties, and a spline-based method for efficiently sampling over waveform uncertainty.  

This package relies on bilby and is essentially an addon to bilby. For more information, check out `Bilby's Documentation <https://lscsoft.docs.ligo.org/bilby/index.html>`_. This work is in collaboration with and derived from past work by Dr. Jocelyn Read. See `Jocelyn Read 2023 Class. Quantum Grav. 40 135002 <https://arxiv.org/abs/2301.06630v2>`_.

Check out the :doc:`installation` section for instructions. General information is present in the Content tab, tutorials are in the Tutorials tab, and code documentation is in the API tab.

.. note::

   This project is under active development and may change drastically.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   notebooks/WFU_Sampling_Tutorial
   notebooks/Posterior_Analysis_Tutorial
   notebooks/Parameterization_Tutorial

.. toctree::
   :maxdepth: 1
   :caption: API:

   fd_model_difference
   parameterization
   recovery_from_parameterization
   uncertainties_from_parameterization
   WFU_dA_prior
   WFU_dphi_prior
   maxL
   Q_factor
   WaveformGeneratorWFU
