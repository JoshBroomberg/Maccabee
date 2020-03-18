========
Maccabee
========

Introduction
------------

Maccabee provides a new mechanism for benchmarking causal inference methods. This method uses sampled Data Generating Processes defined over empirical covariates to provide more realistic, less biased evaluation of estimator performance across the distributional problem space of causal inference.

If you're ready to get started see :doc:`installation` and :doc:`usage`. If you'd like to learn more about the motivation and design of Maccabee, see the :doc:`design` page.

Package Design Principles
-------------------------

Fundamentally, this package only succeeds if it provides a useful and usable way to benchmark new methods for causal inference developed by its users. Maccabee’s features are focused around four design principles to achieve this end:

* **Minimal imposition on method design:** attention has been paid to ensuring model developers can use their own empirical data and models with Maccabee painlessly. This includes support for benchmarking models written in both Python and R to avoid the need for language translation.

* **Quickstart but powerful customization:** The package includes high-quality data and pre-tuned parameters. This means that little boilerplate code is required to run a benchmark and receive results. This helps new users understand, and get value out of, the package quickly. At the same time, there is a large control surface to give advanced users the tools they need to support heavily-customized benchmarking processes.

* **Support for optimized, parallel execution:** valid Monte Carlo benchmarks require large sample sizes. In turn, this requires effecient, optimized code and the ability to access and utilize sufficient computational power. Maccabee provides code compilation for sampled DGPs - which greatly improves execution time - and parallelization tools that enable execution across multiple cores. Together, these tools make large-sample benchmarks feasible.

* **Smooth side-by-side support of old and new approaches:** Maccabee allows for user-specified DGPs to be used side by side with the sampled DGPs enabled by the package. This allows users to switch between/compare the new and old approaches while using a single benchmarking tool. It also allows users to exploit the advanced functionality outlined above even if they don’t use the core sampling functionality.

Table of Contents
-----------------

.. toctree::
  :maxdepth: 2
  :caption: Installation

  installation.rst

.. toctree::
  :maxdepth: 2
  :caption: Usage

  usage.rst
  advanced.rst

.. toctree::
  :maxdepth: 2
  :caption: Design

  design.rst

.. toctree::
  :maxdepth: 2
  :caption: Reference

  reference.rst
