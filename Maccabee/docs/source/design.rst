Design and Implementation
=========================

In depth explanations of the theoretical underpinnings of the Maccabee benchmarking process can be found in Chapters 5 and 6 of the :download:`theory paper </maccabee-theory-paper.pdf>`.

This page documents the implementation which is to operationalize Maccabee's Monte Carlo Benchmarking. ... The second subsection then shows how the sampling procedure is implemented in the functions and classes that make up the package. This is a description at the level of the functional components used concretize, and sample from, the statistical model described in the first subsection.

Package Design Principles
-------------------------

Fundamentally, this package only succeeds if it provides a useful and usable way to benchmark new methods for causal inference developed by its users. Maccabee’s features are focused around four design principles to achieve this end:

* **Minimal imposition on method design:** attention has been paid to ensuring model developers can use their own empirical data and models with Maccabee painlessly. This includes support for benchmarking models written in both Python and R to avoid the need for language translation.

* **Quickstart but powerful customization:** The package includes high-quality data and pre-tuned parameters. This means that little boilerplate code is required to run a benchmark and receive results. This helps new users understand, and get value out of, the package quickly. At the same time, there is a large control surface to give advanced users the tools they need to support heavily-customized benchmarking processes.

* **Support for optimized, parallel execution:** valid Monte Carlo benchmarks require large sample sizes. In turn, this requires effecient, optimized code and the ability to access and utilize sufficient computational power. Maccabee provides code compilation for sampled DGPs - which greatly improves execution time - and parallelization tools that enable execution across multiple cores. Together, these tools make large-sample benchmarks feasible.

* **Smooth side-by-side support of old and new approaches:** Maccabee allows for user-specified DGPs to be used side by side with the sampled DGPs enabled by the package. This allows users to switch between/compare the new and old approaches while using a single benchmarking tool. It also allows users to exploit the advanced functionality outlined above even if they don’t use the core sampling functionality.

Objects
-------

The figure below...

The bolded text signifies Maccabee classes/modules and link to detailed documentation for the relevant component.

.. image:: design/maccabee-design-implementation-fig.png

To perform a **Benchmark** (:mod:`~maccabee.benchmarking`), one or more sets of **Sampling Parameters** (:mod:`~maccabee.parameters`) are using by the **DGP Sampler** (:mod:`~maccabee.data_generation.data_generating_process_sampler`) to sample **DGPs** (:mod:`~maccabee.data_generation.data_generating_process`) at a specific location in the :term:`distributional problem space`. **Data sets** (:mod:`~maccabee.data_generation.generated_data_set`) are then sampled from the sampled DGPs. The location of these data sets in the problem space is evaluated using **Data Metrics** (:mod:`~maccabee.data_analysis.data_metrics`). **Causal Models** (:mod:`~maccabee.modeling.models`) are used to generate estimates for a selected causal estimands. The performance of the models is evaluated against the ground truth from the sampled data sets using **Performance Metrics** (:mod:`~maccabee.modeling.performance_metrics`). The results of repeated DGP and data set samples are aggregated and returned to the user.

Advanced Implementation
-----------------------

- Aggregation in the benchmarking section
- Abstract Syntax Trees for equation construction
- Parallelism
- Good OOP practices throughout.

Glossary of Terms
-----------------

TODO: finish these.

.. glossary::

    Causal Model
      A causal model implements a mathematical estimator which extracts a causal estimand from an observational data set.

    Data Metric
      Data Metrics are real-valued functions which measure some distributional property of a generated data set. Each data metric measures the position of the data set along some well-defined 'axis' of the distributional problem space. Each axis may have more than one corresponding data metric.

    DGP
      A Data Generating Process describes the mathematical process which gives rise to a set of observed data - covariates, treatment assignments, and outcomes - and the corresponding unobserved/oracle data, primarily the treatment effect.

      Concretely, a DGP relates the DGP Variables - defined in the constants group :class:`~maccabee.constants.Constants.DGPVariables` - through a series of stochastic/deterministic functions. The nature of these functions defines the location of the resultant data sets in the :term:`distributional problem space`.

    Distributional Problem Space
      The performance of causal estimators depends on distributional properties of the observed data. The space of all possible distributional properties forms the distributional problem space. The performance of an estimator across the space and in specific regions is of interest to researchers.

    Distributional Problem Space Axis
      The :term:`distributional problem space` is defined by axes which represent the distributional properties and the values they can take on. The cartesian product of the values the axes can take out is the extent of the problem space.

    Distributional Setting
      A location in the :term:`distributional problem space` characterized by a specific position along each :term:`distributional problem space axis`.

    DSL
      TODO - domain specific language.

    DGP Variable
      DGP variables are the variables over which the DGP is defined. See chapter 3 and 4 of the theory work.

    Observable DGP Variable
      DGP variables which are available for causal inference.

    Oracle DGP Variable
      DGP variables which are not available for causal inference but which can be thought of as 'existing' during the data generation process. This includes potential outcomes, treatment effect, outcome noise etc.

    Parameter Specification File
      A file used to specify a set of DGP sampling parameters. The specification conforms to the schema laid out in the :term:`parameter schema file`.

    Default Parameter Specification File
      The file which specifies the default set of DGP sampling parameters. This is laid out as a standard :term:`parameter specification file`.

    Parameter Schema File
      The file which defines all of the DGP sampling parameters by providing names, types, validity conditions, and descriptions. The :term:`parameter specification file` specifies DGP sampling parameters that conform to the schema laid out in this file.

    Performance Metric
      Performance Metrics are real-valued functions which measure the quality of a causal estimator by comparing the estimand value to the ground truth. A performance metric may be well defined for a single estimand value but typically, in the context of this package, they are defined over a sample of estimand values with each estimand value corresponding to an estimate of the causal effect/s in a generated data set.

    Transformed Covariate
      TODO - transformed covariate

    YML
      YAML is a human-readable data-serialization language. It is commonly used for configuration files and in applications where data is being stored or transmitted (Wikipedia).
