Motivation, Approach, and Design
================================

Motivation
----------

Most existing approaches to benchmarking methods for causal inference fall into one of two categories:

1. **Empirical methods** use real, observed covariate and outcome data. This data is typically drawn from randomized experiments so that ground truth effect values are known (although experiments only provide average effect estimates). In these benchmarks, there is a true Data Generating Process (:term:`DGP`) but it is latent and its properties are unknown. It is unclear if the data meets typicaly causal inference assumptions, especially after modifications like replacing the random control group with non-random data to create a heuristically-valid *observational* setting.

2. **Synthetic methods** use artifical, generated covariate and outcome data. This data is drawn from manually specified distributions and functions. In this case, the DGP is known and the true effects - both at the average and individual level - can be recovered directly from the DGP. It is known from the DGP specification if causal assumptions are met.

The common flaw present in both benchmarking designs is that the sample size is, effectively, one. A new causal inference method is usually tested against a single DGP. In empirical datasets, the properties of this DGP and, most importantly, its place in the :term:`distributional problem space` is unknown. So the causal inference method is validated against a single sample from an unknown location in the problem space. In synthetic datasets, the DGP and the location is problem space is known but, in the standard approach, there is only one DGP per distributional setting. In the best case, either of these approaches risks missing important aspects of estimator performance in the selected region of distributional problem space. In the worst case, the results are biased - either by failure of empirical data to conform to basic causal assumptions or by manual specification using unrealistic data or unrepresentative/simplistic versions of DGP from the selected region (unintentially). In either scenario, benchmarking results are unlikely to generalize as indicators of consistent, real-world performance.

A New Approach
--------------

The approach proposed by this package is designed to mitigate the flaws above and improve the validity of benchmarks. This is done by more thoroughly exploring the distributional problem space (relative to empirical benchmarks) and reducing potential for bias by simultaneously using realistic covariates (relative to synthetic benchmarks) and synthetic DGPs constructed by repeatedly sampling from a parameterized *distribution* of functions. The sampling of Data Generating Processes defined over real covarieyes allows one to more robustly evaluate the distribution over method performance in a region of the :term:`distributional problem space`.

In slightly more detail, Maccabee samples treatment assignment and outcome functions from the class of *generalized, additive functions*. These are functions made up of the linear combination of (different) non-linear terms. Each term may contain one or few convariates and takes the form of an two or three way interaction, polynomial power, or discontinuous jump/kink. Covariates may appear in multiple terms. By controlling the probability with which different term types appear in the treatment assignment and outcome function, as well as the overlap of terms between the functions, it is possible to sample functions that represent a wide (and controllable) range of linearity and confounding. Numerical function normalization and manipulation allows further control over the expected treatment probability, the distribution of covariates across the treated/control groups and the heterogeneity of the treatment effect. Finally, control over the treatment effect, noise and co-efficient distribution provides control over the scale and signal-to-noise ratio of the DGP functions.

This is a high-level picture of how Maccabee works. The accompanying theory paper formalizes the various properties discussed above into a set of axes which define a distributional problem space. By controlling the sampling location along each axis, it is possible to generate DGPs that are - in expectation - located in different regions of the space. This, in turn, allows for better evaluation of new causal inference methods. Both by evaluating them across more regions in the problem space and more robustly in each region by constructing multiple, structirally similar DGPs to average out function-specific effects.

Technical Design and Components
-------------------------------

The section above provided an overview of the theoretical approach taken by Maccabee. This section outlines the objects which are used to operationalize this approach. At a 10000 feet, the Maccabee package works as described below. Each of the objects which are represented by a Maccabee class are bolded. Click the bolded name for access to detailed class documentation.

TODO: supplement/replace with a figure.

To perform a **Benchmark** (:mod:`~maccabee.benchmarking`), one or more sets of **Sampling Parameters** (:mod:`~maccabee.parameters`) are using by the **DGP Sampler** (:mod:`~maccabee.data_generation.data_generating_process_sampler`) to sample **DGPs** (:mod:`~maccabee.data_generation.data_generating_process`) at a specific location in the :term:`distributional problem space`. **Data sets** (:mod:`~maccabee.data_generation.generated_data_set`) are then sampled from the sampled DGPs. The location of these data sets in the problem space is evaluated using **Data Metrics** (:mod:`~maccabee.data_analysis.data_metrics`). **Causal Models** (:mod:`~maccabee.modeling.models`) are used to generate estimates for a selected causal estimands. The performance of the models is evaluated against the ground truth from the sampled data sets using **Performance Metrics** (:mod:`~maccabee.modeling.performance_metrics`). The results of repeated DGP and data set samples are aggregated and returned to the user.


TODO: cover:

* Flexible parameter specification
* DGP Sampling
* DGP spec - DSL

Design Principles
-----------------

Fundamentally, this package only succeeds if it provides a useful and usable way to benchmark new methods for causal inference developed by its users. Maccabee’s features are focused around four design principles to achieve this end:

* **Minimal imposition on method design:** attention has been paid to ensuring model developers can use their own empirical data and models with Maccabee painlessly. This includes support for benchmarking models written in both Python and R to avoid the need for language translation.

* **Quickstart but powerful customization:** The package includes high-quality data and pre-tuned parameters. This means that little boilerplate code is required to run a benchmark and receive results. This helps new users understand, and get value out of, the package quickly. At the same time, there is a large control surface to give advanced users the tools they need to support heavily-customized benchmarking processes.

* **Support for advanced functionality:** all Monte Carlo benchmarking requires access to sufficient computational power and a way to persist and compare results. Maccabee provides seamless integration with cluster computing tools to run large benchmarks on public cloud/private compute platforms as well as providing tools for result persistence and management which work both locally and with cluster computing.

* **Smooth side-by-side support of old and new approaches:** most users may feel initial discomfort using only the novel benchmarking approach proposed in the theoretical work. Maccabee allows for concrete, user-specified DGPs to be used side by side with the new approach. This allows users to switch between/compare the new and old approaches while using a single benchmarking tool. It also allows users to exploit the advanced functionality outlined above even if they don’t use the core sampling functionality. The hope is that users who start with concrete DGPs will transition to the newer (and theoretically superior) sampling approaches.

Glossary
--------

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
