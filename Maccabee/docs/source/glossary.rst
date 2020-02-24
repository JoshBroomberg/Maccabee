:orphan:

Glossary of Terms
=================

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
