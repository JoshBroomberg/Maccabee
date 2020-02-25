Design and Implementation
=========================

The goal of this page is to explain how Maccabee's theoretical approach to benchmarking - as laid out in Chapter 5 of the :download:`theory paper </maccabee-theory-paper.pdf>` - is implemented using the functions and classes of the Maccabee package. This page therefore assumes a base level of familiarity with the theoretical approach although a brief summary is provided in the first section below.

Benchmarking Approach Summary
-----------------------------

The graphical model below represents the complete statistical model of a Maccabee Monte Carlo benchmark.

.. image:: design/maccabee-design-graphical-model-fig.png

Briefly summarizing the sampling procedure implied by this model:

* A set of :math:`M` DGPs is sampled based on supplied *DGP Sampling Parameters*.

* For each DGP, :math:`N` sets of individual observation variables consisting of observed covariates and a set of associated treatment assignment, outcome and causal effect variables (both observed and unobserved). The complete set of variables is defined in :data:`~maccabee.constants.Constants.DGPVariables`.

* For each DGP, causal estimand values are sampled (perhaps deterministically) at either the individual observation (for individual effects) or dataset level (for average effects). These values are conditioned on all :math:`N` of the :math:`X`, :math:`T` and :math:`Y` observations.

* :math:`M` Individual or Average Performance Metric values are calculated (deterministically sampled) at the dataset level by combining the causal effect estimate values with the appropriate ground truth value(s). Optionally, :math:`M` Data Metrics are calculated by combining some/all of the covariate data with the observed and oracle outcome data.

Implementation Overview
-----------------------

The section above used a graphical model to describe the benchmarking approach at the level of the data (random variables). This section describes the components used to implement, and sample from, this model. This description is at the level of implemented functions and classes.

.. image:: design/maccabee-design-implementation-fig.png

Core Sampling Execution Flow
++++++++++++++++++++++++++++

The figure below shows how all of Maccabee's classes and functions fit together to perform a single sample of all of the random variables that appear in the graphical model above. Modules containing the closely related components are indicated using boxes. From top to bottom:

* A *data source builder* function from the :mod:`~maccabee.data_sources.data_source_builders` module is used to build a :class:`~maccabee.data_sources.data_sources.DataSource` instance. This class encapsulates the code needed to load and prepare empirical or synthetic covariate observations.

|

* A *parameter store builder* function from the :mod:`~maccabee.parameters.parameter_store_builders` module is used to build a :class:`~maccabee.parameters.parameter_store.ParameterStore` instance. This class encapsulates the code used to modify and access the DGP sampling parameters.

|

* The two instances from the steps above are provided to a :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler` instance. This class encapsulates the code required to sample Data Generating Processes defined over the covariate data from the :class:`~maccabee.data_sources.data_sources.DataSource` instance based on the parameters in the :class:`~maccabee.parameters.parameter_store.ParameterStore` instance.

|

* The :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler` instance is used to sample :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` instance. The :class:`~maccabee.data_generation.data_generating_process.SampledDataGeneratingProcess` subclass, produced by the :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler`, encapsulates the logic needed to sample datasets given the sampled components of a DGP as described in :download:`theory paper </maccabee-theory-paper.pdf>`.

|

* The :class:`~maccabee.data_generation.data_generating_process.SampledDataGeneratingProcess` instance is used to sample  :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance. The :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` class encapsulates the logic used to access generated DGP variables (the observed and unobserved variables listed in :data:`~maccabee.constants.Constants.DGPVariables` over which the DGP is defined). This includes logic to access ground truth estimand values derived from the DGP variables.

|

* A :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance is passed to a :class:`~maccabee.modeling.models.CausalModel` instance. This class encapsulates the (user-supplied) modeling logic that estimates causal estimands. The causal model class provides abstracts the details of the model and allows for simple external access to one or more estimands.

|

* The estimated causal estimand values (from the :class:`~maccabee.modeling.models.CausalModel` instance) and the ground truth values (from the :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance) are passed to *performance metric functions* from the :mod:`~maccabee.modeling.performance_metrics` module (this is a submodule of the :mod:`~maccabee.modeling` module). Given the relative simplicity of the performance metric calculation, the functions from the :mod:`~maccabee.modeling.performance_metrics` module are used directly by code outside the module.

|

* Optionally, the :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instance is passed to *data metric functions* from the :mod:`~maccabee.data_analysis` module. The data metric code is complex enough that the calculation of data metrics using *data metric functions* is encapsulated by the :func:`~maccabee.data_analysis.data_analysis.calculate_data_axis_metrics` function.

Sampling Execution Flow Notes
+++++++++++++++++++++++++++++

A few details are missing from the description of the sampling execution flow in the section above.

Firstly, most of the process above is not implemented directly by users. Rather, it is implemented in *benchmarking functions* from the :mod:`~maccabee.benchmarking` module. The exact process above is implemented in the :func:`~maccabee.benchmarking.benchmarking.benchmark_model_using_sampled_dgp` which accepts a :class:`~maccabee.data_sources.data_sources.DataSource` instance and a :class:`~maccabee.parameters.parameter_store.ParameterStore` instance from the user and then implements the rest of the process (sampling many :class:`~maccabee.data_generation.data_generating_process.SampledDataGeneratingProcess` instances and many :class:`~maccabee.data_generation.generated_data_set.GeneratedDataSet` instances from each DGP). The other functions in the :mod:`~maccabee.benchmarking` module support different use cases and these covered in the tutorials.

Second, there is nuance around how covariate data is handled relative to the formal statistical model. In the model, the covariates are sampled directly from a DGP. In the package, covariate sampling is encapsulated in a :class:`~maccabee.data_sources.data_sources.DataSource` instance which is provided to the :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler` instance. This is done for two reasons.

1. This encapsulates the complex logic needed to load empirical datasets or sample stochastic joint distributions and then normalize the resultant observations. Under the hood, the :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` samples covariates from the :class:`~maccabee.data_sources.data_sources.DataSource` as one would expect.

2. A sample of the covariate data is actually used by the :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler` when normalizing the sampled treatment and outcome functions. This means the :class:`~maccabee.data_generation.data_generating_process_sampler.DataGeneratingProcessSampler` needs access to covariate data **before** DGPs can be sampled.

If the user defines a custom :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` class to represent a concrete DGP, then the choice of whether to use a :class:`~maccabee.data_sources.data_sources.DataSource` or sample covariates directly in the :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` is up to the user.

Finally, it is worth discussing the philosophy behind the choice to use classes vs functions to represent different components. In general, code that is stateless (doesn't preserve any information between runs) is implemented using functions. This applies to the *_builder* functions, metric calculation functions, and the benchmarking functions. Note that, where possible, code executed directly by users is designed to be stateless to allow for execution without the overhead of instance creation and management. Code that is stateful, and called repeatedly, is implemented using classes. Both the functional and class based components are customizable. For example, users can inject their own performance/data metrics as demonstrated in :doc:`/advanced/custom-metrics` and subclass the :class:`~maccabee.data_generation.data_generating_process.DataGeneratingProcess` class to benchmark using concrete DGPs as demonstrated in :doc:`/usage/concrete-dgp`.

DGP Sampling Parameterization
-----------------------------

The theory paper outlined the detailed steps used to sample DGPs. This section explains how the paramterization of these steps is implemented in Maccabee. The DGP sampling procedure from the theory paper is summarized below, with the concrete Maccabee parameters given alongside each step. For full explanations of each step, see the :download:`theory paper </maccabee-theory-paper.pdf>`. For details on the specification and constraints for each parameter see the :download:`parameter_schema.yml </../../maccabee/parameters/parameter_schema.yml>` file. For more on how to use the parameters listed below to run benchmarks see the usage tutorials and the documentation for the :mod:`~maccabee.parameters` module.

*Note: in the current implementation, functional parameters (sampling distributions) are hard coded and therefore are indicated with lower case function names. Dynamic functional parameters are planned for future releases.*

1. A set of potential confounders is selected from all of the covariates based on a single, uniform selection probability. This selection probability is the ``POTENTIAL_CONFOUNDER_SELECTION_PROBABILITY`` parameter.

2. The treatment assignment and untreated outcome functions are sampled from two independently parameterized distributions. Each distribution is parameterized by the probability of selecting each *term type* from the universe of valid terms for that *term type* given the covariates. These probabilities are given in the ``TREAT_MECHANISM_COVARIATE_SELECTION_PROBABILITY`` and ``OUTCOME_MECHANISM_COVARIATE_SELECTION_PROBABILITY`` dictionaries respectively. The co-efficients in all of the terms of two functions are initialized by drawing samples from the global co-efficient distribution using the functional parameter ``sample_subfunction_constants``.

3. An alignment (confounding) adjustment is performed. This step involves randomly selecting terms to add/remove from the sampled treatment assignment and untreated outcome functions to meet a target alignment parameter. This parameter specifies the target probability for a term in the untreated outcome function to also appears in the treatment assignment function. This target probability is the ``ACTUAL_CONFOUNDER_ALIGNMENT`` parameter.

4. The treatment effect mechanism is then sampled. First, a constant treatment effect is sampled from a parameterized, global treatment effect distribution using the functional parameter ``sample_treatment_effect``. Then, terms from the untreated outcome function are sampled based on a treatment effect heterogeneity parameter. This parameter specifies the uniform probability that a term in the untreated outcome function appears in the treatment effect function. This probability is the ``TREATMENT_EFFECT_HETEROGENEITY`` parameter.

5. Numerical normalization and linear modification (by multiplicative/additive constants) of the treatment assignment function is used to meet target parameters for the expected max value of the treatment probability. This expected value is the ``TARGET_PROPENSITY_SCORE`` parameter.

6. Numerical normalization and linear modification (by multiplicative/additive constants) of the untreated outcome function is applied to adjust the signal-to-noise ratio of the observed outcome. This is an automatic adjustment without parameterization.

7. Finally, some components of the DGP sampling occur at data generation time. On each instance of data generation:

  1. A fixed proportion of the covariates from a data source sample are sampled. This proportion is given by the ``OBSERVATION_PROBABILITY`` parameter.
  2. Observed outcome noise is sampled from a parameterized, global outcome noise distribution using the functional parameter ``sample_outcome_noise``.
  3. A parameterized target proportion of the units in the treated and control groups with the lowest/highest probability of treatment respectively are swapped between the groups. This proportion is controlled by the ``FORCED_IMBALANCE_ADJUSTMENT`` parameter.

Implementation Details
----------------------

This section of the documentation covers the details of the implementation in Maccabee.

- Pandas for data management
- Abstract Syntax Trees for equation construction
- Process-based Parallelism
- OOP practices.
