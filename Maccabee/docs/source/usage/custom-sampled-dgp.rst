Customizing the DGP Sampling Procedure
=======================================


The benchmarking tool outlined in :doc:`sampled-dgp` provides a powerful default mechanism for the abstract sample-based benchmark process: sampling DGPs given parameter levels, sampling data from these DGPs, fitting models and calculating performance metrics for estimates produced.

However, it may be useful to customize the DGP sampling process. For example, to gain precise control over the value of the DGP sampling parameters or for the purpose of debugging/diagnosis. The code below demonstrates and end-to-end version of the flow outlined above.

.. code-block:: python

  from maccabee.data_sources import build_random_normal_datasource
  from maccabee.constants import Constants
  from maccabee.parameters import build_parameters_from_axis_levels
  from maccabee.data_generation import DataGeneratingProcessSampler

  # Build the data source
  covar_data_source = build_random_normal_datasource(
      n_covars = 10, n_observations=2000)

  # Build the parameters
  dgp_params = build_parameters_from_axis_levels({
      Constants.AxisNames.OUTCOME_NONLINEARITY: Constants.AxisLevels.LOW,
      Constants.AxisNames.TREATMENT_NONLINEARITY: Constants.AxisLevels.LOW,
  })

  # Build a DGP Sampler.
  dgp_sampler = DataGeneratingProcessSampler(
      parameters=dgp_params,
      data_source=covar_data_source)

  # Sample a DGP.
  dgp = dgp_sampler.sample_dgp()

  # Generate a dataset.
  dataset = dgp.generate_dataset()

.. code-block:: python

  from maccabee.modeling.models import LinearRegressionCausalModel

  model = LinearRegressionCausalModel(dataset)
  model.fit()


>>> dataset.ATE
1.5876

>>> model.estimate(estimand=Constants.Model.ATE_ESTIMAND)
1.5763

There are two new objects in present in this code.

1. The ``build_parameters_from_axis_levels`` takes a dictionary of parameter names and level values and builds a ``ParameterStore`` object.

2. The ``DataGeneratingProcessSampler`` takes a ``ParameterStore`` object and a ``DataSource`` and exposes the DGP sampling process through the instance method ``sample_dgp()``.

See the relevant subsections of the :doc:`/reference` section for details on how to interact with these objects.
