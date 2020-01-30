Design Principles
===================

Introducing Maccabee
---------------------

The primary purpose of the package is to facilitate the use of a new approach to Monte Carlo benchmarking for causal inference methods: the combination of empirical covariates and sampled synthetic treatment and outcome functions. Based on the theoretical work which accompanies the package, this approach should improve the validity of benchmarks by more thoroughly exploring the distributional problem space (relative to empirical benchmarks) and reducing specification bias (relative to synthetic benchmarks) while still using realistic covariate distributions.

Design Principles
------------------

Fundamentally, this package only succeeds if it provides a useful and usable way to benchmark new methods for causal inference developed by its users. Maccabee’s features are focused around four design principles to achieve this end:

* **Minimal imposition on method design:** attention has been paid to ensuring model developers can bring their own empirical data and models with very little additional effort. This includes support for benchmarking models written in both Python and R to avoid the need for language translation.

* **Quickstart but powerful customization:** very little boilerplate code is needed to run a benchmark and receive results using the data and parameters supplied with the package. This helps new users understand, and get value out of, the package quickly. At the same time, there is a large control surface to give advanced users the tools they need to support heavily-customized benchmarking processes.

* **Support for advanced functionality:** all Monte Carlo benchmarking requires access to sufficient computational power and a way to persist and compare results. Maccabee provides seamless integration with cluster computing tools to run large benchmarks on public cloud/private compute platforms as well as providing tools for result persistence and management which work both locally and with cluster computing.

* **Smooth side-by-side support of old and new approaches:** most users may feel initial discomfort using only the novel benchmarking approach proposed in the theoretical work. Maccabee allows for concrete, user-specified DGPs to be used side by side with the new approach. This allows users to switch between/compare the new and old approaches while using a single benchmarking tool. It also allows users to exploit the advanced functionality outlined above even if they don’t use the core sampling functionality. The hope is that users who start with concrete DGPs will transition to the newer (and theoretically superior) sampling approaches.
