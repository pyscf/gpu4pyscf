# Contributing Guidelines

We welcome contributions from everyone. Please follow the guidelines below when
adding features to the library for PySCF functions running on GPU devices.

## Principles

* Functions in GPU4PySCF are meant to be performance-optimized versions of
  the features implemented in PySCF. The primary focus of this library is on
  performance. It is understandable and acceptable to have some API differences
  and incompatibilities with the PySCF CPU code.

* GPU4PySCF functions may process input data or objects created by PySCF. It is
  important to consider this possibility and perform necessary data type
  conversions in GPU4PySCF functions. However, the data or objects produced by
  GPU4PySCF do not need to be directly consumed by PySCF functions. In other
  words, you only need to ensure that GPU4PySCF supports PySCF functions and not
  the other way around.

* GPU4PySCF uses CuPy arrays as the default array container. Please ensure that
  functions in GPU4PySCF can handle any possible mixing of CuPy and NumPy
  arrays, as many NumPy ufunc operations do not support this mixing.

## Naming Conventions

* If the GPU module has a corresponding one in PySCF, please consider using
  the same function names and similar function signatures.

## Additional Suggestions

* If applicable, Please consider providing the `to_cpu` method to convert the
  GPU4PySCF object to the corresponding PySCF object, and also providing the
  `to_gpu` method in the corresponding classes in PySCF.

* Inheriting classes from the corresponding classes in PySCF is not required.
  If you choose to inherit classes from PySCF, please mute the unsupported
  methods. You can overwrite the unsupported methods with a dummy method that
  raise a `NotImplementedError`. Alternatively, you can assign the Python
  keyword `None` or `NotImplemented` to these methods.

* When writing unit tests, please consider including tests:
  - To verify the values returned by the functions against the corresponding values in PySCF.
  - To ensure proper handling of data conversions for those produced by PySCF CPU code.

* While examples or documentation are not mandatory, it is highly recommended to
  include examples of how to invoke the new module.

* CUDA compute capability 70 (sm_70) is required. Please avoid using features
  that are only available on CUDA compute capability 80 or newer. The CUDA code
  should be compiled and run using CUDA 11 and CUDA 12 toolkits.

Thank you for your contributions!
