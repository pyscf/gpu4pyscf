# Q-Chem v6.1 vs GPU4PySCF v1.0

## Direct SCF Energy with B3LYP/*

`-1` indicates that the computation is too expensive for Q-Chem.

Difference from Q-Chem v6.1:

|   mol |   sto-3g |    6-31g |   def2-svp |   def2-tzvpp |   def2-tzvpd |
|------:|---------:|---------:|-----------:|-------------:|-------------:|
|   002 | 1.41e-07 | 7.80e-08 |   4.97e-08 |     1.10e-08 |     1.01e-08 |
|   003 | 4.76e-07 | 6.28e-07 |   1.44e-07 |     1.74e-07 |     7.44e-08 |
|   004 | 3.43e-06 | 8.23e-07 |   1.30e-06 |     1.56e-06 |     2.28e-06 |
|   005 | 3.79e-05 | 5.51e-06 |   7.45e-06 |     1.49e-06 |     2.87e-06 |
|   006 | 1.04e-04 | 5.75e-05 |   5.68e-05 |     4.83e-07 |     2.47e-08 |
|   007 | 2.31e-04 | 2.11e-04 |   1.61e-04 |     2.36e-05 |     8.05e-06 |
|   008 | 3.78e-04 | 6.53e-04 |   3.98e-04 |     1.14e-04 |     1.92e-05 |
|   009 | 6.41e-04 | 1.16e-03 |   6.78e-04 |     2.24e-04 |    -1.00e+00 |
|   010 | 9.30e-04 | 2.02e-03 |   1.09e-03 |     4.74e-04 |    -1.00e+00 |

Speedup over Q-Chem v6.1:

|   mol |   sto-3g |   6-31g |   def2-svp |   def2-tzvpp |   def2-tzvpd |
|------:|---------:|--------:|-----------:|-------------:|-------------:|
|   002 |    0.000 |   0.000 |      1.135 |        0.242 |        0.158 |
|   003 |    2.636 |   0.909 |      1.519 |        1.367 |        1.637 |
|   004 |    2.731 |   3.197 |      3.033 |        2.684 |        4.464 |
|   005 |    4.633 |   4.564 |      4.983 |        4.770 |        7.540 |
|   006 |    8.928 |   6.462 |      7.570 |        5.794 |        8.443 |
|   007 |    9.645 |   6.382 |      8.953 |        6.613 |       11.092 |
|   008 |    9.193 |   5.312 |      8.497 |        6.760 |        9.769 |
|   009 |    8.273 |   5.037 |      9.411 |        6.702 |       -1.000 |
|   010 |    5.786 |   5.178 |     10.253 |        7.975 |       -1.000 |

## Direct SCF Gradient with B3LYP/*


Difference from Q-Chem v6.1:

|   mol |   sto-3g |    6-31g |   def2-svp |   def2-tzvpp |   def2-tzvpd |
|------:|---------:|---------:|-----------:|-------------:|-------------:|
|   002 | 3.00e-06 | 1.11e-06 |   9.35e-07 |     1.78e-06 |     1.49e-06 |
|   003 | 1.13e-05 | 4.89e-06 |   8.47e-06 |     5.70e-06 |     5.29e-06 |
|   004 | 1.76e-05 | 1.32e-05 |   1.15e-05 |     1.03e-05 |     1.01e-05 |
|   005 | 4.14e-05 | 2.99e-05 |   2.66e-05 |     2.74e-05 |     2.92e-05 |
|   006 | 1.00e-04 | 4.36e-05 |   4.35e-05 |     4.86e-05 |     5.08e-05 |
|   007 | 1.59e-04 | 7.61e-05 |   7.76e-05 |     7.24e-05 |     7.64e-05 |
|   008 | 2.76e-04 | 1.42e-04 |   1.57e-04 |     1.07e-04 |     1.11e-04 |
|   009 | 3.33e-04 | 1.74e-04 |   1.97e-04 |     1.31e-04 |    -1.00e+00 |
|   010 | 4.30e-04 | 2.29e-04 |   2.65e-04 |     1.61e-04 |    -1.00e+00 |

Speedup over Q-Chem v6.1:

|   mol |   sto-3g |   6-31g |   def2-svp |   def2-tzvpp |   def2-tzvpd |
|------:|---------:|--------:|-----------:|-------------:|-------------:|
|   002 |    0.122 |   1.378 |      1.158 |        0.244 |        0.253 |
|   003 |    4.140 |   2.029 |      0.709 |        0.750 |        1.070 |
|   004 |    7.218 |   4.289 |      3.093 |        1.356 |        2.636 |
|   005 |   22.362 |   8.240 |      4.141 |        1.669 |        3.320 |
|   006 |   28.631 |  11.549 |      5.022 |        1.742 |        3.282 |
|   007 |   36.823 |  14.574 |      5.829 |        1.761 |        3.255 |
|   008 |   77.679 |  19.387 |      8.060 |        1.813 |        3.040 |
|   009 |   85.545 |  23.899 |      9.672 |        1.912 |       -1.000 |
|   010 |   85.379 |  25.826 |     11.208 |        2.016 |       -1.000 |

## Direct SCF Energy with */def2-tzvpp


Difference from Q-Chem v6.1:

|   mol |       HF |      LDA |      PBE |      M06 |    B3LYP |   wB97m-v |
|------:|---------:|---------:|---------:|---------:|---------:|----------:|
|   002 | 4.62e-11 | 1.28e-08 | 1.29e-08 | 9.42e-07 | 1.10e-08 |  7.39e-08 |
|   003 | 1.01e-09 | 6.56e-08 | 2.49e-08 | 3.90e-06 | 1.74e-07 |  3.05e-07 |
|   004 | 2.98e-09 | 9.17e-07 | 5.20e-07 | 4.49e-05 | 1.56e-06 |  2.84e-06 |
|   005 | 9.13e-09 | 1.83e-06 | 1.83e-06 | 1.39e-05 | 1.49e-06 |  3.90e-06 |
|   006 | 1.73e-08 | 8.76e-07 | 1.07e-06 | 2.17e-05 | 4.83e-07 |  1.56e-05 |
|   007 | 2.86e-08 | 2.08e-05 | 2.13e-05 | 2.55e-06 | 2.36e-05 |  1.40e-05 |
|   008 | 4.96e-08 | 1.34e-04 | 1.33e-04 | 1.53e-04 | 1.14e-04 |  3.18e-05 |
|   009 | 6.80e-08 | 2.65e-04 | 2.67e-04 | 1.23e-04 | 2.24e-04 |  7.58e-05 |
|   010 | 9.56e-08 | 5.63e-04 | 5.67e-04 | 5.12e-04 | 4.74e-04 |  2.06e-04 |

Speedup over Q-Chem v6.1:

|   mol |    HF |   LDA |   PBE |   M06 |   B3LYP |   wB97m-v |
|------:|------:|------:|------:|------:|--------:|----------:|
|   002 | 0.706 | 0.123 | 0.161 | 0.158 |   0.182 |     0.459 |
|   003 | 0.921 | 0.597 | 0.694 | 1.929 |   1.368 |     2.720 |
|   004 | 1.925 | 1.174 | 1.756 | 3.736 |   2.673 |     4.879 |
|   005 | 3.310 | 1.974 | 2.362 | 5.437 |   4.777 |     5.866 |
|   006 | 3.706 | 2.884 | 3.067 | 7.113 |   5.855 |     5.819 |
|   007 | 4.106 | 2.932 | 2.959 | 7.927 |   6.632 |     6.136 |
|   008 | 4.624 | 3.178 | 3.562 | 7.401 |   6.766 |     6.481 |
|   009 | 4.893 | 3.739 | 3.993 | 8.568 |   6.755 |     6.557 |
|   010 | 5.278 | 3.576 | 3.599 | 8.733 |   8.033 |     6.556 |

## Direct SCF Gradient with */def2-tzvpp


Difference from Q-Chem v6.1:

|   mol |       HF |      LDA |      PBE |      M06 |    B3LYP |   wB97m-v |
|------:|---------:|---------:|---------:|---------:|---------:|----------:|
|   002 | 1.79e-07 | 1.46e-06 | 2.74e-06 | 3.51e-05 | 1.78e-06 |  4.00e-06 |
|   003 | 5.68e-07 | 5.26e-06 | 6.23e-06 | 2.88e-04 | 5.70e-06 |  1.72e-05 |
|   004 | 1.90e-07 | 1.11e-05 | 1.42e-05 | 5.27e-04 | 1.03e-05 |  2.77e-05 |
|   005 | 2.77e-07 | 3.25e-05 | 3.46e-05 | 1.18e-03 | 2.74e-05 |  7.74e-05 |
|   006 | 3.41e-07 | 5.83e-05 | 5.82e-05 | 1.08e-03 | 4.86e-05 |  8.82e-05 |
|   007 | 4.87e-07 | 8.73e-05 | 8.52e-05 | 1.54e-03 | 7.24e-05 |  1.24e-04 |
|   008 | 6.95e-07 | 1.25e-04 | 1.22e-04 | 1.96e-03 | 1.07e-04 |  1.79e-04 |
|   009 | 8.16e-07 | 1.53e-04 | 1.48e-04 | 2.33e-03 | 1.31e-04 |  2.22e-04 |
|   010 | 2.97e-07 | 1.89e-04 | 1.84e-04 | 2.86e-03 | 1.61e-04 |  2.69e-04 |

Speedup over Q-Chem v6.1:

|   mol |    HF |   LDA |   PBE |   M06 |   B3LYP |   wB97m-v |
|------:|------:|------:|------:|------:|--------:|----------:|
|   002 | 0.485 | 0.235 | 0.303 | 0.298 |   0.277 |     0.248 |
|   003 | 0.412 | 0.298 | 0.458 | 0.815 |   0.743 |     0.819 |
|   004 | 0.627 | 0.338 | 0.582 | 1.383 |   1.446 |     1.165 |
|   005 | 0.618 | 0.352 | 0.542 | 1.679 |   1.647 |     1.179 |
|   006 | 0.564 | 0.391 | 0.579 | 1.738 |   1.735 |     1.063 |
|   007 | 0.528 | 0.461 | 0.608 | 1.770 |   1.804 |     0.994 |
|   008 | 0.519 | 0.641 | 0.730 | 1.845 |   1.830 |     0.955 |
|   009 | 0.491 | 0.821 | 0.888 | 1.919 |   1.938 |     0.959 |
|   010 | 0.474 | 0.964 | 1.074 | 2.021 |   2.036 |     0.932 |
