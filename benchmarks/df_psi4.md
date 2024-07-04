# Psi4 v1.8 vs GPU4PySCF v1.0

## DF SCF Energy with B3LYP/*


Difference from Psi4 v1.8:

| mol               |   sto-3g |    6-31g |   def2-svp |   def2-tzvpp |   def2-tzvpd |
|:------------------|---------:|---------:|-----------:|-------------:|-------------:|
| 020_Vitamin_C     | 3.94e-06 | 2.92e-06 |   3.87e-06 |     3.97e-06 |     6.44e-06 |
| 031_Inosine       | 5.60e-06 | 7.62e-06 |   7.70e-06 |     8.83e-06 |     1.11e-05 |
| 033_Bisphenol_A   | 6.17e-06 | 4.85e-06 |   7.90e-06 |     6.77e-06 |     8.91e-06 |
| 037_Mg_Porphin    | 4.61e-06 | 6.88e-05 |   1.27e-06 |     1.98e-07 |     5.48e-07 |
| 042_Penicillin_V  | 4.28e-06 | 4.41e-06 |   5.73e-06 |     7.50e-06 |     1.00e-05 |
| 045_Ochratoxin_A  | 7.56e-06 | 7.76e-06 |   9.46e-06 |     9.32e-06 |     1.22e-05 |
| 052_Cetirizine    | 1.25e-05 | 1.29e-05 |   1.37e-05 |     1.89e-05 |     2.23e-05 |
| 057_Tamoxifen     | 6.77e-06 | 6.21e-06 |   8.05e-06 |     8.46e-06 |     1.10e-05 |
| 066_Raffinose     | 1.29e-06 | 9.27e-07 |   3.81e-06 |     2.70e-07 |     4.43e-06 |
| 084_Sphingomyelin | 9.02e-06 | 1.04e-05 |   1.15e-05 |     1.39e-05 |     1.58e-05 |

Speedup over Psi4 v1.8:

| mol               |   sto-3g |   6-31g |   def2-svp |   def2-tzvpp |   def2-tzvpd |
|:------------------|---------:|--------:|-----------:|-------------:|-------------:|
| 020_Vitamin_C     |    2.036 |   2.303 |      1.682 |        6.050 |        7.489 |
| 031_Inosine       |    3.043 |   4.348 |      4.568 |       10.717 |       14.537 |
| 033_Bisphenol_A   |    3.647 |   4.340 |      4.592 |       12.260 |       14.499 |
| 037_Mg_Porphin    |    4.653 |   6.469 |      5.975 |       12.207 |       15.290 |
| 042_Penicillin_V  |    4.402 |   5.982 |      5.168 |       11.653 |       14.475 |
| 045_Ochratoxin_A  |    4.683 |   6.738 |      5.254 |       11.583 |       14.132 |
| 052_Cetirizine    |    5.579 |   6.838 |      8.951 |       14.425 |       18.188 |
| 057_Tamoxifen     |    5.910 |   8.096 |      8.857 |       13.108 |       15.729 |
| 066_Raffinose     |    7.403 |   9.194 |      9.873 |       12.816 |       15.032 |
| 084_Sphingomyelin |    4.357 |   8.717 |     10.092 |       13.111 |       14.858 |

## DF Gradient with B3LYP/*


Difference from Psi4 v1.8:

| mol               |   sto-3g |    6-31g |   def2-svp |   def2-tzvpp |   def2-tzvpd |
|:------------------|---------:|---------:|-----------:|-------------:|-------------:|
| 020_Vitamin_C     | 2.05e-05 | 1.53e-05 |   1.63e-05 |     1.54e-05 |     1.60e-05 |
| 031_Inosine       | 1.90e-05 | 3.74e-05 |   2.21e-05 |     1.68e-05 |     1.78e-05 |
| 033_Bisphenol_A   | 1.44e-05 | 1.57e-05 |   1.69e-05 |     1.48e-05 |     1.48e-05 |
| 037_Mg_Porphin    | 3.39e-05 | 3.85e-05 |   3.36e-05 |     3.88e-05 |     3.80e-05 |
| 042_Penicillin_V  | 3.33e-05 | 2.76e-05 |   2.86e-05 |     2.45e-05 |     2.69e-05 |
| 045_Ochratoxin_A  | 2.75e-05 | 2.11e-05 |   2.65e-05 |     2.47e-05 |     2.55e-05 |
| 052_Cetirizine    | 2.58e-05 | 2.30e-05 |   2.43e-05 |     2.38e-05 |     2.89e-05 |
| 057_Tamoxifen     | 3.21e-05 | 1.87e-05 |   2.40e-05 |     1.94e-05 |     2.67e-05 |
| 066_Raffinose     | 3.88e-05 | 3.40e-05 |   3.65e-05 |     3.86e-05 |     3.57e-05 |
| 084_Sphingomyelin | 8.62e-05 | 6.58e-05 |   6.80e-05 |     6.69e-05 |     6.91e-05 |

Speedup over Psi4 v1.8:

| mol               |   sto-3g |   6-31g |   def2-svp |   def2-tzvpp |   def2-tzvpd |
|:------------------|---------:|--------:|-----------:|-------------:|-------------:|
| 020_Vitamin_C     |   11.820 |  17.117 |     11.993 |       23.867 |       27.485 |
| 031_Inosine       |   18.390 |  25.608 |     17.521 |       36.825 |       43.393 |
| 033_Bisphenol_A   |   14.898 |  22.793 |     14.532 |       33.466 |       39.945 |
| 037_Mg_Porphin    |   18.411 |  25.384 |     15.312 |       34.411 |       38.112 |
| 042_Penicillin_V  |   22.417 |  28.357 |     17.625 |       39.145 |       41.711 |
| 045_Ochratoxin_A  |   23.806 |  29.375 |     17.688 |       38.940 |       42.077 |
| 052_Cetirizine    |   23.829 |  30.125 |     32.081 |       42.579 |       46.687 |
| 057_Tamoxifen     |   22.314 |  32.323 |     35.035 |       39.943 |       44.103 |
| 066_Raffinose     |   26.787 |  36.922 |     37.953 |       42.626 |       44.503 |
| 084_Sphingomyelin |   36.525 |  39.186 |     44.866 |       56.275 |       54.244 |

## DF Hessian with B3LYP/*


Difference from Psi4 v1.8:

| mol   |
|-------|

Speedup over Psi4 v1.8:

| mol   |
|-------|

## DF SCF Energy with */def2-tzvpp


Difference from Psi4 v1.8:

| mol               |      PBE |      M06 |    B3LYP |
|:------------------|---------:|---------:|---------:|
| 020_Vitamin_C     | 1.33e-06 | 9.42e-06 | 3.97e-06 |
| 031_Inosine       | 3.94e-06 | 6.23e-06 | 8.83e-06 |
| 033_Bisphenol_A   | 8.92e-08 | 1.21e-05 | 6.77e-06 |
| 037_Mg_Porphin    | 1.18e-05 | 3.36e-04 | 1.98e-07 |
| 042_Penicillin_V  | 1.10e-06 | 2.82e-05 | 7.50e-06 |
| 045_Ochratoxin_A  | 2.84e-06 | 7.09e-07 | 9.32e-06 |
| 052_Cetirizine    | 1.97e-05 | 5.55e-05 | 1.89e-05 |
| 057_Tamoxifen     | 4.40e-07 | 5.13e-05 | 8.46e-06 |
| 066_Raffinose     | 1.75e-05 | 4.66e-05 | 2.70e-07 |
| 084_Sphingomyelin | 6.49e-06 | 1.85e-05 | 1.39e-05 |

Speedup over Psi4 v1.8:

| mol               |    PBE |    M06 |   B3LYP |
|                   |    scf |    scf |     scf |
|:------------------|-------:|-------:|--------:|
| 020_Vitamin_C     |  5.914 |  8.831 |   6.799 |
| 031_Inosine       |  8.944 | 11.555 |  10.830 |
| 033_Bisphenol_A   | 10.984 | 13.277 |  12.233 |
| 037_Mg_Porphin    | 12.392 | 14.231 |  12.043 |
| 042_Penicillin_V  | 10.971 | 12.977 |  11.411 |
| 045_Ochratoxin_A  | 12.511 | 12.896 |  10.330 |
| 052_Cetirizine    | 15.735 | 15.302 |  14.497 |
| 057_Tamoxifen     | 14.090 | 15.049 |  12.898 |
| 066_Raffinose     | 13.939 | 14.076 |  12.827 |
| 084_Sphingomyelin | 13.738 | 14.585 |  13.022 |

## DF Gradient with */def2-tzvpp


Difference from Psi4 v1.8:

| mol               |      PBE |      M06 |    B3LYP |
|:------------------|---------:|---------:|---------:|
| 020_Vitamin_C     | 1.64e-05 | 9.86e-05 | 1.54e-05 |
| 031_Inosine       | 1.80e-05 | 1.95e-04 | 1.69e-05 |
| 033_Bisphenol_A   | 1.57e-05 | 1.50e-04 | 1.50e-05 |
| 037_Mg_Porphin    | 5.27e-05 | 6.12e-04 | 3.86e-05 |
| 042_Penicillin_V  | 2.51e-05 | 2.70e-04 | 2.47e-05 |
| 045_Ochratoxin_A  | 2.76e-05 | 2.31e-04 | 2.45e-05 |
| 052_Cetirizine    | 2.47e-05 | 2.19e-04 | 2.38e-05 |
| 057_Tamoxifen     | 1.93e-05 | 2.48e-04 | 2.20e-05 |
| 066_Raffinose     | 3.12e-05 | 4.36e-04 | 3.85e-05 |
| 084_Sphingomyelin | 7.20e-05 | 3.89e-04 | 6.69e-05 |

Speedup over Psi4 v1.8:

| mol               |    PBE |    M06 |   B3LYP |
|:------------------|-------:|-------:|--------:|
| 020_Vitamin_C     | 26.538 | 33.006 |  24.628 |
| 031_Inosine       | 33.583 | 34.866 |  36.376 |
| 033_Bisphenol_A   | 35.681 | 45.892 |  34.088 |
| 037_Mg_Porphin    | 33.516 | 45.192 |  34.262 |
| 042_Penicillin_V  | 36.408 | 48.302 |  38.405 |
| 045_Ochratoxin_A  | 40.876 | 46.000 |  39.111 |
| 052_Cetirizine    | 44.134 | 50.615 |  41.878 |
| 057_Tamoxifen     | 37.867 | 51.285 |  40.409 |
| 066_Raffinose     | 38.103 | 48.984 |  42.362 |
| 084_Sphingomyelin | 46.478 | 66.586 |  55.514 |

## DF Hessian with */def2-tzvpp


Difference from Psi4 v1.8:

| mol   |
|-------|

Speedup over Psi4 v1.8:

| mol   |
|-------|