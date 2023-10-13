# Benchmark details

Two types of machines for the following benchmarks
- A100-SXM4-80G with Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz
- V100-SXM4-32G with Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz

CUDA and GPU driver
- Driver Version: 450.191.01   
- CUDA Version: 11.7

Q-Chem is running on Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz

'nan' indicates failed jobs with Q-Chem.

negative value indicates failed jobs with GPU4PySCF.

Raw data for density fitting can be found `benchmarks/df/organic`

Raw data for direct SCF can be found in `benchmarks/scf/water_clusters`

# Density fitting SCF single-point energy (v0.6.0)

## GPU4PySCF v0.6.0 on Nvidia A100-SXM4-80G vs Q-Chem 6.1 on 32 CPU threads

B3LYP with different basis

| mol               |   natm |   sto-3g |   6-31g |   def2-svp |   def2-tzvpp |   def2-tzvpd |
|:------------------|-------:|---------:|--------:|-----------:|-------------:|-------------:|
| 020_Vitamin_C     |     20 |     0.92 |    1.5  |       2.13 |         5.94 |         8.36 |
| 031_Inosine       |     31 |     4.74 |    7.12 |      10.98 |        17.02 |        21.17 |
| 033_Bisphenol_A   |     33 |     4.53 |    6.24 |       7    |        16.55 |        20.96 |
| 037_Mg_Porphin    |     37 |     7.38 |    9.9  |      13.88 |        16.9  |        23.39 |
| 042_Penicillin_V  |     42 |     5.9  |    8.19 |      11.43 |        16.41 |        20.11 |
| 045_Ochratoxin_A  |     45 |     6.94 |   10.06 |      12.9  |        15.33 |        21.62 |
| 052_Cetirizine    |     52 |     7.15 |    9.86 |      13.85 |        17.34 |        23.24 |
| 057_Tamoxifen     |     57 |     7.48 |    8.95 |      13.19 |        19.26 |        24.22 |
| 066_Raffinose     |     66 |     8.22 |   10.12 |      14.98 |        15.28 |        16.1  |
| 084_Sphingomyelin |     84 |   nan    |    9.69 |      14.83 |        17.82 |        20.33 |
| 095_Azadirachtin  |     95 |    16.06 |   17.18 |      24.22 |        23.29 |       nan    |
| 113_Taxol         |    113 |    20.11 |   18.04 |      23.38 |        24    |       nan    |
| 168_Valinomycin   |    168 |    23.43 |   19.41 |     nan    |       nan    |       nan    |

def2-tzvpp with different xc functionals

| mol               |   natm |    LDA |    PBE |   B3LYP |    M06 |   wB97m-v |
|:------------------|-------:|-------:|-------:|--------:|-------:|----------:|
| 020_Vitamin_C     |     20 |   2.86 |   6.09 |   13.11 |  11.58 |     17.46 |
| 031_Inosine       |     31 |  13.14 |  15.87 |   16.57 |  25.89 |     26.14 |
| 033_Bisphenol_A   |     33 |  12.31 |  16.88 |   16.54 |  28.45 |     28.82 |
| 037_Mg_Porphin    |     37 |  13.85 |  19.03 |   20.53 |  28.31 |     30.27 |
| 042_Penicillin_V  |     42 |  10.34 |  13.35 |   15.34 |  22.01 |     24.2  |
| 045_Ochratoxin_A  |     45 |  13.34 |  15.3  |   19.66 |  27.08 |     25.41 |
| 052_Cetirizine    |     52 |  17.79 |  17.44 |   19    |  24.41 |     25.87 |
| 057_Tamoxifen     |     57 |  14.7  |  16.57 |   18.4  |  24.86 |     25.47 |
| 066_Raffinose     |     66 |  13.77 |  14.2  |   20.47 |  22.94 |     25.35 |
| 084_Sphingomyelin |     84 |  14.24 |  12.82 |   15.96 |  22.11 |     24.46 |
| 095_Azadirachtin  |     95 |   5.58 |   7.72 |   24.18 |  26.84 |     25.21 |
| 113_Taxol         |    113 |   5.44 |   6.81 |   24.58 |  29.14 |    nan    |
| 168_Valinomycin   |    168 | nan    | nan    |  nan    | nan    |    nan    |

## GPU4PySCF v0.6.0 on Nvidia V100-SXM4-32G vs Q-Chem 6.1 on 32 CPU threads
B3LYP with different basis

| mol               |   natm |   sto-3g |   6-31g |   def2-svp |   def2-tzvpp |   def2-tzvpd |
|:------------------|-------:|---------:|--------:|-----------:|-------------:|-------------:|
| 020_Vitamin_C     |     20 |     0.52 |    0.93 |       1.23 |         3.98 |         4.88 |
| 031_Inosine       |     31 |     0.97 |    1.92 |       3.03 |         6.79 |         8.19 |
| 033_Bisphenol_A   |     33 |     1.16 |    1.89 |       2.09 |         6.72 |         8.31 |
| 037_Mg_Porphin    |     37 |     1.79 |    3.55 |       4.49 |         7.64 |        10.55 |
| 042_Penicillin_V  |     42 |     1.37 |    2.62 |       3.63 |         7.69 |         9.24 |
| 045_Ochratoxin_A  |     45 |     1.58 |    3.23 |       4.12 |         7.27 |         9.88 |
| 052_Cetirizine    |     52 |     1.83 |    3.61 |       4.72 |         8.63 |        11.32 |
| 057_Tamoxifen     |     57 |     1.92 |    3.3  |       4.59 |         9.72 |         7.87 |
| 066_Raffinose     |     66 |     2.31 |    4.04 |       5.75 |         6.09 |         5.54 |
| 084_Sphingomyelin |     84 |   nan    |    3.29 |       4.92 |         7.32 |         8    |
| 095_Azadirachtin  |     95 |     4.63 |    8.46 |      10.55 |        13.83 |       nan    |
| 113_Taxol         |    113 |     6.55 |   10.1  |       9.43 |        12.31 |       nan    |
| 168_Valinomycin   |    168 |     9.23 |   11.66 |     nan    |       nan    |       nan    |

def2-tzvpp with different xc functionals

| mol               |   natm |   LDA |   PBE |   B3LYP |   M06 |   wB97m-v |
|:------------------|-------:|------:|------:|--------:|------:|----------:|
| 020_Vitamin_C     |     20 |  1.89 |  3.3  |    8.18 |  5.95 |     10.58 |
| 031_Inosine       |     31 |  4.64 |  5.95 |    6.41 |  9.48 |     13.15 |
| 033_Bisphenol_A   |     33 |  4.85 |  6.64 |    6.58 | 11.04 |     14.72 |
| 037_Mg_Porphin    |     37 |  5.61 |  8.6  |    9.01 | 12.34 |     16.56 |
| 042_Penicillin_V  |     42 |  4.36 |  6.17 |    7.09 | 10.62 |     14.28 |
| 045_Ochratoxin_A  |     45 |  5.47 |  6.97 |    8.74 | 12.05 |     14.14 |
| 052_Cetirizine    |     52 |  8.43 |  8.51 |    9.16 | 12.44 |     15.37 |
| 057_Tamoxifen     |     57 |  6.79 |  8.41 |    9.98 | 13.44 |     15.67 |
| 066_Raffinose     |     66 |  3.22 |  4.31 |    8.11 | 10.58 |     13.22 |
| 084_Sphingomyelin |     84 |  3.34 |  3.97 |    6.52 |  8.63 |     12.11 |
| 095_Azadirachtin  |     95 |  3.35 |  4.74 |   14.29 | 16.52 |     15.05 |
| 113_Taxol         |    113 |  3.12 |  4.1  |   12.59 | 15.74 |    nan    |
# Density fitting gradient (v0.6.0)

## GPU4PySCF v0.6.0 on Nvidia A100-SXM4-80G vs Q-Chem 6.1 on 32 CPU threads

B3LYP with different basis

| mol               |   natm |   sto-3g |   6-31g |   def2-svp |   def2-tzvpp |   def2-tzvpd |
|:------------------|-------:|---------:|--------:|-----------:|-------------:|-------------:|
| 020_Vitamin_C     |     20 |     3.13 |    4.12 |       5.98 |         9.71 |        10.7  |
| 031_Inosine       |     31 |    11.42 |    9.82 |      13.23 |        16.47 |        16.16 |
| 033_Bisphenol_A   |     33 |    13.28 |   10.3  |      11.94 |        16.08 |        16.02 |
| 037_Mg_Porphin    |     37 |    13.75 |   10.54 |      15.87 |        18.33 |        19.89 |
| 042_Penicillin_V  |     42 |    13.3  |   10.7  |      14.07 |        17.2  |        18.81 |
| 045_Ochratoxin_A  |     45 |    14.68 |   11.33 |      16.28 |        19.79 |        20.94 |
| 052_Cetirizine    |     52 |    21.46 |   14.62 |      19.55 |        20.51 |        21.93 |
| 057_Tamoxifen     |     57 |    20.97 |   16.37 |      18.78 |        20.27 |        21.96 |
| 066_Raffinose     |     66 |    25.4  |   17.78 |      25.71 |        23.88 |        22.38 |
| 084_Sphingomyelin |     84 |   nan    |   17.46 |      20.9  |        23.64 |        26.52 |
| 095_Azadirachtin  |     95 |    39.13 |   32.27 |      40.78 |        39.94 |       nan    |
| 113_Taxol         |    113 |    48.57 |   42.77 |      51.57 |        49.03 |       nan    |
| 168_Valinomycin   |    168 |    87.81 |   72.58 |     nan    |       nan    |       nan    |

def2-tzvpp with different xc functionals

| mol               |   natm |    LDA |    PBE |   B3LYP |    M06 |   wB97m-v |
|:------------------|-------:|-------:|-------:|--------:|-------:|----------:|
| 020_Vitamin_C     |     20 |   5.02 |   7.04 |   10.55 |   9.28 |     11.11 |
| 031_Inosine       |     31 |   7.3  |  10.03 |   15.12 |  12.62 |     13.9  |
| 033_Bisphenol_A   |     33 |   7.58 |  11.1  |   15.55 |  12.64 |     14    |
| 037_Mg_Porphin    |     37 |   7.47 |  11.34 |   18.05 |  15.81 |     14.85 |
| 042_Penicillin_V  |     42 |   6.03 |   8.96 |   17.4  |  14.47 |     13.81 |
| 045_Ochratoxin_A  |     45 |   7.51 |   9.33 |   19.51 |  17.2  |     14.55 |
| 052_Cetirizine    |     52 |   8.32 |   9.7  |   20.8  |  16.46 |     15.7  |
| 057_Tamoxifen     |     57 |   8.91 |   9.61 |   20.61 |  16.2  |     15    |
| 066_Raffinose     |     66 |   8.52 |   9.46 |   24.2  |  18.63 |     17.13 |
| 084_Sphingomyelin |     84 |   8.51 |   9.49 |   23.62 |  21.63 |     17.66 |
| 095_Azadirachtin  |     95 |   7.69 |   9.48 |   42.24 |  34.01 |     23.93 |
| 113_Taxol         |    113 |   8.08 |   9.05 |   51.03 |  40.13 |    nan    |
| 168_Valinomycin   |    168 | nan    | nan    |  nan    | nan    |    nan    |

## GPU4PySCF v0.6.0 on Nvidia V100-SXM4-32G vs Q-Chem 6.1 on 32 CPU threads

B3LYP with different basis

| mol               |   natm |   sto-3g |   6-31g |   def2-svp |   def2-tzvpp |   def2-tzvpd |
|:------------------|-------:|---------:|--------:|-----------:|-------------:|-------------:|
| 020_Vitamin_C     |     20 |     1.43 |    2.4  |       3.3  |         5.46 |         5.54 |
| 031_Inosine       |     31 |     3.01 |    4.22 |       5.14 |         7.06 |         6.84 |
| 033_Bisphenol_A   |     33 |     3.31 |    4.18 |       4.38 |         6.75 |         6.89 |
| 037_Mg_Porphin    |     37 |     4.13 |    5.08 |       6.42 |         7.89 |         8.54 |
| 042_Penicillin_V  |     42 |     4.05 |    5.06 |       5.89 |         7.88 |         8.39 |
| 045_Ochratoxin_A  |     45 |     4.59 |    5.42 |       6.8  |         8.6  |         8.93 |
| 052_Cetirizine    |     52 |     6.11 |    7.04 |       7.97 |         9.13 |         9.53 |
| 057_Tamoxifen     |     57 |     6.17 |    8.05 |       7.74 |         9.3  |         9.22 |
| 066_Raffinose     |     66 |     7.9  |    9.49 |      10.82 |        10.51 |         9.58 |
| 084_Sphingomyelin |     84 |   nan    |    7.64 |       7.99 |         9.56 |        10.39 |
| 095_Azadirachtin  |     95 |    13.3  |   17.59 |      16.25 |        17.93 |       nan    |
| 113_Taxol         |    113 |    17.55 |   23.43 |      20.81 |        21.54 |       nan    |
| 168_Valinomycin   |    168 |    31.21 |   38.79 |     nan    |       nan    |       nan    |

def2-tzvpp with different xc functionals

| mol               |   natm |   LDA |   PBE |   B3LYP |   M06 |   wB97m-v |
|:------------------|-------:|------:|------:|--------:|------:|----------:|
| 020_Vitamin_C     |     20 |  3.19 |  4.28 |    5.9  |  4.82 |      5.84 |
| 031_Inosine       |     31 |  3.21 |  4.5  |    6.55 |  5.52 |      6.39 |
| 033_Bisphenol_A   |     33 |  3.55 |  4.87 |    6.61 |  5.51 |      6.48 |
| 037_Mg_Porphin    |     37 |  3.19 |  5.2  |    8.32 |  7.26 |      6.81 |
| 042_Penicillin_V  |     42 |  3.15 |  4.35 |    8.11 |  7.23 |      6.97 |
| 045_Ochratoxin_A  |     45 |  3.32 |  4.29 |    8.99 |  8.04 |      6.92 |
| 052_Cetirizine    |     52 |  3.51 |  4.6  |    9.41 |  8.18 |      7.57 |
| 057_Tamoxifen     |     57 |  3.86 |  4.66 |    9.56 |  8.4  |      7.51 |
| 066_Raffinose     |     66 |  3.4  |  4.32 |   10.94 |  9.4  |      8.29 |
| 084_Sphingomyelin |     84 |  3.15 |  3.81 |    9.66 |  8.97 |      8.03 |
| 095_Azadirachtin  |     95 |  3.32 |  4.37 |   18.47 | 16.01 |      1.68 |
| 113_Taxol         |    113 |  3.12 |  1.19 |   22.53 | 16.94 |    nan    |

# Density fitting hessian (v0.6.0)

coming soon..

# Direct SCF single-point energy (v0.6.0)
def2-tzvpp with different xc functionals

|   mol |   natm |    LDA |    PBE |   B3LYP |    M06 |   wB97m-v |
|------:|-------:|-------:|-------:|--------:|-------:|----------:|
|     2 |      3 |   0.22 |   0.32 |    0.27 |   0.25 |      0.69 |
|     3 |     15 |   0.68 |   0.25 |    1.58 |   2.61 |      4.84 |
|     4 |     30 |   1.59 |   2.63 |    4.09 |   6.93 |      8.17 |
|     5 |     60 |   2.86 |   3.64 |    7.15 |   8.44 |      9.44 |
|     6 |     96 |   4.34 |   4.39 |    7.75 |  10.58 |      9.87 |
|     7 |    141 |   4.07 |   4.1  |    8.87 |  10.47 |     10.13 |
|     8 |    228 |   4.34 |   4.58 |    9.39 |  10.48 |      9.36 |
|     9 |    300 |   5.05 |   5.21 |    9.35 |  11.36 |    nan    |
|    10 |    417 |   4.91 | nan    |  nan    | nan    |    nan    |

# Direct SCF gradient (v0.6.0) 
def2-tzvpp with different xc functionals

|   mol |   natm |    LDA |    PBE |   B3LYP |    M06 |   wB97m-v |
|------:|-------:|-------:|-------:|--------:|-------:|----------:|
|     2 |      3 |   0.82 |   0.89 |    0.75 |   0.82 |      0.6  |
|     3 |     15 |   0.39 |   0.19 |    1.46 |   1.52 |      1.47 |
|     4 |     30 |   0.56 |   1.04 |    2.07 |   2.25 |      1.89 |
|     5 |     60 |   0.54 |   0.87 |    2.42 |   2.4  |      1.77 |
|     6 |     96 |   0.6  |   0.87 |    2.36 |   2.51 |      1.53 |
|     7 |    141 |   0.93 |   1.1  |    2.61 |   2.59 |      1.55 |
|     8 |    228 |   1.92 |   1.9  |    3.37 |   3.39 |      1.83 |
|     9 |    300 |   2.26 |   2.02 |    3.06 |   3.59 |    nan    |
|    10 |    417 |   2.46 | nan    |  nan    | nan    |    nan    |